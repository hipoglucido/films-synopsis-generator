import os
import preprocessor
import generator
import settings
import model
from sklearn.externals import joblib
import pickle as pk
import numpy as np
from sklearn.model_selection import train_test_split
import random
from keras.preprocessing import sequence

def test_generator():

    synopses, genres = load_preprocessed_data(settings.INPUT_PREPROCESSED_FILMS)
    print(synopses[:20])
    print(genres[:20])
    X_train, X_val, y_train, y_val = train_test_split(
        synopses, genres, test_size=settings.VALIDATION_SPLIT)
    c= generator.Generator(X_train, y_train)
    c.load_genre_binarizer()
    c.load_indexes()
    #a = g.generate().__next__()
    #g.get_train_val_generators()
    from time import sleep

    while 1:
        for a,b in c.generate():
            print(a[0].shape,a[1].shape,b.shape)
            continue
            for i in range(a[0].shape[0]):
                s = str(c.to_synopsis(a[1][i]))
                if len(s) > 100:
                    print(s)
                continue
                print(c.to_genre(a[0][i]),a[0][i].shape)
                print(c.to_synopsis(a[1][i]),len(a[1][i]),type(a[1][i][0]))
                print(c.index_to_word[b[i]],type(b[i]))
                print('_______________________________________')

            
def generate_files():

    p = preprocessor.Preprocessor()
    df = p.load_dataset()
    
    p.preprocess_synopses(df)
    p.preprocess_genres(df)
    p.build_indexes()
    if settings.USE_W2V:
        p.generate_embedding_weights()
    p.filter_dataset()
    p.encode_genres()
    p.encode_synopses()
    p.save_data()

def check_paths():
    if not os.path.exists(settings.DATA_DIR):
        os.makedirs(settings.DATA_DIR)

    if not os.path.exists(settings.OTHERS_DIR):
        os.makedirs(settings.OTHERS_DIR)

    if not os.path.exists(settings.WEIGHTS_DIR):
        os.makedirs(settings.WEIGTHS_DIR)

    if not os.path.exists(settings.TENSORBOARD_LOGS_DIR):
        os.makedirs(settings.TENSORBOARD_LOGS_DIR)

def check_nltk_resources():
    nltk.download('averaged_perceptron_tagger')


def train_network():

    synopses, genres = load_preprocessed_data(settings.INPUT_PREPROCESSED_FILMS)
    X_train, X_val, y_train, y_val = train_test_split(
        synopses, genres, test_size=settings.VALIDATION_SPLIT)

    network = model.Network()
    network.load_generators(X_train, X_val, y_train, y_val)       # Synopses and genres as parameter
    
    network.build()
    #network.load_weights()
    network.compile()
    network.train()


def load_preprocessed_data(path):
    """
    Loads preprocessed lists of synopses and genres
    """
    films_preprocessed = joblib.load(path)
#    films_preprocessed = pk.load(path)
    genres = films_preprocessed[0]
    synopses = films_preprocessed[1]
    settings.logger.info("Loaded preprocessed films from " + str(path))
    return synopses, genres
    
def get_predictions_greedy(g, n, encoded_genres):
    print("Greedy search mode")
    previous_words = [g.word_to_index[sample_start()]]
    for i in range(settings.MAX_SYNOPSIS_LEN):
        padded_previous_words = sequence.pad_sequences([previous_words], maxlen=settings.MAX_SYNOPSIS_LEN, padding='post', value = g.word_to_index[settings.PAD_TOKEN])
        next_word_probs = n.model.predict([encoded_genres,padded_previous_words])[0]
        sorted_words = np.argsort(next_word_probs)
        best_word = sorted_words[-1]
        previous_words.append(best_word)
        #print(g.to_synopsis(previous_words))
        #print(next_word_probs)
        #print(next_word_probs.sum())
        #print(previous_words.shape)
        #print(encoded_genres.shape)
    previous_words = g.to_synopsis(previous_words)
    return previous_words
def sample_start():
    possible_starts = ['la','el','en','durante','cuando','son','las','eran']
    start = random.sample(possible_starts, 1)[0]
    return start
        
def get_predictions_beam(g, n, encoded_genres):
    try:
        beam_size = int(input("Introduce an integer for the beamsize: "))
    except:
        get_predictions_beam(g, n, encoded_genres)
    model = n.model
    start = [g.word_to_index[sample_start()]]
    synopses = [[start,0.0]]
    while(len(synopses[0][0]) < 150):
        temp_synopses = []
        for synopsis in synopses:
            
            partial_synopsis = sequence.pad_sequences([synopsis[0]], maxlen=settings.MAX_SYNOPSIS_LEN, padding='post', value = g.word_to_index[settings.PAD_TOKEN])
            next_words_pred = model.predict([encoded_genres, np.asarray(partial_synopsis)])[0]
            next_words = np.argsort(next_words_pred)[-beam_size:]
            for word in next_words:
                new_partial_synopsis, new_partial_synopsis_prob = synopsis[0][:], synopsis[1]
                new_partial_synopsis.append(word)
                new_partial_synopsis_prob+=next_words_pred[word]
                temp_synopses.append([new_partial_synopsis,new_partial_synopsis_prob])
        synopses = temp_synopses
        synopses.sort(key = lambda l:l[1])
        synopses = synopses[-beam_size:]
    synopses = [g.to_synopsis(s[0]) for s in synopses]
    return synopses

def get_predictions(g, n):
    possible_genres = list(g.mlb.classes_)
    print("Possible film genres: ",','.join(possible_genres)) 
    input_line = input("Insert a comma separated set of genres (r for random): ")
    randomly = input_line == 'r'
    if randomly:
        n_genres = random.randint(1,7)
        input_genres = random.sample(possible_genres, n_genres)
    else:
        input_genres = input_line.split(',')
        for ig in input_genres:
            if ig not in possible_genres:
                print(ig + " is not a possible genre")
                get_predictions(g, n)
    print("Input genres: ",', '.join(input_genres))
    encoded_genres = g.mlb.transform([input_genres])
    mode = input("Input g or b for greedy or beam search mode: ")
    if mode == 'g':
        syn = [get_predictions_greedy(g, n, encoded_genres)]
    else:
        syn = get_predictions_beam(g, n, encoded_genres)
    for s in syn:
        print("Synopsis: ",s)
    get_predictions(g, n)
    
def interface():
    settings.logger.info("Starting user interface...")
    n = model.Network()
    n.load_embeddings()
    n.build()
    n.load_weights()
    g = generator.Generator(None, None)
    g.load_indexes()
    g.load_genre_binarizer()
    get_predictions(g, n)

if __name__ == '__main__':
    #check_nltk_resources()
    #check_paths()
    #generate_files()
    #test_generator()
    train_network()
    #interface()
