"""
Data loading, preprocessing and generation
"""

import settings

import os
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.externals import joblib

class Generator():

    def __init__(self):
        """
        Loads the films csv (genres and synopsis)
        Preprocesses both of them and keeps them in memory
        Initialises
        """
        #Attributes list (initialised afterwards in this method)
        self.index_to_word = None #{ 3:'hola', 3002: 'cielo' ...
        self.word_to_index = None #{'hola': 3, 'cielo': 3002, ...
        self.mlb = None
        self.genres_features = None
        self.synopses = None
    def preprocess_and_save(self):
        df = self.load_dataset()
        
        self.preprocess_synopses(df)
        #self.load_genre_binarizer()
        self.preprocess_genres(df)
        self.filter_dataset()
        
        
    def initialize(self):
        #Read dataset into DataFrame

        
        settings.logger.info("Vocabulary size: "+str(settings.VOCABULARY_SIZE))
        
        settings.logger.info("Building word indexes")
        #print(str(unique).encode('latin1'))
        self.word_to_index = {}
        self.index_to_word = {}
        for i, word in enumerate(unique):
            self.word_to_index[word]=i
            self.index_to_word[i]=word

        settings.logger.info("Maximum synopsis length allowed: "+str(settings.MAX_SYNOPSIS_LEN))
        
    def preprocess_synopses(self, df):
        settings.logger.info("Preprocessing synopses...")
        #Keep the synopsis as a list
        
        self.synopses = list(df['Synopsis'].map(self.tokenize).values)
        
        from collections import defaultdict
            
        word_freqs = defaultdict(int)
        for synopsis in self.synopses:
            for word in synopsis.split():
                word_freqs[word] += 1
                
        word_freqs = list(word_freqs.items())
        most_frequent = sorted(word_freqs, key = lambda x: x[1], reverse = True)
        settings.logger.info("Most frequent words: " + str(most_frequent[:10]))
        
        most_frequent = most_frequent[:settings.VOCABULARY_SIZE]
        knwown_words = [w[0] for w in most_frequent]
        settings.logger.info("Only "+str(len(knwown_words))+" words will be considered (VOCABULARY_SIZE)")
        
        #Substitute any unkown word with <unk>
        def map_unkown_tokens(synopsis):
            new_synopsis = []
            for word in synopsis.split():
                if word in knwown_words:
                    new_synopsis.append(word)
                else:
                    new_synopsis.append(settings.UNKNOWN_TOKEN)
            return new_synopsis
        self.synopses = [map_unkown_tokens(synopsis) for synopsis in self.synopses]
        
    def filter_dataset(self):
        """
        This is meant to be the last preprocessing step. It reduces the
        dataset and should be called after preprocess_synopsis and
        preprocess_genres
        """
        for genres, synopsis in zip(self.genres, self.synopses):
            print(genres, synopsis)
        #settings.logger.info("Total tokens in corpus after preprocessing : "+str(corpus_tokens_count))
        
        
    
    def encode_genres(self):   
        #Preprocess genres (multilabel)
        self.genres_features = self.mlb.transform(self.genres)
        settings.logger.info(str(len(self.mlb.classes_))+" different genres found:"+str(self.mlb.classes_)[:100]+"...")
        #I am not sure when to use .encode('latin1')
        settings.logger.debug('Genres vector shape: '+str(self.genres_features.shape)) 
 
    
    def load_dataset(self):
        import pandas as pd
        if settings.USE_SMALL_DATASET:
            nrows = 100
        else:
            nrows = None
        df = pd.read_csv(filepath_or_buffer  = os.path.join(settings.DATA_DIR,'synopsis_genres.csv'),sep = '#',encoding = 'latin_1',index_col = 'ID', nrows = nrows)
        df = df[df['Synopsis'].notnull() & df['Genre'].notnull()]
        settings.logger.info(str(df.info()))
        #settings.logger.info(str(df[['Genre','Synopsis']][:5]).encode('latin1'))
        return df
        
    def generate_embedding_weights(self):
        df = self.load_dataset()
        self.preprocess_synopses(df)
        
        settings.logger.info('Generating word-to-index dictionary...')
        id_count = 1 #0 is reserved for the masking
        word_to_index = {}
        for synopsis in self.synopses:
            for word in synopsis:
                if word not in word_to_index:
                    word_to_index[word] = id_count
                    id_count += 1
        joblib.dump(word_to_index, settings.WORD_TO_INDEX_PATH)
        settings.logger.info('Loading Word2Vec model from '+settings.WORD2VEC_MODEL_PATH)
        model = pd.read_csv(settings.WORD2VEC_MODEL_PATH, sep = ' ', header = None, \
                            index_col = 0, skiprows = 1, nrows = 2000)
        settings.logger.info('Generating embedding weights matrix...')
        vocab_dim = 300 # dimensionality of your word vectors
        n_symbols = len(word_to_index) + 1 # adding 1 to account for 0th index (for masking)
        embedding_weights = np.zeros((n_symbols+1,vocab_dim))
        for word,index in word_to_index.items():
            try:
                embedding_weights[index,:] = model.loc[word].values
            except:
                print('NOT in w2v',word)
        file = pd.HDFStore(settings.EMBEDDING_WEIGHTS_PATH)
        file.append("embedding_weights", embedding_weights)
        file.close()
        
    def preprocess_genres(self, df):
        
        self.genres = list(df['Genre'].map(lambda x: x.split('|')))
        
        from collections import defaultdict
            
        genre_freqs = defaultdict(int)
        for genres_ in self.genres:
            for genre in genres_:
                genre_freqs[genre] += 1
        
        genre_freqs = list(genre_freqs.items())        
        most_frequent = sorted(genre_freqs, key = lambda x: x[1], reverse = True)
        settings.logger.info("Most frequent genres: " + str(most_frequent[:10]))
        
        knwown_genres = [g[0] for g in most_frequent][:settings.MAX_GENERES]
        
        settings.logger.info("Only "+str(len(knwown_genres))+" genres will be considered (MAX_GENERES)")
        
        def delete_unkown_genres(fgenres):
            return [genre for genre in fgenres if genre in knwown_genres]
            
        self.genres = [delete_unkown_genres(fgenres) for fgenres in self.genres]      
        
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(self.genres)
        #settings.logger.info(str(len(self.mlb.classes_))+" different genres found:"+str(self.mlb.classes_)[:100]+"...")
        #I am not sure when to use .encode('latin1')        
        filepath = os.path.join(settings.OTHERS_DIR, 'genre_binarizer_'+str(len(self.mlb.classes_))+'_classes.pkl')
        joblib.dump(self.mlb, filepath)
        settings.logger.info(filepath+' saved')
        
        
    def load_genre_binarizer(self):
        filepath = settings.GENRE_BINARIZER_PATH
        self.mlb = joblib.load(filepath)
        settings.logger.info(filepath+' loaded')
        
    def tokenize(self,s):
        """
        Tokenize the synopsis, example:
        'HOLAA!! ¿Qué tal estás?'  -> 'holaa ! ! ¿ qué tal estás ?<eos>'
        """
        return ' '.join(re.findall(r"[\w]+|[^\s\w]", s)).lower() + settings.EOS_TOKEN

    def to_genre(self, vector):
        """
        [0,0,1,0,1...] -> 'drama|comedia'
        """
        return '|'.join(self.mlb.inverse_transform(vector[None,:])[0])

    def to_synopsis(self, vectors):
        """
        Converts a vector of words (i.e. a vector of vector) into its
        text representation
        """
        return ' '.join([self.index_to_word[i] for i in vectors if i != 0])

    def generate(self):
        """
        Generate batches to feed the network.
        Batches are comprised of ((genre, previous_words), next_words))
        """
        from keras.preprocessing import sequence
        
        #Initialize batch variables
        previous_words_batch = []
        next_word_batch = []
        genres_batch = []
        
        settings.logger.info("Generating data...")
        
        #Keep track of how many batches have been fed
        batches_fed_count = 0 
        #Keep track of the current batch that is being built
        current_batch_size = 0
        while 1:
            synopsis_counter = -1
            #Iterate over all the synopsis
            
            for synopsis in self.synopses:
                synopsis_counter+=1
                genre = self.genres_features[synopsis_counter]
                #Itearte over synopsis' words
                splitted_synopsis = synopsis.split()
                
                for i in range(len(splitted_synopsis)-1):
                    #Grab next word and add it to the current batch
                    next_word = synopsis.split()[i+1]
                    next = np.zeros(settings.VOCABULARY_SIZE)
                    next[self.word_to_index[next_word]] = 1
                    next_word_batch.append(next)
                    
                    #Grab previous words and add them to the current batch
                    previous_words = [self.word_to_index[word] for word in splitted_synopsis[:i+1]]
                    previous_words_batch.append(previous_words)
                    
                    #Add the genre to the batch
                    genres_batch.append(genre)
                    
                    #Increment batch size
                    current_batch_size+=1
                    
                    
                    if current_batch_size<settings.BATCH_SIZE:
                        #Keep building the batch
                        continue
                    #Batch is ready
                    next_word_batch = np.asarray(next_word_batch)
                    genres_batch = np.asarray(genres_batch)
                    
                    #Padd previous words of synopses
                    previous_words_batch = sequence.pad_sequences(previous_words_batch, maxlen=settings.MAX_SYNOPSIS_LEN, padding='post')
                    #print(len(genres_batch),genres_batch[0].shape,previous_words_batch.shape,next_word_batch.shape)
                    #print(next_word_batch.mean())
                    if 0:
                        for j in range(batch_size):
                            print(str(self.to_genre(genres_batch[j])).encode('latin1'))
                            print(str(self.to_synopsis(previous_words_batch[j])).encode('latin1'))
                            print(str(self.to_synopsis(np.nonzero(next_word_batch[j])[0])).encode('latin1'))
                            print("************")
                        print("____________________________________________")
                    #Yield batch
                    yield [[genres_batch, previous_words_batch], next_word_batch]
                    batches_fed_count+=1
                    #settings.logger.info("Batches yielded: "+str(batches_fed_count))
                    
                    #Reset variables
                    previous_words_batch = []
                    next_word_batch = []
                    genres_batch = []
                    current_batch_size = 0

