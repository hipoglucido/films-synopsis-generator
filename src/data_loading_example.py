import sqlite3
import pandas as pd
import settings
import os
import re
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

#import cPickle as pickle
from sklearn.preprocessing import MultiLabelBinarizer



EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        #self.encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
        self.variable_initializer()
        
    def tokenize(self,s):
        return ' '.join(re.findall(r"[\w]+|[^\s\w]", s))
        
    def variable_initializer(self):
        self.df = pd.read_csv(filepath_or_buffer  = os.path.join(settings.RAW_DATA_DIR,'synopsis_genres.csv'),sep = '#',encoding = 'latin_1',index_col = 'ID',nrows = 50)
        df = self.df
        df.info()
        print(df.head())
        nb_samples = df.shape[0]
        self.caps = list(df['Synopsis'].map(self.tokenize).values)
        print(self.caps)
        caps = self.caps
        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print("Vocabulary size: "+str(self.vocab_size))
        print("Maximum caption length: "+str(self.max_cap_len))
        print("Variables initialization done!")
        print("Processing genres...")
        self.mlb = MultiLabelBinarizer()
        self.genres_features = self.mlb.fit_transform(df['Genre'].map(lambda x: x.split('|')))
        print(len(self.mlb.classes_)," different genres found:",str(self.mlb.classes_)[:100],"...")
        
        print(self.genres_features.shape)
    def to_genre(self, x):
        return '|'.join(self.mlb.inverse_transform(x[None,:])[0])
    def to_text(self, x):
        return ' '.join([self.index_word[i] for i in x if i != 0])
    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        caps = []
        imgs = []
        
        imgs = self.genres_features
        caps = self.caps
        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = imgs[image_counter]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next = np.zeros(self.vocab_size)
                    next[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print("yielding count: "+str(gen_count))
                        print(len(images),images[0].shape,partial_caps.shape,next_words.shape)
                        #print(next_words.mean())
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        #base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        #base_model.trainable=False
        image_model = Sequential()
        #image_model.add(base_model)
        #image_model.add(Flatten())
        image_model.add(Dense(EMBEDDING_DIM, input_dim = len(self.mlb.classes_), activation='relu'))

        image_model.add(RepeatVector(self.max_cap_len))

        lang_model = Sequential()
        lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(LSTM(256,return_sequences=True))
        lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

        model = Sequential()
        model.add(Merge([image_model, lang_model], mode='concat'))
        model.add(LSTM(1000,return_sequences=False))
        model.add(Dense(self.vocab_size))
        model.add(Activation('softmax'))

        print("Model created!")

        if(ret_model==True):
            return model

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def get_word(self,index):
        return self.index_word[index]    

        
        
from keras.callbacks import ModelCheckpoint

def train_model(weight = None, batch_size=32, epochs = 10):

    cg = CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    print(cg.total_samples/batch_size)
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list)
    try:
        model.save('WholeModel.h5', overwrite=True)
        model.save_weights('Weights.h5',overwrite=True)
    except:
        print("Error in saving model.")
    print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=50)
'''
c = CaptionGenerator()

from time import sleep 
t = 5000
for a,b in c.data_generator():
    t-=1
    for i in range(a[0].shape[0]):
        print(c.to_genre(a[0][i]))
        print(c.to_text(a[1][i]))
        print(c.to_text(np.nonzero(b[i])[0]))
        print('_______________________________________')
    
    #sleep(0.1)
'''