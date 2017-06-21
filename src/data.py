"""
Data loading, preprocessing and generation
"""

import settings

import os
import numpy as np
import re

from keras.preprocessing import sequence
from sklearn.preprocessing import MultiLabelBinarizer

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
        
        #Read dataset into DataFrame
        import pandas as pd
        df = pd.read_csv(filepath_or_buffer  = os.path.join(settings.DATA_DIR,'synopsis_genres.csv'),sep = '#',encoding = 'latin_1',index_col = 'ID')
        df = df[df['Synopsis'].notnull() & df['Genre'].notnull()]
        df.info()
        settings.logger.info(str(df.head()).encode('latin1'))
        
        #Keep the synopsis as a list
        self.synopses = list(df['Synopsis'].map(self.tokenize).values)

        corpus_tokens_count=0
        for synopsis in self.synopses:
            corpus_tokens_count += len(synopsis.split())-1
        settings.logger.info("Total tokens in corpus after preprocessing : "+str(corpus_tokens_count))
        
        #Calculate unique words
        words = [txt.split() for txt in self.synopses]
        unique = []
        for word in words:
            unique.extend(word)
        unique = list(set(unique))
        settings.logger.info("Unique words in corpus: "+str(len(unique)))
        if settings.VOCABULARY_SIZE is None:
            settings.VOCABULARY_SIZE = len(unique)
        else:
            #TODO max vocabulary restriction
            settings.logger.warning("Vocabulary size restriction not implemented yet, \
                                    using full vocabulary instead...")
            settings.VOCABULARY_SIZE = len(unique)
        settings.logger.info("Vocabulary size: "+str(settings.VOCABULARY_SIZE))
        
        settings.logger.info("Building word indexes")
        #print(str(unique).encode('latin1'))
        self.word_to_index = {}
        self.index_to_word = {}
        for i, word in enumerate(unique):
            self.word_to_index[word]=i
            self.index_to_word[i]=word


        settings.logger.info("Maximum synopsis length allowed: "+str(settings.MAX_SYNOPSIS_LEN))

        
        #Preprocess genres (multilabel)
        self.mlb = MultiLabelBinarizer()
        self.genres_features = self.mlb.fit_transform(df['Genre'].map(lambda x: x.split('|')))
        settings.logger.info(str(len(self.mlb.classes_))+" different genres found:"+str(self.mlb.classes_)[:100]+"...")
        #I am not sure when to use .encode('latin1')
        settings.logger.debug('Genres vector shape: '+str(self.genres_features.shape))        
        
    def tokenize(self,s):
        """
        Tokenize the synopsis, example:
        'HOLAA!! ¿Qué tal estás?'  -> 'holaa ! ! ¿ qué tal estás ?'
        """
        return ' '.join(re.findall(r"[\w]+|[^\s\w]", s)).lower()

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

