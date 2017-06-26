
import settings

import os
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from sklearn.externals import joblib

from time import strftime


class Generator():
    def __init__(self, synopses, genres):
        self.synopses = synopses
        self.genres = genres

    # def get_train_val_generators(self, ):
    #
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         self.synopses, self.genres, test_size = settings.VALIDATION_SPLIT, random_seed = 42)
    #     generator_train = self.generate(X_train, y_train)
    #     #self.generator_train.initialize()
    #     generator_val = self.generate(X_test, y_test)
    #     return generator_train, generator_vala

    def load_indexes(self):
        self.word_to_index = joblib.load(os.path.join(settings.WORD_TO_INDEX_PATH))
        self.index_to_word = joblib.load(os.path.join(settings.INDEX_TO_WORD_PATH))
        settings.logger.info("Loaded index dictionaries")

    def load_genre_binarizer(self):
        filepath = settings.GENRE_BINARIZER_PATH
        self.mlb = joblib.load(filepath)
        settings.logger.info(filepath + ' loaded')

    def to_genre(self, vector):
        """
        [0,0,1,0,1...] -> 'drama|comedia'
        """

        return '|'.join(self.mlb.inverse_transform(vector[None, :])[0])

    def to_synopsis(self, vector):
        """
        Converts a vector of words (i.e. a vector of vectors) into its
        text representation
        """
        return ' '.join([self.index_to_word[i] for i in vector])

    def generate(self):
        """
        Generate batches to feed the network.
        Batches are comprised of ((genre, previous_words), next_words))
        """
        # from keras.preprocessing import sequence
        # Initialize batch variables
        previous_words_batch = []
        next_word_batch = []
        genres_batch = []

        settings.logger.info("Generating data...  ")

        # Keep track of how many batches have been fed
        batches_fed_count = 0
        while 1:
            # Keep track of the current batch that is being built
            current_batch_size = 0

            synopsis_counter = -1
            # Iterate over all the synopsis

            for synopsis in self.synopses:
                synopsis_counter += 1
                genre = self.genres[synopsis_counter]
                # Itearte over synopsis' words
                if len(synopsis) > settings.MAX_SYNOPSIS_LEN:
                    synopsis = synopsis[:settings.MAX_SYNOPSIS_LEN]
                    synopsis[-1] = self.word_to_index[settings.EOS_TOKEN]

                for i in range(len(synopsis) - 1):
                    # Grab next word and add it to the current batch
                    next_word = np.zeros(settings.VOCABULARY_SIZE)
                    next_word[synopsis[i + 1]] = 1
                    next_word_batch.append(next_word)

                    # Grab previous words and add them to the current batch
                    previous_words = [word for word in synopsis[:i + 1]]
                    #print(9999,len(previous_words))
                    pad_units = settings.MAX_SYNOPSIS_LEN - len(previous_words)
                    padding = [self.word_to_index[settings.PAD_TOKEN] for i in range(pad_units)]
                    previous_words.extend(padding)
                    next_sentence = np.asarray(previous_words)
                    previous_words_batch.append(next_sentence)

                    # Add the genre to the batch
                    genres_batch.append(genre)

                    # Increment batch size
                    current_batch_size += 1

                    if current_batch_size < settings.BATCH_SIZE:
                        # Keep building the batch
                        continue
                    # Batch is ready
                    next_word_batch = np.asarray(next_word_batch)
                    genres_batch = np.asarray(genres_batch)
                    previous_words_batch = np.asarray(previous_words_batch)

                    # Padd previous words of synopses
                    '''
                    #previous_words_batch = sequence.pad_sequences(previous_words_batch,
                                                                  maxlen=settings.MAX_SYNOPSIS_LEN, padding='post')
                    '''
                    # print(len(genres_batch),genres_batch[0].shape,previous_words_batch.shape,next_word_batch.shape)
                    # print(next_word_batch.mean())
                    
                    if 0:
                        for j in range(settings.BATCH_SIZE):
                            print(str(self.to_genre(genres_batch[j])).encode('latin1'))
                            print(str(self.to_synopsis(previous_words_batch[j])).encode('latin1'))
                            print(str(self.to_synopsis(np.nonzero(next_word_batch[j])[0])).encode('latin1'))
                            print("************")
                        print("____________________________________________")
                    
                    yield ([genres_batch, previous_words_batch], next_word_batch)
                    batches_fed_count += 1
                    
                    # settings.logger.info("Batches yielded: "+str(batches_fed_count))

                    # Reset variables
                    previous_words_batch = []
                    next_word_batch = []
                    genres_batch = []
                    current_batch_size = 0

