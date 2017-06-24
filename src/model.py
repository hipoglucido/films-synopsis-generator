"""
Training
"""
import settings
import data

import os
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge
#from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.externals import joblib

class Network():
    """
    Class for training
    """
    def __init__(self):
        self.model = None
        self.generator = None

    def load_generators(self, synopses, genres):
        X_train, X_test, y_train, y_test = train_test_split(
            synopses, genres, test_size = settings.VALIDATIN_SPLIT)
        self.generator_train = data.Generator(X_train, y_train)
        #self.generator_train.initialize()
        self.generator_val = data.Generator(X_test, y_test)
        #self.generator_val.initialize()

    def build(self):
        settings.logger.info("Building model...")
        if self.embedding_weights is None:
            self.load_embeddings()

        #with tf.device("/gpu:1"):
        genres_model = Sequential()
        genres_model.add(Dense(units = settings.EMBEDDING_DIM,
                              input_dim = len(self.generator.mlb.classes_),
                              activation='relu'))
        genres_model.add(RepeatVector(settings.MAX_SYNOPSIS_LEN))
        #with tf.device("/gpu:0"):
        synopsis_model = Sequential()
        synopsis_model.add(Embedding(input_dim = settings.VOCABULARY_SIZE + 1 + 2,  # Extra +2 because UNK and EOS have own entry.
                                output_dim = settings.EMBEDDING_DIM,
                                weights = [self.embedding_weights],
                                input_length=settings.MAX_SYNOPSIS_LEN),
                                trainable = False)
        synopsis_model.add(LSTM(units = settings.EMBEDDING_DIM,
                            return_sequences=True))
        synopsis_model.add(TimeDistributed(Dense(settings.EMBEDDING_DIM)))
        #with tf.device("/gpu:1"):
        self.model = Sequential()
        self.model.add(Merge([genres_model, synopsis_model], mode='concat'))
        self.model.add(LSTM(units = 1000,return_sequences=False))
        self.model.add(Dense(settings.VOCABULARY_SIZE))
        self.model.add(Activation('softmax'))
        
        if settings.PRINT_MODEL_SUMMARY:
            self.model.summary()

    def load_weights(self):
        self.model.load_weights(settings.WEIGHTS_PATH)
        settings.logger.info("Loaded weights "+settings.WEIGHTS_PATH)

    def load_embeddings(self):
        self.word_to_index = joblib.load(settings.WORD_TO_INDEX_PATH)
        self.embedding_weights = joblib.load(settings.EMBEDDING_WEIGHTS_PATH)
            
    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    def train(self):
        """
        Train the model.
        """
        assert self.generator is not None

        weights_name = 'LSTM_weights-{epoch:03d}-tloss{loss:.4f}.hdf5'
        file_name = os.path.join(settings.WEIGHTS_DIR)
        
        #Add callbacks
        callbacks_list = []
        checkpoint = ModelCheckpoint(settings.WEIGHTS_PATH, monitor='loss', save_best_only=True, mode='min')
        callbacks_list.append(checkpoint)
        
        tf_logs = TensorBoard(
                    log_dir = settings.TENSORBOARD_LOGS_DIR,
                    histogram_freq = 1,
                    write_graph = True)
        callbacks_list.append(tf_logs)
        
        #Fit the model
        self.model.fit_generator(
                            generator = self.generator_train.generate(),
                            steps_per_epoch=settings.STEPS_PER_EPOCH,
                            epochs=settings.EPOCHS,
                            validation_data=self.generator_train.generate(),
                            validation_steps=settings.STEPS_VAL,
                            workers=1,
                            callbacks=callbacks_list)

        #validation_data = None, validation_steps = None