"""
Training
"""
import settings
import generator

import os
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge
#from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

import tensorflow as tf
from sklearn.externals import joblib

class Network():
    """
    Class for training
    """
    def __init__(self):
        settings.logger.info("Starting network...")
        self.model = None
        self.generator_train = None
        self.generator_val = None

    def load_generators(self, X_train, X_val, y_train, y_val):
        self.train_generator = generator.Generator(X_train,y_train)
        self.train_generator.load_indexes()
        self.train_generator.load_genre_binarizer()
        self.val_generator = generator.Generator(X_val, y_val)
        self.val_generator.load_indexes()
        self.val_generator.load_genre_binarizer()
        #self.generator_val.initialize()
        
        
    def build(self, use_embeddings = True):
        settings.logger.info("Building model...")
        self.use_embeddings = use_embeddings
        #with tf.device("/gpu:1"):
        genres_model = Sequential()
        genres_model.add(Dense(units = settings.EMBEDDING_DIM,
                              input_dim = settings.MAX_GENRES,
                              activation='relu'))
        genres_model.add(RepeatVector(settings.MAX_SYNOPSIS_LEN))
        #with tf.device("/gpu:0"):
        synopsis_model = Sequential()
        if use_embeddings:
            self.load_embeddings()
            synopsis_model.add(Embedding(input_dim = settings.VOCABULARY_SIZE + 1,
                                    output_dim = settings.EMBEDDING_DIM,
                                    weights = [self.embedding_weights],
                                    input_length=settings.MAX_SYNOPSIS_LEN,
                                    trainable = False))
        else:
            synopsis_model.add(Embedding(input_dim = settings.VOCABULARY_SIZE, 
                                output_dim = settings.EMBEDDING_DIM,
                                input_length=settings.MAX_SYNOPSIS_LEN))
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
        settings.logger.info("Weight embedding matrix loaded "+str(self.embedding_weights.shape))
            
    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    def train(self):
        """
        Train the model.
        """

        weights_name = 'LSTM_w2v'+str(self.use_embeddings)+'_v'+str(settings.VOCABULARY_SIZE)+'_g'+str(settings.MAX_GENRES)+'_w-{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5'
        file_path = os.path.join(settings.WEIGHTS_DIR,weights_name)
        
        #Add callbacks
        callbacks_list = []
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, mode='auto')
        callbacks_list.append(checkpoint)
        
        tf_logs = TensorBoard(
                    log_dir = settings.TENSORBOARD_LOGS_DIR,
                    histogram_freq = 1,
                    write_graph = True)
        callbacks_list.append(tf_logs)
        
        #Fit the model
        self.model.fit_generator(
                            generator = self.train_generator.generate(),
                            steps_per_epoch=settings.STEPS_PER_EPOCH,
                            epochs=settings.EPOCHS,
                            validation_data=self.val_generator.generate(),
                            validation_steps=settings.STEPS_VAL,
                            workers=1,
                            callbacks=callbacks_list)

        #validation_data = None, validation_steps = None