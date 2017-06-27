"""
Project settings
"""

import os
import logging


## DIRECTORIES
"""
.
├── data
│   ├── input
│   ├── tensorboard_logs
│   └── weights
└── src
"""
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OTHERS_DIR = os.path.join(DATA_DIR, "others")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
PREDICTIONS_DIR = os.path.join(OTHERS_DIR, "predictions")
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tensorboard_logs')

'''
LSTM_w2v0_v20000_g10_w-008-tloss6.5773-vloss6.7394
LSTM_w2v1_v20000_g10_w-008-tloss7.0115-vloss6.8510
LSTM_w2v0_v7000_g8_w-005-tloss5.3248-vloss5.2801
LSTM_w2v1_v7000_g8_w-005-tloss4.4168-vloss4.3784
'''
## TRAINING
BATCH_SIZE = 128
EPOCHS = 10000
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'LSTM_w2v0_v20000_g10_w-008-tloss6.5773-vloss6.7394.hdf5')
STEPS_PER_EPOCH = 10000
VALIDATION_SPLIT = 0.2
STEPS_VAL = 2000
OPTIMIZER = 'rmsprop'

## PREPROCESSING
MAX_SYNOPSIS_LEN = 150

VOCABULARY_SIZE = int(WEIGHTS_PATH.split('_v')[1].split('_g')[0]) #Integer, inferred from WEIGHTS_PATH if executing on prediction time
MAX_GENRES = int(WEIGHTS_PATH.split('_g')[1].split('_w')[0]) #Integer, inferred from WEIGHTS_PATH if executing on prediction time
EOS_TOKEN = '<eos>'
UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
DIGIT_TOKEN = 'DIGITO'
MINIMUM_KNOWN_PERC_TOKENS_PER_SYNOPSIS = 0.7


## OTHER CONSTANTS
EMBEDDING_DIM = 300#128
USE_W2V = int(WEIGHTS_PATH.split('_w2v')[1].split('_v')[0]) #1 or 0, inferred from WEIGHTS_PATH if executing on prediction time
SEED = 2017

## DEBUGGING
USE_SMALL_DATASET = 0
USE_SMALL_WORD2VEC = 1
PRINT_MODEL_SUMMARY = 1


##FILES
#This assertion should only take place at prediction time, not when generating_files() is called

assert any ([MAX_GENRES == 25 and VOCABULARY_SIZE == 50000,
             MAX_GENRES == 10 and VOCABULARY_SIZE == 20000,
             MAX_GENRES == 8 and VOCABULARY_SIZE == 7000])

if MAX_GENRES == 10:
    GENRE_BINARIZER_PATH = os.path.join(OTHERS_DIR, '20170626T023708_genre_binarizer_10_classes.pkl')
elif MAX_GENRES == 25:
    GENRE_BINARIZER_PATH = os.path.join(OTHERS_DIR, '20170625T052119_genre_binarizer_25_classes.pkl')
elif MAX_GENRES == 8:
    GENRE_BINARIZER_PATH = os.path.join(OTHERS_DIR, '20170626T104150_genre_binarizer_8_classes.pkl')
if VOCABULARY_SIZE == 20000:
    WORD_TO_INDEX_PATH = os.path.join(OTHERS_DIR, '20170626T023708_20001_word_to_index.pkl')
    INDEX_TO_WORD_PATH = os.path.join(OTHERS_DIR, '20170626T023708_20001_index_to_word.pkl')
    EMBEDDING_WEIGHTS_PATH = os.path.join(OTHERS_DIR, '20170626T025245_20001_embedding_weights.pkl')
    INPUT_PREPROCESSED_FILMS = os.path.join(DATA_DIR,"20170626T025251_v20000_109214_preprocessed_films.pkl")
elif VOCABULARY_SIZE == 50000:
    WORD_TO_INDEX_PATH = os.path.join(OTHERS_DIR, '20170625T052119_50001_word_to_index.pkl')
    INDEX_TO_WORD_PATH = os.path.join(OTHERS_DIR, '20170625T052119_50001_index_to_word.pkl')
    EMBEDDING_WEIGHTS_PATH = os.path.join(OTHERS_DIR, '20170625T054945_50001_embedding_weights.pkl')
    INPUT_PREPROCESSED_FILMS = os.path.join(DATA_DIR,"20170625T054951_v50000_113347_preprocessed_films.pkl")
elif VOCABULARY_SIZE == 7000:
    WORD_TO_INDEX_PATH = os.path.join(OTHERS_DIR, '20170626T104150_7000_word_to_index.pkl')
    INDEX_TO_WORD_PATH = os.path.join(OTHERS_DIR, '20170626T104150_7000_index_to_word.pkl')
    EMBEDDING_WEIGHTS_PATH = os.path.join(OTHERS_DIR, '20170626T104824_7001_embedding_weights.pkl')
    INPUT_PREPROCESSED_FILMS = os.path.join(DATA_DIR,"20170626T104827_v7000_104056_preprocessed_films.pkl")
WORD2VEC_MODEL_PATH = os.path.join(OTHERS_DIR, 'SBW-vectors-300-min5.txt')


## LOGGING
# create logger
logger = logging.getLogger('NC')
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

logger.info("Starting...")
logger.info("Vocabulary size: "+str(VOCABULARY_SIZE))
logger.info("Genres: "+str(MAX_GENRES))
logger.info("W2V: "+str(USE_W2V))
logger.info("weights: "+str(WEIGHTS_PATH))
