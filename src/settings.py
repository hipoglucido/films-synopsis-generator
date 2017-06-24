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
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tensorboard_logs')

##FILES
GENRE_BINARIZER_PATH = os.path.join(OTHERS_DIR, 'genre_binarizer_378_classes.pkl')
WORD_TO_INDEX_PATH = os.path.join(OTHERS_DIR, 'word_to_index.pkl')
EMBEDDING_WEIGHTS_PATH = os.path.join(OTHERS_DIR, 'embedding_weights.hdf')
WORD2VEC_MODEL_PATH = os.path.join(OTHERS_DIR, 'SBW-vectors-300-min5.txt')
INPUT_PREPROCESSED_FILMS = os.path.join(DATA_DIR,"991_preprocessed_films.pkl")

## TRAINING
BATCH_SIZE = 32
EPOCHS = 100
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'LSTM_weights-058-tloss5.0896.hdf5')
STEPS_PER_EPOCH = 1000000

## PREPROCESSING
MAX_SYNOPSIS_LEN = 150
VOCABULARY_SIZE = 50000 #None will use the whole corpus vocabulary (151852)
MAX_GENERES = 25
EOS_TOKEN = '<eos>'
UNKNOWN_TOKEN = '<unk>'
MINIMUM_KNOWN_PERC_TOKENS_PER_SYNOPSIS = 0.9


## OTHER CONSTANTS
EMBEDDING_DIM = 300#128


## DEBUGGING
USE_SMALL_DATASET = 1
USE_SMALL_WORD2VEC = 0
PRINT_MODEL_SUMMARY = 1

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


