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
INPUT_DIR = os.path.join(DATA_DIR, "input")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tensorboard_logs')

## TRAINING
BATCH_SIZE = 32
EPOCHS = 100
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'LSTM_weights-058-tloss5.0896.hdf5')
STEPS_PER_EPOCH = 1000000

## OTHER CONSTANTS
EMBEDDING_DIM = 128
MAX_SYNOPSIS_LEN = 150
VOCABULARY_SIZE = None #None will use the whole corpus vocabulary
WORD_EMBEDDINGS_PATH = os.path.join(DATA_DIR,"SBW-vectors-300-min5.txt")

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


