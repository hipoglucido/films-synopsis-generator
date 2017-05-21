# Directory settings

import os




ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
#ROOT_DIR = '..'#'/home/vgarciacazorla/NC/films-synopsis-generator'
DATA_DIR = os.path.join(ROOT_DIR, "data")


## Directories
INPUT_DIR = os.path.join(DATA_DIR, "input")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
WORD_EMBEDDINGS_PATH = os.path.join(DATA_DIR,"SBW-vectors-300-min5.txt")
TENSORBOARD_LOGS_DIR = os.path.join(DATA_DIR, 'tensorboard_logs')


