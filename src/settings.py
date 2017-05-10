# Directory settings

import os

#ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
ROOT_DIR = '/home/vgarciacazorla/NC/films-synopsis-generator'
DATA_DIR = os.path.join(ROOT_DIR, "data")


## Input directories
INPUT_DIR = os.path.join(DATA_DIR, "input")
DATASETS_DIR = os.path.join(INPUT_DIR, "datasets")
RAW_DATA_DIR = os.path.join(DATASETS_DIR,"raw")
TRAIN_DATA_DIR = os.path.join(DATASETS_DIR,"train")
TEST_DATA_DIR = os.path.join(DATASETS_DIR,"test")
INPUT_WEIGHTS_DIR = os.path.join(INPUT_DIR, "weights")

##Output directories
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
TENSORBOARD_LOGS_DIR = os.path.join(OUTPUT_DIR, 'tensorboard_logs')
OUTPUT_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")

