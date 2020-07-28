import os


CHECKPOINT_PATH = "./models/model.{epoch:02d}-{val_loss:.4f}.hdf5"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
LOG_FILE = "./logs/training.log"
TENSORBOARD_LOGS="./tensorboard"
CUDA_VISIBLE_DEVICES=""

#Training Data Path
DATA_PATH = "./data/emnist-byclass.mat"

#Stop the training process if the validation accuracy don't improve for 10 continuoes epochs.
EARLY_STOP_PATIENCE = 10

#Batch size
BATCH_SIZE = 256

#Number of epochs
EPOCH = 100

#Total Number of prediction classed
NUM_CLASSES = 62

