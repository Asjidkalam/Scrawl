import os
from define_model import define_model
from constants import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

from keras.models import load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.optimizers import Adam

from tensorflow.keras import backend as K
from distutils.version import LooseVersion as LV
from keras import __version__
print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
from scipy.io import loadmat
import cv2
import pickle

class Model:

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        return


    def load_data(self,data_path=None):

        """Loading the EMINST dataset"""
        if data_path is None:
            data = loadmat(os.path.abspath(os.path.join(os.getcwd(), DATA_PATH)))
        else:
            data = loadmat(os.path.abspath(os.path.join(os.getcwd(), data_path)))

        # Loading Training Data
        X_train = data["dataset"][0][0][0][0][0][0]
        y_train = data["dataset"][0][0][0][0][0][1]
        X_train = X_train.astype('float32')
        X_train /= 255.0

        ##Loading Testing Data
        X_test = data["dataset"][0][0][1][0][0][0]
        y_test = data["dataset"][0][0][1][0][0][1]
        X_test = X_test.astype('float32')
        X_test /= 255.0

        # one-hot encoding:
        Y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
        Y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

        # input image dimensions
        img_rows, img_cols = 28, 28


        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

        print('Intermediate X_train:', X_train.shape)
        print('Intermediate X_test:', X_test.shape)

        # Reshaping all images into 28*28 for pre-processing
        X_train = X_train.reshape(X_train.shape[0], 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 28, 28)

        # for train data
        for t in range(X_train.shape[0]):
            X_train[t] = np.transpose(X_train[t])

        # for test data
        for t in range(X_test.shape[0]):
            X_test[t] = np.transpose(X_test[t])

        print('Process Complete: Rotated and reversed test and train images!')

        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_train = X_train.reshape(X_train.shape[0], 784, )

        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 784, )

        print('EMNIST data loaded: train:', len(X_train), 'test:', len(X_test))
        print('Flattened X_train:', X_train.shape)
        print('Y_train:', Y_train.shape)
        print('Flattened X_test:', X_test.shape)
        print('Y_test:', Y_test.shape)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def character_model(self):

        self.model = define_model(NUM_CLASSES)

    def loadmodel(self,path=None):

        if path is not None and os.path.exists(path):
            self.model = load_model(path)
        else:
            print("Unable to find model at the specified path")
        return


    def train(self,pretrained_model_path=None):

        cb_checkpoint = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_weights_only=False, period=1)
        cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
        reduce_on_plateau = ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.1, patience=20, verbose=1)
        cb_tensorboard = TensorBoard(log_dir=TENSORBOARD_LOGS)
        csv_logger = CSVLogger(LOG_FILE)

        callback_values = [cb_checkpoint,cb_early_stopper,reduce_on_plateau,csv_logger,cb_tensorboard]

        if pretrained_model_path is not None and os.path.exists(os.path.abspath(os.path.join(os.getcwd(),pretrained_model_path))):
            self.loadmodel(pretrained_model_path)
            if self.model is not None:
                print("Starting training from the pretrained model")

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(0.00001), metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_split=0.1,
                                 epochs=EPOCH, callbacks=callback_values,batch_size=BATCH_SIZE)


    def test(self,model_path=None):
        if model_path is None and self.model is None:
            print("No model found at specified path")
            exit()
        self.loadmodel(model_path)
        accuracy = self.model.evaluate(self.X_test,self.Y_test,batch_size=BATCH_SIZE)
        print("Accuracy on test data is {}".format(accuracy[1]))

    def predict(self, img_path=None,model_path=None):

        if model_path is None:
            print("No model found at specified path")
            exit()

        if img_path is None:
            print("Invalid image path provided. Unable to make a prediction")
            exit()
        self.loadmodel(model_path
                       )
        try:
            with open(os.path.abspath(os.path.join(os.getcwd(), "../data/mapping.pkl"))) as f:
                mapping = pickle.load(f)
        except Exception as e:
            print(e)
            mapping = ['0', '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        img = cv2.imread(img_path, 0)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th3 = cv2.subtract(255, th3)
        pred_img = th3
        pred_img = cv2.resize(pred_img, (28, 28))
        pred_img = pred_img / 255.0
        pred_img = pred_img.reshape(1,784)
        prediction = mapping[self.model.predict_classes(pred_img)[0]]
        print("\n\nPredicted Value : {}".format(prediction))
