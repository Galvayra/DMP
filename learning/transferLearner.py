from DMP.utils.arg_fine_tuning import *
from DMP.learning.score import MyScore
from DMP.learning.variables import IMAGE_RESIZE, NUM_CHANNEL_OF_IMAGE
from DMP.learning.neuralNet import TensorModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras import models, layers
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import os

ALIVE_DIR = 'alive'
DEATH_DIR = 'death'
BATCH_SIZE = 32


class TransferLearner(TensorModel):
    def __init__(self, is_cross_valid=True):
        super().__init__(is_cross_valid=is_cross_valid)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()

    def transfer_learning(self, x_train, y_train, x_test, y_test):
        #
        # train_gen = ImageDataGenerator()
        # valid_gen = ImageDataGenerator()

        model = self.__cnn_model(img_shape=(IMAGE_RESIZE, IMAGE_RESIZE, NUM_CHANNEL_OF_IMAGE), num_cnn_layers=2)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)

        print(model.evaluate(x_test, y_test))

        # pre_trained_vgg = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
        # pre_trained_vgg.trainable = False
        # pre_trained_vgg.summary()
        #
        # additional_model = models.Sequential()
        # additional_model.add(pre_trained_vgg)
        # additional_model.add(layers.Flatten())
        # additional_model.add(layers.Dense(4096, activation='relu'))
        # additional_model.add(layers.Dense(2048, activation='relu'))
        # additional_model.add(layers.Dense(1024, activation='relu'))
        # additional_model.add(layers.Dense(4, activation='softmax'))
        # additional_model.summary()

    def __cnn_model(self, img_shape, num_cnn_layers):
        NUM_FILTERS = 32
        KERNEL = (3, 3)
        # MIN_NEURONS = 20
        MAX_NEURONS = 120

        model = Sequential()

        for i in range(1, num_cnn_layers + 1):
            if i == 1:
                model.add(Conv2D(NUM_FILTERS * i, KERNEL, input_shape=img_shape, activation='relu', padding='same'))
            else:
                model.add(Conv2D(NUM_FILTERS * i, KERNEL, activation='relu', padding='same'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(int(MAX_NEURONS), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(int(MAX_NEURONS / 2), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        print(model.summary())

        return model
