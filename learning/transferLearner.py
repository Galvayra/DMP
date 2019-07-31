from DMP.utils.arg_fine_tuning import *
from DMP.learning.score import MyScore
from DMP.learning.variables import IMAGE_RESIZE, NUM_CHANNEL_OF_IMAGE
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


class TransferLearner(MyScore):
    def __init__(self, is_cross_valid=True):
        super().__init__()
        self.__name_of_log = str()
        self.__name_of_tensor = str()
        self.__is_cross_valid = is_cross_valid

    @property
    def name_of_log(self):
        return self.__name_of_log

    @name_of_log.setter
    def name_of_log(self, name):
        self.__name_of_log = name

    @property
    def name_of_tensor(self):
        return self.__name_of_tensor

    @name_of_tensor.setter
    def name_of_tensor(self, name):
        self.__name_of_tensor = name

    @property
    def is_cross_valid(self):
        return self.__is_cross_valid

    def __set_name_of_log(self):
        name_of_log = self.name_of_log + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_log)

        if DO_SHOW:
            print("======== Directory for Saving ========")
            print("   Log File -", name_of_log)

    def __set_name_of_tensor(self):
        name_of_tensor = self.name_of_tensor + "fold_" + str(self.num_of_fold)

        if self.is_cross_valid:
            os.mkdir(name_of_tensor)

        if DO_SHOW:
            print("Tensor File -", name_of_tensor, "\n\n\n")

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
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        print(model.summary())

        return model
