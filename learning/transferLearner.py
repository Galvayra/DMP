from DMP.utils.arg_fine_tuning import *
from DMP.learning.neuralNet import TensorModel
from DMP.learning.variables import IMAGE_RESIZE
from keras.applications import VGG19, VGG16, ResNet50
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from DMP.learning.variables import KEY_TEST
import numpy as np

ALIVE_DIR = 'alive'
DEATH_DIR = 'death'
BATCH_SIZE = 32
DO_FINE_TUNING = True


class TransferLearner(TensorModel):
    def __init__(self):
        super().__init__(is_cross_valid=True)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()
        self.trained_model = None
        self.custom_model = None

    def load_pre_trained_model(self, input_tensor):
        w_size, h_size, n_channel = input_tensor.shape
        input_tensor = Input(shape=(w_size, h_size, n_channel))
        self.trained_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    def transfer_learning(self, x_train, y_train, x_test, y_test):
        self.__init_custom_model()
        self.__training(x_train, y_train)
        self.__predict_model(x_test, y_test)

    def training_end_to_end(self, x_train, y_train, x_test, y_test):
        self.__init_cnn_model(img_shape=(IMAGE_RESIZE, IMAGE_RESIZE, 3), num_cnn_layers=2)
        self.__training(x_train, y_train)
        self.__predict_model(x_test, y_test)

    @staticmethod
    def __get_y_predict(history):
        y_predict = list()

        for regress in history:
            if regress > 0.5:
                y_predict.append(1.0)
            else:
                y_predict.append(0.0)

        return np.array(y_predict)

    def __init_custom_model(self):
        self.num_of_fold += 1

        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in self.trained_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block2_pool'].output

        # Stacking a new simple convolutional network on top of it
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.7)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Creating new model. Please note that this is NOT a Sequential() model.
        self.custom_model = Model(input=self.trained_model.input, output=x)

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in self.custom_model.layers[:7]:
            layer.trainable = DO_FINE_TUNING

        # if DO_SHOW:
        #     print(self.custom_model.summary())

    def __training(self, x_train, y_train):
        # set file names for saving
        self.set_name_of_log()
        self.set_name_of_tensor()

        board = TensorBoard(log_dir=self.name_of_log + "/fold_" + str(self.num_of_fold),
                            histogram_freq=0, write_graph=True, write_images=True)
        ckpt = ModelCheckpoint(filepath=self.get_name_of_tensor() + '/model')
        self.custom_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        self.custom_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, callbacks=[board])
        # self.custom_model.save_weights(self.get_name_of_tensor() + '/dl_model.h5')

    def __predict_model(self, x_test, y_test):
        h = self.custom_model.predict(x_test, batch_size=BATCH_SIZE)
        y_predict = self.__get_y_predict(h)

        self.compute_score(y_test, y_predict, h)
        self.set_score(target=KEY_TEST)
        self.show_score(target=KEY_TEST)

    def __init_cnn_model(self, img_shape, num_cnn_layers):
        self.num_of_fold += 1
        NUM_FILTERS = 32
        KERNEL = (3, 3)

        self.custom_model = Sequential()

        # feature map 1
        for i in range(1, num_cnn_layers + 1):
            if i == 1:
                self.custom_model.add(Conv2D(NUM_FILTERS * i, KERNEL, input_shape=img_shape, activation='relu',
                                             padding='same'))
            else:
                self.custom_model.add(Conv2D(NUM_FILTERS * i, KERNEL, activation='relu', padding='same'))
        self.custom_model.add(MaxPooling2D(pool_size=(2, 2)))

        # feature map 2
        for i in range(1, num_cnn_layers + 1):
            self.custom_model.add(Conv2D(NUM_FILTERS * i, KERNEL, activation='relu', padding='same'))
        self.custom_model.add(MaxPooling2D(pool_size=(2, 2)))

        # feature map 3
        for i in range(1, num_cnn_layers + 1):
            self.custom_model.add(Conv2D(NUM_FILTERS * i, KERNEL, activation='relu', padding='same'))
        self.custom_model.add(MaxPooling2D(pool_size=(2, 2)))

        # FC
        self.custom_model.add(Flatten())
        self.custom_model.add(Dense(2048, activation='relu'))
        self.custom_model.add(Dropout(0.25))
        self.custom_model.add(Dense(1024, activation='relu'))
        self.custom_model.add(Dropout(0.25))
        self.custom_model.add(Dense(256, activation='relu'))
        self.custom_model.add(Dropout(0.25))
        self.custom_model.add(Dense(1, activation='sigmoid'))
        self.custom_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        if DO_SHOW:
            self.custom_model.summary()

