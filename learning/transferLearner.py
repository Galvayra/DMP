from DMP.utils.arg_fine_tuning import *
from DMP.learning.neuralNet import TensorModel
from keras.applications import VGG19, VGG16, ResNet50
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import Adam

ALIVE_DIR = 'alive'
DEATH_DIR = 'death'
BATCH_SIZE = 32


class TransferLearner(TensorModel):
    def __init__(self, is_cross_valid=True):
        super().__init__(is_cross_valid=is_cross_valid)
        self.num_of_input_nodes = int()
        self.num_of_output_nodes = int()
        self.trained_model = None

    def load_pre_trained_model(self, input_tensor):
        w_size, h_size, n_channel = input_tensor.shape
        input_tensor = Input(shape=(w_size, h_size, n_channel))
        self.trained_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    def transfer_learning(self, x_train, y_train, x_test, y_test):
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in self.trained_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block2_pool'].output

        # Stacking a new simple convolutional network on top of it
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.7)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Creating new model. Please note that this is NOT a Sequential() model.
        custom_model = Model(input=self.trained_model.input, output=x)

        if DO_SHOW:
            print(custom_model.summary())

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in custom_model.layers[:7]:
            layer.trainable = False

        custom_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        custom_model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)
        print(custom_model.evaluate(x_test, y_test))

    # def training_end_to_end(self):
    #     model = self.__cnn_model(img_shape=(IMAGE_RESIZE, IMAGE_RESIZE, NUM_CHANNEL_OF_IMAGE), num_cnn_layers=2)
    #     model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH)
    #     print(model.evaluate(x_test, y_test))

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
