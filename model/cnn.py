from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras import backend as K

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K


class CnnClassifier():

    def __init__(self, batch_size=1, epochs=1, smooth=1., channels=3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.smooth = smooth
        self.channels = channels
        self._graph = None

    def fit(self, X, y):
        self._graph = self._build_network(X)

        self._graph.fit(X, y,
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        return self

    def predict(self, X):
        return self._graph.predict(X, verbose=1)

    def _build_network(self, X):
        base_cnn = Sequential()
        base_cnn.add(BatchNormalization(
            input_shape = (256, 256, self.channels), 
            name = 'NormalizeInput')
        )

        base_cnn.add(Conv2D(8, kernel_size=(3,3), padding='same'))
        base_cnn.add(Conv2D(8, kernel_size=(3,3), padding='same'))
        # use dilations to get a slightly larger field of view
        base_cnn.add(Conv2D(16, kernel_size=(3,3), dilation_rate=2, padding='same'))
        base_cnn.add(Conv2D(16, kernel_size=(3,3), dilation_rate=2, padding='same'))
        base_cnn.add(Conv2D(32, kernel_size=(3,3), dilation_rate=3, padding='same'))

        # the final processing
        base_cnn.add(Conv2D(16, kernel_size=(1,1), padding='same'))
        base_cnn.add(Conv2D(3, kernel_size=(1,1), padding='same', activation='sigmoid'))
        base_cnn.summary()    

        def dice_coef(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + self.smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)

        def dice_coef_loss(y_true, y_pred):
            return -dice_coef(y_true, y_pred)


        base_cnn.compile(
            optimizer='adam', 
            loss=dice_coef_loss, 
            metrics=[dice_coef, 'acc', 'mse']
        )
        return base_cnn

class UnetClassifier(CnnClassifier):

    def __init__(self, batch_size=1, epochs=1, smooth=1., channels=3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.smooth = smooth
        self.channels = channels
        self._graph = None

    def _build_network(self, X):
        inputs = Input(X.shape[1:])
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
        return model

   