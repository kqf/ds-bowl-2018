from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras import backend as K

class CnnClassifier():

    def __init__(self, batch_size=1000, epochs=1, smooth=1., channels=3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.smooth = smooth
        self.channels = channels
        self._graph = None

    def fit(self, X, y):
        self._graph = self._build_network(X)

        self._graph.fit(X, y,
            steps_per_epoch=self.batch_size,
            epochs=3
        )
        return self

    def predict(self, X):
        pass

    def _build_network(self, X):
        base_cnn = Sequential()
        base_cnn.add(BatchNormalization(
            input_shape = (None, None, self.channels), 
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
        base_cnn.add(Conv2D(1, kernel_size=(1,1), padding='same', activation='sigmoid'))
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