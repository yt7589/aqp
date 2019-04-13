import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class QciLinearRegression(object):
    def __init__(self):
        self.name = 'QciLinearRegression'
        self.loss = 'mean_squared_error'
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.epoch = 5000

    def load_dataset(self):
        train_x = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
        train_y = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
        validate_x = np.array([], dtype=float)
        validate_y = np.array([], dtype=float)
        test_x = np.array([], dtype=float)
        test_y = np.array([], dtype=float)
        return train_x, train_y, validate_x, validate_y, test_x, test_y

    def train(self):
        train_x, train_y, validate_x, validate_y, test_x, test_y = self.load_dataset()
        model = self.build_model()
        model.compile(loss=self.loss, optimizer=self.optimizer)
        history = model.fit(train_x, train_y, epochs=self.epoch, verbose=False)
        plt.title('linear regression training process')
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.plot(history.history['loss'])
        plt.show()
        model.save('./work/aqt003_qiclr')
        weights = np.array(model.get_weights())
        print(weights)

    def build_model(self):
        layer1 = tf.keras.layers.Dense(units=1, input_shape=[1])
        model = tf.keras.Sequential([layer1])
        return model

    def predict(self, data):
        model = tf.keras.models.load_model('./work/aqt003_qiclr')
        rst = model.predict(data)
        return rst

if '__main__' == __name__:
    lr = QciLinearRegression()
    lr.train()
