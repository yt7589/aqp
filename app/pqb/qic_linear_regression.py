import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class QciLinearRegression(object):
    def __init__(self, learning_rate=0.01, epoch=50000, patience=10, 
                train_x=None, train_y=None, 
                validate_x=None, validate_y=None, 
                test_x=None, test_y=None):
        self.name = 'QciLinearRegression'
        self.loss = 'mean_squared_error'
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.patience = patience
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        if train_x is not None:
            self.train_x = train_x
            self.train_y = train_y
            self.validate_x = validate_x
            self.validate_y = validate_y
            self.test_x = test_x
            self.test_y = test_y
        else:
            self.train_x, self.train_y, self.validate_x, \
                        self.validate_y, self.test_x, \
                        self.test_y = self.generate_dataset()

    def generate_dataset(self):
        train_x = np.array([-40, -10, 0, 8, 15, 22, 38, 20, 9, 13], dtype=np.float32)
        train_y = np.array([-40, 14, 32, 46, 59, 72, 100, 68, 48.2, 55.4], dtype=np.float32)
        validate_x = np.array([], dtype=float)
        validate_y = np.array([], dtype=float)
        test_x = np.array([], dtype=float)
        test_y = np.array([], dtype=float)
        return train_x, train_y, validate_x, validate_y, test_x, test_y

    def train(self):
        model = self.build_model()
        model.compile(loss=self.loss, optimizer=self.optimizer)
        class PrintDot(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('epoch:{0}...{1}!'.format(epoch, logs))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        print('I am ok')
        print('x:{0}; y:{0}'.format(self.train_x.shape, self.train_y.shape))
        history = model.fit(self.train_x, self.train_y, 
                    epochs=self.epoch, validation_split = 0.1,  
                    verbose=False, callbacks=[early_stop, PrintDot()])
        plt.title('linear regression training process')
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.plot(history.history['loss'])
        #plt.show()
        plt.savefig('/content/drive/My Drive/aqp/aqt003_001.png', format='png')
        model.save('./work/aqt003_qiclr')
        weights = np.array(model.get_weights())
        print(weights)
        return weights

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
