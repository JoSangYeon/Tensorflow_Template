import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from dataset import *
from model import *
from learning import *
from inference import *

batch_size=32
EPOCHS = 10


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    train_loader = MyDataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    valid_loader = MyDataLoader(x_test, y_test, batch_size=batch_size)

    model = MyModel_1(keras.layers.Input(shape=(28, 28, 1)))
    model.summary()

    device = 'GPU:0' if tf.test.is_gpu_available() else 'CPU:0'
    optimizer = keras.optimizers.Adam()
    criterion = 'MSE'
    metrics = ['MSE', 'MAE']

    print(device)
    model.compile(loss=criterion, optimizer=optimizer, metrics=metrics)
    with tf.device(f'/device:{device}'):
        history = model.fit(train_loader, validation_data=valid_loader, epochs=EPOCHS)


if __name__ == '__main__':
    main()