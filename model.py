import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def MyModel_1(input_x : keras.layers.Input):
    x = keras.layers.Conv2D(32, 3)(input_x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    output_x = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.models.Model(input_x, output_x)
    return model


def main():
    input_shape = keras.layers.Input(shape=(28, 28, 1))
    model = MyModel_1(input_shape)

    model.summary()


if __name__ == '__main__':
    main()