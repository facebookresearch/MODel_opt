import tensorflow as tf

"""
Reimplementations of simple models in TF2 that
don't have standard implementations in TFM.
"""


def alexnet(input_shape, num_classes):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(
                filters=96,
                kernel_size=(11, 11),
                strides=(4, 4),
                activation="relu",
                input_shape=(227, 227, 3),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=384,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


class VGGBlock(tf.keras.models.Sequential):
    def __init__(self, n, m):
        super().__init__()
        for i in range(m):
            self.add(
                tf.keras.layers.Conv2D(
                    filters=n,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                )
            )
        self.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))


class VGGDense(tf.keras.models.Sequential):
    def __init__(self, n, m=2):
        super().__init__()
        for i in range(m):
            self.add(tf.keras.layers.Dense(units=n, activation="relu"))


class VGG11(tf.keras.models.Sequential):
    def __init__(self, input_shape, classes, filters=64):
        super().__init__()
        self.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        self.add(VGGBlock(n=filters * 1, m=1))
        self.add(VGGBlock(n=filters * 2, m=1))
        self.add(VGGBlock(n=filters * 4, m=2))
        self.add(VGGBlock(n=filters * 8, m=2))
        self.add(VGGBlock(n=filters * 8, m=2))
        self.add(tf.keras.layers.Flatten())
        self.add(VGGDense(n=filters * 64))
        self.add(tf.keras.layers.Dense(units=classes, activation="softmax"))


def vgg11(input_shape, num_classes):
    return VGG11(input_shape, num_classes)
