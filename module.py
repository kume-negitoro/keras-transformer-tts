import keras
from keras.layers import Dense, Permute, BatchNormalization, Lambda, ReLU, Dropout
from keras.constraints import max_norm
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.models import Sequential

from functools import reduce
from itertools.chain import from_iterable


def pipe(*fns):
    return lambda x: reduce(lambda f1, f2: f2(f1), fns, x)


def Linear(in_dim, out_dim, bias=True, w_init="linear"):
    return Dense(
        out_dim,
        use_bias=bias,
        input_shape=(in_dim,),
        kernel_initializer="glorot_uniform",
    )


def Conv(
    in_channels,
    out_channels,
    kernel_size=1,
    stride=1,
    padding="valid",
    dilation=1,
    bias=True,
    w_init="linear",
):
    return Conv1D(
        kernel_size=kernel_size, strides=stride, dilation_rate=dilation, use_bias=bias
    )


def EncoderPrenet(embedding_size, num_hidden):
    return Sequential(
        [
            Embedding(),
            Permute((1, 3, 2)),
            Sequential(
                [
                    Conv(
                        in_channels=num_hidden,
                        out_channels=num_hidden,
                        kernel_size=5,
                        w_init="relu",
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Dropout(0.2),
                ]
            ),
            Sequential(
                [
                    Conv(
                        in_channels=num_hidden,
                        out_channels=num_hidden,
                        kernel_size=5,
                        w_init="relu",
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Dropout(0.2),
                ]
            ),
            Sequential(
                [
                    Conv(
                        in_channels=num_hidden,
                        out_channels=num_hidden,
                        kernel_size=5,
                        w_init="relu",
                    ),
                    BatchNormalization(),
                    ReLU(),
                    Dropout(0.2),
                ]
            ),
            Permute((1, 3, 2)),
            Linear(num_hidden, num_hidden),
        ]
    )


class FNN(Layer):
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden

    def call(self, input):
        return Sequential(
            [
                Permute((2, 3)),
                Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init="relu"),
                ReLU(),
                Conv(num_hidden * 4, num_hidden, kernel_size=1),
                Permute((2, 3)),
                Lambda(lambda x: x + input),
                Dense(num_hidden, kernel_constraint=max_norm(num_hidden)),
            ]
        )(input)

def PostConvNet(num_hidden):
    return Sequential(from_iterable([
        Conv(in_channels=hp.num_mels * hp.outputs_per_step, out_channels=num_hidden, kernel_size=5, padding='same', w_init='tanh'),

    ]))