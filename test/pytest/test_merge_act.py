import pytest
import hls4ml
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, Concatenate, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
import math
from tensorflow.keras import backend as K

test_root_path = Path(__file__).parent

@pytest.mark.parametrize("activation_function", [Activation(activation='relu', name='Activation'),
                                                Activation(activation='sigmoid', name='Activation')
												])

@pytest.mark.parametrize('backend', ['Vivado'])
def test_activations(activation_function, backend):
    model = tf.keras.models.Sequential()
    model.add(Dense(64,
              input_shape=(1,),
              name='Dense',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=None))
    model.add(activation_function)

    model.compile(optimizer='adam', loss='mse')
    X_input = np.random.rand(100,1)
    keras_prediction = model.predict(X_input)
    config = hls4ml.utils.config_from_keras_model(model)
    config['Model']['Strategy'] = 'Resource'
    output_dir = str(test_root_path / 'hls4mlprj_keras_api_activations_{}_{}'.format(activation_function.__class__.__name__, backend))
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=output_dir, backend=backend)
    hls_model.compile()
    hls_prediction = hls_model.predict(X_input)

    np.testing.assert_allclose(hls_prediction, keras_prediction, rtol=1e-2, atol=0.01)

    # assert len(model.layers) + 1 == len(hls_model.get_layers())

    # assert list(hls_model.get_layers())[2].attributes['class_name'] == activation_function.__class__.__name__