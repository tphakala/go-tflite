import numpy as np
import tensorflow as tf


def fizzbuzz(i):
    if i % 15 == 0:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    elif i % 5 == 0:
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif i % 3 == 0:
        return np.array([0, 1, 0, 0], dtype=np.float32)
    else:
        return np.array([1, 0, 0, 0], dtype=np.float32)


def bin(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)], dtype=np.float32)


trX = np.array([bin(i, 7) for i in range(1, 101)])
trY = np.array([fizzbuzz(i) for i in range(1, 101)])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_dim=7),
    tf.keras.layers.Activation('tanh'),
    tf.keras.layers.Dense(4, input_dim=64),
    tf.keras.layers.Activation('softmax'),
])

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(trX, trY, epochs=3600, batch_size=64)


def representative_dataset_gen():
    for i in range(100):
        yield [trX[i: i + 1]]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('fizzbuzz_model_quant.tflite', 'wb') as f:
    f.write(tflite_model)
