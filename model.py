import tensorflow as tf

import tensorflow_addons as tfa

node_1605430f35f94411aaf6b97eae005e19 = tf.keras.layers.Input(shape=(28, 28, 1))
node_2fbbd8e5b0a5456faa2d47f7026b139f = tf.keras.layers.Conv2D(filters=16, kernel_size=(16, 16), strides=(1, 1), padding='same')(node_1605430f35f94411aaf6b97eae005e19)
node_39ce8c39bacb4fb392c2372fb81a0b7e = tf.keras.layers.Activation(activation="relu")(node_2fbbd8e5b0a5456faa2d47f7026b139f)
node_2c8a6d78d0204888942f16317f2a079f = tf.keras.layers.Dropout(rate=0.5)(node_39ce8c39bacb4fb392c2372fb81a0b7e)
node_71914b8774b64700b38dc3e8e7a62caa = tf.keras.layers.Flatten()(node_2c8a6d78d0204888942f16317f2a079f)
node_020cdce94de241ac9556bb0b0022c1f2 = tf.keras.layers.Dense(units=10)(node_71914b8774b64700b38dc3e8e7a62caa)
node_96afcbc0a4ba4ed9b02b579068f166f0 = tf.keras.layers.Activation(activation="softmax")(node_020cdce94de241ac9556bb0b0022c1f2)
model = tf.keras.Model(inputs=node_1605430f35f94411aaf6b97eae005e19, outputs=node_96afcbc0a4ba4ed9b02b579068f166f0)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
