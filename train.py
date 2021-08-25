import tensorflow as tf


def fit_model(model, data, label, validation_split=0.3):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1,
                                                                   factor=0.25, min_lr=3e-07)
    remote_monitor = tf.keras.callbacks.RemoteMonitor(root='http://localohst:8080', path='/publish/epoch/end',
                                                      field='data', headers=None, send_as_json=True)

    model.fit(
        data, label,
        epochs=10,
        batch_size=32,
        validation_split=validation_split,
        callbacks=[remote_monitor, learning_rate_reduction, early_stop]
    )


