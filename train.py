import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

NUM_CLASSES = 4

X_dataset = np.loadtxt('data.csv', delimiter=',', dtype='float32', usecols=list(range(0, 42)))
Y_dataset = np.loadtxt('data.csv', delimiter=',', dtype='int32', usecols=(42))

X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, train_size=0.85, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(21, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint('classifier.hdf5', verbose=1, save_weights_only=False)

es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=100000,
    batch_size=84,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

model = tf.keras.models.load_model('classifier.hdf5')

predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))
print(predict_result[0])
print(np.sum(predict_result[0]))