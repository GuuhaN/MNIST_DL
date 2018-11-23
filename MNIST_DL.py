# Import all the libraries
import tensorflow as tf
import matplotlib.pyplot as matlib
import numpy as np

mnist = tf.keras.datasets.mnist #28x28 pixels images of hand-written digits 0-9

# Load the MNIST dataset into arrays of training images & test images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to easier classification for the training session
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Creating a model for the network
model = tf.keras.models.Sequential()

# Hidden layers
model.add(tf.keras.layers.Flatten())

# Dense parameters (units aka how many nodes, activation algorithm)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# Build the neural network
model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy'
                ,metrics=['accuracy'])

# Train the network - if already trained, comment those both out
# x_train & y_train = training program, epoch = how many iterations of the network it goes through
model.fit(x_train, y_train, epochs=3) 
# Save training model to reuse for other applications
model.save('test-tensorflow')
# Load the trained model
new_model = tf.keras.models.load_model('test-tensorflow')
predictions = new_model.predict([x_test])

# Predict number
predictNumber = 0

# Get the value of the predictions made in the network
print('Predicted number: '+ str(np.argmax(predictions[predictNumber])))

matlib.imshow(x_test[predictNumber])
matlib.show()