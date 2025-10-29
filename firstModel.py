#first model
import tensorflow as tf
import matplotlib.pyplot as plt

#creating X and Y values
X = tf.range(-100,100,5)
Y = 2*X+5

#Spliting the data into training data and test data
X_train = X[:30]
Y_train = Y[:30]
X_test = X[30:]
Y_test = Y[30:]

#visualizing the data before training
plt.figure(figsize=(11,8))
plt.scatter(X_train,Y_train, c="b", label="TrainingData")
plt.scatter(X_test,Y_test,c="g", label="TestingData")
plt.legend()
plt.title("Before training")
plt.show()

#creating a model for prediction
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1],activation="relu",name="InputLayer"),
    tf.keras.layers.Dense(10, name="HiddenLayer"), 
    tf.keras.layers.Dense(1,name="OutputLayer")
    ])

#compiling the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=["mae"])


model.fit(X_train, Y_train, epochs=100)
Y_pred = model.predict(X_test)

#visualizing after prediction
plt.figure(figsize=(11,8))
plt.scatter(X_train, Y_train, c="b",label="TrainingData")
plt.scatter(X_test, Y_test, c="g", label="TestingData")
plt.scatter(X_test, Y_pred, c="r", label="Predictions")
plt.legend()
plt.title("After training")
plt.show()