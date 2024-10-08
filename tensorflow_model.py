import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np

# Define MNIST Model in TensorFlow
class TFMNISTClassifier(tf.keras.Model):
    def __init__(self, output_size = 10):
        super(TFMNISTClassifier, self).__init__()
        # Define the layers of the model
        self.conv1 = Conv2D(32, kernel_size=5, padding='same', activation='relu')
        self.pool1 = MaxPooling2D(pool_size=2)
        self.conv2 = Conv2D(64, kernel_size=5, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=2)
        self.flatten = Flatten()
        self.dense1 = Dense(1000, activation='relu')
        self.outputs = Dense(output_size, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.outputs(x)
        return x

# Training function
def train_tf():
    def custom_train_tf(model, train_dataset, epochs, cfg):
        print("Starting training...")
        optimizer = Adam(learning_rate=cfg.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(train_dataset, epochs=epochs, verbose=1)
        return model
    return custom_train_tf

# Testing function
def test_tf():
    def custom_test_tf(model, test_dataset):
        print("Starting evaluation...")

        results = model.evaluate(test_dataset, verbose=1)
        loss, accuracy = results[:2]

        # Predict classes to calculate F1 score
        all_labels = []
        all_predictions = []
        for images, labels in test_dataset:
            preds = model.predict(images)
            preds = np.argmax(preds, axis=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(preds)

        f1 = f1_score(all_labels, all_predictions, average='weighted')
        metrics = {'f1_score': f1}

        return loss, accuracy, metrics
    return custom_test_tf
