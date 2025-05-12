import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import Sequence
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (200, 200)  # Resize images to this size
AGE_LABELS = ["0-2", "3-12", "13-19", "20-30", "31-40", "41-50", "51-60", "61-70", "71-80", "80+"]
AGE_MAP = {age: i for i, age in enumerate(AGE_LABELS)}

# Path to UTKFace dataset
dataset_path = "E:/age_prediction/UTKFace/"  # Replace with your actual path

# Generator to load data in batches
class AgeDataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=32, image_size=(200, 200), shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.images[k] for k in batch_indexes]
        images, labels = self.__data_generation(batch_images)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_images):
        images = []
        labels = []
        for image_name in batch_images:
            parts = image_name.split("_")
            age = int(parts[0])  # Extract age from filename

            # Classify age into categories
            if age <= 2:
                label = 0
            elif age <= 12:
                label = 1
            elif age <= 19:
                label = 2
            elif age <= 30:
                label = 3
            elif age <= 40:
                label = 4
            elif age <= 50:
                label = 5
            elif age <= 60:
                label = 6
            elif age <= 70:
                label = 7
            elif age <= 80:
                label = 8
            else:
                label = 9

            # Load image
            img_path = os.path.join(self.dataset_path, image_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, self.image_size)  # Resize to the desired size
            images.append(img)
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        labels = np.eye(len(AGE_LABELS))[labels]  # One-hot encode the labels
        images = images / 255.0  # Normalize images
        return images, labels


# Split dataset into training and testing (80% train, 20% test)
train_gen = AgeDataGenerator(dataset_path, batch_size=32, image_size=IMAGE_SIZE, shuffle=True)
test_gen = AgeDataGenerator(dataset_path, batch_size=32, image_size=IMAGE_SIZE, shuffle=False)

# Define the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))  # 10 age categories

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
history = model.fit(train_gen, validation_data=test_gen, epochs=10)

# Save the trained model to a file
model.save('age_model.h5')
