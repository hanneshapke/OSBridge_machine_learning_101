import os
from PIL import Image
import numpy as np

import pickle
import gzip

from sklearn.cross_validation import train_test_split


_directory = '/Users/hannes/PetImages/training_set/'
categories = ['Cat/', 'Dog/']
max_samples = 5


data = list()
classification = list()

for animal in categories:
    print (animal)
    directory = _directory + animal
    _files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in _files[:max_samples]:
        try:
            image = Image.open(directory + f)
            image.load()
            image_matrix = np.asarray(image, dtype="int32").reshape(1, 200, 200)
            image_classification = 1 if animal == 'Cat/' else 0
            data.append(image_matrix)
            classification.append(image_classification)
        except IOError:
            continue

# shuffle
from sklearn.utils import shuffle
def reshuffle_dataset(X_data_set, y_data_set):
    return shuffle(np.stack(X_data_set, axis=0), np.stack(y_data_set, axis=0), random_state=0)

data, classification, = reshuffle_dataset(data, classification)

# create split set
X_train, X_test, y_train, y_test = train_test_split(data, classification, test_size=0.20, random_state=42)

#store the object
f = gzip.open('petsTrainingData.pklz', 'wb')
pickle.dump([X_train, X_test, y_train, y_test], f)
f.close()
