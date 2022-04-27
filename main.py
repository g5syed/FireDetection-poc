import cv2  # used to grayscale and resize images
import numpy as np  # dealing with arrays
import os  # deal with directories
from random import shuffle  # mixing up our currently ordered data that might lead our network astray in training
from tqdm import tqdm  # a nice pretty percentage bar for tasks.
# import the model from sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

TRAIN_DIR = 'data/train'
IMG_SIZE = 50


def label_img(img):
    word_label = img.split('_')[0]
    # conversion to binary array [fire,no fire]
    if word_label == 'fire':
        return [1, 0]
    elif word_label == 'no fire':
        return [0, 1]
    else:
        print('error')


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # print(path, '/t/t', img.shape)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        cv2.waitKey(1)
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


train_data = create_train_data()

# Split training and testing data
train = train_data[:100]
test = train_data[100:]


def reshape_response(Y):
    train_Y = []
    for i in Y:
        if i[0] == 1 and i[1] == 0:
            train_Y.append(1)
        else:
            train_Y.append(0)
    n = len(train_Y)
    train_Y = np.array(train_Y).reshape((n, 1))
    return train_Y


def flatten_features(X):
    container = []
    m, n, p, q = np.shape(X)
    for i in range(0, m):
        temp = X[i].flatten()
        container.append(temp)
    flat_array = np.vstack(container)
    return flat_array


# Create data arrays, split into feature and response
trainX = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainX = flatten_features(trainX)
trainY = [i[1] for i in train]
trainY = reshape_response(trainY)

testX = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testX = flatten_features(testX)
testY = [i[1] for i in test]
testY = reshape_response(testY)

# Make an instance of the model
logreg = LogisticRegression()
# Fit a model on the training data
logreg.fit(trainX, trainY.ravel())

# Predict labels for the new data
predictions = logreg.predict(testX)

# Measure Model Performance: accuracy score
score = logreg.score(testX, testY)
print(score)

#Confusion Matrix
cm = metrics.confusion_matrix(testY, predictions)
print(cm)

plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, cmap= 'Blues_r')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
