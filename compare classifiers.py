from keras.datasets import mnist
from matplotlib import pyplot
from skimage.feature import hog
import numpy as np
from skimage import data, exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,


(train_X, train_y), (test_X, test_y) =mnist.load_data()
print ('X_train: ' + str(train_X.shape))
print ('Y_train: ' + str(train_y.shape))
print ('X_test: ' + str(test_X.shape))
print ('Y_test: ' + str(test_y.shape))

hog_images_train = []
hog_features_train = []
hog_images_test = []
hog_features_test = []

for i in range(1000):
  #pyplot.subplot(330 + 1 + i)
  #pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
  #pyplot.show()
  fd, hog_image = hog(train_X[i], orientations=8, pixels_per_cell=(2, 2),
                	cells_per_block=(2, 2), visualize=True)
  # pyplot.subplot(330 + 1 + i)
  # pyplot.imshow(hog_image, cmap=pyplot.get_cmap('gray'))
  hog_images_train.append(hog_image)
  hog_features_train.append(fd)
for i in range(200):
  fd, hog_image = hog(test_X[i], orientations=8, pixels_per_cell=(2, 2),
                	cells_per_block=(2, 2), visualize=True)
  hog_images_test.append(hog_image)
  hog_features_test.append(fd)
    

knn_Classifier = KNeighborsClassifier(n_neighbors=3)
knn_Classifier.fit(hog_features_train,train_y[:1000])
predicted_y1 = knn_Classifier.predict(hog_features_test)
knn_acc = accuracy_score(test_y[:200], predicted_y1)
print("Accuracy of knn Classifier: "+str(knn_acc))
print(classification_report(y_test, predicted_y1))

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(hog_features_train,train_y[:1000])
predicted_y2 = naive_bayes_classifier.predict(hog_features_test)
naive_acc = accuracy_score(test_y[:200], predicted_y2)
print("Accuracy of naive bayes classifier: "+str(naive_acc))
print(classification_report(y_test, predicted_y2))

svm_classifier = SVC()
svm_classifier.fit(hog_features_train,train_y[:1000])
predicted_y3 = svm_classifier.predict(hog_features_test)
svm_acc = accuracy_score(test_y[:200], predicted_y3)
print("Accuracy of svm classifier: "+str(svm_acc))
print(classification_report(y_test, predicted_y3))

if knn_acc == max(knn_acc,naive_acc,svm_acc):
  print("knn is the most accurate classifier with accuracy = " + str(knn_acc))
if naive_acc == max(knn_acc,naive_acc,svm_acc):
  print("naive bayes is the most accurate classifier with accuracy = " + str(naive_acc))
else:
  print("svm is the most accurate classifier with accuracy = " + str(svm_acc))