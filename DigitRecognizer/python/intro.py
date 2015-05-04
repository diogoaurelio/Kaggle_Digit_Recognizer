from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import numpy as np
from sklearn import decomposition #PCA
import random
import numpy as np


path = '/Users/DiogoAdmin/Documents/Developer/Machine Learning/Kaggle/DigitRecognizer'
train_csv = path + '/train.csv'
test_csv = path + '/test.csv'
train = pd.read_csv(train_csv)
labels = train.ix[:,0].values.astype('int32') #first column contains the labels, the rest are the features
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv(test_csv).values).astype('float32')

#convert labels to binary class Matrix
#class_binary_matrix = MultiLabelBinarizer().fit_transform(classes)
#print class_binary_matrix
#print Y_train

tmp = []
for i in labels: 
    if i not in tmp:
        tmp.append(i)
classes = np.array(sorted(tmp))
#class_binary_matrix
Y_train = [ [0 for i in classes] for j in classes]
for i in classes:
    Y_train[i][i] = 1

n_samples, n_features = train.shape
print 'Number of samples is: '+ str(n_samples)  #42k
print 'Number of features is: '+ str(n_features)

# pre-processing: divide by max and substract mean
#scale = np.max(X_train)
#X_train /= scale
#X_test /= scale
#
#mean = np.std(X_train)
#X_train -= mean
#X_test -= mean
#
#input_dim = X_train.shape[1]
#print 'Input dim: '+str(input_dim)
#nb_classes = len(tmp)
#print 'nb_classes: '+str(nb_classes)



#digits = train
# visualizing
#fig = plt.figure(figsize=(8,6))
#for i in range(15):
#    ax = fig.add_subplot(3, 5, i+1, xticks=[], yticks=[])
#    ax.imshow(lfw_people.images[i], cmap=plt.cm.bone)
#    ax.text(0, 7, str(digits.target[i]))



# Cross validation
#from sklearn.cross_validation import train_test_split
#X_train, X_test, Y_train, Ytest = train_test_split(train, random_state=0)

#from sklearn.cross_validation import train_test_split
from sklearn import decomposition
pca = decomposition.RandomizedPCA(n_components=150, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print X_train_pca.shape
print X_test_pca.shape


from sklearn import svm
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, Y_train)

#from sklearn import metrics
#Y_pred = clf.predict(X_test_pca)
#print(metrics.classification_report(Y_test, Y_pred, target_names=lfw_people.target_names))
#
#print(metrics.confusion_matrix(Y_test, Y_pred))
#
#print(metrics.f1_score(Y_test, Y_pred))

