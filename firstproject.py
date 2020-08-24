import scipy
print('Scipy: {}'.format(scipy._version_))
import numpy
print('Numpy: {}'.format(numpy._version_))
import matplotlib
print('Matplotlib: {}'.format(matplotlib._version_))
import pandas
print('Pandas: {}'.format(pandas._version_))
import sklearn
print('Sklearn: {}'.format(sklearn._version_))

import pandas 
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classfifcation_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecessionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection 
from sklearn.ensemble import VotingClassifier

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master.iris.csv"
names = ['sepal-length','sepal-width','petal-length','class']
dataset = read_csv(url , names=names)

#dimensions of the dataset 
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())

#univarient plots-box and whsiker plots 
dataset.plot(kind='box' , subplots=True , layout=(2,2), sharex=False , sharey=False)
pyplot.show()

#histogram 
dataset.hist()
pyplot.show()

#multivariate plots 
scatter_matrix(dataset)
pyplot.show()

#creating a validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]
x_train,x_validation,y_train,y_validaton = train_test_split(x,y,test size=0.2 , random_state = 1)

model = []
models.append(('LR',LogisticRegression(solver = 'libllinear' . multi_class = 'ovr')))
models.append(('LDA' , LinearDiscrimantAnalysis()))
models.append (('KNN' , KNieghborsClassifier()))
models.append(('SVM' , SVC(gamma = 'auto')))

results = []
names = []
for name, model in models:
    Kfold = StratifiedKFold(n_splits = 10 , random_state = 1)
    cv _results = cross_val_score(model, x_train,y_train,cv = Kfold,scoring = 'accuracy')
    results.append(cv_results)
    names.append(names)
    print('%s: %f (%f)' % (name, cv_results.mean(),cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Camparison')
pyplot.show()
#prediction
model = SVC(gamma = 'auto')
model.fit(x_train,y_train)
predictions = model.predict(x_validation)
#evaluate the predictions
print(accuracy_score(y_validaton,predictions))
print(confusion_matrix(y_validation,predictions))
print(classfication_report(y_validation,predictions))






