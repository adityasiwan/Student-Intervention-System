
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"


n_students = len(student_data.index)

n_features = len(student_data.columns)-1

n_passed = student_data.passed.str.contains('yes').sum()

n_failed = student_data.passed.str.contains('no').sum() #or we can do n_failed=n_students-n_passed

grad_rate = (float(n_passed)/float(n_students))*100

print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

feature_cols = list(student_data.columns[:-1])

target_col = student_data.columns[-1] 

print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

X_all = student_data[feature_cols]
y_all = student_data[target_col]

print "\nFeature values:"
print X_all.head()


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():
        
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


from sklearn.cross_validation import train_test_split

num_train = 300

num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=num_train, test_size=num_test, random_state=42)

print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    train_classifier(clf, X_train, y_train)
    
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    print "\n"


from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

clf_A = tree.DecisionTreeClassifier()
clf_B = svm.SVC()
clf_C = GaussianNB()

X_train_100 = X_train.sample(n=100)
y_train_100 = y_train.sample(n=100)

X_train_200 = X_train.sample(n=200)
y_train_200 = y_train.sample(n=200)

X_train_300 = X_train.sample(n=300)
y_train_300 = y_train.sample(n=300)


train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)


from sklearn import svm, grid_search, datasets
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
parameters = [{'C': [1, 10, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
clf = SVC()

f1_scorer = make_scorer(f1_score, pos_label='yes')

grid_obj = grid_search.GridSearchCV(clf,param_grid = parameters, scoring = f1_scorer)

grid_obj.fit(X_train, y_train)

clf = grid_obj.best_estimator_

print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

