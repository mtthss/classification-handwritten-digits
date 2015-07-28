__author__ = 'matteo'

# import
from preprocess import load_my_data

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


######################################
# PREPROCESSING AND DATA EXPLORATION #
######################################

# initialization
path1 = "train_data"
path2 = "test_data"

# load data
(X_tr,y_tr) = load_my_data(path1)
(X_tst,y_tst) = load_my_data(path2)

print X_tr.shape, y_tr.shape
print X_tst.shape, y_tst.shape


###################################
# EVALUATE HYPERPARAMETER SETTING #
###################################

n_folds = 5

best_score = 0
best_modelLR = None
regularization = ['l2','l1']
for t in regularization:
    clf = LogisticRegression(dual=False, penalty=t)
    scores = cross_validation.cross_val_score(clf, X_tr, y_tr, cv=n_folds)
    if scores.mean()>best_score:
        best_modelLR = t
        best_score = scores.mean()
    print("\nLogReg, "+t+" regularized; CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

best_score = 0
best_modelRF = None
estimators = [20,40,80,160]
for num in estimators:
    clf = RandomForestClassifier(n_estimators=180,min_samples_split=4)
    scores = cross_validation.cross_val_score(clf, X_tr, y_tr, cv=n_folds)
    if scores.mean()>best_score:
        best_modelRF = num
        best_score = scores.mean()
    print("\nRandForests, "+ str(num) +" estimators; CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


best_score = 0
best_modelSVM = None
degree = [1,2,3,4]
for d in degree:
    clf = svm.SVC(kernel='poly', degree=d, C=1)
    scores = cross_validation.cross_val_score(clf, X_tr, y_tr, cv=n_folds)
    if scores.mean()>best_score:
        best_modelSVM = d
        best_score = scores.mean()
    print("\nKernelSVM, poly degree: "+ str(d) +"; CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


##############
# TESTING LR #
##############

print '\nLogistic Regression, '+best_modelLR+' regularization'
clf = LogisticRegression(dual=False, penalty=best_modelLR)
clf.fit(X_tr,y_tr)
ypred = clf.predict(X_tst)

cm = confusion_matrix(y_tst, ypred)
f1_score_macro = f1_score(y_tst, ypred, average='macro')
precision_macro = precision_score(y_tst, ypred, average='macro')
recall_macro = recall_score(y_tst, ypred, average='macro')

print "Confusion Matrix:\n", cm
print "F1 score: ", f1_score_macro
print "Precision: ", precision_macro
print "Recall", recall_macro


##############
# TESTING RF #
##############

print '\nRandom Forests, '+str(num)+' estimators'
clf = RandomForestClassifier(n_estimators=best_modelRF,min_samples_split=4)
clf.fit(X_tr,y_tr)
ypred = clf.predict(X_tst)

cm = confusion_matrix(y_tst, ypred)
f1_score_macro = f1_score(y_tst, ypred, average='macro')
precision_macro = precision_score(y_tst, ypred, average='macro')
recall_macro = recall_score(y_tst, ypred, average='macro')

print "Confusion Matrix:\n", cm
print "F1 score: ", f1_score_macro
print "Precision: ", precision_macro
print "Recall", recall_macro


###############
# TESTING SVM #
###############

print '\nSVM, '+str(best_modelSVM)+' degree'
clf = svm.SVC(kernel='poly', degree=best_modelSVM, C=1)
clf.fit(X_tr,y_tr)
ypred = clf.predict(X_tst)

cm = confusion_matrix(y_tst, ypred)
f1_score_macro = f1_score(y_tst, ypred, average='macro')
precision_macro = precision_score(y_tst, ypred, average='macro')
recall_macro = recall_score(y_tst, ypred, average='macro')

print "Confusion Matrix:\n", cm
print "F1 score: ", f1_score_macro
print "Precision: ", precision_macro
print "Recall", recall_macro