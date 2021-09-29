from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, hamming_loss, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

'''
# KNN params for grid search
param_grid_knn = dict(n_neighbors = list(range(1,15)),weights = ['auto','ball_tree','kd_tree','brute'],algorithm=['uniform','distance'],leaf_size=list(range(1,5)))
# SVM params for grid search
param_grid_svm = dict(gamma=['scale', 'auto'], C=[10, 100, 1000, 10000, 100000], kernel=['poly','rbf'])
# SGD params for grid search
param_grid_sgd ={'alpha':(0.00001, 0.0001, 0.001, 0.01, 0.1), 'max_iter': (100, 1000, 10000), 'loss':('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'), 'learning_rate':('constant', 'optimal', 'invscaling', 'adaptive')}
# RF params for grid search
param_grid_rf ={'n_estimators':range(10,71,5), 'criterion':('gini', 'entropy'), 'max_features':('auto', 'sqrt', 'log2'), 'class_weight' :('balanced', 'balanced_subsample')}
'''
# grid search for every machine learning model
def choose_best_params(classifier, param_dict, X_train, y_train, scoring='accuracy'):
    param_grid_input = param_dict
    if(classifier == 'KNN'):
        gridKNN = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_input, scoring=scoring, cv=5)
        gridKNN.fit(X_train,y_train)
        best_score = gridKNN.best_score_
        best_params = gridKNN.best_params_
    elif(classifier == 'SVM'):
        gridSVM = GridSearchCV(SVC(probability=True), param_grid=param_grid_input, scoring=scoring, cv=5)
        gridSVM.fit(X_train,y_train)
        best_score = gridSVM.best_score_
        best_params = gridSVM.best_params_
    elif(classifier == 'RF'):
        gridRF= GridSearchCV(RandomForestClassifier(min_samples_split=100, min_samples_leaf=20, max_depth=8), param_grid=param_grid_input, scoring=scoring, cv=5)
        gridRF.fit(X_train,y_train)
        best_score = gridRF.best_score_
        best_params = gridRF.best_params_
    elif(classifier == 'SGD'):
        gridSGD= GridSearchCV(SGDClassifier(eta0 = 0.0001, class_weight = 'balanced'), param_grid=param_grid_input, scoring=scoring, cv=5)
        gridSGD.fit(X_train,y_train)
        best_score = gridSGD.best_score_
        best_params = gridSGD.best_params_
    else:
        raise ValueError("Incorrect data format, please input 'SVM' or 'KNN' or 'RF' or 'SGD'")
    return best_score, best_params

# train a knn binary classifier return metrics
def knn_binary_classifier(X_train, X_test, y_train, y_test, n_neighbors, weights, algorithm, leaf_size):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, algorithm=algorithm)
    cross_score = cross_val_score(knn_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(knn_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
    y_test_prob = knn_clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    f1 = f1_score( y_test, y_test_pred)
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    fpr, tpr, thersholds = roc_curve(y_test, y_test_prob, pos_label=1)
    return cross_score, conf_mx, f1, p, r, fpr, tpr
    
# train a svm binary classifier return metrics
def svm_binary_classifier(X_train, X_test, y_train, y_test, gamma, C, kernel):
    svm_clf = SVC(probability=True, gamma=gamma, C=C, kernel=kernel) 
    cross_score = cross_val_score(svm_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(svm_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
    y_test_prob = svm_clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    f1 = f1_score( y_test, y_test_pred)
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    fpr, tpr, thersholds = roc_curve(y_test, y_test_prob, pos_label=1)
    return cross_score, conf_mx, f1, p, r, fpr, tpr

# train a sgd binary classifier return metrics
def sgd_binary_classifier(X_train, X_test, y_train, y_test, max_iter, loss, alpha, learning_rate):
    sgd_clf = SGDClassifier(class_weight='balanced', eta0=0.0001, max_iter=max_iter, loss=loss, alpha=alpha, learning_rate=learning_rate)
    cross_score = cross_val_score(sgd_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
    y_test_prob = sgd_clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    f1 = f1_score( y_test, y_test_pred)
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    fpr, tpr, thersholds = roc_curve(y_test, y_test_prob, pos_label=1)
    return cross_score, conf_mx, f1, p, r, fpr, tpr
    
# train a rf binary classifier return metrics    
def rf_binary_classifier(X_train, X_test, y_train, y_test, class_weight, criterion, max_features, n_estimators):
    rf_clf= RandomForestClassifier(class_weight=class_weight, criterion=criterion, max_features=max_features, n_estimators=n_estimators)
    cross_score = cross_val_score(rf_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(rf_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
    y_test_prob = rf_clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    f1 = f1_score( y_test, y_test_pred)
    p = precision_score(y_test, y_test_pred)
    r = recall_score(y_test, y_test_pred)
    fpr, tpr, thersholds = roc_curve(y_test, y_test_prob, pos_label=1)
    return cross_score, conf_mx, f1, p, r, fpr, tpr
    
# train a knn multi classifier return metrics    
def knn_multi_classifier(X_train, X_test, y_train, y_test, n_neighbors, weights, algorithm, leaf_size):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, algorithm=algorithm)
    cross_score = cross_val_score(knn_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(knn_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
#     y_test_pred = knn_clf.predict(X_test)
    f1 = f1_score( y_test, y_test_pred, average='macro')
    p = precision_score(y_test, y_test_pred, average='macro')
    r = recall_score(y_test, y_test_pred, average='macro')
    kappa = cohen_kappa_score(y_test, y_test_pred)
    return cross_score, conf_mx, f1, p, r, kappa

# train a svm binary classifier return metrics    
def svm_multi_classifier(X_train, X_test, y_train, y_test, gamma, C, kernel):
    svm_clf = SVC(gamma=gamma, C=C, kernel=kernel) 
    cross_score = cross_val_score(svm_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(svm_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
#     svm_clf.fit(X_train, y_train)
#     y_test_pred = svm_clf.predict(X_test)
    f1 = f1_score( y_test, y_test_pred, average='macro')
    p = precision_score(y_test, y_test_pred, average='macro')
    r = recall_score(y_test, y_test_pred, average='macro')
    kappa = cohen_kappa_score(y_test,y_test_pred)
    return cross_score, conf_mx, f1, p, r, kappa

# train a sgd binary classifier return metrics
def sgd_multi_classifier(X_train, X_test, y_train, y_test, max_iter, loss, alpha, learning_rate):
    sgd_clf = SGDClassifier(class_weight='balanced', eta0=0.0001, max_iter=max_iter, loss=loss, alpha=alpha, learning_rate=learning_rate)
    cross_score = cross_val_score(sgd_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
#     y_test_pred = sgd_clf.predict(X_test)
    f1 = f1_score( y_test, y_test_pred, average='macro')
    p = precision_score(y_test, y_test_pred, average='macro')
    r = recall_score(y_test, y_test_pred, average='macro')
    kappa = cohen_kappa_score(y_test,y_test_pred)
    return cross_score, conf_mx, f1, p, r, kappa

# train a rf binary classifier return metrics    
def rf_multi_classifier(X_train, X_test, y_train, y_test, class_weight, criterion, max_features, n_estimators):
    rf_clf = RandomForestClassifier(class_weight=class_weight, criterion=criterion, max_features=max_features, n_estimators=n_estimators)
    cross_score = cross_val_score(rf_clf, X_train, y_train, cv=4)
    y_test_pred = cross_val_predict(rf_clf, X_test, y_test, cv=4)
    conf_mx = confusion_matrix(y_test, y_test_pred)
#     y_test_pred = rf_clf.predict(X_test)
    f1 = f1_score( y_test, y_test_pred, average='macro')
    p = precision_score(y_test, y_test_pred, average='macro')
    r = recall_score(y_test, y_test_pred, average='macro')
    kappa = cohen_kappa_score(y_test,y_test_pred)
    return cross_score, conf_mx, f1, p, r, kappa