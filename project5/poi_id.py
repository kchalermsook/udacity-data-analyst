#!/usr/bin/python

import sys
import pickle
import pprint

sys.path.append("../tools/")

import numpy as np
import math
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform as sp_rand


def maxRecall_scorer(clf, features, labels):
    cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        return recall
    except:
        return 0


# Algolithm Setting
###4.1  Naive Bayes
from sklearn.naive_bayes import GaussianNB

n_clf = GaussianNB()
n_clf_s = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', n_clf)])
n_clf_m = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', n_clf)])
n_clf_r = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('classifier', n_clf)])
n_clf_ma = Pipeline(steps=[
    ('scaler', MaxAbsScaler()),
    ('classifier', n_clf)])
###4.2  Decision Tree
# param_grid = {'alpha': sp_rand()}
# RandomizedSearchCV(estimator=n_clf, param_distributions=param_grid, n_iter=100)
# criterion = np.array(['gini','entropy'])

from sklearn import tree

d_clf = tree.DecisionTreeClassifier()
d_clf_s = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', d_clf)])
d_clf_m = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', d_clf)])
d_clf_r = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('classifier', d_clf)])
d_clf_ma = Pipeline(steps=[
    ('scaler', MaxAbsScaler()),
    ('classifier', d_clf)])
# param_grid = {
#     'min_samples_split': [2, 3,4],
#     'splitter': ['best', 'random']
# }
# d_clf = GridSearchCV(estimator=d_clf, param_grid=param_grid, cv= 5)
###4.3. Random Forest
from sklearn.ensemble import RandomForestClassifier

r_clf = RandomForestClassifier(n_jobs=-1, max_features='auto', n_estimators=200, oob_score=True)
r_clf_s = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', r_clf)])
r_clf_m = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', r_clf)])
r_clf_r = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('classifier', r_clf)])
r_clf_ma = Pipeline(steps=[
    ('scaler', MaxAbsScaler()),
    ('classifier', r_clf)])
# r_clf = RandomForestClassifier()
# param_grid = {
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# r_clf = GridSearchCV(estimator=r_clf, param_grid=param_grid, cv= 5)

# r_clf = Pipeline(steps=[
#     ('scaler', StandardScaler()),
#     ('classifier', RandomForestClassifier(n_jobs=-1, max_features='auto', n_estimators=200, oob_score=True))])

### 4.4 SGDClassifier http://scikit-learn.org/stable/tutorial/machine_learning_map/
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss="log", penalty="l2")
sgd_clf_s = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', sgd_clf)])
sgd_clf_m = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier', sgd_clf)])
sgd_clf_r = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('classifier', sgd_clf)])
sgd_clf_ma = Pipeline(steps=[
    ('scaler', MaxAbsScaler()),
    ('classifier', sgd_clf)])
### 4.5 LogisticRegression
from sklearn.linear_model import LogisticRegression

lgr =  LogisticRegression(tol=0.05, C=10 ** -8, penalty='l2', random_state=42)

l_clf_s = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier',lgr)])
l_clf_m = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('classifier',lgr)])
l_clf_r = Pipeline(steps=[
    ('scaler', RobustScaler()),
    ('classifier',lgr)])
l_clf_ma = Pipeline(steps=[
    ('scaler', MaxAbsScaler()),
    ('classifier',lgr)])
# l_clf = LogisticRegression(tol=0.05, C=10 ** -8, penalty='l2', random_state=42)
# l_parameters = {
#               'tol': [0.1, 0.05, 0.01, 0.001],
#               # 'C': [1, 10, 100],
#               # 'penalty': ['l1', 'l2'],
#               # 'random_state': [10, 20, 30, 40]
#               }
# l_clf = GridSearchCV(l_clf, l_parameters, scoring= maxRecall_scorer)
all_clf = []
# all_clf = all_clf + [n_clf, d_clf, r_clf, sgd_clf]
all_clf = all_clf + [r_clf]

NUM_FEATURE_START = 3
NUM_FEATURE_END = 4

max_algo_dict = {}


def add_features(data_dict, features_list):
    """
    Given the data dictionary of people with features, adds some features to
    """
    for name in data_dict:
        record = data_dict[name]
        have_to_ratio = True
        if record['exercised_stock_options'] == 'NaN':
            record

        if record['to_messages'] == 'NaN' or record['from_poi_to_this_person'] == 'NaN':
            have_to_ratio = False
        have_from_ratio = True
        if record['from_messages'] == 'NaN' or record['from_this_person_to_poi'] == 'NaN':
            have_from_ratio = False

        if have_to_ratio:
            record['to_ratio'] = float(record['from_poi_to_this_person']) / record['to_messages']
        else:
            record['to_ratio'] = 'NaN'

        if have_from_ratio:
            record['from_ratio'] = float(record['from_this_person_to_poi']) / record['from_messages']
        else:
            record['from_ratio'] = 'NaN'

    # print "finished"
    return data_dict


def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,
                                                                                                     test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))

    print "done.\n"
    print "class : {}".format(clf.__class__.__name__)
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    print "accuracy:    {}".format(mean(accuracy))
    return mean(precision), mean(recall), mean(accuracy)


def evaluate_clf_s(clf, features, labels, folds=1000):
    cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        # print clf.best_params_
        predictions = clf.predict(features_test)

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        algo_name = clf.__class__.__name__

        print "done.\n"
        print "class : {}".format(clf.__class__.__name__)
        print "precision: {}".format(precision)
        print "recall:    {}".format(recall)
        print "accuracy:    {}".format(accuracy)
        print '### feature importance'
        # print(clf.best_score_)
        # params = PARAMS_DICT[clf]
        # for eachParam in  params:
        #     clf.best_estimator_[eachParam]
        return precision, recall, accuracy
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


def train_and_test(features, labels, features_list):
    max_acc = 0
    max_precision = 0
    max_recall = 0
    target_num_feature = 0
    target_clf = None
    max_features_list = features_list
    for num in range(NUM_FEATURE_START, NUM_FEATURE_END):

        best_features = get_k_best(num, features, labels, features_list)

        # new_features_list = [target_label] + best_features.keys()

        # add new feature here to see the result
        new_features_list = features_list
        # extract data again with only interested features

        new_data = featureFormat(my_dataset, new_features_list)
        new_labels, new_features = targetFeatureSplit(new_data)
        print "Current Number of Features : {}".format(num)
        precision, recall, accuracy, max_clf = best_algo(all_clf, new_features, new_labels, new_features_list)
        if (recall > 0.3 and precision > 0.3 and  recall > max_recall):
            max_acc = accuracy
            max_precision = precision
            max_recall = recall
            target_num_feature = num
            target_clf = max_clf
            max_features_list = new_features_list

    return target_clf, max_acc, max_precision, max_recall, target_num_feature, max_features_list


def best_algo(clf_arr, new_features, new_labels, new_features_list):
    max_recall = 0
    max_clf = None
    max_precision = 0
    max_acc = 0
    for clf in clf_arr:
        precision, recall, accuracy = evaluate_clf_s(clf, new_features, new_labels)
        algo_name = clf.__class__.__name__
        if algo_name in max_algo_dict:
            if (precision + recall > max_algo_dict[algo_name]['precision'] +
                max_algo_dict[algo_name]['recall']):
                max_algo_dict[algo_name]['precision'] = precision
                max_algo_dict[algo_name]['recall'] = recall
                max_algo_dict[algo_name]['accuracy'] = accuracy
                max_algo_dict[algo_name]['features'] = new_features_list

        else:
            max_algo_dict[algo_name] = {}
            max_algo_dict[algo_name]['precision'] = precision
            max_algo_dict[algo_name]['recall'] = recall
            max_algo_dict[algo_name]['accuracy'] = accuracy
            max_algo_dict[algo_name]['features'] = new_features_list

        if (recall > 0.3 and precision > 0.3 and  recall + precision > max_recall + max_precision):
            max_recall = recall
            max_clf = clf
            max_acc = accuracy
            max_precision = precision
    return max_precision, max_recall, max_acc, max_clf


def get_k_best(num_top_features, features, labels, features_list):
    k_best = SelectKBest(k=num_top_features)
    k_best.fit(features, labels)
    results_list = zip(features_list[1:], k_best.scores_)
    results_list = list(reversed(sorted(results_list, key=lambda x: x[1])))
    print results_list
    best_features = dict(results_list[:num_top_features])
    return best_features


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
all_finance_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                        'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                        'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
all_email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                      'shared_receipt_with_poi']
added_features = ['from_ratio']
target_label = 'poi'
selected_features = ['exercised_stock_options', 'total_stock_value', 'bonus']
all_features = [target_label] + all_finance_features + all_email_features
features_list = [target_label] + selected_features
# features_list = all_features + added_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Dataset lenght : {}".format(len(data_dict))

numPoi = sum(data_dict[d]['poi'] is True for d in data_dict)
print "Number of POI/ Non POI : {}/{}".format(numPoi, len(data_dict) - numPoi)
print "Number of Finance Features : {}".format(len(all_finance_features))
print "Number of Email Features : {}".format(len(all_email_features))

### Task 2: Remove outliers
# Print Employee name by sorting using key.
nan_arr = {}
total_cal = {}
complete_data_by_employee = {}
for employee in sorted(data_dict):
    # print employee
    if (employee == 'TOTAL'):
        print pprint.pprint(data_dict[employee])
    # pprint.pprint(data_dict[employee])
    current_data = data_dict[employee]
    num_incomplete = 0
    for each_feature in all_features:
        if (current_data[each_feature] == 'NaN'):
            num_incomplete = num_incomplete + 1
            if each_feature in nan_arr:
                nan_arr[each_feature] = nan_arr[each_feature] + 1
            else:
                nan_arr[each_feature] = 1
    complete_data_by_employee[employee] = num_incomplete
import operator

nan_arr = sorted(nan_arr.items(), key=operator.itemgetter(1))
complete_data_by_employee = sorted(complete_data_by_employee.items(), key=operator.itemgetter(1))
print "Number of NaN data for each feature : ", nan_arr
print "Number of Incomplete data by employee : ", complete_data_by_employee

# Looking through all the names, I would like to know what is "THE TRAVEL AGENCY IN THE PARK" and "TOTAL"
pprint.pprint(data_dict["THE TRAVEL AGENCY IN THE PARK"])
pprint.pprint(data_dict["TOTAL"])
# the result show that this person seems to have many NaN value. "Total" seems to be the total row..

###Remove these two key from the dict
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# ratio of to

data_dict = add_features(data_dict, features_list)

# for employee in sorted(data_dict):
# print employee
# pprint.pprint(data_dict[employee])
my_dataset = data_dict

### Extract features and labels from dataset for local testing

# Scale features
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

## Task 4: Try a varity of classifiers
## Please name your classifier clf for easy export below.
## Note that if you want to do PCA or other multi-stage operations,
## you'll need to use Pipelines. For more info:
## http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# reference http://scikit-learn.org/stable/tutorial/machine_learning_map/

l_clf, max_acc, max_precision, max_recall, target_num_feature, features_list = train_and_test(features, labels,
                                                                                              features_list)

print "target num features:    {}".format(target_num_feature)
print "target accuracy:    {}".format(max_acc)
print "target max precision:    {}".format(max_precision)
print "target recall:    {}".format(max_recall)
print "target algo: {}".format(l_clf.__class__.__name__)
print "target feature list : {}".format(features_list)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print max_algo_dict
clf = l_clf
dump_classifier_and_data(clf, my_dataset, features_list)
