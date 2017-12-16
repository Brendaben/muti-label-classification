import my_pet
import logging
import sys
from os import path
from skmultilearn import dataset
from skmultilearn.problem_transform import *
from skmultilearn.adapt import MLkNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from datasets import *
from time import time

def get_classifiers():
    binary_relevance = BinaryRelevance(GaussianNB())
    classifier_chain = ClassifierChain(GaussianNB())
    label_powerset = LabelPowerset(GaussianNB())
    decision_tree = DecisionTreeClassifier(random_state=0)
    knn = MLkNN(k=20)
    random_forest = RandomForestClassifier(max_depth=2, random_state=0)
    clfs = [binary_relevance, classifier_chain, label_powerset, decision_tree, knn, random_forest]
    names = ['binary_relevance', 'classifier_chain', 'label_powerset', 'decision_tree', 'knn', 'random_forest']
    return clfs, names

def main(args):
    X_train, y_train, X_test, y_test = generate_random_data()
    dataset_name = 'random_generated'
    if len(args) > 0:
        dataset_name = args[0]
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)

    clfs, clf_names = get_classifiers()
    logger = my_pet.create_logger('../log/{}.log'.format(dataset_name))
    logger.info('dataset_name: {}'.format(dataset_name))
    for i, clf in enumerate(clfs):
        logger.info('classifier name: {}'.format(clf_names[i]))
        fit_time = time()
        clf.fit(X_train, y_train)
        fit_time = time() - fit_time
        logger.info('\t fit_time: {}s'.format(fit_time))
        
        y_pred_train = clf.predict(X_train)
        accuracy_train = metrics.accuracy_score(y_train, y_pred_train)

        y_pred_test = clf.predict(X_test)
        accuracy_test = metrics.accuracy_score(y_test, y_pred_test)
            
        logger.info('\t accuracy_train: {}'.format(accuracy_train))
        logger.info('\t accuracy_test: {}'.format(accuracy_test))

    if dataset_name == 'random_generated':
        logger.info('X_train: \n{}'.format(X_train))
        logger.info('y_train: \n{}'.format(y_train))
        logger.info('X_test: \n{}'.format(X_test))
        logger.info('y_test: \n{}'.format(y_test))
    logger.info('*'*75)
if __name__ == '__main__':
    main(sys.argv[1:])