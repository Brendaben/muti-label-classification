import logging
from os import path
from skmultilearn import dataset
from sklearn import datasets, model_selection

def load_dataset(dataset_name):
    logger = logging.getLogger()
    path_train = '../data/mulan/{}/{}-train.arff'.format(dataset_name, dataset_name)
    path_test = '../data/mulan/{}/{}-test.arff'.format(dataset_name, dataset_name)
    if (not path.exists(path_train)) or (not path.exists(path_test)):
        logger.debug('data set \"{}\" not found.'.format(dataset_name))
        return None, None, None, None
    X_train, y_train = dataset.load_from_arff(path_train, labelcount = 14, endian="little")
    logger.debug(X_train)
    X_test, y_test = dataset.load_from_arff(path_test, labelcount = 14, endian="little")
    return X_train.toarray(), y_train.toarray(), X_test.toarray(), y_test.toarray()

def generate_random_data():
    X, y = datasets.make_multilabel_classification(sparse = True, n_labels = 20,
        return_indicator = 'sparse', allow_unlabeled = False)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train.toarray(), y_train.toarray(), X_test.toarray(), y_test.toarray()
