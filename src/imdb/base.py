__all__ = [
    'CLASSIFIER_LOGISTIC_REGRESSION', 'CLASSIFIER_LINEAR_SVG',
    'ReviewAnalyzer']


import re
from os import path
from typing import List, Tuple, Callable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ..config import Config


CLASSIFIER_LOGISTIC_REGRESSION = 'lr'
CLASSIFIER_LINEAR_SVG = 'lv'
CLASSIFIERS = {
    CLASSIFIER_LOGISTIC_REGRESSION: LogisticRegression,
    CLASSIFIER_LINEAR_SVG: LinearSVC,
}


class ReviewAnalyzer(object):
    __TRAIN_FILENAME__: str = None
    __TEST_FILENAME__: str = None
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    train_initial: List[Tuple[str, bool]] = []
    test_initial: List[Tuple[str, bool]] = []
    accuracy: float = 0.0
    classfier_class: str = CLASSIFIERS[CLASSIFIER_LOGISTIC_REGRESSION]

    def __init__(
        self,
        train_dataset_filename: str = None,
        test_dataset_filename: str = None,
        classfier_mode: str = CLASSIFIER_LOGISTIC_REGRESSION,
        **kwargs):
        """Constructor
        """
        self.__TRAIN_FILENAME__ = train_dataset_filename
        self.__TEST_FILENAME__ = test_dataset_filename
        self.classfier_class = CLASSIFIERS[classfier_mode]\
            if classfier_mode in CLASSIFIERS\
            else CLASSIFIERS[CLASSIFIER_LOGISTIC_REGRESSION]

    def __preprocess(self, line: str) -> str:
        line = self.REPLACE_NO_SPACE.sub("", line)
        line = self.REPLACE_WITH_SPACE.sub(" ", line)
        return re.sub(r'\s+', ' ', line).strip()

    def import_dataset(
        self,
        tran_dataset_filename: str = None,
        test_dataset_filename: str = None):
        """method to read dataset
        """
        if tran_dataset_filename:
            self.__TRAIN_FILENAME__ = tran_dataset_filename

        if test_dataset_filename:
            self.__TEST_FILENAME__ = test_dataset_filename
        
        if not self.__TRAIN_FILENAME__ or not self.__TEST_FILENAME__ or\
                not path.isfile(self.__TRAIN_FILENAME__) or\
                not path.isfile(self.__TEST_FILENAME__):
            raise ImportError(tran_dataset_filename, test_dataset_filename)

        with open(self.__TRAIN_FILENAME__, mode='r') as f:
            self.train_initial = [
                self.__preprocess(line) for line in f.readlines()]

        with open(self.__TEST_FILENAME__, mode='r') as f:
            self.test_initial = [
                self.__preprocess(line) for line in f.readlines()]

    def vectorize(
        self,
        binary: bool = True,
        ngram_range: Tuple[int, int] = (1, 1)):
        """method to vectorize dataset
        """
        if not self.train_initial or not self.test_initial:
            raise Exception("Dataset is blank!")

        self.cv = CountVectorizer(binary=binary, ngram_range=ngram_range)
        self.cv.fit(self.train_initial)
        self.X = self.cv.transform(self.train_initial)
        self.X_test = self.cv.transform(self.test_initial)

    def train(self):
        target = [
            1 if i < len(self.train_initial) / 2 else 0
            for i in range(len(self.train_initial))]

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, target, train_size=0.75)

        # NOTE: Find the best accuracy
        for c in [0.01]:  #, 0.05, 0.1, 0.15, 0.2]:
            lr = self.classfier_class(C=c, max_iter=10000)
            lr.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, lr.predict(X_val))
            if accuracy > self.accuracy:
                self.accuracy = accuracy
            print ("Accuracy for C=%s: %s" % (c, accuracy))

        # NOTE: Train the final model
        self.final_model = self.classfier_class(
            C=self.accuracy, max_iter=10000)
        self.final_model.fit(self.X, target)
        final_accuracy = accuracy_score(target, self.final_model.predict(self.X_test))
        print ("Final Accuracy: %s" % final_accuracy)

        # NOTE: Testing with discriminating samples for both of P & N
        feature_to_coef = {
            word: coef for word, coef in zip(
                self.cv.get_feature_names(), self.final_model.coef_[0]
            )
        }

        for best_positive in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1], 
            reverse=True)[:5]:
            print (best_positive)

        for best_negative in sorted(
            feature_to_coef.items(), 
            key=lambda x: x[1])[:5]:
            print (best_negative)
        
        print("Training completed!")
        # return final_model, final_accuracy

    def build(
        self,
        tran_dataset_filename: str = None,
        test_dataset_filename: str = None,
        binary: bool = True,
        ngram_range: Tuple[int, int] = (1, 1)):
        """starting point of ml based sentiment analysis model
        """
        self.import_dataset(
            tran_dataset_filename=tran_dataset_filename,
            test_dataset_filename=test_dataset_filename)

        self.vectorize(binary=binary, ngram_range=ngram_range)
        return self.train()

    def test(self, sample: str):
        item = self.cv.transform([sample])
        self.final_model.predict(item)

