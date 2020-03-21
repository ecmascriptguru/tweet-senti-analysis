import os
# from src.vader import Demo
from src.imdb import ReviewAnalyzer, CLASSIFIER_LINEAR_SVG
from src.config import Config


def test_imdb_reviews():
    toolkit = ReviewAnalyzer(classfier_mode=CLASSIFIER_LINEAR_SVG)
    train = os.path.join(Config.BASE_DIR, '.tmp', 'full_train.txt')
    test = os.path.join(Config.BASE_DIR, '.tmp', 'full_test.txt')
    toolkit.build(train, test, ngram_range=(1, 2))
    toolkit.test("Very good")
    print(accuracy)


def run():
    test_imdb_reviews()


if __name__ == "__main__":
    run()