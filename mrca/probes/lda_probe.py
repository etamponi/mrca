import numpy
from sklearn.lda import LDA

__author__ = 'Emanuele Tamponi'


class LDAProbe(object):

    def __init__(self):
        pass

    def __call__(self, x, y, inputs, labels):
        classes = numpy.unique(labels)
        if len(classes) == 1:
            if y == classes[0]:
                return 1
            else:
                return -1
        lda = LDA().fit(inputs, labels)
        prob = lda.predict_proba([x])[0][lda.classes_.tolist().index(y)]
        return 2 * prob - 1

    def prepare(self, inputs, labels):
        pass
