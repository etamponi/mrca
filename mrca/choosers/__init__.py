__author__ = 'Emanuele Tamponi'


def absolute_size(size, inputs):
    if isinstance(size, int):
        return size
    else:
        return int(size * (len(inputs) - 1))
