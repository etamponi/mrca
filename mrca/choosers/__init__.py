__author__ = 'Emanuele Tamponi'


def absolute_size(size, total):
    if isinstance(size, int):
        return size
    else:
        return int(size * total)
