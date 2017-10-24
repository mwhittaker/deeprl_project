"""Miscellaneous utilities"""

import pickle

def save_object(obj, filename):
    """Pickle an object to a file, overwriting if the file exists"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load an object from a pickled file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)
