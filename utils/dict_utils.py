import copy
import json
import pickle


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __deepcopy__(self, name):
        copy_dict = dict()
        for key, value in self.items():
            if hasattr(value, '__deepcopy__'):
                copy_dict[key] = copy.deepcopy(value)
            else:
                copy_dict[key] = value
        return ObjDict(copy_dict)

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)


