from types import SimpleNamespace


class BaseConfig(SimpleNamespace):
    name = 'BaseConfig'

    def __str__(self):
        desc = []
        for k, v in self.__dict__.items():
            desc.append(f'\n\t{k}={v}')
        return f'{self.name}(' + ', '.join(desc) + ')'

