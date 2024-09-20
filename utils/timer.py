import time

from utils.dict_utils import ObjDict


class Timer:
    def __init__(self, dec=1):
        self.dec = dec
        self.rec = ObjDict()
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def record(self, desc):
        rec_time = time.time()
        self.rec[desc] = rec_time - self.start_time
        self.start_time = rec_time
        return self.rec[desc]*self.dec

    def record_desc(self, desc):
        return f'{desc}: {self.record(desc)}'

    def __str__(self):
        desc = ''
        for k, v in self.rec.items():
            desc += f'{k}: {v*self.dec}\n'
        return desc
