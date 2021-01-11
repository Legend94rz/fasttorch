from queue import Queue
import threading
from threading import Thread, Lock
from multiprocessing.pool import ThreadPool
import torch as T
from torch.utils.data import DataLoader
import random


def _worker_init():
    local = threading.current_thread()
    local.__dict__['stream'] = T.cuda.Stream()


def _worker_main(args):
    batch, device = args
    with T.cuda.stream(threading.current_thread().__dict__['stream']):
        return [b.to(device) for b in batch]


class AsynchronousLoader:
    def __init__(self, dataloader, device, q_size=10):
        self.dataloader = dataloader
        self.device = device
        self.q_size = q_size
        self.inq = Queue()
        self.outq = Queue()
        self.front_buf = None     # list of tuple: ( (c1, c2, ..., cn), is_finish )
        self.idx = 0
        self.fill_threads = Thread(target=self._fill, args=(q_size, q_size), daemon=True)
        self.fill_threads.start()

    def _fill(self, njob, nworker):
        p = ThreadPool(nworker, initializer=_worker_init)
        while True:
            _ = self.inq.get()
            is_finish = False
            lst = []
            try:
                for _ in range(njob):
                    lst.append((next(self.it), self.device))
            except StopIteration:
                is_finish = True
            self.outq.put((p.map(_worker_main, lst), is_finish))

    def __iter__(self):
        self.idx = 0
        self.it = iter(self.dataloader)
        self.inq.put(0)
        self.front_buf = self.outq.get()
        return self

    def __next__(self):
        if self.idx == 0:
            self.inq.put(0)
        if self.idx >= len(self.front_buf[0]):
            self.outq.get()  # clear queue
            raise StopIteration
        res = self.front_buf[0][self.idx]
        self.idx += 1
        if (self.idx >= len(self.front_buf[0])-1) and not self.front_buf[1]:
            self.front_buf = self.outq.get()
            self.idx = 0
        return res

    def __len__(self):
        return len(self.dataloader)
