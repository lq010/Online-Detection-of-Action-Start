from multiprocessing import Process, Queue, Array
import numpy as np

class ArrayQueue(object):
    def __init__(self, template, maxsize=0):
        if type(template) is not np.ndarray:
            raise ValueError('ArrayQueue(template, maxsize) must use a numpy.ndarray as the template.')
        if maxsize == 0:
            # this queue cannot be infinite, because it will be backed by real objects
            raise ValueError('ArrayQueue(template, maxsize) must use a finite value for maxsize.')

        # find the size and data type for the arrays
        # note: every ndarray put on the queue must be this size
        self.dtype = template.dtype
        self.shape = template.shape
        self.byte_count = len(template.data)

        # make a pool of numpy arrays, each backed by shared memory, 
        # and create a queue to keep track of which ones are free
        self.array_pool = [None] * maxsize
        self.free_arrays = Queue(maxsize)
        for i in range(maxsize):
            buf = Array('c', self.byte_count, lock=False)
            self.array_pool[i] = np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
            self.free_arrays.put(i)

        self.q = Queue(maxsize)

    def put(self, item, *args, **kwargs):
        if type(item) is np.ndarray:
            if item.dtype == self.dtype and item.shape == self.shape and len(item.data)==self.byte_count:
                # get the ID of an available shared-memory array
                id = self.free_arrays.get()
                # copy item to the shared-memory array
                self.array_pool[id][:] = item
                # put the array's id (not the whole array) onto the queue
                new_item = id
            else:
                raise ValueError(
                    'ndarray does not match type or shape of template used to initialize ArrayQueue'
                )
        else:
            # not an ndarray
            # put the original item on the queue (as a tuple, so we know it's not an ID)
            new_item = (item,)
        self.q.put(new_item, *args, **kwargs)

    def get(self, *args, **kwargs):
        item = self.q.get(*args, **kwargs)
        if type(item) is tuple:
            # unpack the original item
            return item[0]
        else:
            # item is the id of a shared-memory array
            # copy the array
            arr = self.array_pool[item].copy()
            # put the shared-memory array back into the pool
            self.free_arrays.put(item)
            return arr