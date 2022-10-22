from __future__ import annotations
from typing import Generic, TypeVar
from sortedcontainers import SortedKeyList

T = TypeVar("T")


class MaxHeapObj(object):
    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class SortedList(Generic[T]):
    __heap: SortedKeyList
    __capacity: int

    def __init__(self, capacity: int = None):
        self.__heap = SortedKeyList(key=lambda x: MaxHeapObj(x))
        self.__capacity = capacity

    def add(self, el: T):
        if self.__capacity is None or len(self.__heap) < self.__capacity:
            self.__heap.add(el)
        else:
            # Drop the worst performing thingy
            self.__heap.add(el)
            popped = self.__heap.pop()

    def __len__(self):
        return len(self.__heap)

    def __getitem__(self, index):
        return self.__heap[index]

    def get_list(self):
        return self.__heap
