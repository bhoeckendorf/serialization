from enum import Enum


class ArrayType(Enum):
    NUMPY = 1
    PYTORCH = 2
    PIL = 3


class PackedArray:

    def __init__(self, arraytype, dtype, shape, buffer, mode=None):
        self.arraytype = arraytype
        self.dtype = dtype
        self.shape = shape
        self.mode = mode
        self.buffer = buffer
