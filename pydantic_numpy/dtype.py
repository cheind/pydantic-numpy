from typing import Any

import numpy as np
from pydantic import ValidationError
from pydantic.fields import ModelField

from pydantic_numpy.ndarray import NDArray


class _BaseDType:
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update({"type": cls.__name__})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val: Any, field: ModelField):
        if field.sub_fields:
            msg = f"{cls.__name__} has no subfields"
            raise ValidationError(msg)
        if not isinstance(val, cls):
            return cls(val)
        return val


class dobble(np.float64, _BaseDType):
    pass


float64 = dobble


class single(np.float32, _BaseDType):
    pass


float32 = single


class half(np.float16, _BaseDType):
    pass


float16 = half


class long(np.int64, _BaseDType):
    pass


int64 = long


class intc(np.int32, _BaseDType):
    pass


int32 = intc


class short(np.int16, _BaseDType):
    pass


int16 = short


class int8(np.int8, _BaseDType):
    pass


byte = int8


class uintp(np.uint64, _BaseDType):
    pass


uint64 = uintp


class uintc(np.uint32, _BaseDType):
    pass


uint32 = uintc


class ushort(np.uint16, _BaseDType):
    pass


uint16 = ushort


class uint8(np.uint8, _BaseDType):
    pass


ubyte = uint8


# NDArray typings

NDArrayFp64 = NDArray[float64]
NDArrayFp32 = NDArray[float32]
NDArrayFp16 = NDArray[float16]

NDArrayInt64 = NDArray[int64]
NDArrayInt32 = NDArray[int32]
NDArrayInt16 = NDArray[int16]
NDArrayInt8 = NDArray[int8]

NDArrayUint64 = NDArray[uint64]
NDArrayUint32 = NDArray[uint32]
NDArrayUint16 = NDArray[uint16]
NDArrayUint8 = NDArray[uint8]

NDArrayBool = NDArray[bool]
