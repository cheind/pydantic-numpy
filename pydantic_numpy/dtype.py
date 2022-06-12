from typing import Any

import numpy as np
from pydantic import ValidationError
from pydantic.fields import ModelField


class BaseDType:
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


class single(np.float32, BaseDType):
    pass


float32 = single


class half(np.float16, BaseDType):
    pass


float16 = half


class intc(np.int32, BaseDType):
    pass


int32 = intc


class short(np.int16, BaseDType):
    pass


int16 = short
