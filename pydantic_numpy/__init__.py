from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import numpy as np
from numpy.lib import NumpyVersion
from pydantic import BaseModel, ValidationError
from pydantic.fields import ModelField

T = TypeVar("T", bound=np.generic)


class NPFileDesc(BaseModel):
    path: Path
    key: Optional[str]


if NumpyVersion(np.__version__) < "1.22.0":

    class NDArray(Generic[T], np.ndarray):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField):
            return _validate(cls, val, field)


else:

    class NDArray(Generic[T], np.ndarray[Any, T]):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField):
            return _validate(cls, val, field)


def _validate(cls: Type, val: Any, field: ModelField) -> np.ndarray:
    if isinstance(val, Mapping):
        val = NPFileDesc(**val)
    if isinstance(val, NPFileDesc):
        val: NPFileDesc
        path = val.path.resolve().absolute()
        key = val.key
        if path.suffix.lower() not in [".npz", ".npy"]:
            raise ValidationError("Expected npz or npy file.")

        try:
            content = np.load(str(path))
        except FileNotFoundError:
            raise ValidationError(f"Failed to load numpy data from file {path}")
        if path.suffix.lower() == ".npz":
            key = key or content.files[0]
            data = content[key]
        else:
            data = content
    else:
        data = val

    if field.sub_fields is not None:
        dtype_field = field.sub_fields[0]
        return np.array(data, dtype=dtype_field.type_)
    else:
        return np.array(data)
