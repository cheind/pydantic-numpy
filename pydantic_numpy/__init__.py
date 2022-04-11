from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import numpy as np
from numpy.lib import NumpyVersion
from pydantic import BaseModel
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
        def validate(cls, val: Any, field: ModelField) -> np.ndarray:
            return _validate(cls, val, field)

    class PotentialNDArray(Generic[T], np.ndarray):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
            try:
                return _validate(cls, val, field)
            except ValueError:
                return None


else:

    class NDArray(Generic[T], np.ndarray[Any, T]):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
            return _validate(cls, val, field)

    class PotentialNDArray(Generic[T], np.ndarray[Any, T]):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
            try:
                return _validate(cls, val, field)
            except ValueError:
                return None


def _validate(cls: Type, val: Any, field: ModelField) -> np.ndarray:
    if isinstance(val, Mapping):
        val = NPFileDesc(**val)
    if isinstance(val, NPFileDesc):
        val: NPFileDesc
        path = val.path.resolve().absolute()
        key = val.key
        if path.suffix.lower() not in [".npz", ".npy"]:
            raise ValueError("Expected npz or npy file.")
        if not path.is_file():
            raise ValueError(f"Path does not exist {path}")
        try:
            content = np.load(str(path))
        except FileNotFoundError:
            raise ValueError(f"Failed to load numpy data from file {path}")
        if path.suffix.lower() == ".npz":
            key = key or content.files[0]
            try:
                data = content[key]
            except KeyError:
                raise ValueError(f"Key {key} not found in npz.")
        else:
            data = content
    else:
        data = val

    if field.sub_fields is not None:
        dtype_field = field.sub_fields[0]
        return np.array(data, dtype=dtype_field.type_)
    else:
        return np.array(data)
