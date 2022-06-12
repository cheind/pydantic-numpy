from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Optional, Type, TypeVar

import numpy as np
from numpy.lib import NumpyVersion
from pydantic import BaseModel, FilePath, validator
from pydantic.fields import ModelField

T = TypeVar("T", bound=np.generic)
nd_array_type = np.ndarray if NumpyVersion(np.__version__) < "1.22.0" else np.ndarray[Any, T]


class NPFileDesc(BaseModel):
    path: FilePath = ...
    key: Optional[str] = None

    @validator("path")
    def absolute(cls, value):
        return value.resolve().absolute()


class _CommonNDArray(ABC):
    @classmethod
    @abstractmethod
    def validate(cls, val: Any, field: ModelField):
        ...

    @classmethod
    def __modify_schema__(cls, field_schema, field: Optional[ModelField]):
        if field and field.sub_fields:
            type_with_potential_subtype = f"np.ndarray[{field.sub_fields[0]}]"
        else:
            type_with_potential_subtype = "np.ndarray"
        field_schema.update({"type": type_with_potential_subtype})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def _validate(cls: Type, val: Any, field: ModelField) -> "NDArray":
        if isinstance(val, Mapping):
            val = NPFileDesc(**val)

        if isinstance(val, NPFileDesc):
            val: NPFileDesc

            if val.path.suffix.lower() not in [".npz", ".npy"]:
                raise ValueError("Expected npz or npy file.")

            if not val.path.is_file():
                raise ValueError(f"Path does not exist {val.path}")

            try:
                content = np.load(str(val.path))
            except FileNotFoundError:
                raise ValueError(f"Failed to load numpy data from file {val.path}")

            if val.path.suffix.lower() == ".npz":
                key = val.key or content.files[0]
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
            return np.asarray(data, dtype=dtype_field.type_)
        return np.asarray(data)


class NDArray(Generic[T], nd_array_type, _CommonNDArray):
    @classmethod
    def validate(cls, val: Any, field: ModelField) -> np.ndarray:
        return cls._validate(val, field)


class PotentialNDArray(Generic[T], nd_array_type, _CommonNDArray):
    """Like NDArray, but validation errors result in None."""

    @classmethod
    def validate(cls, val: Any, field: ModelField) -> Optional[np.ndarray]:
        try:
            return cls._validate(val, field)
        except ValueError:
            return None
