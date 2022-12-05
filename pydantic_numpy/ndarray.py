from __future__ import annotations

import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Generic
from typing import Mapping
from typing import Optional
from typing import TypeVar
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib import NumpyVersion
from pydantic import BaseModel
from pydantic import FilePath
from pydantic import validator
from pydantic.fields import ModelField

if TYPE_CHECKING:
    from pydantic.typing import CallableGenerator

T = TypeVar("T", bound=np.generic)

if sys.version_info < (3, 9) or NumpyVersion(np.__version__) < "1.22.0":
    nd_array_type = np.ndarray
else:
    nd_array_type = np.ndarray[Any, T]


class NPFileDesc(BaseModel):
    path: FilePath = ...
    key: Optional[str] = None

    @validator("path")
    def absolute(cls, value: Path) -> Path:
        return value.resolve().absolute()


class _CommonNDArray(ABC):

    @classmethod
    @abstractmethod
    def validate(cls, val: Any, field: ModelField) -> nd_array_type:
        ...

    @classmethod
    def __modify_schema__(
        cls, field_schema: dict[str, Any], field: ModelField | None
    ) -> None:
        if field and field.sub_fields:
            type_with_potential_subtype = f"np.ndarray[{field.sub_fields[0]}]"
        else:
            type_with_potential_subtype = "np.ndarray"
        field_schema.update({"type": type_with_potential_subtype})

    @classmethod
    def __get_validators__(cls) -> CallableGenerator:
        yield cls.validate

    @staticmethod
    def _validate(val: Any, field: ModelField) -> nd_array_type:
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
    def validate(cls, val: Any, field: ModelField) -> nd_array_type:
        return cls._validate(val, field)


class PotentialNDArray(Generic[T], nd_array_type, _CommonNDArray):
    """Like NDArray, but validation errors result in None."""

    @classmethod
    def validate(cls, val: Any, field: ModelField) -> Optional[nd_array_type]:
        try:
            return cls._validate(val, field)
        except ValueError:
            return None
