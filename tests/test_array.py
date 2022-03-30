from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import BaseSettings, ValidationError, BaseModel
from pydantic_numpy import NDArray, NPFileDesc

JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}


def test_numpy_field(tmpdir):
    class MySettings(BaseSettings):
        K: NDArray[np.float32]

        class Config:
            json_encoders = {np.ndarray: lambda arr: arr.tolist()}

    # Directly specify values
    cfg = MySettings(K=[1, 2])
    assert_allclose(cfg.K, [1.0, 2.0])
    assert cfg.K.dtype == np.float32
    assert cfg.json()

    cfg = MySettings(K=np.eye(2))
    assert_allclose(cfg.K, [[1.0, 0], [0.0, 1.0]])
    assert cfg.K.dtype == np.float32

    # Load from npy
    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npy"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32

    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K=NPFileDesc(path=Path(tmpdir) / "data.npy"))
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32

    np.savez(Path(tmpdir) / "data.npz", values=np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npz", "key": "values"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npz", "key": "values"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy", "key": "nosuchkey"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy"})

    with pytest.raises(ValidationError):
        MySettings(K="absc")

    # Not specifying a dtype will use numpy default dtype resolver

    class MySettingsNoGeneric(BaseSettings):
        K: NDArray

        class Config:
            json_encoders = {np.ndarray: lambda arr: arr.tolist()}

    cfg = MySettingsNoGeneric(K=[1, 2])
    assert_allclose(cfg.K, [1, 2])
    assert cfg.K.dtype == int

    assert cfg.json()

    # Optional test

    class MySettingsOptional(BaseSettings):
        K: Optional[NDArray]

    cfg = MySettingsOptional()

    class MyModelField(BaseModel):
        K: NDArray[np.float32]

        class Config:
            json_encoders = JSON_ENCODERS
            arbitrary_types_allowed = True

    model_field = MyModelField(K=[1.0, 2.0])
    assert model_field.json()

    class MyModel(BaseModel):
        L: dict[str, MyModelField]

        class Config:
            json_encoders = JSON_ENCODERS

    model = MyModel(L={"a": MyModelField(K=[1.0, 2.0])})
    assert model.L["a"].K.dtype == np.dtype("float32")
    assert model.json()
