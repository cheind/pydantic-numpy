from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import BaseModel, ValidationError

import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray, NPFileDesc, PotentialNDArray

JSON_ENCODERS = {np.ndarray: lambda arr: arr.tolist()}


class MySettings(BaseModel):
    K: NDArray[pnd.float32]

    class Config:
        json_encoders = {np.ndarray: lambda arr: arr.tolist()}


def test_init_from_values():
    # Directly specify values
    cfg = MySettings(K=[1, 2])
    assert_allclose(cfg.K, [1.0, 2.0])
    assert cfg.K.dtype == np.float32
    assert cfg.json()

    cfg = MySettings(K=np.eye(2))
    assert_allclose(cfg.K, [[1.0, 0], [0.0, 1.0]])
    assert cfg.K.dtype == np.float32


def test_load_from_npy_path(tmpdir):
    # Load from npy
    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npy"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_load_from_NPFileDesc(tmpdir):
    np.save(Path(tmpdir) / "data.npy", np.arange(5))
    cfg = MySettings(K=NPFileDesc(path=Path(tmpdir) / "data.npy"))
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_load_field_from_npz(tmpdir):
    np.savez(Path(tmpdir) / "data.npz", values=np.arange(5))
    cfg = MySettings(K={"path": Path(tmpdir) / "data.npz", "key": "values"})
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert cfg.K.dtype == np.float32


def test_exceptional(tmpdir):
    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npz", "key": "values"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy", "key": "nosuchkey"})

    with pytest.raises(ValidationError):
        MySettings(K={"path": Path(tmpdir) / "nosuchfile.npy"})

    with pytest.raises(ValidationError):
        MySettings(K="absc")


def test_unspecified_npdtype():
    # Not specifying a dtype will use numpy default dtype resolver

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

    cfg = MySettingsNoGeneric(K=[1, 2])
    assert_allclose(cfg.K, [1, 2])
    assert cfg.K.dtype == int


def test_json_encoders():
    import json

    class MySettingsNoGeneric(BaseModel):
        K: NDArray

        class Config:
            json_encoders = {np.ndarray: lambda arr: arr.tolist()}

    cfg = MySettingsNoGeneric(K=[1, 2])
    jdata = json.loads(cfg.json())

    assert "K" in jdata
    assert type(jdata["K"]) == list
    assert jdata["K"] == list([1, 2])


def test_optional_construction():
    class MySettingsOptional(BaseModel):
        K: Optional[NDArray[pnd.float32]]

    cfg = MySettingsOptional()
    assert cfg.K is None

    cfg = MySettingsOptional(K=[1, 2])
    assert type(cfg.K) == np.ndarray
    assert cfg.K.dtype == np.float32


def test_potential_array(tmpdir):
    class MySettingsPotential(BaseModel):
        K: PotentialNDArray[pnd.float32]

    np.savez(Path(tmpdir) / "data.npz", values=np.arange(5))

    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "data.npz", "key": "values"})
    assert cfg.K is not None
    assert_allclose(cfg.K, [0.0, 1.0, 2.0, 3.0, 4.0])

    # Path not found
    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "nothere.npz", "key": "values"})
    assert cfg.K is None

    # Key not there
    cfg = MySettingsPotential(K={"path": Path(tmpdir) / "data.npz", "key": "nothere"})
    assert cfg.K is None


def test_subclass_basemodel():
    class MyModelField(BaseModel):
        K: NDArray[pnd.float32]

        class Config:
            json_encoders = JSON_ENCODERS

    class MyModel(BaseModel):
        L: dict[str, MyModelField]

        class Config:
            json_encoders = JSON_ENCODERS

    model_field = MyModelField(K=[1.0, 2.0])
    assert model_field.json()

    model = MyModel(L={"a": MyModelField(K=[1.0, 2.0])})
    assert model.L["a"].K.dtype == np.dtype("float32")
    assert model.json()
