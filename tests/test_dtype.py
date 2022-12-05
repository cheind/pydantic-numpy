import numpy as np
from pydantic import BaseModel

import pydantic_numpy.dtype as pnd


def test_float32():
    class MyFloat32Model(BaseModel):
        V: pnd.float32

    model_from_int = MyFloat32Model(V=1)
    assert model_from_int.V == np.float32(1)

    model_from_float = MyFloat32Model(V=1.0)
    assert model_from_float.V == np.float32(1)


def test_int32():
    class MyInt32Model(BaseModel):
        V: pnd.int32

    model_from_int = MyInt32Model(V=1)
    assert model_from_int.V == np.int32(1)

    model_from_float = MyInt32Model(V=1.0)
    assert model_from_float.V == np.int32(1)


def test_complex256():
    class MyComplex256Model(BaseModel):
        V: pnd.complex256

    model_from_int = MyComplex256Model(V=1 + 1j)
    assert model_from_int.V == np.complex256(1 + 1j)

    model_from_float = MyComplex256Model(V=1.0 + 1.j)
    assert model_from_float.V == np.complex256(1 + 1.j)
