import numpy as np
import pytest
from pydantic import BaseModel

import pydantic_numpy.dtype as pnd


@pytest.mark.parametrize("data", (1, 1.0))
@pytest.mark.parametrize(
    "pnp_dtype,np_dtype", (
            (pnd.float16, np.float16),
            (pnd.float32, np.float32),
            (pnd.float64, np.float64),
            (pnd.float128, np.float128),
            (pnd.int8, np.int8),
            (pnd.int16, np.int16),
            (pnd.int32, np.int32),
            (pnd.int64, np.int64),
            (pnd.uint8, np.uint8),
            (pnd.uint16, np.uint16),
            (pnd.uint32, np.uint32),
            (pnd.uint64, np.uint64),
    )
)
def test_float32(data, pnp_dtype, np_dtype):
    class MyModel(BaseModel):
        V: pnp_dtype

    assert MyModel(V=data).V == np_dtype(data)


@pytest.mark.parametrize("data", (1 + 1j, 1.0 + 1.0j))
@pytest.mark.parametrize(
    "pnp_dtype,np_dtype", (
            (pnd.complex64, np.complex64),
            (pnd.complex128, np.complex128),
            (pnd.complex256, np.complex256),
    )
)
def test_complex256(data, pnp_dtype, np_dtype):
    class MyModel(BaseModel):
        V: pnp_dtype

    assert MyModel(V=data).V == np_dtype(data)
