[![Build Status](https://github.com/cheind/pydantic-numpy/actions/workflows/python-package.yml/badge.svg)](https://github.com/cheind/pydantic-numpy/actions/workflows/python-package.yml)

# pydantic-numpy
This tiny library provides support for integrating numpy `np.ndarray`'s into pydantic models / settings. 

## Usage
For more examples see [test_ndarray.py](./tests/test_ndarray.py)

```python
from pydantic import BaseModel

import pydantic_numpy.dtype as pnd
from pydantic_numpy import NDArray


class MySettings(BaseModel):
    K: NDArray[pnd.float32]


# Instantiate from array
cfg = MySettings(K=[1, 2])
# Instantiate from numpy file
cfg = MySettings(K={"path": "path_to/array.npy"})
# Instantiate from npz file with key
cfg = MySettings(K={"path": "path_to/array.npz", "key": "K"})

cfg.K
# np.ndarray[np.float32]
```

### Subfields
This package also comes with `pydantic_numpy.dtype`, which adds subtyping support such as `NDArray[pnd.float32]`. All subfields must be from this package as numpy dtypes have no [Pydantic support](https://pydantic-docs.helpmanual.io/usage/types/#generic-classes-as-types).


## Install
```shell
pip install pydantic-numpy
```

## History
The original idea originates from [this discussion](https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434), but stopped working for `numpy>=1.22`. This repository picks up where the previous discussion ended
 - added designated repository for better handling of PRs
 - added support for `numpy>1.22`
 - Dtypes are no longer strings but `np.generics`. I.e. `NDArray['np.float32']` becomes `NDArray[np.float32]`
 - added automated tests and continuous integration for different numpy/python versions
 