[![Build Status](https://app.travis-ci.com/cheind/pydantic-numpy.svg?branch=main)](https://app.travis-ci.com/cheind/pydantic-numpy)

# pydantic-numpy
This tiny library provides support for integrating numpy `np.ndarray`'s into pydantic models / settings. 

## Usage
For more examples see [test_array.py](./tests/test_array.py)

```python
import numpy as np
from pydantic import BaseSettings

from pydantic_numpy import NDArray

class MySettings(BaseSettings):
    K: NDArray[np.float32]


# Instantiate from array
cfg = MySettings(K=[1, 2])
# Instantiate from numpy file
cfg = MySettings(K={"path": Path(path_to) / "array.npy"})
# Instantiate from npz file with key
cfg = MySettings(K={"path": Path(path_to) / "array.npz", "key": "K"})

cfg.K
# np.ndarray[np.float32]
```

## Install
```
pip install git+https://github.com/cheind/pydantic-numpy.git
```

## History
The original idea originates from [this discussion](https://gist.github.com/danielhfrank/00e6b8556eed73fb4053450e602d2434), but stopped working for `numpy>=1.22`. This repository picks up where the previous discussion ended
 - added designated repository for better handling of PRs
 - added support for `numpy>1.22`
 - Dtypes are no longer strings but `np.generics`. I.e. `NDArray['np.float32']` becomes `NDArray[np.float32]`
 - added automated tests and continuous integration for different numpy/python versions
 