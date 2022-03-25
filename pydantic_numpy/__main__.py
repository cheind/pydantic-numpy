import numpy as np
from pydantic import BaseSettings
from pathlib import Path
import tempfile

from pydantic_numpy import NDArray


arr = np.random.rand(3, 3)
path_to = Path(tempfile.mkdtemp())
np.save(path_to / "array.npy", arr)
np.savez(path_to / "array.npz", K=arr)


class MySettings(BaseSettings):
    K: NDArray[np.float32]


# Instantiate with array
cfg = MySettings(K=[1, 2])
# Instantiate from numpy file
cfg = MySettings(K={"path": Path(path_to) / "array.npy"})
# Instantiate from npz file with key
cfg = MySettings(K={"path": Path(path_to) / "array.npz", "key": "K"})

print(cfg.K)
# np.ndarray[np.float32]
