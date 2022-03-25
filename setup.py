from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR / "requirements.txt") as f:
    required = f.read().splitlines()

main_ns = {}
with open(THISDIR / "pydantic_numpy" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="pydantic-numpy",
    author="Christoph Heindl",
    description="Seamlessly integrate numpy arrays into pydantic models",
    license="MIT",
    version=main_ns["__version__"],
    packages=find_packages(".", include="pydantic_numpy*"),
    install_requires=required,
    zip_safe=False,
)