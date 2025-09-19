from setuptools import setup, find_packages

setup(
    name="wisdom-idc",
    version="0.1.0",
    description="WISDOM: An explainable, white-box Coverage testing tool for DNNs (PyTorch)",
    author="Shenghao Qiu",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy>=1.24",
        "matplotlib",
        "pytest",
        "setuptools",
        "onnx",
        "pyyaml",
        "opencv-python",
        "scikit-learn>=1.3",
        "tqdm",
        "pandas",
        "captum",
    ],
    include_package_data=True,
)