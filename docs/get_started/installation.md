# Installation

## Manual installation

First of all, you have to choose the version of some packages and install them from their official site manually, mainly some packages related to cuda, and also, you have to choose the cuda version.

### Python

We recommend you use the latest version of Python, which works well generally and may provide a better performance. The minimum supported version of Python is `3.8`.

### Pytorch

Install [Pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You have to choose the version of Pytorch based on the cuda version on your machine. Similarly, we recommend you use the latest version of Pytorch, which works well generally and may provide a better performance. You can skip this step if it's fine to use the latest version of Pytorch and the `pip` will install it in the next section. The minimum supported version of Pytorch is `1.11`.

## Automatic installation

### Install as a package

Use `pip` to install this package.

```bash
pip install lightning-template
```

### Install from source

Generally, you can just use the latest dependencies without specifying their version, so you can use the command as follows to install this project and all required dependencies.

```bash
git clone https://github.com/shenmishajing/lightning_template.git
cd lightning_template
pip install -e ".[deps]"
```
