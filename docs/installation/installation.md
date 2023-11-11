## Installation


### Manual installation

First of all, you have to choose the version of some packages and install them from their official site manually, mainly some packages related to cuda, and also, you have to choose the cuda version. 

#### Python

We recommend you use the latest version of Python, which works well generally and may provide a better performance. The minimum version of Python is `3.8`.

#### Pytorch

Install [Pytorch](https://pytorch.org/get-started/locally/) from their official site manually. You have to choose the version of Pytorch based on the cuda version on your machine. Similarly, we recommend you use the latest version of Pytorch, which works well generally and may provide a better performance. You can skip installing the Pytorch manually in this section and the `pip` will install it with the latest version in the next section if you want to use the latest Pytorch. The minimum version of Pytorch is `1.11`.

### Automatical installation

#### For project template (install as a package)

Use `pip` to install this package.

```bash
pip install lightning-tempalet
```

#### For developers (install from source)

Generally, you can just use the latest packages in `requirements.txt` without specifying their version, so you can use the command as follows to install this project and all required packages.

```bash
pip install -r requirements.txt
pip install -e .
```
