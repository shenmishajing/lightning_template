# coding:utf-8

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="lightning-template",
    version="1.1.7",
    description="A template wrapper for pytorch-lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="shenmishajing",
    author_email="shenmishajing@gmail.com",
    url="https://github.com/shenmishajing/lightning_template",
    project_urls={
        "Code": "https://github.com/shenmishajing/lightning_template",
        "Issue tracker": "https://github.com/shenmishajing/lightning_template/issues",
    },
    python_requires=">=3.8",
    install_requires=[
        "lightning>=2.0.0",
        "jsonargparse[all]",
        "scikit-learn",
        "speed-benchmark",
    ],
    license="MIT License",
    packages=find_packages(),  # 包
    entry_points={
        "console_scripts": [
            "cli = lightning_template.tools.cli:main",
            "lr_finder = lightning_template.tools.model.lr_finder:main",
            "batch_size_finder = lightning_template.tools.model.batch_size_finder:main",
            "model_statistics = lightning_template.tools.model.model_statistics:main",
        ]
    },
    platforms=["all"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: Chinese (Simplified)",
    ],
)
