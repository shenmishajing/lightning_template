# Contribution

## Development

```bash
git clone https://github.com/shenmishajing/lightning_template.git
cd lightning_template
pip install -e ".[all]"
pre-commit install
```

### Build Documents

```bash
sphinx-autobuild docs docs/_build
```

### Build Package

```bash
python -m build
```

See the github action workflow file [python-publish.yml](https://github.com/shenmishajing/lightning_template/blob/master/.github/workflows/python-publish.yml) for details.

## Code Style

The code is formatted using [black](https://github.com/python/black).
