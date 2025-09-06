# Volatility Modelling Toolkit

[![CI](https://github.com/sitmo/volkit/actions/workflows/tests.yml/badge.svg)](https://github.com/sitmo/volkit/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/sitmo/volkit/branch/main/graph/badge.svg)](https://codecov.io/gh/sitmo/volkit)
[![Docs](https://readthedocs.org/projects/volkit/badge/?version=latest)](https://volkit.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/volkit.svg)](https://github.com/sitmo/volkit/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/volkit.svg)](https://pypi.org/project/volkit/)
[![PyPI version](https://img.shields.io/pypi/v/volkit.svg)](https://pypi.org/project/volkit/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A powerfull lightweight toolkit for option pricing, implied volatility, and Greeks.


## Installation

```bash
pip install volkit
```

```{toctree}
:hidden:
:maxdepth: 2

getting-started
api/index
```


```{toctree}
:caption: Examples
:maxdepth: 1

Call and Put Price Table <examples/strikes_table>
Price Table with all Greeks <examples/strikes_table_with_greeks>
Greeks: Analytic vs Numeric <examples/greeks_demo>
Greeks: Surface Plots <examples/greeks_surfaces>
Broadcasting Basics <examples/broadcasting>
Implied Volatilities <examples/implied_vol>
Implied Futures from Options <examples/implied_future>
Datasets: SPXW <examples/spxw_dataset.ipynb>

```


## API
See {doc}`api/index`.
