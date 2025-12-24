# MLOPS

## Installation

``` bash
# install w.o. gui
pip install -e .[no_gui] --config-settings editable_mode=strict

# install w. gui
pip install -e .[gui] --config-settings editable_mode=strict
```

## Package Structure

```
src/mlops
| - shapes
| - | - typing
| - | - | - {shape}.py
| - | - objects
| - | - | - insts.py
| - | - funcs
| - | - | - convert
| - | - | - | - {shape2shape}.py
```