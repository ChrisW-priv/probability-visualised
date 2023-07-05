---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from distributions import *
import numpy as np
import matplotlib.pyplot as plt

# make plots bigger and better
plt.figure(figsize=(5,5), dpi=160)
```

```python
base = np.linspace(-10,10, 1000)
x = dist_normal(-1,2)(base)
y = dist_normal(1,4)(base)

def g(x): return x+y

domain_z, distribution_z = rand_var_transform(base, x, g)
plt.scatter(base, x)
plt.scatter(base, y)
plt.scatter(domain_z, distribution_z)
```

This is wrong! Z does not follow the N(0,6)

```python
z = np.convolve(x,y)
plt.scatter(base, x)
plt.scatter(base, y)
plt.scatter(base, distribution_z)
```
