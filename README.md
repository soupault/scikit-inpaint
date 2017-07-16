# scikit-inpaint

A collection of inpainting methods written in Python.

Installation
------------

Remotely:
```
pip install git+https://github.com/soupault/scikit-inpaint.git --user
```

or using a local copy:
```
git clone git@github.com:soupault/scikit-inpaint.git
cd scikit-inpaint

pip install . --user
```

Usage
-----

```
import skinpaint

img_inpainted = skinpaint.{method}(img_corrupted, mask)
```

You might also find several examples in `./examples/` directory.
