import os
from collections.abc import Collection

import evalica
import pandas as pd
import plotly.io as pio

project = "Evalica"
author = "Dmitry Ustalov"
copyright = "2024\u20132026 Dmitry Ustalov"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "nbsphinx",
]

html_theme = "furo"
html_title = "Evalica"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"

autodoc_type_aliases = {
    "npt.NDArray": "numpy.ndarray",
}

napoleon_custom_sections = [("Quote", "note")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

nbsphinx_execute = "always"
nbsphinx_timeout = 120

# evalica uses TYPE_CHECKING-only imports (Collection), so get_type_hints() fails
# unless we inject them into the module namespace before autodoc runs.
evalica.Collection = Collection  # type: ignore[attr-defined]

# pandas 3.x removed Series.__class_getitem__, so pd.Series[float] is not subscriptable
# at runtime. Inject it so get_type_hints() can evaluate dataclass field annotations.
if not hasattr(pd.Series, "__class_getitem__"):
    pd.Series.__class_getitem__ = classmethod(lambda cls, _item: cls)  # type: ignore[attr-defined]

os.environ.setdefault("PLOTLY_RENDERER", "notebook")
pio.renderers.default = os.environ["PLOTLY_RENDERER"]
