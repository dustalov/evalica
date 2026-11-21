import os

import evalica
import pandas as pd
import plotly.io as pio
from docutils.nodes import TextElement, reference
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.intersphinx import missing_reference

project = "Evalica"
author = "Dmitry Ustalov"
copyright = "2024\u20132026 Dmitry Ustalov"
release = evalica.__version__
version = release

exclude_patterns = ["_build"]

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
html_theme_options = {
    "source_repository": "https://github.com/dustalov/evalica/",
    "source_branch": "master",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/dustalov/evalica",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa: E501
            "class": "",
        },
    ],
}
html_logo = "../Evalica.svg"
html_favicon = "../Evalica.svg"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"

napoleon_custom_sections = [("Quote", "note")]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

nbsphinx_execute = "always"
nbsphinx_timeout = 120

# pandas 3.x removed Series.__class_getitem__, so pd.Series[float] is not subscriptable
# at runtime. Inject it so get_type_hints() can evaluate dataclass field annotations.
if not hasattr(pd.Series, "__class_getitem__"):
    pd.Series.__class_getitem__ = classmethod(lambda cls, _item: cls)  # type: ignore[attr-defined]  # ty: ignore[missing-attribute]

os.environ.setdefault("PLOTLY_RENDERER", "notebook")
pio.renderers.default = os.environ["PLOTLY_RENDERER"]


def resolve_numpy(app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: TextElement) -> reference | None:
    """Resolve NumPy names that annotations reference as py:class but NumPy documents as data or attributes."""
    target = node.get("reftarget", "")
    if node.get("reftype") != "class" or not target.startswith("numpy."):
        return None
    node["reftarget"] = target.replace("numpy._typing._array_like.", "numpy.typing.")
    node["reftype"] = "obj"
    return missing_reference(app, env, node, contnode)


def setup(app: Sphinx) -> None:
    app.connect("missing-reference", resolve_numpy)
