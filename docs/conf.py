import os
import sys
from datetime import datetime

# Ensure package import works in executed notebooks
sys.path.insert(0, os.path.abspath(".."))

project = "volkit"
author = "Thijs van den Berg"
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_nb",  # executable Markdown notebooks
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
# Execute code cells during build so plots are generated
nb_execution_mode = "cache"  # speed up subsequent builds
nb_execution_timeout = 180
nb_execution_raise_on_error = True

# MyST configuration
myst_enable_extensions = ["deflist", "colon_fence"]

html_theme = "furo"
html_title = "volkit"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# We want warnings if execution fails
nb_execution_raise_on_error = True

# Keep the matplotlib backend non-interactive for headless builds
os.environ.setdefault("MPLBACKEND", "Agg")


autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # put type hints into the doc body

# Napoleon (NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Optional: headings in MyST .md files get anchors for easy linking
myst_heading_anchors = 3

# MyST config: add dollar/AMS math
myst_enable_extensions = [
    "deflist",
    "colon_fence",
    "dollarmath",  # <- add this
    "amsmath",  # optional but nice
]

# Cross-links to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
}


#  MathJax inline/block delimiters
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}
