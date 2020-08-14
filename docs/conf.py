import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../pytorch_adaptive_computation_time/"))


# -- Project information -----------------------------------------------------
project = "pytorch-adaptive-computation-time"
copyright = "2020, Aidan Swope"
author = "Aidan Swope"


# -- General configuration ---------------------------------------------------
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinxarg.ext"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
