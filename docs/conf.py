# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'

project = 'OpenMM PME Slicing'
copyright = ("2022, Charlles Abreu. Project based on OpenMM")
author = 'Charlles Abreu'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/craabreu/openmm-pme-slicing/issues/%s', '#'),
    'pr': ('https://github.com/craabreu/openmm-pme-slicing/pull/%s', 'PR #'),
}

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# if not on_rtd:  # only set the theme if we're building docs locally
pygments_style = "sphinx"

html_theme = "alabaster"

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html'],
}
html_short_title = '%s-%s' % (project, version)

# Bibliography file
bibtex_bibfiles = ['refs.bib']

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

linkcheck_ignore = [
    r'^https://doi.org',
]

autodoc_member_order = 'bysource'
add_module_names = False

# External links
extlinks = {'OpenMM': ('http://docs.openmm.org/latest/api-python/generated/openmm.openmm.%s.html',
                       'openmm.%s')}


# Do not skip constructor docstrings
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)