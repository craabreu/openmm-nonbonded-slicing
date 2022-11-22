# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import pmeslicing
import openmm


def valid_method(method):
    return not (method.startswith('_') or method in ['cast', 'thisown', 'isinstance'])


def create_rst_file(cls):
    name = cls.__name__
    methods = list(filter(valid_method, dir(cls)))
    with open(f'pythonapi/{name}.rst', 'w') as f:
        f.writelines([
            f'{name}\n',
            f'='*len(name)+'\n',
            f'\n',
            f'.. autoclass:: {cls.__module__}.{name}\n',
            f'    :members:\n',
            f'    :inherited-members:\n',
            f'    :member-order: alphabetical\n',
            f'    :show-inheritance:\n',
            f'    :exclude-members: thisown\n',
            f'    :special-members: __init__\n',
            f'\n',
            f'    .. autosummary::\n',
        ])
        f.writelines([' '*8 + method + '\n' for method in methods])


create_rst_file(pmeslicing.SlicedPmeForce)

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
copyright = ('2022, Charlles Abreu. Project based on OpenMM')
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
pygments_style = 'sphinx'

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    'logo': 'logo_small.png',
    'logo_name': True,
    'github_button': False,
    'github_user': 'craabreu',
    'github_repo': 'openmm-pme-slicing',
}
html_sidebars = {
   '**': ['about.html', 'globaltoc.html', 'searchbox.html'],
}
html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
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
