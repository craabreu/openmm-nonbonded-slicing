============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/craabreu/openmm-nonbonded-slicing/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * A simple script that reproduces the bug.

Documentation improvements
==========================

Nonbonded Slicing could always use more documentation, whether as part of the
official Nonbonded Slicing docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/craabreu/openmm-nonbonded-slicing/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome

Development
===========

To set up `openmm-nonbonded-slicing` for local development:

1. Fork `openmm-nonbonded-slicing <https://github.com/craabreu/openmm-nonbonded-slicing>`_
   (look for the "Fork" button).

2. Clone your fork locally::

    git clone git@github.com:your_name_here/openmm-nonbonded-slicing.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes, run all the unit tests and the doc builder::

    cd build
    make install
    make PythonInstall
    make test
    make PythonTest
    make doc

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Releasing on PyPI
-----------------

Pushing a ``vX.Y.Z`` tag builds and tests wheels for every OpenMM version listed in
``devtools/wheels/openmm-versions.json`` and, after approval of the ``pypi`` environment,
publishes them as ``X.Y.Z``, ``X.Y.Z.post1``, ... (one release per OpenMM minor version).

To publish an existing tag, or to try a release on TestPyPI first, run the *Wheels* workflow
manually with ``ref`` set to the tag and ``target`` set to ``testpypi`` or ``pypi``.

To support a new OpenMM minor version, add an entry to ``openmm-versions.json`` with the next
post number, the exact OpenMM release to build against and the SWIG version of its PyPI wheel
(the header of ``openmm/openmm.py``), then publish that post-release for the current version.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (``make test`` and ``make PythonTest``) [1]_.
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``docs/changelog.rst`` about the changes.
4. Add yourself to ``docs/authors.rst``.

.. [1] If you cannot run the GPU tests locally, GitHub Actions runs the Reference and OpenCL tests for each pull request;
       GPU tests still need to be run by a maintainer.
