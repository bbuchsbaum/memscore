R Quickstart
============

What you will do
----------------

You will configure the external backend that powers the R wrapper and run the
same scoring workflow from R.

Install
-------

Install the R package from a clone of the repository:

.. code-block:: r

   install.packages("pak")
   pak::pak("local::.")

The R package does not use ``reticulate``. It shells out to one of:

- a ``memscore`` executable on ``PATH``
- a Python interpreter that can run ``python -m memscore``

First success
-------------

Configure the backend and inspect what R will run:

.. code-block:: r

   library(memscore)

   memscore_configure(
     python = "/path/to/python",
     workdir = "/path/to/repo"
   )

   memscore_cli_info()

Score images:

.. code-block:: r

   scores <- memscore_predict(c("image1.jpg", "image2.jpg"))
   scores[, c("id", "memorability")]

Next steps
----------

- ``vignette("memscore", package = "memscore")``
- :doc:`../how-to/benchmark-manifest`
- :doc:`../how-to/study-figrim`

