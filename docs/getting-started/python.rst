Python Quickstart
=================

What you will do
----------------

You will install ``memscore``, score a directory of images with the released
``ResMem`` checkpoint, and see the small public Python API.

Install
-------

For scoring only:

.. code-block:: bash

   python -m pip install .

For benchmarking with frozen CLIP challengers:

.. code-block:: bash

   python -m pip install ".[benchmark]"

First success
-------------

Score a directory of images:

.. code-block:: bash

   memscore predict ./images --recursive --output results/predictions.csv

The output CSV contains:

- ``id``
- ``image_path``
- ``memorability``

Python API
----------

.. code-block:: python

   from memscore import predict_paths

   predictions = predict_paths(["images/example.jpg"])
   print(predictions[0].identifier, predictions[0].score)

Next steps
----------

- :doc:`../how-to/benchmark-manifest`
- :doc:`../how-to/study-figrim`
- :doc:`../api/memscore`

