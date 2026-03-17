Benchmark From A Custom Manifest
================================

When to use this workflow
-------------------------

Use ``benchmark-manifest`` when you already have image paths, memorability
targets, and train/val/test assignments.

Manifest requirements
---------------------

The CSV must contain:

- ``path``
- ``score``
- ``split``

It may also contain:

- ``id``

Relative paths are resolved against ``--root`` when provided.

Command-line workflow
---------------------

.. code-block:: bash

   memscore benchmark-manifest \
     --manifest /path/to/manifest.csv \
     --root /path/to/images \
     --clip-models ViT-B-32 RN50 \
     --regressors ridge \
     --output-json results/manifest.json \
     --output-predictions results/manifest_predictions.csv

What you get back
-----------------

- a JSON summary with ``ResMem`` metrics, challenger metrics, and deltas
- a CSV containing ground truth, ``ResMem`` predictions, and challenger
  predictions for the test split

Python API
----------

.. code-block:: python

   from memscore import benchmark_manifest

   run = benchmark_manifest(
       "manifest.csv",
       root="images",
       clip_models=["ViT-B-32", "RN50"],
       regressors=["ridge"],
   )
   print(run.summary["delta_vs_resmem"])

R API
-----

.. code-block:: r

   benchmark <- memscore::memscore_benchmark_manifest(
     manifest = "manifest.csv",
     root = "images",
     clip_models = c("ViT-B-32", "RN50"),
     regressors = "ridge"
   )

   benchmark$summary$delta_vs_resmem

