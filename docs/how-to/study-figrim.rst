Run The FIGRIM Study
====================

What this workflow does
-----------------------

``study-figrim`` runs repeated stratified train/val/test splits on FIGRIM and
compares the released ``ResMem`` baseline against frozen CLIP challengers.

Command-line workflow
---------------------

.. code-block:: bash

   memscore study-figrim \
     --figrim-root /path/to/figrim \
     --protocols across within \
     --score-types corrected_hit_rate \
     --clip-models ViT-B-32 RN50 ViT-B-16 \
     --clip-tta fivecrop \
     --pca-dims none 128 64 32 \
     --num-splits 10 \
     --output-json results/figrim_study.json \
     --output-csv results/figrim_study_summary.csv

Important knobs
---------------

- ``--clip-models`` controls which frozen backbones are evaluated
- ``--clip-tta`` enables ``none``, ``fivecrop``, or ``tencrop``
- ``--pca-dims`` adds compressed ridge baselines
- ``--pls-dims`` adds supervised PLS baselines
- ``--ensemble-weight-step`` controls the validation-weighted ensemble search

Outputs
-------

- JSON with split-level results and model summaries
- CSV summary table for plotting or paper tables

Programmatic access
-------------------

.. code-block:: python

   from memscore import study_figrim

   study = study_figrim(
       "/path/to/figrim",
       clip_models=["ViT-B-32", "RN50", "ViT-B-16"],
       clip_tta_mode="fivecrop",
   )
   print(study["summary_rows"][0])

.. code-block:: r

   study <- memscore::memscore_study_figrim(
     "/path/to/figrim",
     clip_models = c("ViT-B-32", "RN50", "ViT-B-16"),
     clip_tta = "fivecrop"
   )

   study$study$summary_rows

