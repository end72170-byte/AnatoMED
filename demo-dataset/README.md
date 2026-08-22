# Real demo cases

This directory contains two real, de-identified demonstration cases copied
from the local source dataset:

- `case11/normal`: normal example.
- `case96/abnormal`: abnormal example.

The same cases are provided in the three layouts used by the pipeline:

- `clip-dataset/all/<case>/<condition>/high`: one main view per slot plus video;
- `sam-dataset/all/<case>/<condition>/high`: augmented training views, masks,
  and text annotations;
- `videos/<case>/<condition>/high`: the corresponding video files.

These are real medical images and videos, not synthetic fixtures. They are
included only for controlled local smoke tests and visual demonstrations. Do
not redistribute them or use them outside the applicable data-access,
de-identification, and ethics approvals.
