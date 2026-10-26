# Endometriosis
# GLENDA Lesion Localisation

Detection and segmentation of endometriosis lesions in laparoscopic images using GLENDA v1.5.

## Project objective

Compares pathology classifiers, dense segmentation models, and instance segmentation models for endometriosis lesion detection and delineation on the GLENDA v1.5 dataset, under case-held-out cross-validation.

## Research questions

- **RQ1** — Do pathology classifiers detect lesions, or the source of the frames?
- **RQ2** — Which segmentation paradigm localises lesions best, and does separating "lesion" from "lesion type" help?
- **RQ3** — How should lesion-free frames be handled: a classification gate, training with negatives, or both?
- **RQ4** — What helps the rare classes (ovary, DIE, uterus)?
- **RQ5** — Where and why do models fail?

## Models

| Role | Models |
|---|---|
| Classifier | ResNet-50, EfficientNet-B0 |
| Dense segmentation | U-Net (ResNet-34) |
| Instance segmentation | Mask R-CNN (R50-FPN), YOLO11-seg |

## Data

GLENDA v1.5 — 373 annotated keyframes from 102 cases, 13,438 lesion-free frames from 27 cases. Released by ITEC, University of Klagenfurt, under CC BY-NC 4.0.

https://ftp.itec.aau.at/datasets/GLENDA/index.html

Data is **not** redistributed. Only manifests (file IDs, fold assignments) are committed.

## Workflow

- **Code** — written and debugged in VS Code (local, CPU-only)
- **Training** — run on Google Colab (GPU)
- **Data and checkpoints** — stored in Google Drive
- **Version control** — GitHub

## Structure

```
code/               Python source
data_manifests/     Frozen split manifests and metadata (committed)
notebooks/local/    Local analysis notebooks
notebooks/colab/    Colab training notebooks
results/            Aggregated tables and figures
experiments/        Per-run checkpoints and logs (Drive, gitignored)
```



## Evaluation

5-fold cross-validation grouped by surgical case, repeated across three seeds. All primary metrics reported with case-level bootstrap 95% intervals. Models are not ranked when the interval of a paired difference includes zero.

## Licence

Code: MIT.
Data: CC BY-NC 4.0 (non-commercial use only).



