# Endometriosis
# endomri-seg

Deep learning for finding and outlining pelvic organs (uterus, ovaries) and endometriomas in multi-sequence pelvic MRI, using the public **UT-EndoMRI** dataset.

## What this project does

1. **Classification:** detect whether a slice or patient contains an endometrioma (pretrained CNNs with 2.5D input).
2. **Segmentation baselines:** 2D/3D U-Net, Attention U-Net, and nnU-Net, with sequence (T1, T1FS, T2, T2FS) and loss-function (BCE, Dice, Dice+BCE, Focal/Tversky) ablations.
3. **Detection and instance segmentation:** Mask R-CNN and YOLO-seg, compared against the U-Net baselines.
4. **Cascade:** a detector finds the region, a small U-Net refines the mask. Also tests whether a classification gate helps.
5. **Generalization:** leave-one-center-out testing, compared with human interrater agreement.

## Dataset

UT-EndoMRI (*Scientific Data*, 2025): about 130 patients from two sources, with uterus, ovary, and endometrioma masks across T1, T1FS, T2, and T2FS sequences. Data is not included in this repo; place it under `data/raw/`.

Caveats: no deep infiltrating endometriosis labels, small sample size, and uneven sequence availability. Details should be confirmed in the data audit.

## Evaluation

- Patient-level, center-stratified 5-fold cross-validation (never split by slice)
- Classification: AUROC, AUPRC, sensitivity at fixed specificity
- Segmentation: per-structure Dice, HD95
- Detection: mask AP, lesion sensitivity, false positives per patient
- Paired statistical tests and bootstrap confidence intervals

## Repository layout

```
configs/      experiment configs (YAML)
data/         raw, processed, splits
src/          datasets, models, losses, train, eval, viz
notebooks/    data audit, error analysis
results/      metrics and figures
```

## Scope

Research use only. Clinical framing is limited to organ and endometrioma delineation. A Dice of 0.75 to 0.85 is a reference benchmark, not evidence of clinical readiness.

## Citation

> A Multi-Modal Pelvic MRI Dataset for Deep Learning-Based Pelvic Organ Segmentation in Endometriosis. *Scientific Data*, 2025. DOI: 10.1038/s41597-025-05623-3
