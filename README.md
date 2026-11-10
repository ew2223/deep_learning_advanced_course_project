# Is Label Noise Beneficial for Imbalanced Learning? A Reproduction of LNR

Course project for DD2610 Deep Learning, Advanced Course (KTH). It reimplements
Label-Noise-based Re-balancing (LNR) from

> Hu, Liu, Gong, Wang and Peng. *Learning Imbalanced Data with Beneficial Label Noise.* ICML 2025.

and evaluates it, at a reduced scale, in the paper's two settings:

- **Binary classification** on five KEEL datasets: no re-balancing (baseline), LNR and SMOTE, each
  with a KNN, CART and MLP classifier.
- **Multi-class classification** on long-tailed CIFAR-10 (CIFAR-10-LT, imbalance ratio 100):
  MiSLAS, MiSLAS+SelMix and MiSLAS+LNR.

## Repository structure

```
src/
  binary_keel.py          KEEL experiment: baseline, LNR and SMOTE with KNN, CART and MLP
  keel_Wilcoxon_test.py   paired Wilcoxon signed-rank test of LNR vs. SMOTE on a KEEL run
  multiclass_cifar10.py   CIFAR-10-LT experiment: MiSLAS, MiSLAS+SelMix and MiSLAS+LNR
utils/
  keel_utils.py           loading and preprocessing of KEEL data, label flipping, metrics
  cifar_utils.py          CIFAR-10-LT data, ResNet-32, MiSLAS stage-2 modules, LNR flipping, metrics
  download_keel_data.py   downloads the five KEEL datasets
LICENSE
THIRD_PARTY_LICENSES.md
```

The scripts read their data from `data/` and write their results to `results/`. Both folders are
created locally and are not part of the repository.

## Requirements

Python 3.9 or later with:

```
pip install numpy pandas scikit-learn imbalanced-learn tqdm torch torchvision pillow
```

The KEEL experiment needs only the first five packages.

## Data

**KEEL.** Download the five datasets (abalone-19_vs_10-11-12-13, flare-F, glass1,
led7digit-0-2-4-5-6-7-8-9_vs_1 and yeast-1-2-8-9_vs_7) into `data/keel/`. The script uses paths
relative to `utils/`, so run it from there:

```
cd utils
python download_keel_data.py
cd ..
```

**CIFAR-10.** The dataset is downloaded automatically to `data/cifar10/` on the first run.

**MiSLAS stage-1 checkpoint.** The CIFAR experiment starts from the official MiSLAS stage-1
checkpoint for CIFAR-10-LT with imbalance factor 100. Download
`cifar10_imb001_stage1_mixup10.pth.tar` from the model zoo of the
[MiSLAS repository](https://github.com/Jia-Research-Lab/MiSLAS) and place it in `data/cifar10/`,
or pass its location with `--stage1_checkpoint <path>`.

## Running the experiments

Run all commands from the project root.

**KEEL** (5 datasets, 50 rounds):

```
python src/binary_keel.py
python src/binary_keel.py --datasets glass1 --n_rounds 2      # quick test
```

Each run writes `config.json`, `rounds.csv` and `summary.csv` to `results/keel_<date>_<time>/`.

**Significance test** on a KEEL run:

```
python src/keel_Wilcoxon_test.py --run_dir results/keel_<date>_<time>
```

This adds `significance_lnr_vs_smote.csv` and `significance_summary.txt` to the run folder.

**CIFAR-10-LT** (the setting used in the report: five seeds, both SelMix settings, results at the
best and at the last epoch):

```
python src/multiclass_cifar10.py --seeds 0 1 2 3 4 --epoch_selection best_test \
    --selmix_variants full frozen-mislasopt
```

Each run writes one result file per seed and a summary (mean and standard deviation over the
seeds) to a new folder under `results/`. With `--epoch_selection best_test`, the results at the
last epoch of the same runs are written as well. Without options, the script runs one seed and
evaluates after the last epoch. `python src/multiclass_cifar10.py --help` lists all options.

## Third-party code

Parts of the code are adapted from the following sources, which are named in the docstrings of the
respective functions. These parts remain under the copyright and licence of their authors:

| Source | Used for | Licence |
|---|---|---|
| [MiSLAS](https://github.com/Jia-Research-Lab/MiSLAS) (Zhong et al., CVPR 2021) | stage-2 training, class-balanced sampler, label-aware smoothing, learnable weight scaling, calibration error | MIT, Copyright (c) 2021 Zhisheng Zhong |
| [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) (Cao et al., NeurIPS 2019) | long-tailed CIFAR-10 subset | MIT, Copyright (c) 2019 Kaidi Cao |
| [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) | ResNet-32 | BSD 2-Clause, Copyright (c) 2018 Yerlan Idelbayev |
| [SelMix](https://github.com/val-iisc/SelMix) (Ramasubramanian et al., ICLR 2024) | gain estimation, pair sampling, optimiser settings | no licence file in the repository |
| [LNR](https://github.com/guangzhengh/LNR) (Hu et al., ICML 2025) | settings consulted for details the paper leaves open | no licence file in the repository |

## License

The code written for this project is released under the
[PolyForm Noncommercial License 1.0.0](LICENSE). The third-party parts listed above are not
covered by this licence.

## References

- Hu, Liu, Gong, Wang and Peng. Learning Imbalanced Data with Beneficial Label Noise. ICML 2025.
- Zhong, Cui, Liu and Jia. Improving Calibration for Long-Tailed Recognition. CVPR 2021.
- Ramasubramanian, Rangwani, Takemori, Samanta, Umeda and Radhakrishnan. Selective Mixup
  Fine-Tuning for Optimizing Non-Decomposable Objectives. ICLR 2024.
- Chawla, Bowyer, Hall and Kegelmeyer. SMOTE: Synthetic Minority Over-sampling Technique. JAIR 2002.
- Cao, Wei, Gaidon, Arechiga and Ma. Learning Imbalanced Datasets with Label-Distribution-Aware
  Margin Loss. NeurIPS 2019.
