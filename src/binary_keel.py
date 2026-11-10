"""
Binary KEEL benchmark: Baseline vs. LNR vs. SMOTE with KNN, CART and MLP.

Run from the project root, e.g.
    python src/binary_keel.py                                  # all 5 datasets, 50 rounds
    python src/binary_keel.py --datasets glass1 --n_rounds 2   # quick test
    python src/binary_keel.py --out_dir results/my_run         # custom output folder

Each run writes to results/keel_<YYYY-mm-dd_HH-MM-SS>/ (unless --out_dir is given):
    config.json     the settings of the run
    rounds.csv      one row per dataset x round x method x classifier: F1, G-mean, ROC-AUC
                    and the hyper-parameters selected in that round
    summary.csv     mean and std over rounds per dataset x metric x classifier x method
"""

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Suppress ConvergenceWarning from hardcapped 2000 epochs under noise
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Resolves project root: src/.. -> DD2610_final_project_Sep2026/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
data_root = PROJECT_ROOT / "data"

# Import utility functions
from utils.keel_utils import get_X_y, load_keel, preprocessor, flip_labels, get_g_mean, display_benchmark_table

# ---------- 5 Target Datasets ----------
DATASETS = [
    "abalone-19_vs_10-11-12-13",
    "flare-F",
    "glass1",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "yeast-1-2-8-9_vs_7"
]

MODELS = ['KNN', 'CART', 'MLP']
METRICS = ['F1-score', 'G-mean', 'ROC-AUC']
METHODS = ['Baseline', 'LNR', 'SMOTE']

ROUND_FIELDS = ['dataset', 'round', 'method', 'classifier', 'F1-score', 'G-mean', 'ROC-AUC',
                'knn_k', 't_flip', 't_flip_cv_f1', 'n_flipped', 'smote_k_neighbors', 'smote_gamma']


def test_scores(y_true, y_pred, y_prob) -> list[float]:
    """F1, G-mean and ROC-AUC on one test part."""
    return [
        f1_score(y_true, y_pred, zero_division=0),
        get_g_mean(y_true, y_pred),
        roc_auc_score(y_true, y_prob)
    ]


def run_round(X, y, r: int, k: int, n_jobs: int) -> list[dict]:
    """
    One evaluation round with seed r: split, baseline, LNR and SMOTE for KNN, CART and MLP.
    Returns one dict per method x classifier with the test scores and the selected hyper-parameters.
    """
    rows = []
    processor = preprocessor(X)

    # PREP. 70/30 Stratified Split per Section C.7
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=r
    )

    # 3*4 = 12 folds for cross-validation per Section C.2
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=r)

    # Random uniform values for label flipping, one per training sample, reused across all classifiers and t_flip candidates
    u = np.random.default_rng(r).random(len(y_train))

    # k is fixed by --knn_k (default 5)

    # # PREP. Select best k in [5, 10] for KNN per Table 5; metric: F1 score
    # k_f1_scores = {}
    # for candidate_k in [5, 10]:
    #     knn_pipe = Pipeline([
    #         ('prep', clone(processor)),
    #         ('model', KNeighborsClassifier(n_neighbors=candidate_k)),
    #     ])
    #     scores = cross_val_score(knn_pipe, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    #     k_f1_scores[candidate_k] = np.mean(scores)

    # k = max(k_f1_scores, key=k_f1_scores.get)
    # print(f'\nSelected k = {k} for the KNN classifier (Mean F1: {k_f1_scores[k]:.4f}).')

    # PREP. Model setup with explicit Table 5 parameters
    pipe = {
            'KNN': Pipeline([('prep', clone(processor)), ('model', KNeighborsClassifier(n_neighbors=k))]),
            'CART': Pipeline([('prep', clone(processor)), ('model', DecisionTreeClassifier(random_state=r))]),
            'MLP': Pipeline([('prep', clone(processor)), ('model', MLPClassifier(
                hidden_layer_sizes=(5, 10, 5),
                solver='adam',
                learning_rate_init=0.001,
                max_iter=2000,
                random_state=r
            ))])
        }

    # ---------- 1. Baseline Evaluation ----------
    for name, pipeline in pipe.items():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        rows.append({'method': 'Baseline', 'classifier': name, 'knn_k': k,
                     **dict(zip(METRICS, test_scores(y_test, y_pred, y_prob)))})

    # Flip-rate estimator for the final refit: the baseline MLP, fitted on X_train only
    eta = pipe['MLP'].predict_proba(X_train)[:, 1]

    #---------- 2. LNR Evaluation ---------
    t_flip_candidates = np.linspace(0, 4, 10)

    # Pre-extract fold indices to ensure clean validation evaluation
    fold_splits = list(cv.split(X_train, y_train))

    # Flip-rate estimator refitted on each CV training fold, so it never sees the validation fold.
    # Computed once per fold and reused for every classifier and t_flip candidate.
    fold_etas = []
    for tr_idx, _ in fold_splits:
        cf = clone(pipe['MLP'])
        cf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        fold_etas.append(cf.predict_proba(X_train.iloc[tr_idx])[:, 1])

    for name, pipeline in pipe.items():
        t_f1_scores = {}

        for candidate_t in t_flip_candidates:
            fold_f1s = []

            for (tr_idx, val_idx), eta_tr in zip(fold_splits, fold_etas):
                X_tr = X_train.iloc[tr_idx] if hasattr(X_train, 'iloc') else X_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
                y_va = y_train[val_idx]  # uncorrupted ground truth

                # Flip labels only on training fold
                y_tr_flipped = flip_labels(eta_tr, y_tr, t_flip=candidate_t, u=u[tr_idx])

                clf = clone(pipeline)
                clf.fit(X_tr, y_tr_flipped)

                # Score on TRUE, uncorrupted validation labels
                y_pred_val = clf.predict(X_va)
                fold_f1s.append(f1_score(y_va, y_pred_val, zero_division=0))

            t_f1_scores[candidate_t] = np.mean(fold_f1s)

        best_t_flip = max(t_f1_scores, key=t_f1_scores.get)
        print(f"Selected t_flip = {best_t_flip:.4f} for {name} with LNR (CV F1: {t_f1_scores[best_t_flip]:.4f})")

        # Refit on full X_train using best t_flip
        final_flipped = flip_labels(eta, y_train, t_flip=best_t_flip, u=u)
        final_clf = clone(pipeline)
        final_clf.fit(X_train, final_flipped)

        y_pred = final_clf.predict(X_test)
        y_prob = final_clf.predict_proba(X_test)[:, 1]

        rows.append({'method': 'LNR', 'classifier': name, 'knn_k': k,
                     't_flip': best_t_flip, 't_flip_cv_f1': t_f1_scores[best_t_flip],
                     'n_flipped': int(np.sum(final_flipped != y_train)),
                     **dict(zip(METRICS, test_scores(y_test, y_pred, y_prob)))})

    #---------- 3. SMOTE Evaluation ----------
    # Calculate safe bounds across CV training folds so SMOTE never fails on fold rounding
    fold_min_counts = [np.sum(y_train[tr] == 1) for tr, _ in cv.split(X_train, y_train)]

    # k candidates bounded below minimum fold minority count; -1 to avoid self-neighboring
    max_k = min(20, min(fold_min_counts) - 1)
    k_cand_smote = np.unique(np.linspace(1, max(1, max_k), 3, dtype=int))

    # Determine the minimum gamma that guarantees generating at least 1 sample
    # The 1.01 offset ensures that target_minority strictly rounds up to at least N_minority+1 instead of truncating back down to N_minority due to binary floating-point representation limits
    # Check across all 12 folds
    min_gamma_req = [
        (np.sum(y_train[tr] == 1) + 1.01) / np.sum(y_train[tr] == 0)
        for tr, _ in fold_splits
    ]

    # Check on the full training set (for GridSearchCV refit)
    min_gamma_req.append((np.sum(y_train == 1) + 1.01) / np.sum(y_train == 0))

    # Safe lower bound: strictly higher than the natural ratio across all folds
    safe_min_gamma = max(min_gamma_req)

    # Ensure safe_min_gamma does not exceed 1.0 (balanced)
    safe_min_gamma = min(safe_min_gamma, 0.99)
    gamma_cand = np.linspace(safe_min_gamma, 1.0, 10)

    param_grid_smote = {
        'smote__k_neighbors': k_cand_smote.tolist(),
        'smote__sampling_strategy': gamma_cand.tolist()
    }

    # Define pipeline using imblearn's Pipeline
    smote_pipe = {
        'KNN': ImbPipeline([
            ('prep', clone(processor)),
            ('smote', SMOTE(random_state=r)),
            ('model', KNeighborsClassifier(n_neighbors=k))
        ]),
        'CART': ImbPipeline([
            ('prep', clone(processor)),
            ('smote', SMOTE(random_state=r)),
            ('model', DecisionTreeClassifier(random_state=r))
        ]),
        'MLP': ImbPipeline([
            ('prep', clone(processor)),
            ('smote', SMOTE(random_state=r)),
            ('model', MLPClassifier(hidden_layer_sizes=(5, 10, 5), max_iter=2000, random_state=r))
        ])
    }

    # Fit GridSearchCV
    for name, smote_pipeline in smote_pipe.items():
        # Select hyperparameters for SMOTE (n_neighbors in resampler, re-balance ratio γ)
        grid_search = GridSearchCV(
            estimator=smote_pipeline,
            param_grid=param_grid_smote,
            scoring='f1',
            cv=cv,
            n_jobs=n_jobs
        )
        grid_search.fit(X_train, y_train)

        print(f"Best params for {name} with SMOTE: {grid_search.best_params_}")
        best_pipeline = grid_search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        rows.append({'method': 'SMOTE', 'classifier': name, 'knn_k': k,
                     'smote_k_neighbors': grid_search.best_params_['smote__k_neighbors'],
                     'smote_gamma': grid_search.best_params_['smote__sampling_strategy'],
                     **dict(zip(METRICS, test_scores(y_test, y_pred, y_prob)))})

    return rows


def summarize(dataset_name: str, round_rows: list[dict]) -> list[dict]:
    """Mean and std over rounds per metric x classifier x method (np.std, ddof=0, as before)."""
    records = []
    for metric_name in METRICS:
        for model_name in MODELS:
            stats = {}
            for method_name in METHODS:
                values = np.array([row[metric_name] for row in round_rows
                                   if row['method'] == method_name and row['classifier'] == model_name])
                stats[method_name] = (values.mean(), values.std())

            # Identify the method with the highest mean in this row
            best_method = max(stats, key=lambda m: stats[m][0])

            row_dict = {
                'Dataset': dataset_name,
                'Metric': metric_name,
                'Classifier': model_name,
                'Best': best_method  # Saved for dynamic formatting
            }
            for method_name, (mean_val, std_val) in stats.items():
                row_dict[method_name] = f"{mean_val:.4f} ± {std_val:.4f}"
                row_dict[f'_{method_name}_mean'] = mean_val
                row_dict[f'_{method_name}_std'] = std_val
            records.append(row_dict)
    return records


def write_summary_csv(path: Path, records: list[dict]) -> None:
    """One row per dataset x metric x classifier, with mean and std of each method."""
    fields = ['dataset', 'metric', 'classifier', 'best']
    for m in METHODS:
        fields += [f'{m}_mean', f'{m}_std']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {'dataset': rec['Dataset'], 'metric': rec['Metric'],
                   'classifier': rec['Classifier'], 'best': rec['Best']}
            for m in METHODS:
                row[f'{m}_mean'] = f"{rec[f'_{m}_mean']:.6f}"
                row[f'{m}_std'] = f"{rec[f'_{m}_std']:.6f}"
            writer.writerow(row)


def display_records(records: list[dict]) -> list[dict]:
    """Drop the numeric helper keys before printing."""
    return [{k: v for k, v in rec.items() if not k.startswith('_')} for rec in records]


def relative_to_project(path: Path) -> str:
    """Path relative to the project root if possible, so config.json holds no personal paths."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary KEEL benchmark: Baseline vs. LNR vs. SMOTE.")
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help="KEEL dataset names (zip files under --data_dir). Default: the 5 project datasets.")
    parser.add_argument('--n_rounds', type=int, default=50,
                        help="Number of evaluation rounds; round r uses seed r (default: 50).")
    parser.add_argument('--knn_k', type=int, default=5,
                        help="Number of neighbours of the KNN classifier (default: 5).")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Parallel jobs for SMOTE's GridSearchCV (default: -1 = all cores).")
    parser.add_argument('--data_dir', type=Path, default=data_root / "keel",
                        help="Folder with the KEEL zip files (default: data/keel).")
    parser.add_argument('--out_dir', type=Path, default=None,
                        help="Output folder (default: results/keel_<YYYY-mm-dd_HH-MM-SS>).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir or PROJECT_ROOT / "results" / f"keel_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'data_dir': relative_to_project(args.data_dir),
              'out_dir': relative_to_project(out_dir)}
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saving results to {out_dir}")

    rounds_path = out_dir / "rounds.csv"
    with open(rounds_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=ROUND_FIELDS).writeheader()

    all_dataset_records = []
    total_pbar = tqdm(args.datasets, desc="Total Progress (Datasets)", position=0)

    for dataset_name in total_pbar:
        total_pbar.set_postfix({"Current Dataset": dataset_name})

        file_path = args.data_dir / f"{dataset_name}.zip"
        df = load_keel(str(file_path))
        X, y = get_X_y(df)

        dataset_rows = []
        # Dataset-specific Progress Bar across defined number of rounds
        round_pbar = tqdm(range(args.n_rounds), desc=f"Rounds [{dataset_name}]", position=1, leave=False)

        # Defined number of rounds of evaluation with random data splits
        for r in round_pbar:
            print(f"\n==================== Round {r+1}/{args.n_rounds} for dataset {dataset_name} ====================")
            rows = run_round(X, y, r, k=args.knn_k, n_jobs=args.n_jobs)
            for row in rows:
                row.update({'dataset': dataset_name, 'round': r})

            # Append this round to rounds.csv right away, so an interrupted run keeps its finished rounds
            with open(rounds_path, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=ROUND_FIELDS).writerows(rows)
            dataset_rows += rows

        # ---------- Summary Table for Current Dataset ----------
        records = summarize(dataset_name, dataset_rows)
        all_dataset_records += records
        display_benchmark_table(
            display_records(records),
            title=f"Final Results for {dataset_name}:",
            is_master=False
        )
        # Rewrite summary.csv after each dataset, so it is complete for every finished dataset
        write_summary_csv(out_dir / "summary.csv", all_dataset_records)

    # ---------- Combined Master Summary for All Datasets ----------
    display_benchmark_table(
        display_records(all_dataset_records),
        title=f"ALL {len(args.datasets)} DATASETS BENCHMARK SUMMARY (MEAN ± STD)",
        is_master=True
    )
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
