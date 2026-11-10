"""
Utility functions for loading and preprocessing KEEL datasets.
Intended for applying on binary classification datasets from the KEEL repository.
This project focuses on the following datasets: 
- abalone-19_vs_10-11-12-13
- flare-F
- glass1
- led7digit-0-2-4-5-6-7-8-9_vs_1
- yeast-1-2-8-9_vs_7
"""

import zipfile
import re, sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import confusion_matrix

def load_keel(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        # pick the real .dat file, skipping macOS metadata (__MACOSX/, ._*)
        candidates = [n for n in z.namelist()
              if n.endswith(".dat")
              and not n.startswith("__MACOSX")
              and not n.split("/")[-1].startswith("._")]
        if len(candidates) != 1:
            raise ValueError(f"Expected 1 .dat file in {zip_path}, found: {candidates}")
        name = candidates[0]
        with z.open(name) as f:
            df = pd.read_csv(f, comment="@", header=None, skipinitialspace=True,
                         na_values=["?", "<null>"])
    # strip leading/trailing whitespace from text columns (class label, nominal features)
    for col in df.select_dtypes(include=["object", "string"]):
        df[col] = df[col].str.strip()

    if "flare" in zip_path.lower():
        df.columns = [
            'LargestSpotSize',
            'SpotDistribution',
            'Activity',
            'Evolution',
            'Prev24Hour',
            'HistComplex',
            'BecomeHist',
            'Area',
            'C-class',
            'M-class',
            'X-class',
            'Class',
        ]
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Separate features X as a DataFrame (for preprocessor column inspection)
    and target y as a 1D NumPy array (for positional alignment with eta).
    """
    X = df.iloc[:, :-1]  # Keep as DataFrame so select_dtypes works
    y = df.iloc[:, -1].map({'negative': 0, 'positive': 1}).astype(int).to_numpy()  # NumPy array
    return X, y


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Extract numeric columns only
    num = X.select_dtypes(include=['number']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num)
    ], remainder='drop')
    
    return preprocessor


def flip_labels(eta: np.ndarray, labels: np.ndarray, t_flip: float, u: np.ndarray = None) -> np.ndarray:
    """
    Flip labels based on Algorithm 1 in the original paper.
    """
    flipped_labels = labels.copy()
    
    # Extract 1D array of positional indices for the majority class
    ind_maj = np.flatnonzero(flipped_labels == 0)

    # If no majority samples exist, return labels unchanged
    if len(ind_maj) == 0:
        return flipped_labels
    
    # Standardize posterior probabilities for majority class
    mu = np.mean(eta[ind_maj])
    std = np.std(eta[ind_maj])

    # Guard against zero variance / saturated predictions
    # If std is 0, all majority samples have identical scores; no sample is closer to minority
    if std == 0.0 or np.isnan(std):
        z_maj = np.zeros_like(eta[ind_maj])
    else:
        z_maj = (eta[ind_maj] - mu) / std
    
    # Calculate flip rates (rho) with NaN protection
    rho_maj = np.maximum(np.tanh(z_maj - t_flip), 0.0)
    rho_maj = np.nan_to_num(rho_maj, nan=0.0)
    rho_maj = np.clip(rho_maj, 0.0, 1.0)
    
    # Bernoulli sampling: flip where u < rho (u ~ Uniform(0, 1), one value per sample)
    if u is None:
        u = np.random.random(len(labels))
    flip_indices = ind_maj[u[ind_maj] < rho_maj]
    flipped_labels[flip_indices] = 1
    
    return flipped_labels


def get_g_mean(y_true, y_pred) -> float:
  """Computes G-Mean: sqrt(Sensitivity * Specificity)."""
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
  return np.sqrt(sensitivity * specificity)


def is_notebook() -> bool:
    """Detects whether the code is running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except (NameError, ImportError):
        return False


def display_benchmark_table(records: list[dict], title: str = "", is_master: bool = False):
    """
    Universally formats and displays benchmark tables across all environments:
    - Jupyter Notebook: Renders via IPython Markdown table
    - Interactive Terminal (TTY): Prints properly-aligned table with ANSI bold
    - Headless / File Redirect: Strips ANSI to output clean, aligned plain text
    """
    index_cols = ['Dataset', 'Metric', 'Classifier'] if is_master else ['Metric', 'Classifier']
    
    # ---------------- 1. Jupyter Notebook Rendering ----------------
    if is_notebook():
        from IPython.display import display, HTML
        
        rows = []
        for r in records:
            row = r.copy()
            best = row.get('Best')
            if best and best in row:
                row[best] = f"<b>{row[best]}</b>"
            row.pop('Best', None)
            
            # Drop Dataset key for per-dataset printing
            if not is_master:
                row.pop('Dataset', None)
            rows.append(row)
            
        df = pd.DataFrame(rows).set_index(index_cols)
        
        if title:
            display(HTML(f"<h3>{title}</h3>"))
        display(HTML(df.to_html(escape=False)))
        return

    # ---------------- 2. Terminal & Log File Rendering ----------------
    use_ansi = sys.stdout.isatty()  # True if terminal, False if piped to file
    
    rows = []
    for r in records:
        row = r.copy()
        best = row.get('Best')
        if best and best in row and use_ansi:
            row[best] = f"\033[1m{row[best]}\033[0m"
        row.pop('Best', None)
        
        # Drop Dataset key for per-dataset printing
        if not is_master:
            row.pop('Dataset', None)
        rows.append(row)

    df = pd.DataFrame(rows).set_index(index_cols)

    # Helper: calculate visible width excluding invisible ANSI escape codes
    def visible_len(val):
        return len(re.sub(r'\033\[[0-9;]*m', '', str(val)))

    headers = list(df.index.names) + list(df.columns)
    table_rows = []
    for idx_val, row in df.iterrows():
        idx_tuple = list(idx_val) if isinstance(idx_val, tuple) else [idx_val]
        table_rows.append(idx_tuple + [row[c] for c in df.columns])

    # Compute maximum visible column widths
    col_widths = [visible_len(h) for h in headers]
    for row in table_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], visible_len(val))

    # Format output
    if title:
        print(f"\n{title}")
    
    header_str = "  ".join(f"{str(h):<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)

    for row in table_rows:
        line_parts = []
        for i, val in enumerate(row):
            pad = col_widths[i] - visible_len(val)
            line_parts.append(str(val) + (" " * pad))
        print("  ".join(line_parts))