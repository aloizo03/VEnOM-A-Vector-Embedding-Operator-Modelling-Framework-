import os

import numpy as np
import pandas as pd
import torch

from sklearn.random_projection import GaussianRandomProjection

TABFM_HF_REPO = "google/tabfm-1.0.0-pytorch"

class TabFMEmbedder:

    RAW_DIM = 2048  # row_num_cls (8) * embed_dim (256) in tabfm v1.0.0

    def __init__(self, model_type='classification', checkpoint_path=None,
                 device=None, max_rows=512, max_cols=100, seed=42):
        try:
            from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_loader
        except ImportError as e:
            raise ImportError(
                "The tabfm package is required for --tab-model tabfm. "
                "Install it with: pip install 'tabfm[pytorch]'") from e

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.seed = seed
        self.is_classifier = model_type == 'classification'

        # checkpoint_path=None downloads the weights from the HF hub
        self.model = tabfm_loader.load(model_type=model_type,
                                       checkpoint_path=checkpoint_path,
                                       device=str(self.device))
        self.model.eval()
        # load() casts the model to bfloat16 by default; feature tensors must
        # match that compute dtype or in_linear's matmul dtype-mismatches.
        self.dtype = next(self.model.parameters()).dtype

    def _to_numeric(self, df):
        """Mixed-type DataFrame -> (float array [T, H], categorical flags [H])."""
        columns = []
        cat_flags = []
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                values = series.astype(np.float64).values
                is_cat = False
            elif pd.api.types.is_datetime64_any_dtype(series):
                values = series.astype('int64').astype(np.float64).values
                is_cat = False
            else:
                codes, _ = pd.factorize(series.astype(str))
                values = codes.astype(np.float64)
                is_cat = True

            if np.all(np.isnan(values)):
                values = np.zeros_like(values)
            else:
                values = np.where(np.isnan(values), np.nanmean(values), values)
            columns.append(values)
            cat_flags.append(is_cat)

        x = np.stack(columns, axis=1)
        # Standardise + clip, mirroring TabFM's CustomStandardScaler
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-6
        x = np.clip((x - mean) / std, -100.0, 100.0)
        return x, cat_flags

    def _row_representations(self, x, y, train_size, cat_mask):
        """Forward pass up to row_interactor_2 (skips the 24 ICL blocks)."""
        model = self.model
        try:
            emb = model.cell_embedder(x, y, train_size, cat_mask, d=None)
            emb = model.col_embedder(emb, train_size)
            b, t, _, _ = emb.shape
            cls = model.cls_tokens.expand(b, t, -1, -1)
            emb = torch.cat([cls, emb], dim=2)
            emb = model.row_interactor(emb, d=None)
            emb = model.col_embedder_2(emb, train_size)
            return model.row_interactor_2(emb, d=None)
        except AttributeError as e:
            raise RuntimeError(
                'TabFM internals do not match tabfm==1.0.0; the embedding '
                'extraction path needs to be updated for the installed version.') from e

    def embed_dataset(self, df):
        """Compute one raw RAW_DIM vector for a tabular dataset (DataFrame)."""
        df = df.dropna(axis=1, how='all')
        if len(df) > self.max_rows:
            df = df.sample(n=self.max_rows, random_state=self.seed)
        if df.shape[1] > self.max_cols:
            df = df.iloc[:, :self.max_cols]
        if df.shape[0] == 0 or df.shape[1] == 0:
            return np.zeros(self.RAW_DIM, dtype=np.float64)

        x_np, cat_flags = self._to_numeric(df)

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0).to(self.dtype)
        t = x.shape[1]
        # No task labels: all rows are context with a constant dummy target
        if self.is_classifier:
            y = torch.zeros((1, t), dtype=torch.long, device=self.device)
        else:
            y = torch.zeros((1, t), dtype=torch.float32, device=self.device)
        train_size = torch.tensor([t], dtype=torch.long, device=self.device)
        cat_mask = None
        if any(cat_flags):
            cat_mask = torch.tensor(cat_flags, dtype=torch.bool,
                                    device=self.device).unsqueeze(0)

        with torch.no_grad():
            reps = self._row_representations(x, y, train_size, cat_mask)

        return reps.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float64)

    def embed_file(self, filepath):
        if isinstance(filepath, str) and os.path.exists(filepath):
            df = pd.read_csv(filepath)
        elif isinstance(filepath, pd.DataFrame):
            df = filepath
        else:
            raise FileNotFoundError(f'Cannot read tabular dataset: {filepath}')
        return self.embed_dataset(df)
