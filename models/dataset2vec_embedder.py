import importlib.util
import json
import os

import numpy as np
import pandas as pd

# the cloned https://github.com/hadijomaa/dataset2vec repo at the project root
D2V_REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'dataset2vec')


def _load_repo_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(D2V_REPO_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Dataset2VecEmbedder:

    RAW_DIM = 32  # units_h of the published checkpoints

    def __init__(self, checkpoint_dir=None, split=0, n_batches=10, seed=42):
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                'tensorflow is required for --tab-model dataset2vec. '
                'Install it with: pip install tensorflow') from e
        self.tf = tf

        if not os.path.isdir(D2V_REPO_DIR):
            raise FileNotFoundError(
                f'dataset2vec repo not found at {D2V_REPO_DIR}. Clone it with: '
                'git clone https://github.com/hadijomaa/dataset2vec.git')

        self._modules = _load_repo_module('d2v_modules', 'modules.py')
        self._flatten = _load_repo_module('d2v_dummdataset', 'dummdataset.py').flatten

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(D2V_REPO_DIR, 'checkpoints', f'split-{split}')
        with open(os.path.join(checkpoint_dir, 'configuration.txt')) as f:
            config = json.load(f)

        self.batch_size = config['batch_size']
        self.ninstanc = config['ninstanc']
        self.nclasses = config['nclasses']
        self.nfeature = config['nfeature']
        self.n_batches = n_batches
        self.seed = seed

        nonlinearity = config['nonlinearity_d2v']
        self._function_f = self._modules.FunctionF(
            units=config['units_f'], nhidden=config['nhidden_f'], nonlinearity=nonlinearity,
            architecture=config['architecture_f'], resblocks=config['resblocks_f'], trainable=False)
        self._pool_f = self._modules.PoolF(units=config['units_f'])
        self._function_g = self._modules.FunctionG(
            units=config['units_g'], nhidden=config['nhidden_g'], nonlinearity=nonlinearity,
            architecture=config['architecture_g'], trainable=False)
        self._pool_g = self._modules.PoolG(units=config['units_g'])
        self._function_h = self._modules.FunctionH(
            units=config['units_h'], nhidden=config['nhidden_h'], nonlinearity=nonlinearity,
            architecture=config['architecture_h'], resblocks=config['resblocks_h'], trainable=False)

        # one dummy forward pass so every Dense layer builds its variables
        dummy = self._sample_instance(np.zeros((4, 2)), np.zeros(4, dtype=np.int64), 1,
                                      np.random.default_rng(0))
        self._forward([dummy])
        self._restore_checkpoint(checkpoint_dir)

    def _restore_checkpoint(self, checkpoint_dir):
        reader = self.tf.train.load_checkpoint(os.path.join(checkpoint_dir, 'weights'))
        roots = {0: self._function_f, 1: self._function_g, 2: self._function_h}
        n_params = 0
        for key in reader.get_variable_to_shape_map():
            if not key.startswith('layer_with_weights-'):
                continue
            parts = key.split('/')
            obj = roots[int(parts[0].split('-')[1])]
            i = 1
            while parts[i] == 'block':
                obj = obj.block[int(parts[i + 1])]
                i += 2
            value = reader.get_tensor(key)
            getattr(obj, parts[i]).assign(value)
            n_params += value.size
        expected = sum(int(np.prod(w.shape)) for layer in roots.values() for w in layer.weights)
        if n_params != expected:
            raise RuntimeError(
                f'Dataset2Vec checkpoint restore mismatch: assigned {n_params} '
                f'parameters but the model has {expected}')

    def _prepare(self, df):
        """DataFrame -> (min-max scaled predictors [N, F], class labels [N])."""
        df = df.dropna(axis=1, how='all')
        if df.shape[1] >= 2:
            predictors, target = df.iloc[:, :-1], df.iloc[:, -1]
        else:
            predictors, target = df, None

        columns = []
        for col in predictors.columns:
            series = predictors[col]
            if pd.api.types.is_numeric_dtype(series):
                values = series.astype(np.float64).values
            elif pd.api.types.is_datetime64_any_dtype(series):
                values = series.astype('int64').astype(np.float64).values
            else:
                codes, _ = pd.factorize(series.astype(str))
                values = codes.astype(np.float64)
            if np.all(np.isnan(values)):
                values = np.zeros_like(values)
            else:
                values = np.where(np.isnan(values), np.nanmean(values), values)
            columns.append(values)
        x = np.stack(columns, axis=1)

        # per-column min-max scaling, as in the repo's ptp() / the "minmax"
        # normalisation the checkpoints were trained with
        span = np.ptp(x, axis=0)
        span[span == 0] = 1.0
        x = (x - x.min(axis=0)) / span

        if target is None:
            labels = np.zeros(len(df), dtype=np.int64)
        elif pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
            # continuous target: quantile-bin it into discrete classes
            labels = pd.qcut(target.rank(method='first'), q=10, labels=False).values
        else:
            labels, _ = pd.factorize(target.astype(str))
        return x, labels.astype(np.int64)

    def _sample_instance(self, x, labels, n_labels, rng):
        """One (flattened batch, ninstanc, nfeature, nclasses) sampling item."""
        rows = rng.choice(x.shape[0], size=min(self.ninstanc, x.shape[0]), replace=False)
        feats = rng.choice(x.shape[1], size=min(self.nfeature, x.shape[1]), replace=False)
        classes = rng.choice(n_labels, size=min(self.nclasses, n_labels), replace=False)
        x_sub = x[rows][:, feats]
        # one-hot of only the sampled classes, so high-cardinality targets
        # never materialise a full N x C matrix
        y_sub = (labels[rows][:, None] == classes[None, :]).astype(np.float64)
        flat = self._flatten(x_sub, y_sub)
        return flat, x_sub.shape[0], x_sub.shape[1], y_sub.shape[1]

    def _forward(self, instances):
        tf = self.tf
        x = tf.constant(np.vstack([inst[0] for inst in instances]), dtype=tf.float32)
        ninstanc = tf.constant([inst[1] for inst in instances], dtype=tf.int32)
        nfeature = tf.constant([inst[2] for inst in instances], dtype=tf.int32)
        nclasses = tf.constant([inst[3] for inst in instances], dtype=tf.int32)

        e = self._function_f(x)
        e = self._pool_f(e, nclasses, nfeature, ninstanc)
        e = self._function_g(e)
        e = self._pool_g(e, nclasses, nfeature)
        return self._function_h(e)

    def embed_dataset(self, df):
        """Compute one raw RAW_DIM meta-feature vector for a tabular dataset."""
        if df.shape[0] == 0 or df.shape[1] == 0:
            return np.zeros(self.RAW_DIM, dtype=np.float64)
        x, labels = self._prepare(df)
        n_labels = int(labels.max()) + 1

        rng = np.random.default_rng(self.seed)
        metafeatures = []
        for _ in range(self.n_batches):
            instances = [self._sample_instance(x, labels, n_labels, rng)
                         for _ in range(self.batch_size)]
            metafeatures.append(self._forward(instances).numpy())
        return np.vstack(metafeatures).mean(axis=0).astype(np.float64)

    def embed_file(self, filepath):
        if isinstance(filepath, pd.DataFrame):
            df = filepath
        elif isinstance(filepath, str) and os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f'Cannot read tabular dataset: {filepath}')
        return self.embed_dataset(df)
