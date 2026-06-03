import argparse
import json
import os
from collections import Counter

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Run linear baselines on ABIDE raw h5 connectivity features.')
    parser.add_argument('--dataroot', default=os.path.join(code_dir, 'data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20'),
                        help='Dataset root containing raw/*.h5.')
    parser.add_argument('--output', default=None,
                        help='Optional JSON output path. Defaults to dataroot/linear_baseline_results.json.')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--c', type=float, default=0.1)
    return parser.parse_args()


def load_features(raw_dir):
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.h5'))
    corr = []
    pcorr = []
    abs_pcorr = []
    labels = []
    tri = None

    for filename in files:
        with h5py.File(os.path.join(raw_dir, filename), 'r') as handle:
            corr_mat = np.asarray(handle['corr'])
            pcorr_mat = np.asarray(handle['pcorr'])
            if tri is None:
                tri = np.triu_indices(corr_mat.shape[0], 1)
            corr.append(corr_mat[tri])
            pcorr.append(pcorr_mat[tri])
            abs_pcorr.append(np.abs(pcorr_mat[tri]))
            labels.append(int(np.asarray(handle['label'])[0]))

    features = {
        'corr': np.asarray(corr, dtype=np.float32),
        'pcorr': np.asarray(pcorr, dtype=np.float32),
        'abs_pcorr': np.asarray(abs_pcorr, dtype=np.float32),
    }
    return files, features, np.asarray(labels, dtype=np.int64)


def evaluate_feature_set(x, y, folds, seed, c_value):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(splitter.split(x, y)):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, solver='liblinear',
                               max_iter=1000, class_weight='balanced'),
        )
        clf.fit(x[train_index], y[train_index])
        pred = clf.predict(x[test_index])
        score = clf.decision_function(x[test_index])
        fold_results.append({
            'fold': fold,
            'accuracy': float(accuracy_score(y[test_index], pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y[test_index], pred)),
            'auc': float(roc_auc_score(y[test_index], score)),
        })

    summary = {}
    for metric in ['accuracy', 'balanced_accuracy', 'auc']:
        values = [row[metric] for row in fold_results]
        summary[metric + '_mean'] = float(np.mean(values))
        summary[metric + '_std'] = float(np.std(values))
    return {'folds': fold_results, 'summary': summary}


def main():
    args = parse_args()
    raw_dir = os.path.join(args.dataroot, 'raw')
    output = args.output or os.path.join(args.dataroot, 'linear_baseline_results.json')
    files, features, labels = load_features(raw_dir)

    results = {
        'dataroot': args.dataroot,
        'n_subjects': len(files),
        'label_counts': {str(k): v for k, v in Counter(labels.tolist()).items()},
        'folds': args.folds,
        'seed': args.seed,
        'logistic_regression_c': args.c,
        'features': {},
    }
    for name, x in features.items():
        results['features'][name] = evaluate_feature_set(x, labels, args.folds,
                                                         args.seed, args.c)

    with open(output, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
