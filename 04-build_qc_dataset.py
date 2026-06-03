import argparse
import json
import os
import shutil
from collections import Counter

import deepdish as dd
import h5py
import numpy as np
import pandas as pd


def parse_args():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(code_dir, 'data/ABIDE_pcp/cpac/filt_noglobal')
    default_output = os.path.join(code_dir, 'data/ABIDE_pcp/cpac/filt_noglobal_qc_fd020_perc20')

    parser = argparse.ArgumentParser(description='Build an isolated QC-filtered ABIDE raw dataset.')
    parser.add_argument('--source-root', default=default_source,
                        help='Original ABIDE root containing raw/*.h5 and subject_IDs.txt.')
    parser.add_argument('--output-root', default=default_output,
                        help='Output ABIDE root for the filtered dataset.')
    parser.add_argument('--phenotype', default=os.path.join(code_dir, 'data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'),
                        help='ABIDE phenotype CSV.')
    parser.add_argument('--mean-fd-max', type=float, default=0.2,
                        help='Keep subjects with func_mean_fd below this value.')
    parser.add_argument('--perc-fd-max', type=float, default=20.0,
                        help='Keep subjects with func_perc_fd below this value.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove output root before generating the filtered dataset.')
    return parser.parse_args()


def load_source_ids(source_root):
    ids_path = os.path.join(source_root, 'subject_IDs.txt')
    with open(ids_path) as f:
        return [line.strip() for line in f if line.strip()]


def build_filter(pheno, source_ids, mean_fd_max, perc_fd_max):
    pheno = pheno[pheno['SUB_ID'].isin(source_ids)].copy()
    pheno['func_mean_fd'] = pd.to_numeric(pheno['func_mean_fd'], errors='coerce')
    pheno['func_perc_fd'] = pd.to_numeric(pheno['func_perc_fd'], errors='coerce')

    func_ok = pheno['qc_func_rater_2'].eq('OK') | pheno['qc_func_rater_3'].eq('OK')
    mask = (
        pheno['qc_rater_1'].eq('OK') &
        func_ok &
        (pheno['func_mean_fd'] < mean_fd_max) &
        (pheno['func_perc_fd'] < perc_fd_max)
    )
    selected = pheno.loc[mask].set_index('SUB_ID')
    selected_ids = [subject_id for subject_id in source_ids if subject_id in selected.index]
    return selected, selected_ids


def clean_array(value):
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def write_filtered_raw(source_root, output_root, selected_ids):
    source_raw = os.path.join(source_root, 'raw')
    output_raw = os.path.join(output_root, 'raw')
    os.makedirs(output_raw, exist_ok=True)

    missing = []
    for subject_id in selected_ids:
        source_file = os.path.join(source_raw, subject_id + '.h5')
        output_file = os.path.join(output_raw, subject_id + '.h5')
        if not os.path.exists(source_file):
            missing.append(subject_id)
            continue

        sample = dd.io.load(source_file)
        with h5py.File(output_file, 'w') as handle:
            handle.create_dataset('corr', data=clean_array(sample['corr'][()]))
            handle.create_dataset('pcorr', data=clean_array(sample['pcorr'][()]))
            handle.create_dataset('label', data=np.asarray(sample['label'][()]).astype(np.int64))

    if missing:
        raise RuntimeError('Missing raw h5 files for subjects: %s' % ', '.join(missing))


def main():
    args = parse_args()
    if os.path.exists(args.output_root):
        if args.overwrite:
            shutil.rmtree(args.output_root)
        else:
            raise FileExistsError('%s already exists; pass --overwrite to rebuild it.' % args.output_root)

    source_ids = load_source_ids(args.source_root)
    pheno = pd.read_csv(args.phenotype, dtype={'SUB_ID': str})
    selected, selected_ids = build_filter(pheno, source_ids, args.mean_fd_max, args.perc_fd_max)

    os.makedirs(args.output_root, exist_ok=True)
    write_filtered_raw(args.source_root, args.output_root, selected_ids)

    with open(os.path.join(args.output_root, 'subject_IDs.txt'), 'w') as f:
        f.write('\n'.join(selected_ids) + '\n')

    labels = (selected.loc[selected_ids, 'DX_GROUP'].astype(int) % 2).tolist()
    summary = {
        'source_root': args.source_root,
        'output_root': args.output_root,
        'phenotype': args.phenotype,
        'filters': {
            'qc_rater_1': 'OK',
            'qc_func_rater_2_or_3': 'OK',
            'func_mean_fd_lt': args.mean_fd_max,
            'func_perc_fd_lt': args.perc_fd_max,
        },
        'n_subjects': len(selected_ids),
        'label_counts': {str(k): v for k, v in Counter(labels).items()},
        'n_sites': int(selected.loc[selected_ids, 'SITE_ID'].nunique()),
    }
    with open(os.path.join(args.output_root, 'qc_filter_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
