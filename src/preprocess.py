"""Extract assay labels, prepare new stratified splits, and export endpoint records/FASTA."""
import argparse
import csv
import json
import re
from pathlib import Path

POSITIVE = {'toxic', 'cytotoxic', 'hemolytic', '1'}
NEGATIVE = {'non-toxic', 'non-hemolytic', '0'}
LABELS = POSITIVE | NEGATIVE | {'unknown'}


def parse_label(value):
    value = str(value).strip()
    match = re.fullmatch(r'<result>\s*([^<>]+?)\s*</result>', value)
    label = (match.group(1) if match else value).strip().lower()
    if label not in LABELS:
        raise ValueError(f'Unsupported annotation label: {label!r}')
    return label


def assay_records(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            record = json.loads(line)
            sequence = record.get('sequence', record.get('Sequence'))
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError(f'Missing peptide sequence at line {line_no}')
            yield record


def ensure_new(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def extract_labels(sequence_path, label_path, out_path):
    records = list(assay_records(sequence_path))
    labels = Path(label_path).read_text(encoding='utf-8').splitlines()
    if len(records) != len(labels):
        raise ValueError(f'Assay/label row count mismatch: {len(records)} vs {len(labels)}')
    rows = [(r.get('sequence', r.get('Sequence')), parse_label(label)) for r, label in zip(records, labels)]
    with ensure_new(out_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence', 'label'])
        writer.writerows(rows)
    return len(rows)


def split_data(path, out_dir, seed=42, keep_nonstandard=False, no_header=False, sep=','):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path, sep=sep, header=None if no_header else 'infer',
                     names=['sequence', 'label'] if no_header else None, dtype=str, keep_default_na=False)
    if not {'sequence', 'label'} <= set(df.columns):
        raise ValueError('Input must contain sequence and label columns')
    df = df[['sequence', 'label']].copy()
    if df.isna().any().any():
        raise ValueError('Input contains missing sequences or labels')
    df['sequence'] = df['sequence'].str.replace('\xa0', '', regex=False).str.replace(' ', '', regex=False)
    df['label'] = df['label'].map(parse_label)
    counts = {'input': len(df), 'unknown': int((df.label == 'unknown').sum())}
    df = df[df.label != 'unknown'].copy()
    invalid = df.sequence.str.len().eq(0) | df.sequence.str.contains(r'[^A-Za-z]', regex=True)
    if not keep_nonstandard:
        invalid |= df.sequence.str.contains('[XOUBZ]', case=False, regex=True)
    counts['filtered_sequences'] = int(invalid.sum())
    df = df[~invalid].copy()
    if df.empty:
        raise ValueError('No binary-labelled sequences remain after filtering')
    df['label'] = df.label.map(lambda x: int(x in POSITIVE))
    train, heldout = train_test_split(df, test_size=0.2, stratify=df.label, random_state=seed)
    valid, test = train_test_split(heldout, test_size=0.5, stratify=heldout.label, random_state=seed)
    outputs = {'data_labeled.csv': df, 'train_total.csv': train, 'test_total.csv': heldout,
               'valid.csv': valid, 'test.csv': test}
    folder = Path(out_dir)
    for name in [*outputs, 'split.json']:
        if (folder/name).exists():
            raise FileExistsError(f'Refusing to replace existing data: {folder/name}')
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(folder/name, index=False)
    counts.update(seed=seed, train=len(train), valid=len(valid), test=len(test))
    (folder/'split.json').write_text(json.dumps(counts, indent=2)+'\n')
    return counts


def split_assays(input_path, hemo_path, cyto_path):
    # Assay text identifies hemolysis.
    rows = list(assay_records(input_path))
    if Path(hemo_path).resolve() == Path(cyto_path).resolve():
        raise ValueError('Endpoint output paths must be different')
    hemo_path, cyto_path = ensure_new(hemo_path), ensure_new(cyto_path)
    with hemo_path.open('w', encoding='utf-8') as hemo, cyto_path.open('w', encoding='utf-8') as cyto:
        for row in rows:
            seq = row.get('sequence', row.get('Sequence'))
            key = next((k for k in ['toxicity assays', 'cytotoxicity assays'] if k in row), None)
            if key is None:
                raise ValueError('Expected toxicity assays or cytotoxicity assays key')
            assays = []
            for a in row[key]:
                assays.extend(a if isinstance(a, list) else [a])
            h, c = [], []
            for assay in assays:
                (h if 'hemolysis' in str(assay.get('assay', '')).lower() else c).append(assay)
            hemo.write(json.dumps({'Sequence': seq, 'Hemolysis': h}, ensure_ascii=False)+'\n')
            cyto.write(json.dumps({'Sequence': seq, 'Cytotoxicity': c}, ensure_ascii=False)+'\n')


def csv2fasta(path, out_path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    with ensure_new(out_path).open('w') as handle:
        for i, row in enumerate(rows):
            label = parse_label(row['label'])
            prefix = 'pos' if label in POSITIVE else 'neg' if label in NEGATIVE else 'unknown'
            handle.write(f'>{prefix}_{i}\n{row["sequence"].upper()}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract', help='Join JSONL assay records and one label per line')
    p.add_argument('--assays', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('split', help='Create NEW 80/10/10 stratified splits; published splits are already included')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep-nonstandard', action='store_true')
    p.add_argument('--no-header', action='store_true', help='For a two-column TSV without a header')
    p.add_argument('--sep', choices=['comma', 'tab'], default='comma')
    p = sub.add_parser('split-assays', help='Split combined assay records by assay text')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--hemolysis-output', type=Path, required=True)
    p.add_argument('--cytotoxicity-output', type=Path, required=True)
    p = sub.add_parser('fasta', help='Export uppercase FASTA for external tools')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'extract':
        print(json.dumps({'records': extract_labels(args.assays, args.labels, args.output)}))
    elif args.command == 'split':
        print(json.dumps(split_data(args.input, args.output_dir, args.seed, args.keep_nonstandard,
                                    args.no_header, '\t' if args.sep == 'tab' else ','), indent=2))
    elif args.command == 'split-assays':
        split_assays(args.input, args.hemolysis_output, args.cytotoxicity_output)
    else:
        csv2fasta(args.input, args.output)


if __name__ == '__main__':
    main()
