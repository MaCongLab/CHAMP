"""Run endpoint-specific annotation rounds with the supplied prompt templates."""
import argparse
import json
from collections import Counter
from pathlib import Path

from preprocess import assay_records, parse_label, POSITIVE, NEGATIVE
from prompt import toxicity_prompt, cytotoxicity_prompt, hemolysis_prompt

PROMPTS = {'combined': toxicity_prompt, 'cytotoxicity': cytotoxicity_prompt, 'hemolysis': hemolysis_prompt}
ASSAY_KEYS = {'combined': ('toxicity assays', 'cytotoxicity assays'),
              'cytotoxicity': ('Cytotoxicity', 'cytotoxicity assays'),
              'hemolysis': ('Hemolysis', 'hemolysis assays')}


def canonical_label(value, task):
    label = parse_label(value)
    if label in POSITIVE:
        return {'combined': 'toxic', 'cytotoxicity': 'cytotoxic', 'hemolysis': 'hemolytic'}[task]
    if label in NEGATIVE:
        return 'non-hemolytic' if task == 'hemolysis' else 'non-toxic'
    return 'unknown'


def majority_vote(labels):
    label, count = Counter(labels).most_common(1)[0]
    return label if count > len(labels)/2 else 'unknown'


def build_requests(input_path, task, limit=None):
    rows = list(assay_records(input_path))
    if limit is not None:
        rows = rows[:limit]
    requests = []
    for i, row in enumerate(rows):
        key = next((k for k in ASSAY_KEYS[task] if k in row), None)
        if key is None or not isinstance(row[key], list):
            raise ValueError(f'Record {i+1} must contain an assay list for task {task}')
        assays = row[key]
        requests.append({'record_index': i, 'sequence': row.get('sequence', row.get('Sequence')),
                         'assays_empty': len(assays) == 0,
                         'input': f'<experiment>{assays}</experiment>'})
    if not requests:
        raise ValueError('No assay records selected')
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--task', choices=sorted(PROMPTS), default='combined')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Write request payloads without calling an API')
    args = parser.parse_args()
    if args.rounds < 1 or (args.limit is not None and args.limit < 1):
        parser.error('rounds and limit must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    requests = build_requests(args.input, args.task, args.limit)
    prompt = PROMPTS[args.task]
    client = None
    if not args.dry_run:
        from utils import make_client, GPT_QA
        client = make_client()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'task': args.task, 'model': args.model, 'rounds': args.rounds, 'records': len(requests),
                'dry_run': args.dry_run,
                'tie_rule': 'unknown when no strict majority', 'empty_assay_rule': 'unknown without an API request'}
    (args.output_dir/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (args.output_dir/'prompt.txt').write_text(prompt, encoding='utf-8')
    if args.dry_run:
        with (args.output_dir/'requests.jsonl').open('w', encoding='utf-8') as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False)+'\n')
        print(f'Prepared {len(requests)} records. No API requests sent.')
        return
    rounds = []
    for round_no in range(1, args.rounds+1):
        labels = []
        with (args.output_dir/f'round_{round_no}.txt').open('w', encoding='utf-8') as handle:
            for request in requests:
                if request['assays_empty']:
                    label = 'unknown'
                else:
                    text = GPT_QA(prompt, model_name=args.model, input=request['input'], client=client)
                    label = canonical_label(text, args.task)
                labels.append(label)
                handle.write(f'<result>{label}</result>\n')
                handle.flush()
        rounds.append(labels)
    import csv
    with (args.output_dir/'majority_labels.txt').open('w') as labels_file, \
            (args.output_dir/'majority_labels.csv').open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['sequence', 'label'])
        for request, votes in zip(requests, zip(*rounds)):
            label = majority_vote(votes)
            labels_file.write(f'<result>{label}</result>\n')
            writer.writerow([request['sequence'], label])
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
