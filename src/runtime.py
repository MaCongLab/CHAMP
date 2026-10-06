"""Shared utilities for training, evaluation, and prediction."""
import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = {'combined': '', 'cytotoxicity': 'cytotoxicity_data',
             'hemolysis': 'hemolysis_data', 'cdhit': 'cdhit_data', 'complete': 'complete_data'}


def task_data(task, filename):
    return ROOT / 'data' / TASK_DIRS[task] / filename


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OriginalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sequences, _, labels, cnn_rows = zip(*batch)
        # The tokenizer limit includes special tokens.
        tokens = self.tokenizer(list(sequences), padding=True, return_tensors='pt',
                                truncation=True, max_length=50)
        width = max(len(row) for row in cnn_rows)
        if width < 11:
            raise ValueError('The CNN needs at least one peptide of >=9 residues in each '
                             'batch. Include a longer peptide or choose a larger batch size; '
                             'padding is not changed automatically.')
        cnn = torch.stack([torch.cat((row, torch.ones(width-len(row), dtype=torch.long)))
                           for row in cnn_rows])
        return tokens['input_ids'], tokens['attention_mask'], torch.stack(labels), cnn


def build_model(esm_model_dir, feat_dim=512, checkpoint=None):
    from model import Prot_model

    esm_path = Path(esm_model_dir).expanduser().resolve()
    if not esm_path.is_dir():
        raise FileNotFoundError(f'ESM model directory not found: {esm_path}. See README.md.')
    model = Prot_model(aac_emb_dim=feat_dim, class_num=2, esm_model_path=str(esm_path))
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for key in ('state_dict', 'model_state_dict'):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Checkpoint must be a tensor state_dict')
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model, model.esm_model.tokenizer


def amp_context(device, enabled):
    return torch.amp.autocast('cuda') if enabled and torch.device(device).type == 'cuda' else nullcontext()


def collect_predictions(model, loader, device, amp=False):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for ids, mask, labels, cnn in loader:
            with amp_context(device, amp):
                logits = model(ids.to(device), mask.to(device), cnn.to(device))
            # Preserve positive raw logits and also export probabilities.
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.cpu())
    if not all_logits:
        raise ValueError('No input records to predict')
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    return labels, logits.numpy(), torch.softmax(logits, dim=1)[:, 1].numpy()


def calculate_metrics(labels, predictions, probabilities, positive_logits):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp+fn)) if tp+fn else None
    specificity = float(tn / (tn+fp)) if tn+fp else None
    both = len(np.unique(labels)) == 2
    return {'n': int(len(labels)), 'accuracy': float(accuracy_score(labels, predictions)),
            'sensitivity': sensitivity, 'specificity': specificity,
            'averaged_ss': (sensitivity+specificity)/2 if both else None,
            'mcc': float(matthews_corrcoef(labels, predictions)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'auroc_probability': float(roc_auc_score(labels, probabilities)) if both else None,
            'auroc_positive_logit': float(roc_auc_score(labels, positive_logits)) if both else None,
            'fdr': float(fp/(fp+tp)) if fp+tp else None,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str)+'\n')


def prediction_main(require_labels):
    # Parse before loading an ESM checkpoint. CPU is supported via --device cpu.
    parser = argparse.ArgumentParser(description='Evaluate AMPToxPred' if require_labels else 'Predict peptide toxicity')
    parser.add_argument('--input', type=Path, required=True, help='CSV with sequence; evaluation also requires label')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--esm-model-dir', type=Path, default=ROOT/'models/ESMplusplus_small')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--feat-dim', type=int, default=512)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                        help='CUDA autocast; ignored on CPU')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.exists() or args.output.with_suffix('.metrics.json').exists() or args.output.with_suffix('.run.json').exists():
        parser.error('Output already exists; choose a new output path')
    from Dataset_esm3 import protein_dataset_esm3
    from torch.utils.data import DataLoader

    dataset = protein_dataset_esm3(args.input, maxlen=50, require_labels=require_labels)
    model, tokenizer = build_model(args.esm_model_dir, args.feat_dim, args.checkpoint)
    device = torch.device(args.device)
    model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=OriginalCollator(tokenizer))
    labels, logits, probabilities = collect_predictions(model, loader, device, args.amp)
    predictions = logits.argmax(axis=1)  # 0.5 boundary, exact ties classified as class 0.
    result = dataset.df.copy()
    if (labels == -1).all():
        result = result.drop(columns=['label'])
    result['positive_logit'] = logits[:, 1]
    result['probability'] = probabilities
    result['pred'] = predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    write_json(args.output.with_suffix('.run.json'), vars(args))
    if (labels >= 0).all():
        metrics = calculate_metrics(labels, predictions, probabilities, logits[:, 1])
        write_json(args.output.with_suffix('.metrics.json'), metrics)
        print(json.dumps(metrics, indent=2))
    print(f'Saved predictions to {args.output}')
