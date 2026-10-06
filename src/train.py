"""Train AMPToxPred and save checkpoints selected by validation accuracy."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['combined', 'cytotoxicity', 'hemolysis', 'cdhit', 'complete'], default='combined')
    parser.add_argument('--train-data-path', '--train_data_path', type=Path)
    parser.add_argument('--valid-data-path', '--valid_data_path', type=Path)
    parser.add_argument('--esm-model-dir', type=Path, default=ROOT/'models/ESMplusplus_small')
    parser.add_argument('--output-dir', type=Path, help='New/empty directory; default save_models/<task>')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-k', '--save_k', type=int, default=20)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=512)
    parser.add_argument('--valid-batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--feat-dim', '--feat_dim', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze-esm', action='store_true',
                        help='Freeze encoder parameters. Otherwise use their configured gradient flags.')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.save_k < 1 or args.batch_size < 2 or args.valid_batch_size < 1:
        parser.error('epochs/save-k/valid-batch-size must be positive; training batch-size must be >=2')

    import json
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from Dataset_esm3 import protein_dataset_esm3
    from runtime import (OriginalCollator, amp_context, build_model, calculate_metrics,
                         collect_predictions, seed_everything, task_data, write_json)

    train_path = args.train_data_path or task_data(args.task, 'train_total.csv')
    valid_path = args.valid_data_path or task_data(args.task, 'valid.csv')
    if train_path.resolve() == valid_path.resolve():
        parser.error('Training and validation files must be different')
    output = args.output_dir or ROOT/'save_models'/args.task
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output-dir')
    seed_everything(args.seed)
    training = protein_dataset_esm3(train_path, 50)
    validation = protein_dataset_esm3(valid_path, 50)
    if len(training) < 2 or len(training) % args.batch_size == 1:
        parser.error('The final training batch would have one record, which BatchNorm cannot train on. '
                     'Choose a different --batch-size.')
    model, tokenizer = build_model(args.esm_model_dir, args.feat_dim)
    if args.freeze_esm:
        model.esm_model.requires_grad_(False)
    device = torch.device(args.device)
    model.to(device)
    collator = OriginalCollator(tokenizer)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    valid_loader = DataLoader(validation, batch_size=args.valid_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args), train_data_path=str(train_path.resolve()), valid_data_path=str(valid_path.resolve()),
                    trainable_encoder_parameters=sum(p.numel() for p in model.esm_model.parameters() if p.requires_grad))
    write_json(output/'run.json', metadata)
    best_accuracy, step = -1.0, 0
    with SummaryWriter(str(output/'logs')) as writer, (output/'history.jsonl').open('w') as history:
        for epoch in range(args.epochs):
            model.train()
            model.esm_model.eval()  # Evaluation mode is distinct from parameter freezing.
            total_loss, seen = 0.0, 0
            for ids, mask, labels, cnn in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, use_amp):
                    logits = model(ids.to(device), mask.to(device), cnn.to(device))
                    loss = criterion(logits, labels.to(device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(labels)
                seen += len(labels)
                writer.add_scalar('Loss/train_celoss', float(loss.detach()), step)
                step += 1
            # Validation runs in full precision.
            labels, logits, probabilities = collect_predictions(model, valid_loader, device, amp=False)
            metrics = calculate_metrics(labels, logits.argmax(axis=1), probabilities, logits[:, 1])
            record = {'epoch': epoch, 'train_loss': total_loss/seen, **metrics}
            history.write(json.dumps(record, allow_nan=False)+'\n')
            history.flush()
            writer.add_scalar('Valid/accuracy', metrics['accuracy'], epoch)
            writer.add_scalar('Valid/f1', metrics['f1'], epoch)
            print(json.dumps(record))
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                torch.save(model.state_dict(), output/'var_model_best.ckpt')
                write_json(output/'best_validation.json', record)
            if epoch % args.save_k == 0:
                torch.save(model.state_dict(), output/f'var_model_ep_{epoch}.ckpt')
    print(f'Best validation accuracy: {best_accuracy:.4f}. Checkpoints: {output}')


if __name__ == '__main__':
    main()
