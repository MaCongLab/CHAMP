# CHAMP / AMPToxPred

CHAMP is an antimicrobial peptide dataset with toxicity annotations derived from
experimental assay records. AMPToxPred combines a pretrained protein language
model with TextCNN to predict combined toxicity, cytotoxicity, and hemolysis.

This repository accompanies the CHAMP study and provides the datasets and code
for assay annotation, data processing, model training, evaluation, and prediction.

## Repository structure

```text
CHAMP/
├── data/
│   ├── complete_data/              # Combined toxicity assays and dataset splits
│   │   ├── combined_dataset.txt
│   │   ├── train_total.csv
│   │   ├── valid.csv
│   │   └── test.csv
│   ├── cytotoxicity_data/          # Cytotoxicity assays and dataset splits
│   ├── hemolysis_data/             # Hemolysis assays and dataset splits
│   ├── cdhit_data/                 # CD-HIT benchmark splits
│   └── combined_dataset_gpt-5.txt  # Combined toxicity annotations
├── src/
│   ├── model.py                   # AMPToxPred architecture
│   ├── Dataset_esm3.py            # Sequence loading and encoding
│   ├── runtime.py                 # Batching, checkpoint loading, and metrics
│   ├── train.py                   # Model training
│   ├── test.py                    # Evaluation on labelled sequences
│   ├── predict.py                 # Prediction for new sequences
│   ├── preprocess.py              # Data preparation
│   ├── gpt_predict.py             # LLM-based assay annotation
│   ├── prompt.py                  # Task-specific annotation prompts
│   ├── utils.py
│   └── vocab.txt
├── save_models/
│   ├── toxicity_model-totaldata/var_model_best.ckpt  # Downloaded checkpoint
│   ├── cytotoxicity-model/var_model_best.ckpt
│   └── hemolysis-model/var_model_best.ckpt
├── requirements.txt
└── LICENSE
```

## Installation

Use Python 3.11 or later and run the commands below from the repository root.
Check that `python3 --version` reports a supported version before creating the
environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU execution, install a PyTorch build compatible with your CUDA environment.
Use `--device cpu` in the training and prediction commands for CPU execution.

## Datasets

Each dataset contains `train_total.csv`, `valid.csv`, and `test.csv`:

- **Combined toxicity** (`data/complete_data/`): 4,545 training, 568 validation,
  and 569 test sequences.
- **Cytotoxicity** (`data/cytotoxicity_data/`): 1,620 training, 203 validation,
  and 203 test sequences.
- **Hemolysis** (`data/hemolysis_data/`): 3,062 training, 383 validation,
  and 383 test sequences.
- **CD-HIT benchmark** (`data/cdhit_data/`): 2,256 training, 282 validation,
  and 282 test sequences.

The CSV files contain `sequence` and `label` columns. Label `1` denotes the
positive toxicity endpoint and label `0` denotes its negative class. Unknown
annotations are excluded from binary modelling. Sequence case is preserved;
lowercase amino acid letters denote D-amino acids in the dataset.

Assay records are stored as JSON Lines. The combined toxicity file,
`data/complete_data/combined_dataset.txt`, uses the fields `sequence` and
`toxicity assays`. Endpoint files use `Sequence` with either `Cytotoxicity` or
`Hemolysis`. Each line in `data/combined_dataset_gpt-5.txt` corresponds to the
assay record at the same position in the combined file.

## Model setup

AMPToxPred requires the pretrained `Synthyra/ESMplusplus_small` encoder and a
checkpoint for the toxicity endpoint of interest. Download the specified encoder
revision into `models/ESMplusplus_small` to use the dependency versions in
`requirements.txt`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Synthyra/ESMplusplus_small', revision='8f5693f6f09ed042163d671ecf379137d0869652', local_dir='models/ESMplusplus_small')"
```

The encoder is loaded from a local directory. If it is stored elsewhere, specify
its path with `--esm-model-dir`. Select the task checkpoint with `--checkpoint`:

- Combined toxicity (`AMPToxPred-combined.ckpt`):
  `save_models/toxicity_model-totaldata/var_model_best.ckpt`
- Cytotoxicity (`AMPToxPred-cytotoxicity.ckpt`):
  `save_models/cytotoxicity-model/var_model_best.ckpt`
- Hemolysis (`AMPToxPred-hemolysis.ckpt`):
  `save_models/hemolysis-model/var_model_best.ckpt`

Checkpoint files are excluded from Git. Download the best weights from the
[CHAMP Releases page](https://github.com/MaCongLab/CHAMP/releases) and save each
asset at the path listed above.

The training command below can also produce a new combined toxicity checkpoint
at `outputs/training/combined/var_model_best.ckpt`.

## Prediction

Prepare a CSV file with a `sequence` column, for example `peptides.csv`:

```csv
sequence
GIGKFLHSAKKFGKAFVGEIMNS
KWKLFKKIGAVLKVL
```

Run cytotoxicity prediction:

```bash
python src/predict.py \
  --input peptides.csv \
  --checkpoint save_models/cytotoxicity-model/var_model_best.ckpt \
  --esm-model-dir models/ESMplusplus_small \
  --device cuda --batch-size 64 \
  --output outputs/cytotoxicity_predictions.csv
```

To predict combined toxicity or hemolysis, use the corresponding checkpoint.
The output CSV contains the input columns and three prediction fields:

- `probability`: softmax probability of the positive class.
- `pred`: predicted binary label, selected by the larger of the two logits.
- `positive_logit`: raw model score for the positive class.

The decision boundary is a positive-class probability of 0.5, with exact ties
assigned to class `0`. A `.run.json` file records the inference settings. CUDA
mixed precision is enabled by default and can be disabled with `--no-amp`.

## Evaluation

Evaluate a checkpoint using a CSV with both `sequence` and `label` columns:

```bash
python src/test.py \
  --input data/cytotoxicity_data/test.csv \
  --checkpoint save_models/cytotoxicity-model/var_model_best.ckpt \
  --esm-model-dir models/ESMplusplus_small \
  --device cuda --batch-size 64 \
  --output outputs/cytotoxicity_test.csv
```

For combined toxicity, use `data/complete_data/test.csv` and the combined toxicity
checkpoint. For hemolysis, use `data/hemolysis_data/test.csv` and the hemolysis
checkpoint.

Alongside the predictions, the script saves a `.metrics.json` file containing
accuracy, sensitivity, specificity, their average, MCC, F1 score, false discovery
rate, and confusion-matrix counts. AUROC is reported separately for positive-class
probabilities (`auroc_probability`) and raw positive logits
(`auroc_positive_logit`).

## Training

Train the combined toxicity model using the supplied training and validation
splits:

```bash
python src/train.py \
  --task combined \
  --train-data-path data/complete_data/train_total.csv \
  --valid-data-path data/complete_data/valid.csv \
  --esm-model-dir models/ESMplusplus_small \
  --batch-size 256 --valid-batch-size 128 \
  --epochs 100 --lr 1e-4 --weight-decay 1e-5 \
  --freeze-esm --device cuda --seed 42 \
  --output-dir outputs/training/combined
```

For cytotoxicity or hemolysis, change `--task`, both data paths, and the output
directory to the corresponding endpoint. Custom datasets can also be supplied
through `--train-data-path` and `--valid-data-path`.

Training uses AdamW and cross-entropy loss. The `--freeze-esm` flag freezes the
encoder parameters; the encoder operates in evaluation mode during training.
The checkpoint with the highest validation accuracy is saved as
`var_model_best.ckpt`. Each run also produces periodic checkpoints, validation
metrics, epoch history, TensorBoard logs, and a `run.json` file containing the
run configuration. Use a new or empty output directory for each run.

### Sequence processing

Sequences retain their case and are truncated to 50 residues before encoding.
The CNN uses a 55-token vocabulary and pads sequences to the longest item in each
batch. The encoder tokenizer applies a separate limit of 50 tokens, including
special tokens. Each batch must contain at least one sequence of nine or more
residues to satisfy the CNN pooling dimensions.

Keep the batch size, sequence order, encoder files, checkpoint, and precision
settings consistent when comparing prediction results.

## Assay annotation

Task-specific prompts are defined in `src/prompt.py`. Preview the combined
toxicity annotation inputs without making API requests:

```bash
python src/gpt_predict.py \
  --input data/complete_data/combined_dataset.txt \
  --task combined --limit 3 --dry-run \
  --output-dir outputs/annotation_preview
```

Set `OPENAI_API_KEY` in your environment, then test three annotation rounds on
three records:

```bash
python src/gpt_predict.py \
  --input data/complete_data/combined_dataset.txt \
  --task combined --model gpt-5-mini --rounds 3 --limit 3 \
  --output-dir outputs/combined_annotation
```

Omit `--limit 3` to annotate the entire input file. Each nonempty assay record
can produce one API request per round.

For endpoint-specific annotation, use:

- `--task cytotoxicity` with
  `data/cytotoxicity_data/cytotoxicity_dataset.txt`.
- `--task hemolysis` with `data/hemolysis_data/hemolysis_dataset.txt`.

The output includes the annotations from each round, majority labels in text and
CSV formats, the prompt, and run metadata. Empty assay lists and votes without a
strict majority receive an `unknown` label. API credentials are read from the
environment; `OPENAI_BASE_URL` can be set for a compatible endpoint.

## Data preparation

Convert aligned assay records and annotation labels into a CSV dataset:

```bash
python src/preprocess.py extract \
  --assays data/complete_data/combined_dataset.txt \
  --labels data/combined_dataset_gpt-5.txt \
  --output outputs/labelled.csv
```

Create stratified training, validation, and test splits for a new dataset:

```bash
python src/preprocess.py split \
  --input outputs/labelled.csv \
  --seed 42 --output-dir outputs/new_splits
```

The split command excludes unknown labels, filters sequences containing
X/O/U/B/Z (case-insensitive) or unsupported punctuation, and creates approximately
80/10/10 splits. Use `--keep-nonstandard` to retain alphabetic nonstandard
residues. For a headerless TSV input, add `--no-header --sep tab`.

Additional subcommands include `split-assays` for separating assay records by
endpoint and `fasta` for exporting uppercase FASTA sequences. Run
`python src/preprocess.py --help` for details.

## License

This repository is distributed under the [MIT License](LICENSE).
