"""Evaluate a checkpoint on a CSV containing sequence and label columns."""
if __name__ == '__main__':
    from runtime import prediction_main
    prediction_main(require_labels=True)
