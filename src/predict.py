"""Predict a CSV with a sequence column; labels are optional."""
if __name__ == '__main__':
    from runtime import prediction_main
    prediction_main(require_labels=False)
