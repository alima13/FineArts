from sklearn.metrics import classification_report
import numpy as np

def analyze_results(model, val_gen, class_names):
    val_gen.reset()
    y_true = []
    y_pred = []

    for i in range(len(val_gen)):
        x_batch, y_batch = val_gen[i]
        y_true.extend(np.argmax(y_batch, axis=1))
        preds = model.predict(x_batch)
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
