import matplotlib.pyplot as plt
import numpy as np

def plot_sample_predictions(model, val_gen, class_names):
    x, y = next(val_gen)
    predictions = model.predict(x)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)

    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i])
        plt.title(f'True: {class_names[true_classes[i]]}\\nPred: {class_names[predicted_classes[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
