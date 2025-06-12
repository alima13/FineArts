# 🎨 Art Analysis Using CNN (Savoias Dataset)

This project uses a Convolutional Neural Network (CNN) to classify and analyze art paintings from the [Savoias Dataset](https://www.savoias.com/). The model leverages transfer learning with VGG16.

---

## 📁 Project Structure

main.py # Main training and evaluation script
models/ # Saved models
data/savoias_dataset/ # Dataset directory
utils/ # Utility functions (dataset loading, visualization, analysis)


---

## 🚀 How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Place your dataset**

Place the downloaded Savoias dataset in:

```bash
data/savoias_dataset/
```
3. **Run the model**

```bash
python main.py
```

## 📊 Outputs

Saved Model: in models/art_analysis_model.h5

Sample Visualizations: Plots of predictions on validation set

Classification Report: Accuracy, precision, recall, F1-score

## 🧠 Model Architecture

 ++ Pretrained VGG16

 ++ Dense + Dropout for classification head

## 🔍  Use Cases

SAVOIAS is ideal for:

Training and evaluating models to predict visual or aesthetic complexity.

Analyzing how complexity impacts design, attention, segmentation, captioning, or user experience in art, advertising, UX, interior decoration, etc.

Benchmarking interpretable deep-learning metrics—some recent studies leverage intermediate CNN activations to correlate with human assessments 


## 📌 License

MIT License. For academic and educational purposes.

