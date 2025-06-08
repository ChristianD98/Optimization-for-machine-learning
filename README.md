# Curriculum Learning Across Modalities

This project investigates the effectiveness and generalizability of curriculum learning strategies across different data modalities, model architectures, and optimization methods. We test whether structured learning sequences can improve training efficiency and model performance compared to traditional training approaches.

## 🔍 Overview

Curriculum learning is inspired by the human learning process, where we begin with easier concepts before progressing to more difficult ones. This project evaluates four training strategies:

1. **Vanilla** - Traditional random batch sampling
2. **Curriculum** - Samples easier examples first, gradually introducing harder ones
3. **Anti-curriculum** - Reverse ordering (harder examples first)
4. **Challenger** - Our novel curriculum approach that dynamically adjusts difficulty

## 📊 Experiments

We test these strategies across three distinct modalities:

| Modality | Dataset | Model Architecture | Task |
|----------|---------|-------------------|------|
| Text | IMDB Reviews | MLP with embeddings | Sentiment Classification |
| Image | MNIST | multi-CNN | Digit Classification |
| Sound | ESC-50 | multi-CNN | Environmental Sound Classification |

Each experiment evaluates performance using both SGD and Adam optimizers to assess generalizability across optimization methods.

## 🏗 Project Structure

```
Optimization-for-machine-learning/
├── text/
│   ├── main_text.py          # Text experiments runner
│   ├── text_model.py         # IMDB model implementation
│   ├── scoring_text.py       # Text difficulty scoring functions
│   └── data/
│       └── text_data.py      # IMDB dataset loader
├── image/
│   ├── main.py               # Image experiments runner
│   ├── ConvNet.py            # MNIST model implementation
│   ├── scoring.py            # Image difficulty scoring
│   └── data.py               # MNIST dataset loader
├── sound/
│   ├── main_reproduce_paper.py   # Sound experiments runner
│   └── main_train_networks.py    # ESC-50 model implementation
├── pacing.py                 # Core curriculum implementation
└── README.md
```

## 🧠 Curriculum Implementation

The core of our curriculum learning implementation is in the `PacingGenerator` class, which:
- Scores training examples by difficulty
- Implements different pacing strategies (vanilla, curriculum, anti, challenger)
- Dynamically adjusts the training data distribution during epochs

## 🚀 Running Experiments

### Text Experiments (IMDB)

To run text classification experiments with various curriculum strategies:

```bash
python text/main_text.py
```

Modify the `mode` and `optimizer` variables in the script to test different configurations.

### Image Experiments (MNIST)

To run image classification experiments:

```bash
python image/main.py
```

### Sound Experiments (ESC-50)

To run sound classification experiments with all strategies:

```bash
python sound/main_reproduce_paper.py
```

This script will automatically run all combinations of curriculum strategies and optimizers.

## 📈 Results

Results are saved in modality-specific folders with the naming convention `results_{mode}/{data_type}/`:

- `summary_batch_metrics.csv` - Per-batch training loss and accuracy
- `test_results_summary.csv` - Final test set performance
- `periodic_validation_summary.csv` - Validation performance during training


## 🧪 Experiment Configuration

Each experiment can be configured with the following parameters:

- `mode`: Training strategy (vanilla, curriculum, anti, challenger)
- `curriculum_epochs`: Number of epochs to run curriculum pacing
- `starting_fraction`: Initial percentage of data to use
- `inc`: Increment factor for data inclusion
- `step_length`: Number of batches between curriculum steps
- `batch_size`: Size of training batches
- `optimizer`: SGD or Adam