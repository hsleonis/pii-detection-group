# PII Detection Model Training

A comprehensive, modular framework for training and evaluating Named Entity Recognition (NER) models specifically designed for Personally Identifiable Information (PII) detection. This refactored codebase transforms the original Jupyter notebook into reusable, production-ready Python modules.

## 🌟 Features

### **Modular Architecture**
- **Clean separation of concerns** with dedicated modules for data handling, model training, and evaluation
- **Reusable components** that can be easily imported and integrated into other projects
- **Type hints and comprehensive documentation** for better code maintainability

### **Enhanced NER Model**
- **Extended SimpletTransformers NERModel** with advanced training metrics tracking
- **Real-time loss monitoring** with training, validation, and test loss tracking
- **Comprehensive training statistics** including duration, best metrics, and convergence analysis
- **Automatic model checkpointing** and best model preservation

### **Advanced Data Management**
- **Flexible data loading** with CSV, JSON, and Parquet support
- **Intelligent dataset splitting** with document-level splits to prevent data leakage
- **Class balancing utilities** for handling imbalanced datasets
- **Comprehensive data validation** and quality checks

### **Robust Training Pipeline**
- **Configurable training parameters** with sensible defaults
- **Early stopping and learning rate scheduling** for optimal convergence
- **Mixed precision training** support for faster training on modern GPUs
- **Evaluation during training** with detailed progress tracking

### **Enhanced Evaluation Metrics**
- **Improved seqeval integration** with better entity extraction
- **Comprehensive classification reports** with per-class metrics
- **Training history visualization** with loss curves and performance plots
- **Statistical analysis** of training progression

## 📁 Project Structure

```
pii-detection/
├── seqeval_utils.py          # Enhanced sequence evaluation utilities
├── ner_upgraded.py           # Extended NER model with advanced tracking
├── model_trainer.py          # High-level model training interface
├── data_utils.py             # Data loading and preprocessing utilities
├── main.py                   # Complete usage example and pipeline
├── requirements.txt          # Project dependencies
└── README.md                # This documentation
```

## 🚀 Quick Start

### Installation

1. **Clone or download** the project files
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```python
from data_utils import load_wikipii_dataset
from model_trainer import ModelTrainer
from seqeval_utils import apply_seqeval_patches

# Apply enhanced evaluation metrics
apply_seqeval_patches()

# Load and preprocess dataset
dataset = load_wikipii_dataset('your_dataset.csv')

# Initialize trainer
trainer = ModelTrainer(dataset)

# Train model
trainer.train('bert-base-cased', ratio=70)

# Evaluate
predictions, outputs = trainer.evaluate()

# Make predictions
results = trainer.predict(["John Smith lives in New York"])
```

### Complete Pipeline

```python
from main import PIIDetectionPipeline

# Initialize pipeline
pipeline = PIIDetectionPipeline('wikipii_x.csv', 'outputs')

# Run complete training and evaluation pipeline
results = pipeline.run_complete_pipeline(
    model_name='bert-base-cased',
    train_ratio=70,
    config_overrides={'num_train_epochs': 3, 'learning_rate': 2e-5}
)
```
