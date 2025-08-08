#!/usr/bin/env python3
"""
Complete Usage Example for PII Detection Model Training
This script demonstrates how to use the refactored classes and functions.
"""

import os
import sys
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_utils import DataLoader, load_wikipii_dataset, quick_dataset_analysis
from model_trainer import ModelTrainer
from ner_upgraded import NERUpgraded
from seqeval_utils import apply_seqeval_patches
from simpletransformers.ner import NERArgs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply seqeval patches for improved metrics
apply_seqeval_patches()


class PIIDetectionPipeline:
    """
    Complete pipeline for PII detection model training and evaluation.
    """
    
    def __init__(self, data_path: str, output_dir: str = "outputs"):
        """
        Initialize the PII detection pipeline.
        
        Args:
            data_path: Path to the dataset CSV file
            output_dir: Directory for outputs and model checkpoints
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_loader = DataLoader()
        self.model_trainer = None
        self.dataset = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 Initializing PII Detection Pipeline")
        print(f"   📂 Data path: {data_path}")
        print(f"   📁 Output directory: {output_dir}")

    def load_and_analyze_data(self) -> Dict[str, Any]:
        """
        Load dataset and perform initial analysis.
        
        Returns:
            Analysis results
        """
        print("\n" + "="*50)
        print("📊 STEP 1: DATA LOADING AND ANALYSIS")
        print("="*50)
        
        # Load dataset
        self.dataset = load_wikipii_dataset(self.data_path, preprocess=True)
        
        # Perform analysis
        analysis = quick_dataset_analysis(self.dataset)
        
        # Print summary
        self.data_loader.print_dataset_summary(self.dataset, "WikiPII Dataset")
        
        # Print analysis results
        print("\n🔍 Analysis Results:")
        for rec in analysis['recommendations']:
            print(f"   {rec}")
        
        return analysis

    def setup_training_configuration(self, **config_overrides) -> NERArgs:
        """
        Setup training configuration with sensible defaults.
        
        Args:
            **config_overrides: Configuration parameters to override
            
        Returns:
            Configured NERArgs object
        """
        print("\n" + "="*50)
        print("⚙️  STEP 2: TRAINING CONFIGURATION")
        print("="*50)
        
        # Default configuration
        default_config = {
            'num_train_epochs': 3,
            'learning_rate': 1e-4,
            'train_batch_size': 16,
            'eval_batch_size': 32,
            'max_seq_length': 128,
            'overwrite_output_dir': True,
            'save_model_every_epoch': True,
            'evaluate_during_training': True,
            'evaluate_during_training_steps': 100,
            'logging_steps': 50,
            'save_steps': 500,
            'output_dir': self.output_dir,
            'best_model_dir': os.path.join(self.output_dir, 'best_model'),
            'tensorboard_dir': os.path.join(self.output_dir, 'tensorboard'),
            'early_stopping_patience': 3,
            'early_stopping_delta': 0.001,
            'use_early_stopping': True,
            'fp16': False,  # Set to True if using modern GPU
            'dataloader_num_workers': 2
        }
        
        # Apply overrides
        config = {**default_config, **config_overrides}
        
        # Create NERArgs
        args = NERArgs()
        for key, value in config.items():
            setattr(args, key, value)
        
        print("📋 Training Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        return args

    def initialize_trainer(self, args: NERArgs) -> None:
        """
        Initialize the model trainer with the dataset and configuration.
        
        Args:
            args: Training arguments
        """
        print("\n" + "="*50)
        print("🎯 STEP 3: TRAINER INITIALIZATION")
        print("="*50)
        
        self.model_trainer = ModelTrainer(
            df=self.dataset,
            args=args,
            sim_model_id='bert'  # Simplified model name
        )
        
        # Print dataset info
        dataset_info = self.model_trainer.get_dataset_info()
        print(f"📊 Dataset Information:")
        for key, value in dataset_info.items():
            if key != 'label_distribution':  # Skip detailed distribution for brevity
                print(f"   {key}: {value}")

    def train_model(self, model_name: str = 'bert-base-cased', train_ratio: float = 70) -> Any:
        """
        Train the NER model.
        
        Args:
            model_name: HuggingFace model identifier
            train_ratio: Training data percentage
            
        Returns:
            Training results
        """
        print("\n" + "="*50)
        print("🏋️  STEP 4: MODEL TRAINING")
        print("="*50)
        
        if self.model_trainer is None:
            raise ValueError("Trainer not initialized. Call initialize_trainer() first.")
        
        # Train the model
        training_results = self.model_trainer.train(
            model_id=model_name,
            ratio=train_ratio,
            output_dir=self.output_dir
        )
        
        return training_results

    def evaluate_model(self, evaluation_segments: int = 1) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            evaluation_segments: Number of segments for evaluation
            
        Returns:
            Evaluation results
        """
        print("\n" + "="*50)
        print("🧪 STEP 5: MODEL EVALUATION")
        print("="*50)
        
        if self.model_trainer is None:
            raise ValueError("Trainer not initialized.")
        
        # Evaluate model
        predictions, model_outputs = self.model_trainer.evaluate(
            split=evaluation_segments,
            verbose=True
        )
        
        # Get evaluation metrics
        eval_results = self.model_trainer.get_evaluation_result()
        
        return {
            'metrics': eval_results,
            'predictions': predictions,
            'model_outputs': model_outputs
        }

    def generate_training_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Returns:
            Training report with metrics and visualizations
        """
        print("\n" + "="*50)
        print("📈 STEP 6: TRAINING REPORT")
        print("="*50)
        
        if self.model_trainer is None or not self.model_trainer._is_trained:
            raise ValueError("Model not trained yet.")
        
        # Get training history
        history = self.model_trainer.get_training_history()
        
        # Get training statistics
        stats = self.model_trainer.model.get_training_stats() if self.model_trainer.model else {}
        
        report = {
            'training_stats': stats,
            'training_history': history,
            'model_info': {
                'model_id': self.model_trainer._model_id,
                'model_name': self.model_trainer._model_name,
                'is_trained': self.model_trainer._is_trained,
                'is_evaluated': self.model_trainer._is_evaluated
            }
        }
        
        # Print summary
        print("📊 Training Summary:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return report

    def create_visualizations(self, save_plots: bool = True) -> None:
        """
        Create training and evaluation visualizations.
        
        Args:
            save_plots: Whether to save plots to files
        """
        print("\n" + "="*50)
        print("📊 STEP 7: VISUALIZATION")
        print("="*50)
        
        if self.model_trainer is None or self.model_trainer.model is None:
            print("⚠️  No training data available for visualization")
            return
        
        # Setup plotting
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PII Detection Training Analysis', fontsize=16, fontweight='bold')
        
        # Training loss plot
        train_losses = self.model_trainer.model.train_loss_list
        if train_losses:
            axes[0, 0].plot(train_losses, 'b-', alpha=0.7)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Evaluation metrics plot
        eval_results = self.model_trainer.model.eval_loss_list
        if eval_results:
            # Extract F1 scores if available
            f1_scores = [result.get('eval_f1', 0) for result in eval_results if isinstance(result, dict)]
            if f1_scores:
                axes[0, 1].plot(f1_scores, 'g-', marker='o')
                axes[0, 1].set_title('Validation F1 Score')
                axes[0, 1].set_xlabel('Evaluation Step')
                axes[0, 1].set_ylabel('F1 Score')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Label distribution
        if self.dataset is not None:
            label_counts = self.dataset['labels'].value_counts()
            top_labels = label_counts.head(10)  # Show top 10 labels
            axes[1, 0].bar(range(len(top_labels)), top_labels.values)
            axes[1, 0].set_title('Top 10 Label Distribution')
            axes[1, 0].set_xlabel('Label')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_xticks(range(len(top_labels)))
            axes[1, 0].set_xticklabels(top_labels.index, rotation=45, ha='right')
        
        # Training progress summary
        if train_losses:
            # Moving average of training loss
            window_size = max(1, len(train_losses) // 20)
            if len(train_losses) > window_size:
                moving_avg = pd.Series(train_losses).rolling(window=window_size).mean()
                axes[1, 1].plot(moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
                axes[1, 1].plot(train_losses, 'b-', alpha=0.3, label='Raw Loss')
                axes[1, 1].set_title('Training Loss Trend')
                axes[1, 1].set_xlabel('Batch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'training_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {plot_path}")
        
        plt.show()

    def make_predictions(self, sample_sentences: List[str]) -> Dict[str, Any]:
        """
        Make predictions on sample sentences.
        
        Args:
            sample_sentences: List of sentences to analyze
            
        Returns:
            Prediction results
        """
        print("\n" + "="*50)
        print("🔮 STEP 8: SAMPLE PREDICTIONS")
        print("="*50)
        
        if self.model_trainer is None or not self.model_trainer._is_trained:
            raise ValueError("Model not trained yet.")
        
        # Make predictions
        predictions, raw_outputs = self.model_trainer.predict(sample_sentences)
        
        # Display results
        print("🔮 Prediction Results:")
        for i, (sentence, pred) in enumerate(zip(sample_sentences, predictions)):
            print(f"\n   Sentence {i+1}: {sentence}")
            print(f"   Entities: {[f'{entity['word']}({entity['entity']})' for entity in pred]}")
        
        return {
            'sentences': sample_sentences,
            'predictions': predictions,
            'raw_outputs': raw_outputs
        }

    def run_complete_pipeline(
        self,
        model_name: str = 'bert-base-cased',
        train_ratio: float = 70,
        config_overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run the complete PII detection pipeline.
        
        Args:
            model_name: HuggingFace model to use
            train_ratio: Training data percentage
            config_overrides: Training configuration overrides
            
        Returns:
            Complete pipeline results
        """
        if config_overrides is None:
            config_overrides = {}
        
        print("🚀 Starting Complete PII Detection Pipeline")
        print("="*60)
        
        results = {}
        
        try:
            # Step 1: Load and analyze data
            results['data_analysis'] = self.load_and_analyze_data()
            
            # Step 2: Setup configuration
            args = self.setup_training_configuration(**config_overrides)
            results['config'] = args.__dict__
            
            # Step 3: Initialize trainer
            self.initialize_trainer(args)
            
            # Step 4: Train model
            results['training'] = self.train_model(model_name, train_ratio)
            
            # Step 5: Evaluate model
            results['evaluation'] = self.evaluate_model()
            
            # Step 6: Generate report
            results['report'] = self.generate_training_report()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            # Step 8: Sample predictions
            sample_sentences = [
                "John Smith lives in New York and his email is john@example.com",
                "The meeting is scheduled for March 15th, 2024",
                "Please contact Sarah Johnson at (555) 123-4567"
            ]
            results['predictions'] = self.make_predictions(sample_sentences)
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            logger.exception("Pipeline execution failed")
            raise
        
        return results


def main():
    """Main function demonstrating the complete pipeline."""
    
    # Configuration
    DATA_PATH = "wikipii_x.csv"  # Update with your data path
    OUTPUT_DIR = "pii_detection_outputs"
    MODEL_NAME = "bert-base-cased"
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Please update DATA_PATH with the correct path to your dataset.")
        return
    
    # Custom training configuration
    config_overrides = {
        'num_train_epochs': 2,  # Reduced for demo
        'train_batch_size': 8,  # Smaller batch size for limited GPU memory
        'eval_batch_size': 16,
        'learning_rate': 2e-5,
        'fp16': False,  # Set to True if using modern GPU
        'evaluate_during_training_steps': 50,
        'logging_steps': 25
    }
    
    # Initialize and run pipeline
    pipeline = PIIDetectionPipeline(DATA_PATH, OUTPUT_DIR)
    
    try:
        results = pipeline.run_complete_pipeline(
            model_name=MODEL_NAME,
            train_ratio=70,
            config_overrides=config_overrides
        )
        
        print("\n📊 Final Results Summary:")
        print(f"   Training batches: {results['report']['training_stats']['total_training_batches']}")
        print(f"   Final training loss: {results['report']['training_stats']['final_training_loss']:.4f}")
        print(f"   Evaluation metrics: {results['evaluation']['metrics']}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
