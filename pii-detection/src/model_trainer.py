"""
Model Training Manager for PII Detection
Handles dataset preparation, model configuration, training, and evaluation.
"""

import random
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from simpletransformers.ner import NERArgs
from ner_upgraded import NERUpgraded


class ModelTrainer:
    """
    High-level interface for training and evaluating NER models.
    
    This class provides a simplified interface for:
    - Dataset management and splitting
    - Model configuration
    - Training with progress tracking
    - Evaluation and metrics calculation
    - Result visualization preparation
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        args: Optional[NERArgs] = None, 
        sim_model_id: str = ''
    ) -> None:
        """
        Initialize the model trainer.
        
        Args:
            df: Training dataset as pandas DataFrame
            args: NER training arguments
            sim_model_id: Simplified model identifier for naming
        """
        # Configure model arguments
        if args is None:
            self.args = self._create_default_args()
        else:
            self.args = args
        
        # Set default attributes if not present
        self._set_default_attributes()
        
        # Set dataset
        self.set_dataset(df)
        
        # Model configuration
        self.sim_model_id = sim_model_id
        self._model_id = ''
        
        # Training state tracking
        self._is_trained = False
        self._is_evaluated = False
        self.result: Optional[Dict[str, Any]] = None
        
        # Data splits
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        # Model instance
        self.model: Optional[NERUpgraded] = None

    def _create_default_args(self) -> NERArgs:
        """Create default NER arguments."""
        args = NERArgs()
        args.evaluate_during_training = False
        args.evaluate_during_training_steps = 3
        args.num_train_epochs = 3
        args.learning_rate = 1e-4
        args.overwrite_output_dir = True
        args.train_batch_size = 128
        args.eval_batch_size = 128
        return args

    def _set_default_attributes(self) -> None:
        """Set default attributes if not present in args."""
        if not hasattr(self.args, 'split_column'):
            self.args.split_column = 'doc_id'
        
        if not hasattr(self.args, 'label_column'):
            self.args.label_column = 'labels'
            
        if not hasattr(self.args, 'labels_list'):
            self.args.labels_list = []

    def set_dataset(self, df: pd.DataFrame) -> None:
        """
        Set the training dataset.
        
        Args:
            df: Dataset as pandas DataFrame
        """
        self.df = df.copy()
        
        # Extract unique labels if not already set
        if len(self.args.labels_list) == 0:
            self.args.labels_list = self.df[self.args.label_column].unique().tolist()
            
        print(f"📊 Dataset loaded: {len(self.df)} samples, {len(self.args.labels_list)} unique labels")
        print(f"🏷️  Labels: {self.args.labels_list}")

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            return {}
            
        info = {
            'total_samples': len(self.df),
            'unique_labels': len(self.args.labels_list),
            'labels_list': self.args.labels_list,
            'split_column_range': (
                self.df[self.args.split_column].min(), 
                self.df[self.args.split_column].max()
            ),
            'label_distribution': self.df[self.args.label_column].value_counts().to_dict()
        }
        
        if hasattr(self, 'train_data') and self.train_data is not None:
            info['train_samples'] = len(self.train_data)
            
        if hasattr(self, 'test_data') and self.test_data is not None:
            info['test_samples'] = len(self.test_data)
            
        return info

    def _train_test_split(self, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset based on the split column.
        
        Args:
            ratio: Training ratio (0-100)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if not (0 < ratio < 100):
            raise ValueError("Ratio must be between 0 and 100")
            
        split_column = self.args.split_column
        max_id = self.df[split_column].max()
        split_threshold = np.ceil((max_id * ratio) / 100)
        
        train_data = self.df[self.df[split_column] < split_threshold].copy()
        test_data = self.df[self.df[split_column] >= split_threshold].copy()
        
        print(f"📊 Dataset split ({ratio}% train):")
        print(f"   🏋️  Training: {len(train_data)} samples")
        print(f"   🧪 Testing: {len(test_data)} samples")
        
        return train_data, test_data

    def set_model(self, model_id: str) -> None:
        """
        Initialize the NER model.
        
        Args:
            model_id: HuggingFace model identifier
        """
        self._model_id = model_id
        
        # Create simplified model name
        if self.sim_model_id != '':
            self._model_name = self.sim_model_id
        else:
            self._model_name = self._model_id.split('-')[0]
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"🚀 CUDA available: Using GPU acceleration")
        else:
            print(f"⚠️  CUDA not available: Using CPU (training will be slower)")
        
        # Initialize enhanced NER model
        self.model = NERUpgraded(
            model_type=self._model_name,
            model_name=self._model_id,
            labels=self.args.labels_list,
            args=self.args,
            use_cuda=cuda_available
        )
        
        print(f"✅ Model initialized: {self._model_id}")

    def train(
        self, 
        model_id: str, 
        ratio: float = 70, 
        output_dir: str = "outputs"
    ) -> Any:
        """
        Train the NER model.
        
        Args:
            model_id: HuggingFace model identifier
            ratio: Train/test split ratio (default: 70%)
            output_dir: Directory to save model outputs
            
        Returns:
            Training output from the model
        """
        print(f"🚀 Starting training with model: {model_id}")
        
        # Split dataset
        self.train_data, self.test_data = self._train_test_split(ratio)
        
        # Initialize model if needed
        if self._model_id != model_id or self.model is None:
            self.set_model(model_id)
        
        # Prepare evaluation data if enabled
        eval_data = None
        if self.args.evaluate_during_training:
            # Use a random subset of test data for evaluation
            split_column = self.args.split_column
            eval_start_id = random.randint(
                self.test_data[split_column].min(), 
                self.test_data[split_column].max()
            )
            eval_data = self.test_data[
                self.test_data[split_column] >= eval_start_id
            ].head(20)
            
            print(f"📊 Evaluation during training enabled with {len(eval_data)} samples")
        
        # Set output directory
        self.args.output_dir = output_dir
        
        try:
            # Train the model
            print(f"🏋️  Training started...")
            training_output = self.model.train_model(
                self.train_data, 
                eval_data=eval_data
            )
            
            if training_output is not None:
                self._is_trained = True
                print(f"✅ Training completed successfully!")
                
                # Print training statistics
                stats = self.model.get_training_stats()
                print(f"📈 Training Statistics:")
                print(f"   • Total batches: {stats['total_training_batches']}")
                print(f"   • Average loss: {stats['avg_training_loss']:.4f}")
                print(f"   • Final loss: {stats['final_training_loss']:.4f}")
                if stats['training_duration']:
                    print(f"   • Duration: {stats['training_duration']}")
            else:
                print(f"❌ Training failed!")
                
            return training_output
            
        except Exception as e:
            print(f"❌ Training failed with error: {str(e)}")
            raise

    def evaluate(self, split: int = 1, verbose: bool = True) -> Tuple[List, List]:
        """
        Evaluate the trained model.
        
        Args:
            split: Number of evaluation segments (for large datasets)
            verbose: Whether to print evaluation progress
            
        Returns:
            Tuple of (predictions, model_outputs)
        """
        if not self._is_trained:
            raise ValueError(
                "Model needs to be trained first. Use: model.train('model-name', ratio=70)"
            )
            
        if self.test_data is None:
            raise ValueError("Test data not available. Run training first.")
        
        print(f"🧪 Starting evaluation with {split} segment(s)...")
        
        all_predictions = []
        all_model_outputs = []
        
        prev_idx = 0
        segment_size = len(self.test_data) // split
        
        for i in range(1, split + 1):
            if i == split:  # Last segment gets remaining data
                next_idx = len(self.test_data)
            else:
                next_idx = prev_idx + segment_size
            
            if verbose:
                print(f"📊 Evaluating segment {i}/{split} (samples {prev_idx}-{next_idx})")
            
            segment_data = self.test_data.iloc[prev_idx:next_idx]
            
            try:
                result, model_outputs, predictions = self.model.eval_model(
                    segment_data, 
                    verbose=verbose
                )
                
                self.result = result
                all_predictions.extend(predictions)
                all_model_outputs.extend(model_outputs)
                
                if verbose:
                    print(f"   ✅ Segment {i} completed")
                    
            except Exception as e:
                print(f"   ❌ Segment {i} failed: {str(e)}")
                raise
            
            prev_idx = next_idx
        
        if self.result is not None:
            self._is_evaluated = True
            print(f"✅ Evaluation completed!")
            self._print_evaluation_summary()
        
        return all_predictions, all_model_outputs

    def _print_evaluation_summary(self) -> None:
        """Print evaluation results summary."""
        if self.result:
            print(f"📈 Evaluation Results:")
            for key, value in self.result.items():
                if isinstance(value, (int, float)):
                    if 'loss' in key.lower():
                        print(f"   • {key}: {value:.6f}")
                    else:
                        print(f"   • {key}: {value:.4f}")
                else:
                    print(f"   • {key}: {value}")

    def get_evaluation_result(self) -> Optional[Dict[str, Any]]:
        """
        Get evaluation results.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self._is_evaluated:
            raise ValueError("Model needs to be evaluated first. Use: model.evaluate()")
        
        return self.result

    def predict(
        self, 
        sentences: List[str], 
        split_on_space: bool = True
    ) -> Tuple[List, List]:
        """
        Make predictions on new sentences.
        
        Args:
            sentences: List of sentences to predict
            split_on_space: Whether to split sentences on spaces
            
        Returns:
            Tuple of (predictions, raw_outputs)
        """
        if not self._is_trained:
            raise ValueError("Model needs to be trained first.")
            
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        print(f"🔮 Making predictions on {len(sentences)} sentence(s)...")
        
        try:
            predictions, raw_outputs = self.model.predict(
                sentences, 
                split_on_space=split_on_space
            )
            print(f"✅ Predictions completed!")
            return predictions, raw_outputs
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            raise

    def prepare_for_visualization(
        self, 
        predictions: List, 
        model_outputs: List
    ) -> Tuple[List, List]:
        """
        Prepare prediction results for visualization.
        
        Args:
            predictions: Model predictions
            model_outputs: Model outputs/logits
            
        Returns:
            Tuple of (flattened_predictions, averaged_outputs)
        """
        # Flatten predictions
        flattened_preds = [tag for pred_out in predictions for tag in pred_out]
        
        # Average model outputs
        averaged_outputs = [
            np.mean(logits, axis=0) for output in model_outputs for logits in output
        ]
        
        print(f"📊 Prepared {len(flattened_preds)} predictions for visualization")
        
        return flattened_preds, averaged_outputs

    def save_model(self, output_dir: str = "saved_model") -> str:
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        if not self._is_trained or self.model is None:
            raise ValueError("No trained model to save.")
            
        try:
            self.model.save_model(output_dir)
            print(f"💾 Model saved to: {output_dir}")
            return output_dir
            
        except Exception as e:
            print(f"❌ Failed to save model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> None:
        """
        Load a previously saved model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            # Extract model info from path or use current settings
            if self.model is None:
                raise ValueError("Model configuration needed. Call set_model() first.")
            
            # Load the model
            self.model = NERUpgraded(
                model_type=self._model_name,
                model_name=model_path,
                labels=self.args.labels_list,
                args=self.args,
                use_cuda=torch.cuda.is_available()
            )
            
            self._is_trained = True
            print(f"📂 Model loaded from: {model_path}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise

    def get_training_history(self) -> Dict[str, List]:
        """
        Get training history for plotting.
        
        Returns:
            Dictionary with training metrics history
        """
        if self.model is None:
            return {}
        
        history = {
            'train_loss': self.model.train_loss_list,
            'eval_results': self.model.eval_loss_list,
            'test_results': self.model.test_loss_list
        }
        
        # Add training statistics
        if hasattr(self.model, 'get_training_stats'):
            history['stats'] = self.model.get_training_stats()
        
        return history

    def reset_training_state(self) -> None:
        """Reset training state for fresh training."""
        self._is_trained = False
        self._is_evaluated = False
        self.result = None
        self.train_data = None
        self.test_data = None
        
        if self.model is not None:
            self.model.reset_loss_tracking()
        
        print("🔄 Training state reset")

    def __repr__(self) -> str:
        """String representation of the trainer."""
        status = []
        if hasattr(self, 'df') and self.df is not None:
            status.append(f"dataset={len(self.df)} samples")
        if self._model_id:
            status.append(f"model={self._model_id}")
        if self._is_trained:
            status.append("trained=True")
        if self._is_evaluated:
            status.append("evaluated=True")
        
        return f"ModelTrainer({', '.join(status)})"
