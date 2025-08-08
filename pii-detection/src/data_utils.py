"""
Data Loading and Preprocessing Utilities for PII Detection
Handles CSV loading, data validation, and preprocessing.
"""

import os
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utility class for loading and preprocessing PII detection datasets.
    """
    
    def __init__(self, word_column: str = 'words', label_column: str = 'labels'):
        """
        Initialize the data loader.
        
        Args:
            word_column: Name of the column containing words/tokens
            label_column: Name of the column containing NER labels
        """
        self.word_column = word_column
        self.label_column = label_column
        
    def load_csv(
        self, 
        file_path: str, 
        encoding: str = 'utf-8',
        sort_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            encoding: File encoding (default: utf-8)
            sort_columns: Columns to sort by (e.g., ['doc_id', 'sentence_id'])
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            print(f"📂 Loading dataset from: {file_path}")
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Sort if specified
            if sort_columns:
                available_columns = [col for col in sort_columns if col in df.columns]
                if available_columns:
                    df = df.sort_values(by=available_columns, ascending=True)
                    print(f"📊 Data sorted by: {available_columns}")
            
            print(f"✅ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {str(e)}")
            raise

    def validate_dataset(
        self, 
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset format and content.
        
        Args:
            df: Dataset DataFrame
            required_columns: List of required column names
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        if required_columns is None:
            required_columns = [self.word_column, self.label_column]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Data type validation
        if self.word_column in df.columns:
            non_string_words = df[self.word_column].apply(
                lambda x: not isinstance(x, str) and pd.notna(x)
            ).sum()
            if non_string_words > 0:
                validation_results['warnings'].append(
                    f"{non_string_words} non-string values in {self.word_column}"
                )
        
        if self.label_column in df.columns:
            non_string_labels = df[self.label_column].apply(
                lambda x: not isinstance(x, str) and pd.notna(x)
            ).sum()
            if non_string_labels > 0:
                validation_results['warnings'].append(
                    f"{non_string_labels} non-string values in {self.label_column}"
                )
        
        # Calculate statistics
        validation_results['stats'] = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_labels': df[self.label_column].nunique() if self.label_column in df.columns else 0,
            'label_distribution': df[self.label_column].value_counts().to_dict() if self.label_column in df.columns else {}
        }
        
        return validation_results

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset for NER training.
        
        Args:
            df: Raw dataset DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("🔄 Preprocessing dataset...")
        df_processed = df.copy()
        
        # Convert columns to string type
        if self.word_column in df_processed.columns:
            df_processed[self.word_column] = df_processed[self.word_column].astype('str')
            print(f"   ✅ {self.word_column} converted to string type")
        
        if self.label_column in df_processed.columns:
            df_processed[self.label_column] = df_processed[self.label_column].astype('str')
            print(f"   ✅ {self.label_column} converted to string type")
        
        # Remove rows with NaN values in essential columns
        essential_columns = [self.word_column, self.label_column]
        before_count = len(df_processed)
        
        for col in essential_columns:
            if col in df_processed.columns:
                df_processed = df_processed.dropna(subset=[col])
        
        after_count = len(df_processed)
        if before_count != after_count:
            print(f"   🧹 Removed {before_count - after_count} rows with NaN values")
        
        print(f"✅ Preprocessing completed: {len(df_processed)} samples ready")
        return df_processed

    def get_label_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed label statistics.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dictionary with label statistics
        """
        if self.label_column not in df.columns:
            return {}
        
        label_counts = df[self.label_column].value_counts()
        total_samples = len(df)
        
        stats = {
            'total_samples': total_samples,
            'unique_labels': len(label_counts),
            'label_counts': label_counts.to_dict(),
            'label_percentages': (label_counts / total_samples * 100).round(2).to_dict(),
            'most_common_label': label_counts.index[0],
            'least_common_label': label_counts.index[-1],
            'label_imbalance_ratio': label_counts.iloc[0] / label_counts.iloc[-1] if len(label_counts) > 1 else 1.0
        }
        
        # Identify entity types (B-, I- prefixes)
        entity_types = set()
        for label in label_counts.index:
            if label.startswith(('B-', 'I-')):
                entity_type = label[2:]  # Remove B- or I- prefix
                entity_types.add(entity_type)
        
        stats['entity_types'] = list(entity_types)
        stats['num_entity_types'] = len(entity_types)
        
        return stats

    def split_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: Optional[str] = None,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df: Dataset DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_column: Column to stratify split by
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, rtol=1e-9):
            raise ValueError("Split ratios must sum to 1.0")
        
        if train_ratio <= 0 or val_ratio < 0 or test_ratio <= 0:
            raise ValueError("Split ratios must be positive (val_ratio can be 0)")
        
        print(f"📊 Splitting dataset: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        
        # Prepare stratification
        stratify_data = None
        if stratify_column and stratify_column in df.columns:
            stratify_data = df[stratify_column]
            print(f"   🎯 Stratifying by: {stratify_column}")
        
        # First split: separate test set
        if val_ratio == 0:
            # Only train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                stratify=stratify_data,
                random_state=random_state + 1
            )
        
        print(f"✅ Dataset split completed:")
        print(f"   🏋️  Training: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        if not val_df.empty:
            print(f"   🔍 Validation: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
        print(f"   🧪 Test: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
        
        return train_df, val_df, test_df

    def create_document_based_split(
        self,
        df: pd.DataFrame,
        doc_id_column: str = 'doc_id',
        train_ratio: float = 0.7,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset based on document IDs to avoid data leakage.
        
        Args:
            df: Dataset DataFrame
            doc_id_column: Column containing document IDs
            train_ratio: Training set ratio
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if doc_id_column not in df.columns:
            raise ValueError(f"Document ID column '{doc_id_column}' not found")
        
        print(f"📄 Creating document-based split ({train_ratio:.1%} train)")
        
        # Get unique document IDs
        unique_docs = df[doc_id_column].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_docs)
        
        # Split document IDs
        train_doc_count = int(len(unique_docs) * train_ratio)
        train_docs = set(unique_docs[:train_doc_count])
        test_docs = set(unique_docs[train_doc_count:])
        
        # Create splits based on document membership
        train_df = df[df[doc_id_column].isin(train_docs)].copy()
        test_df = df[df[doc_id_column].isin(test_docs)].copy()
        
        print(f"✅ Document-based split completed:")
        print(f"   📄 Training docs: {len(train_docs)}, samples: {len(train_df)}")
        print(f"   📄 Test docs: {len(test_docs)}, samples: {len(test_df)}")
        
        return train_df, test_df

    def balance_dataset(
        self,
        df: pd.DataFrame,
        method: str = 'undersample',
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Balance dataset to handle class imbalance.
        
        Args:
            df: Dataset DataFrame
            method: Balancing method ('undersample', 'oversample')
            target_column: Column to balance (default: label_column)
            
        Returns:
            Balanced DataFrame
        """
        if target_column is None:
            target_column = self.label_column
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        print(f"⚖️  Balancing dataset using {method} method...")
        
        # Get class counts
        class_counts = df[target_column].value_counts()
        print(f"   Original distribution: {dict(class_counts)}")
        
        if method == 'undersample':
            # Undersample to smallest class
            min_count = class_counts.min()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_df = df[df[target_column] == class_label]
                sampled_df = class_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(sampled_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        elif method == 'oversample':
            # Oversample to largest class
            max_count = class_counts.max()
            balanced_dfs = []
            
            for class_label in class_counts.index:
                class_df = df[df[target_column] == class_label]
                current_count = len(class_df)
                
                if current_count < max_count:
                    # Oversample with replacement
                    additional_samples = max_count - current_count
                    oversampled = class_df.sample(n=additional_samples, replace=True, random_state=42)
                    combined_df = pd.concat([class_df, oversampled], ignore_index=True)
                else:
                    combined_df = class_df
                
                balanced_dfs.append(combined_df)
            
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Report results
        new_class_counts = balanced_df[target_column].value_counts()
        print(f"   Balanced distribution: {dict(new_class_counts)}")
        print(f"   Dataset size: {len(df)} → {len(balanced_df)} samples")
        
        return balanced_df

    def export_dataset(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'csv',
        **kwargs
    ) -> str:
        """
        Export dataset to file.
        
        Args:
            df: DataFrame to export
            output_path: Output file path
            format: Export format ('csv', 'json', 'parquet')
            **kwargs: Additional arguments for export function
            
        Returns:
            Path to exported file
        """
        print(f"💾 Exporting dataset to: {output_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False, **kwargs)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            print(f"✅ Dataset exported successfully: {len(df)} samples")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {str(e)}")
            raise

    def create_sample_dataset(
        self,
        df: pd.DataFrame,
        sample_size: int = 1000,
        stratify: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a smaller sample of the dataset for testing.
        
        Args:
            df: Source DataFrame
            sample_size: Number of samples to extract
            stratify: Whether to maintain class distribution
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if sample_size >= len(df):
            print(f"⚠️  Sample size ({sample_size}) >= dataset size ({len(df)}), returning full dataset")
            return df.copy()
        
        print(f"🎲 Creating sample dataset: {sample_size} samples")
        
        if stratify and self.label_column in df.columns:
            # Stratified sampling
            sample_df = df.groupby(self.label_column, group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), max(1, int(sample_size * len(x) / len(df)))),
                    random_state=random_state
                )
            ).reset_index(drop=True)
            
            # If we didn't get enough samples, fill with random sampling
            if len(sample_df) < sample_size:
                remaining_size = sample_size - len(sample_df)
                remaining_indices = df.index.difference(sample_df.index)
                additional_samples = df.loc[remaining_indices].sample(
                    n=min(remaining_size, len(remaining_indices)),
                    random_state=random_state
                )
                sample_df = pd.concat([sample_df, additional_samples], ignore_index=True)
        else:
            # Random sampling
            sample_df = df.sample(n=sample_size, random_state=random_state)
        
        print(f"✅ Sample created: {len(sample_df)} samples")
        return sample_df

    def print_dataset_summary(self, df: pd.DataFrame, title: str = "Dataset Summary") -> None:
        """
        Print a comprehensive dataset summary.
        
        Args:
            df: Dataset DataFrame
            title: Summary title
        """
        print(f"\n📊 {title}")
        print("=" * (len(title) + 4))
        
        # Basic info
        print(f"Shape: {df.shape[0]} samples × {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\n❌ Missing values:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"   {col}: {count} ({count/len(df)*100:.1f}%)")
        else:
            print(f"\n✅ No missing values")
        
        # Label statistics
        if self.label_column in df.columns:
            label_stats = self.get_label_statistics(df)
            print(f"\n🏷️  Label Statistics:")
            print(f"   Unique labels: {label_stats['unique_labels']}")
            print(f"   Entity types: {label_stats['num_entity_types']} ({label_stats['entity_types']})")
            print(f"   Most common: {label_stats['most_common_label']} ({label_stats['label_percentages'][label_stats['most_common_label']]:.1f}%)")
            print(f"   Least common: {label_stats['least_common_label']} ({label_stats['label_percentages'][label_stats['least_common_label']]:.1f}%)")
            print(f"   Imbalance ratio: {label_stats['label_imbalance_ratio']:.1f}:1")
        
        print()


# Convenience functions for common operations
def load_wikipii_dataset(file_path: str, preprocess: bool = True) -> pd.DataFrame:
    """
    Load and preprocess WikiPII dataset.
    
    Args:
        file_path: Path to the CSV file
        preprocess: Whether to apply preprocessing
        
    Returns:
        Loaded and optionally preprocessed DataFrame
    """
    loader = DataLoader()
    df = loader.load_csv(file_path, sort_columns=['doc_id', 'sentence_id'])
    
    if preprocess:
        df = loader.preprocess_dataset(df)
    
    return df


def quick_dataset_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform quick analysis of a NER dataset.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        Dictionary with analysis results
    """
    loader = DataLoader()
    
    # Validation
    validation = loader.validate_dataset(df)
    
    # Label statistics
    label_stats = loader.get_label_statistics(df)
    
    # Summary
    analysis = {
        'validation': validation,
        'label_statistics': label_stats,
        'recommendations': []
    }
    
    # Generate recommendations
    if validation['is_valid']:
        analysis['recommendations'].append("✅ Dataset format is valid")
    else:
        analysis['recommendations'].append(f"❌ Dataset issues: {validation['errors']}")
    
    if label_stats.get('label_imbalance_ratio', 1) > 10:
        analysis['recommendations'].append("⚠️  High class imbalance detected - consider balancing")
    
    if label_stats.get('num_entity_types', 0) < 2:
        analysis['recommendations'].append("⚠️  Low entity type diversity")
    
    return analysisstate
            )
            val_df = pd.DataFrame()  # Empty validation set
        else:
            # Train-temp split (temp will be further split into val and test)
            temp_ratio = val_ratio + test_ratio
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                stratify=stratify_data,
                random_state=random_state
            )
            
            # Split temp into validation and test
            val_size_in_temp = val_ratio / temp_ratio
            temp_stratify = temp_df[stratify_column] if stratify_data is not None else None
            
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size_in_temp),
                stratify=temp_stratify,
                random_state=random_