"""
Enhanced NER Model with Training Loss Tracking
Extends simpletransformers NERModel with additional training metrics.
"""

import os
import math
import logging
from datetime import datetime
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm, trange

from simpletransformers.ner import NERModel, NERArgs
from simpletransformers.ner.ner_utils import flatten_results
from torch.optim import AdamW
from transformers.optimization import (
    Adafactor,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class NERUpgraded(NERModel):
    """
    Enhanced NER Model with training metrics tracking.
    
    Extends the base NERModel to include:
    - Training loss tracking per batch
    - Evaluation loss tracking
    - Test loss tracking
    - Enhanced training visualization support
    """
    
    def __init__(
        self,
        model_type: str,
        model_name: str,
        labels: Optional[List[str]] = None,
        weight: Optional[torch.Tensor] = None,
        args: Optional[NERArgs] = None,
        use_cuda: bool = True,
        cuda_device: int = -1,
        onnx_execution_provider: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the enhanced NER model.
        
        Args:
            model_type: Type of the model (e.g., 'bert', 'roberta')
            model_name: Name/path of the pretrained model
            labels: List of NER labels
            weight: Class weights for loss computation
            args: NER training arguments
            use_cuda: Whether to use CUDA
            cuda_device: CUDA device ID
            onnx_execution_provider: ONNX execution provider
            **kwargs: Additional arguments
        """
        super().__init__(
            model_type=model_type,
            model_name=model_name,
            labels=labels,
            weight=weight,
            args=args,
            use_cuda=use_cuda,
            cuda_device=cuda_device,
            onnx_execution_provider=onnx_execution_provider,
            **kwargs
        )

        # Initialize loss tracking lists
        self.train_loss_list: List[float] = []
        self.eval_loss_list: List[Dict[str, Any]] = []
        self.test_loss_list: List[Dict[str, Any]] = []
        
        # Training metadata
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None

    def reset_loss_tracking(self) -> None:
        """Reset all loss tracking lists."""
        self.train_loss_list.clear()
        self.eval_loss_list.clear()
        self.test_loss_list.clear()
        self.training_start_time = None
        self.training_end_time = None

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary containing training metrics and statistics.
        """
        stats = {
            'total_training_batches': len(self.train_loss_list),
            'total_eval_steps': len(self.eval_loss_list),
            'total_test_steps': len(self.test_loss_list),
            'avg_training_loss': np.mean(self.train_loss_list) if self.train_loss_list else None,
            'min_training_loss': np.min(self.train_loss_list) if self.train_loss_list else None,
            'max_training_loss': np.max(self.train_loss_list) if self.train_loss_list else None,
            'final_training_loss': self.train_loss_list[-1] if self.train_loss_list else None,
            'training_duration': None
        }
        
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time
            stats['training_duration'] = str(duration)
            stats['training_duration_seconds'] = duration.total_seconds()
        
        return stats

    def save_loss_history(self, output_dir: str, filename: str = "loss_history.csv") -> str:
        """
        Save loss history to CSV file.
        
        Args:
            output_dir: Output directory path
            filename: Name of the CSV file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Create DataFrame with loss history
        max_len = max(
            len(self.train_loss_list),
            len(self.eval_loss_list),
            len(self.test_loss_list)
        )
        
        data = {
            'batch_idx': list(range(max_len)),
            'train_loss': self.train_loss_list + [None] * (max_len - len(self.train_loss_list)),
            'eval_loss': [None] * max_len,
            'test_loss': [None] * max_len
        }
        
        # Fill eval and test losses at appropriate indices
        for i, eval_result in enumerate(self.eval_loss_list):
            if i < max_len:
                data['eval_loss'][i] = eval_result.get('eval_loss', None)
        
        for i, test_result in enumerate(self.test_loss_list):
            if i < max_len:
                data['test_loss'][i] = test_result.get('eval_loss', None)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath

    def train(
        self,
        train_dataset,
        output_dir: str,
        show_running_loss: bool = True,
        eval_data=None,
        test_data=None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Enhanced training method with loss tracking.
        
        This method extends the base training to include comprehensive loss tracking
        and improved logging capabilities.
        """
        self.training_start_time = datetime.now()
        
        try:
            result = self._train_with_tracking(
                train_dataset=train_dataset,
                output_dir=output_dir,
                show_running_loss=show_running_loss,
                eval_data=eval_data,
                test_data=test_data,
                verbose=verbose,
                **kwargs
            )
            
            self.training_end_time = datetime.now()
            
            # Save training statistics
            stats = self.get_training_stats()
            stats_path = os.path.join(output_dir, "training_stats.json")
            
            import json
            with open(stats_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_stats = {}
                for k, v in stats.items():
                    if isinstance(v, np.floating):
                        json_stats[k] = float(v)
                    elif isinstance(v, np.integer):
                        json_stats[k] = int(v)
                    else:
                        json_stats[k] = v
                json.dump(json_stats, f, indent=2)
            
            # Save loss history
            self.save_loss_history(output_dir)
            
            if verbose:
                print(f"✅ Training completed. Stats saved to: {stats_path}")
                print(f"📊 Loss history saved to: {os.path.join(output_dir, 'loss_history.csv')}")
            
            return result
            
        except Exception as e:
            self.training_end_time = datetime.now()
            logger.error(f"Training failed: {str(e)}")
            raise

    def _train_with_tracking(
        self,
        train_dataset,
        output_dir: str,
        show_running_loss: bool = True,
        eval_data=None,
        test_data=None,
        verbose: bool = True,
        **kwargs
    ):
        """
        Internal training method with enhanced loss tracking.
        
        This is a modified version of the parent class training method
        that includes detailed loss tracking and logging.
        """
        model = self.model
        args = self.args

        # Initialize tensorboard writer
        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        # Setup optimizer
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer, t_total)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Training variables
        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        
        train_iterator = trange(
            int(args.num_train_epochs), 
            desc="Epoch", 
            disable=args.silent, 
            mininterval=0
        )
        
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
            
        if args.wandb_project and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        # Mixed precision setup
        if self.args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        # Main training loop
        for _ in train_iterator:
            model.train()
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            
            for step, batch in enumerate(batch_iterator):
                inputs = self._get_inputs_dict(batch)

                # Forward pass with optional mixed precision
                if self.args.fp16:
                    with amp.autocast():
                        loss, *_ = self._calculate_loss(
                            model, inputs, 
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                else:
                    loss, *_ = self._calculate_loss(
                        model, inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

                if args.n_gpu > 1:
                    loss = loss.mean()

                current_loss = loss.item()
                
                # Track training loss
                self.train_loss_list.append(current_loss)

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epoch {epoch_number}/{args.num_train_epochs}. "
                        f"Running Loss: {current_loss:9.4f}"
                    )

                # Backward pass
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                
                # Optimization step
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    # Logging
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                        )
                        logging_loss = tr_loss
                        
                        if (args.wandb_project or self.is_sweeping) and WANDB_AVAILABLE:
                            wandb.log({
                                "Training loss": current_loss,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            })

                    # Evaluation during training
                    if (args.evaluate_during_training and 
                        args.evaluate_during_training_steps > 0 and
                        global_step % args.evaluate_during_training_steps == 0):
                        
                        self._handle_evaluation_step(
                            eval_data, test_data, output_dir, global_step,
                            training_progress_scores, current_loss, verbose, **kwargs
                        )

            epoch_number += 1

        return global_step, tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores

    def _setup_optimizer(self, model) -> torch.optim.Optimizer:
        """Setup optimizer with proper parameter grouping."""
        args = self.args
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        
        # Handle custom parameter groups
        for group in args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        # Handle custom layer parameters
        for group in args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            
            params_d = []
            params_nd = []
            
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            
            group_d["params"] = params_d
            group_nd["params"] = params_nd
            
            optimizer_grouped_parameters.extend([group_d, group_nd])

        # Add remaining parameters
        if not args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n not in custom_parameter_names
                        and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n not in custom_parameter_names
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ])

        # Create optimizer
        if args.optimizer == "AdamW":
            return AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=args.adam_betas,
            )
        elif args.optimizer == "Adafactor":
            return Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    def _setup_scheduler(self, optimizer, t_total):
        """Setup learning rate scheduler."""
        args = self.args
        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        scheduler_map = {
            "constant_schedule": lambda: get_constant_schedule(optimizer),
            "constant_schedule_with_warmup": lambda: get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            ),
            "linear_schedule_with_warmup": lambda: get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            ),
            "cosine_schedule_with_warmup": lambda: get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles
            ),
            "cosine_with_hard_restarts_schedule_with_warmup": lambda: get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles
            ),
            "polynomial_decay_schedule_with_warmup": lambda: get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end, 
                power=args.polynomial_decay_schedule_power
            ),
        }

        if args.scheduler in scheduler_map:
            return scheduler_map[args.scheduler]()
        else:
            raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    def _handle_evaluation_step(
        self, eval_data, test_data, output_dir, global_step,
        training_progress_scores, current_loss, verbose, **kwargs
    ):
        """Handle evaluation during training step."""
        if eval_data is None:
            return

        output_dir_current = os.path.join(output_dir, f"checkpoint-{global_step}")
        os.makedirs(output_dir_current, exist_ok=True)

        # Evaluate on validation data
        results, _, _ = self.eval_model(
            eval_data,
            verbose=verbose and self.args.evaluate_during_training_verbose,
            wandb_log=False,
            output_dir=output_dir_current,
            **kwargs,
        )

        # Track evaluation results
        self.eval_loss_list.append(results)

        if training_progress_scores is not None:
            training_progress_scores["global_step"].append(global_step)
            training_progress_scores["train_loss"].append(current_loss)
            for key in results:
                training_progress_scores[key].append(results[key])

            # Test evaluation if provided
            if test_data is not None:
                test_results, _, _ = self.eval_model(
                    test_data,
                    verbose=verbose and self.args.evaluate_during_training_verbose,
                    silent=self.args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )
                
                # Track test results
                self.test_loss_list.append(test_results)
                
                for key in test_results:
                    training_progress_scores["test_" + key].append(test_results[key])

            # Save progress report
            if training_progress_scores:
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(self.args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

    def predict(self, to_predict: List[str], split_on_space: bool = True) -> tuple:
        """
        Enhanced prediction method with better error handling.
        
        Args:
            to_predict: List of sentences to predict
            split_on_space: Whether to split sentences on spaces
            
        Returns:
            Tuple of (predictions, raw_outputs)
        """
        try:
            predictions, raw_outputs = super().predict(to_predict, split_on_space)
            return predictions, raw_outputs
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        """
        Enhanced evaluation with better metrics tracking.
        
        Args:
            eval_data: Evaluation dataset
            output_dir: Directory to save evaluation results
            verbose: Whether to print verbose output
            silent: Whether to suppress output
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (results, model_outputs, predictions)
        """
        try:
            results, model_outputs, predictions = super().eval_model(
                eval_data, output_dir=output_dir, verbose=verbose, silent=silent, **kwargs
            )
            
            if verbose and not silent:
                print(f"📊 Evaluation completed. Results: {results}")
            
            return results, model_outputs, predictions
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise