import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def init_wandb(project_name="career-diagnostic-nlp", run_name="gap-model-training-v1"):
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 32
        }
    )
    print(f"WandB Tracking Started: {run_name}. Siap pamer metrik ke tim!")

class ElitePerformanceTracker(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, threshold=0.5):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        y_pred_probs = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_probs > self.threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_val, y_pred, average='macro', zero_division=0
        )

        wandb.log({
            "val_macro_precision": precision,
            "val_macro_recall": recall,
            "val_macro_f1": f1
        })

        print(f"\n[Epoch {epoch+1} Selesai] F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
