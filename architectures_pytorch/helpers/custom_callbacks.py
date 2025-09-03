import numpy as np


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from architectures_pytorch.helpers.constants import hyperparameters
from architectures_pytorch.helpers.constants import selected_model
from architectures_pytorch.helpers.constants import threshold


hyperparameters = hyperparameters[selected_model]
# Runs at the end of every epoch and prints the confusion matrix


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, epoch_counter, time, y_test) -> None:
        super().__init__()
        self.test_dataset = test_dataset
        self.epoch_counter = epoch_counter
        self.time = time
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.test_dataset)
        classes = np.argmax(predictions, axis=1)
        print(f"{confusion_matrix(self.y_test, classes)}")
        print(
            f"f1 score: {f1_score(self.y_test, classes, average='weighted')}")
        self.model.evaluate(self.test_dataset)
        print(f"\n\n")
        # Only save model on the last epoch
        if epoch == hyperparameters["num_epochs"] - 1:  # -1 because epochs are 0-indexed
            # Create base directory if it doesn't exist
            import os
            base_dir = f"saved_models/{selected_model}/{threshold}"
            os.makedirs(base_dir, exist_ok=True)
            
            export_path_keras = ""
            if selected_model == "vision_transformer":
                export_path_keras = f"{base_dir}/{int(self.time)}-tl{hyperparameters['transformer_layers']}-pd{hyperparameters['projection_dim']}-p{hyperparameters['patch_size']}.h5"
                self.model.inner_model.save_weights(export_path_keras)
            elif selected_model == "vit":
                export_path_keras = f"{base_dir}/{int(self.time)}-tl{hyperparameters['transformer_layers']}-pd{hyperparameters['projection_dim']}-p{hyperparameters['patch_size']}.h5"
        #self.epoch_counter += 1
