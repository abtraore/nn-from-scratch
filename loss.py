import numpy as np


class Loss:

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


class CCE(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # Avoid division by 0 and mean dragging towards any direction.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Not one-hot encoded.
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # One-hot encoded.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        # Return negative log-likelihoods.
        return -np.log(correct_confidences)
