### LIBRARIES ###
# Global libraries
import torch

### FUNCTION DEFINITIONS ###
def todevice(tensor, device):
    """Sends the given tensor to the given device.

    Args:
        tensor: torch.Tensor (or list/tuple of tensors)
            tensor (or list of tensors) to send
        device: str
            device to which the tensor should be sent
    Returns:
        tensor: torch.Tensor (or list/tuple of tensors)
            tensor (or list of tensors) sent to the device
    """
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)


def metrics(preds, targets):
    """Computes metrics on the correct predictions.

    Args:
        preds: torch.Tensor (in {0, 1})
            predictions
        targets: torch.Tensor (in {0, 1})
            ground-truth values
    Returns:
        accuracy: float
            accuracy of the predictions
        precision: float
            precision of the predictions
        recall: float
            recall of the predictions
        f1_score: float
            F1 score of the predictions
    """
    # target_true = torch.sum(targets == 0).float()
    # correct_true = torch.sum(preds == )

    # accuracy = torch.sum(preds == targets) / len(preds)
    # precision = correct_true / target_true
    # recall = correct_true / target_true
    # f1_score = 2 * precision * recall / (precision + recall)
    # return accuracy, precision, recall, f1_score
    pass