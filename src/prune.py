import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from big_model import BigMNISTModel

def apply_global_pruning(model, amount=0.3):
    """
    Globally prunes 30% of weights in Conv2d and Linear layers by default.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    return model

def measure_sparsity(model):
    """
    Returns the percentage of zero weights in the model.
    """
    num_zeros = 0
    num_elements = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            num_zeros += torch.sum(module.weight == 0).item()
            num_elements += module.weight.nelement()
    return 100. * num_zeros / num_elements

if __name__ == "__main__":
    # Load pre-trained big model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BigMNISTModel().to(device)
    model.load_state_dict(torch.load("big_model.pth", map_location=device))

    # Prune
    model = apply_global_pruning(model, amount=0.5)
    print(f"Model Sparsity: {measure_sparsity(model):.2f}%")
    
    # Save pruned model
    torch.save(model.state_dict(), "big_model_pruned.pth")