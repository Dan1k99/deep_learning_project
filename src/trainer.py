import torch
import time

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    # DEBUG: Track what the model is actually predicting
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect first batch stats for debugging
            if len(all_preds) == 0:
                print(f"DEBUG - True Labels: {labels[:10].tolist()}")
                print(f"DEBUG - Predicted:   {predicted[:10].tolist()}")

    return 100 * correct / total

def train_baseline(model, train_loader, epochs, device, lr=0.001):
    """
    Standard training loop for Task A (Phase 2) [cite: 23-26].
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting Baseline Training (Task A)...")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs} completed.")
    
    return model

def train_constrained(model, train_loader, epochs, device, projector, lr=0.001):
    """
    Modified training loop for Task B with Gradient Intervention (Phase 4).
    [cite: 59, 67-76]
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    # Note: Optimization on Task B often requires lower LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    print(f"Starting Constrained Training (Method: {type(projector).__name__})...")
    
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Computes param.grad [cite: 71]

            # --- INTERVENTION START [cite: 72] ---
            # Loop through layers and project gradients BEFORE optimizer step
            for name, param in model.named_parameters():
                if param.grad is not None: # Apply to all relevant layers (Conv/Linear)
                     # Apply the cleaning function specific to the projector
                     projected_grad = projector.project_gradient(name, param.grad)
                     if projected_grad is not None:
                         param.grad = projected_grad
            # --- INTERVENTION END ---

            optimizer.step() # Update with cleaned gradients [cite: 76]
            
    return model