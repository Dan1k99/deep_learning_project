import torch
import time

def evaluate(model, loader, device, task_classes=None):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Applies a task-specific output mask when evaluating on a subset of classes;
            # logits for classes outside the current task are suppressed to negative infinity
            # so they are never selected as the predicted class.
            if task_classes is not None:
                mask = torch.full_like(outputs, float('-inf'))
                mask[:, task_classes] = 0
                outputs = outputs + mask

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


#def train_baseline(model, train_loader, epochs, device, lr=0.001):
#    """
#    Standard training loop for Task A (Phase 2).
#    """
#    model.to(device)
#    model.train()
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#    print("Starting Baseline Training (Task A)...")
#    for epoch in range(epochs):
#        for inputs, labels in train_loader:
#            inputs, labels = inputs.to(device), labels.to(device)
#
#            optimizer.zero_grad()
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#            loss.backward()
#            optimizer.step()
#
#        print(f"Epoch {epoch+1}/{epochs} completed.")
#
#    return model

def train_baseline(model, train_loader, epochs, device, lr=0.001):
    """
    Optimized training loop for Task A (Expert) using Adam.
    Incorporates weight decay and a cosine annealing scheduler to mitigate
    overfitting and improve final generalization performance.
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Adam optimizer with L2 weight decay (1e-4) selected for Task A;
    # weight decay acts as a regularizer to counteract overfitting on the training split.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing scheduler smoothly reduces the learning rate over training,
    # allowing the optimizer to settle into a sharper minimum in later epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("Starting Baseline Training (Task A) - Optimized Adam...")

    history = {'loss': [], 'accuracy': [], 'lr': []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accumulates per-batch correct predictions for epoch-level accuracy reporting
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Advances the cosine annealing schedule by one epoch
        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f}")

    return model, history

#def train_constrained(model, train_loader, epochs, device, projector, lr=0.001):
#    """
#    Modified training loop for Task B with Gradient Intervention (Phase 4).
#    """
#    model.to(device)
#    model.train()
#    criterion = torch.nn.CrossEntropyLoss()
#    # A lower learning rate is applied for Task B to reduce the risk of catastrophic forgetting
#    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#    print(f"Starting Constrained Training (Method: {type(projector).__name__})...")
#
#    for epoch in range(epochs):
#        for inputs, labels in train_loader:
#            inputs, labels = inputs.to(device), labels.to(device)
#
#            optimizer.zero_grad()
#            outputs = model(inputs)
#            loss = criterion(outputs, labels)
#            loss.backward()  # Populates param.grad for all parameters
#
#            # --- GRADIENT INTERVENTION ---
#            # Iterates through all model parameters and applies the subspace projection
#            # constraint to each gradient before the optimizer step, ensuring that
#            # weight updates remain orthogonal to the Task A subspace.
#            for name, param in model.named_parameters():
#                if param.grad is not None:  # Projection is applied to all Conv/Linear layers
#                     # Delegates gradient cleaning to the projector's specific implementation
#                     projected_grad = projector.project_gradient(name, param.grad)
#                     if projected_grad is not None:
#                         param.grad = projected_grad
#            # --- END GRADIENT INTERVENTION ---
#
#            optimizer.step()  # Updates weights using the projected, subspace-constrained gradients
#
#        print(f"Epoch {epoch+1}/{epochs} completed.")
#
#    return model

def train_constrained(model, train_loader, epochs, device, projector, lr=0.001):
    """
    Modified training loop for Task B with Gradient Intervention (Phase 4).
    Employs Adam with mild weight decay to maintain regularization consistency
    with the Task A training regime while applying subspace gradient projection.
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    # Adam with weight decay (1e-4) mirrors the Task A optimizer configuration,
    # ensuring consistent regularization strength across both training phases.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting Constrained Training (Method: {type(projector).__name__})...")

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Populates param.grad for all parameters

            # --- GRADIENT INTERVENTION ---
            # Iterates through all model parameters and applies the subspace projection
            # constraint to each gradient before the optimizer step, ensuring that
            # weight updates remain orthogonal to the Task A subspace.
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Delegates gradient cleaning to the projector's specific implementation
                    projected_grad = projector.project_gradient(name, param.grad)
                    if projected_grad is not None:
                        param.grad = projected_grad
            # --- END GRADIENT INTERVENTION ---

            optimizer.step()  # Updates weights using the projected, subspace-constrained gradients

            running_loss += loss.item()

            # Accumulates per-batch correct predictions for epoch-level accuracy reporting
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    return model, history