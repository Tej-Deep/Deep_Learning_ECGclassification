import torch

def define_optimizer(model, lr, alpha):
    # Define optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=alpha)
    optimizer.zero_grad()
    return optimizer

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def predict(model, inputs, notes, device):
    outputs = model.forward(inputs, notes)
    predicted = torch.sigmoid(outputs)
    predicted = (predicted>0.5).float() 
    return outputs, predicted

def display_train(epoch, num_epochs, i, model, correct, total, loss, train_loader, valid_loader, device):
    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}')
    train_accuracy = correct/total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}')
    valid_accuracy = eval_valid(model, valid_loader, epoch, num_epochs, device)
    return train_accuracy, valid_accuracy

def eval_valid(model, valid_loader, epoch, num_epochs, device):
    # Compute model train accuracy on test after all samples have been seen using test samples
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels, notes in valid_loader:
            # Get images and labels from test loader
            inputs = inputs.transpose(1,2).float().to(device)
            labels = labels.float().to(device)
            notes = notes.to(device)

            # Forward pass and predict class using max
            # outputs = model(inputs)
            _, predicted = predict(model, inputs, notes, device) #torch.max(outputs.data, 1)

            # Check if predicted class matches label and count numbler of correct predictions
            total += labels.size(0)
            correct += torch.nn.functional.cosine_similarity(labels,predicted).sum().item() # (predicted == labels).sum().item()

    # Compute final accuracy and display
    valid_accuracy = correct/total
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {valid_accuracy:.4f}')
    return valid_accuracy


def eval_test(model, test_loader, device):
    # Compute model test accuracy on test after training
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels, notes in test_loader:
            # Get images and labels from test loader
            inputs = inputs.transpose(1,2).float().to(device)
            labels = labels.float().to(device)
            notes = notes.to(device)

            # Forward pass and predict class using max
            # outputs = model(inputs)
            _, predicted = predict(model, inputs, notes, device)#torch.max(outputs.data, 1)

            # Check if predicted class matches label and count numbler of correct predictions
            total += labels.size(0)
            correct += torch.nn.functional.cosine_similarity(labels,predicted).sum().item() # (predicted == labels).sum().item()

    # Compute final accuracy and display
    test_accuracy = correct/total
    print(f'Ended Training, Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy