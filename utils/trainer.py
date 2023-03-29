import torch 
from .helper_functions import define_optimizer, predict, display_train, eval_test
from tqdm import tqdm



def trainer(model, train_loader, test_loader, valid_loader, num_epochs = 10, lr = 0.01, alpha = 0.99, eval_interval = 10):
    
    # Use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # History for train acc, test acc
    train_accs = []
    valid_accs = []
    
    # Define optimizer
    optimizer = define_optimizer(model, lr, alpha)
    
    # Training model
    for epoch in range(num_epochs):
        # Go trough all samples in train dataset
        model.train()
        correct = 0
        total = 0
        for i, (inputs, labels, notes) in tqdm(enumerate(train_loader)):
            # Get from dataloader and send to device
            inputs = inputs.transpose(1,2).float().to(device)
            # print(labels.shape)
            labels = labels.float().to(device)
            notes = notes.to(device)
            # print(labels.shape)


            # Forward pass
            outputs, predicted = predict(model, inputs, notes, device)
            # print(predicted.shape, labels.shape)
            
            # Check if predicted class matches label and count numbler of correct predictions
            total += labels.size(0)
            correct += torch.nn.functional.cosine_similarity(labels,predicted).sum().item()  #(predicted == labels).sum().item()

            # Compute loss
            # we use outputs before softmax function to the cross_entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Display losses over iterations and evaluate on validation set
            if (i+1) % eval_interval == 0:
                train_accuracy, valid_accuracy = display_train(epoch, num_epochs, i, model, \
                                                               correct, total, loss, \
                                                               train_loader, valid_loader, device)
        if(len(train_loader)%eval_interval!=0):
            train_accuracy, valid_accuracy = display_train(epoch, num_epochs, i, model, \
                                                                    correct, total, loss, \
                                                                    train_loader, valid_loader, device)
        # Append accuracies to list at the end of each iteration
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)
    # Evaluate on test after training has completed
    test_acc = eval_test(model, test_loader, device)
    # Return
    return train_accs, valid_accs, test_acc