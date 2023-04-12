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
        best_valid_loss = float("inf") 
        for i, (inputs, labels, notes) in enumerate(train_loader):
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
            #TODO: change acc criteria
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
                train_accuracy, valid_accuracy, valid_loss = display_train(epoch, num_epochs, i, model, \
                                                               correct, total, loss, \
                                                               train_loader, valid_loader, device)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(),  f'./ckpt_mid/{model.name}_best_lr_{lr}.pt')
                
        if(len(train_loader)%eval_interval!=0):
            train_accuracy, valid_accuracy, valid_loss = display_train(epoch, num_epochs, i, model, \
                                                                    correct, total, loss, \
                                                                    train_loader, valid_loader, device)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(),  f'./ckpt_mid/{model.name}_best_lr_{lr}.pt')
        # Append accuracies to list at the end of each iteration
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)
        # torch.save(model.state_dict(), f'./ckpt_mid/{model.name}_epoch_{epoch}_lr_{lr}.pt')
    # Load best_model
    model.load_state_dict(torch.load(f'./ckpt_mid/{model.name}_best_lr_{lr}.pt'))
    # Evaluate on test after training has completed
    test_acc = eval_test(model, test_loader, device)
    # Return
    return train_accs, valid_accs, test_acc