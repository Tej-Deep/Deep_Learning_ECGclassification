import torch 
from .helper_functions import define_optimizer, predict, display_train, eval_test
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_model(model, optimizer, valid_loss, epoch, path='model.pt'):
    torch.save({'valid_loss': valid_loss,
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict()
                }, path)
    tqdm.write(f'Model saved to ==> {path}')


def save_metrics(train_loss_list, valid_loss_list, global_steps_list, path='metrics.pt'):
    torch.save({'train_loss_list': train_loss_list,
                'valid_loss_list': valid_loss_list,
                'global_steps_list': global_steps_list,
                }, path)

def plot_losses(metrics_save_name='metrics', save_dir='./'):
    path = f'{save_dir}metrics_{metrics_save_name}.pt'
    state = torch.load(path)

    train_loss_list = state['train_loss_list']
    valid_loss_list = state['valid_loss_list']
    global_steps_list = state['global_steps_list']

    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def trainer(model, train_loader, test_loader, valid_loader, num_epochs = 10, lr = 0.01, alpha = 0.99, eval_interval = 10, model_save_name='', save_dir='./'):
    
    # Use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # History for train acc, test acc
    train_accs = []
    valid_accs = []
    global_step = 0
    train_loss_list = []
    valid_loss_list = [] 
    global_steps_list = []
    best_valid_loss = float("inf") 

    
    # Define optimizer
    optimizer = define_optimizer(model, lr, alpha)
    
    # Training model
    for epoch in range(num_epochs):
        # Go trough all samples in train dataset
        model.train()
        running_loss = 0
        correct = 0
        total = 0
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
            # correct += torch.nn.functional.cosine_similarity(labels,predicted).sum().item()  #(predicted == labels).sum().item()
            values, indices = torch.max(outputs,dim=1)
            correct += sum(1 for s, i in enumerate(indices)
                             if labels[s][i] == 1)
            # Compute loss
            # we use outputs before softmax function to the cross_entropy loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)
            running_loss += loss.item()*len(labels)
            global_step += 1*len(inputs)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Display losses over iterations and evaluate on validation set
            if (i+1) % eval_interval == 0:
                train_accuracy, valid_accuracy, valid_loss = display_train(epoch, num_epochs, i, model, \
                                                               correct, total, loss, \
                                                               train_loader, valid_loader, device)
                
                average_train_loss = running_loss / total
                # average_valid_loss = valid_loss
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_model(model, optimizer, best_valid_loss, epoch, path=f'{save_dir}model_{model_save_name}.pt')
                    save_metrics(train_loss_list, valid_loss_list, global_steps_list, path=f'{save_dir}metrics_{model_save_name}.pt')
                    # torch.save(model.state_dict(),  f'./ckpt_mid/{model.name}_best_lr_{lr}.pt')
                
                
        if(len(train_loader)%eval_interval!=0):
            train_accuracy, valid_accuracy, valid_loss = display_train(epoch, num_epochs, i, model, \
                                                                    correct, total, loss, \
                                                                    train_loader, valid_loader, device)
            
            average_train_loss = running_loss / total
            # average_valid_loss = valid_loss/len(valid_loader.dataset)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(valid_loss)
            global_steps_list.append(global_step)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_model(model, optimizer, best_valid_loss, epoch, path=f'{save_dir}model_{model_save_name}.pt')
                save_metrics(train_loss_list, valid_loss_list, global_steps_list, path=f'{save_dir}metrics_{model_save_name}.pt')
                # torch.save(model.state_dict(),  f'./ckpt_mid/{model.name}_best_lr_{lr}.pt')
        # Append accuracies to list at the end of each iteration
        train_accs.append(train_accuracy)
        valid_accs.append(valid_accuracy)
        # torch.save(model.state_dict(), f'./ckpt_mid/{model.name}_epoch_{epoch}_lr_{lr}.pt')
    save_metrics(train_loss_list, valid_loss_list, global_steps_list,
                 path=f'{save_dir}metrics_{model_save_name}.pt')
    # Load best_model
    checkpoint = torch.load(f'{save_dir}model_{model_save_name}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # Evaluate on test after training has completed
    test_acc = eval_test(model, test_loader, device)
    # Return
    return train_accs, valid_accs, test_acc