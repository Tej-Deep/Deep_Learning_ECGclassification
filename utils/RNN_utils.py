import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pywt
import os


def display_eval(epoch, epochs, tlength, global_step, tcorrect, tsamples, t_valid_samples, average_train_loss, average_valid_loss, total_acc_val):
    tqdm.write(
        f'Epoch: [{epoch + 1}/{epochs}], Step [{global_step}/{epochs*tlength}] | Train Loss: {average_train_loss: .3f} \
        | Train Accuracy: {tcorrect / tsamples: .3f} \
        | Val Loss: {average_valid_loss: .3f} \
        | Val Accuracy: {total_acc_val / t_valid_samples: .3f}')


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


def train_RNN(epochs, train_loader, valid_loader, model, loss_fn, optimizer, eval_every=0.25, best_valid_loss=float("Inf"), device='cuda', model_save_name='', save_dir='./'):
    model.train()

    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    wavelet = 'db4'
    level = 3

    for epoch in range(epochs):
        running_loss = 0.0
        t_correct = 0
        t_samples = 0
        for images, labels, notes in tqdm(train_loader):
            optimizer.zero_grad()

            coeffs = pywt.wavedec(images, wavelet, level=level, axis=1)
            threshold = 0.1 * \
                torch.median(torch.abs(torch.from_numpy(coeffs[-1])))
            denoised_coeffs = [pywt.threshold(
                data=c, mode='hard', value=threshold) for c in coeffs]
            images = pywt.waverec(denoised_coeffs, wavelet, axis=1)

            images = torch.tensor(images).float().to(device)
            labels = labels.to(device)
            notes = notes.to(device)

            output = model(images, notes)

            loss = loss_fn(output, labels.float())
            running_loss += loss.item()*len(labels)
            loss.backward()
            global_step += 1*len(images)

            optimizer.step()

            values, indices = torch.max(output, dim=1)
            t_correct += sum(1 for s, i in enumerate(indices)
                             if labels[s][i] == 1)
            t_samples += len(indices)

            if (global_step % (int(eval_every*len(train_loader.dataset)))) < train_loader.batch_size:
                model.eval()
                valid_running_loss = 0.0
                total_acc_val = 0
                with torch.no_grad():

                    for images, labels, notes in valid_loader:

                        coeffs = pywt.wavedec(
                            images, wavelet, level=level, axis=1)
                        threshold = 0.1 * \
                            torch.median(
                                torch.abs(torch.from_numpy(coeffs[-1])))
                        denoised_coeffs = [pywt.threshold(
                            data=c, mode='hard', value=threshold) for c in coeffs]
                        images = pywt.waverec(denoised_coeffs, wavelet, axis=1)

                        images = torch.tensor(images).float().to(device)
                        labels = labels.to(device)
                        notes = notes.to(device)
                        output = model(images, notes)

                        loss = loss_fn(output, labels.float()).item()
                        valid_running_loss += loss*len(images)
                        values, indices = torch.max(output, dim=1)
                        total_acc_val += sum(1 for s,
                                             i in enumerate(indices) if labels[s][i] == 1)

                # evaluation
                average_train_loss = running_loss / t_samples
                average_valid_loss = valid_running_loss / \
                    len(valid_loader.dataset)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                display_eval(epoch, epochs, len(train_loader.dataset), global_step, t_correct, t_samples, len(
                    valid_loader.dataset), average_train_loss, average_valid_loss, total_acc_val)

                # resetting running values
                model.train()

                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_model(model, optimizer, best_valid_loss, epoch,
                               path=f'{save_dir}model_{model_save_name}.pt')
                    save_metrics(train_loss_list, valid_loss_list,
                                 global_steps_list, path=f'{save_dir}metrics_{model_save_name}.pt')

    save_metrics(train_loss_list, valid_loss_list, global_steps_list,
                 path=f'{save_dir}metrics_{model_save_name}.pt')
    print("Training complete.")
    return model


def evaluate_RNN(model, test_loader, device="cuda"):
    model.eval()
    y_pred = []
    y_true = []

    wavelet = 'db4'
    level = 3

    total_acc_test = 0
    with torch.no_grad():
        for images, labels, notes in test_loader:
            coeffs = pywt.wavedec(images, wavelet, level=level, axis=1)
            threshold = 0.1 * \
                torch.median(torch.abs(torch.from_numpy(coeffs[-1])))
            denoised_coeffs = [pywt.threshold(
                data=c, mode='hard', value=threshold) for c in coeffs]
            images = pywt.waverec(denoised_coeffs, wavelet, axis=1)

            images = torch.tensor(images).float().to(device)
            labels = labels.to(device)
            notes = notes.to(device)
            output = model(images, notes)

            values, indices = torch.max(output, dim=1)
            y_pred.extend(indices.tolist())
            y_true.extend(labels.tolist())
            total_acc_test += sum(1 for s,
                                  i in enumerate(indices) if labels[s][i] == 1)

    test_accuracy = total_acc_test / len(test_loader.dataset)
    print(f'Test Accuracy: {test_accuracy: .3f}')

    return test_accuracy


def rename_with_acc(save_name, save_dir, acc):
    acc = round(acc*100)
    # Rename model
    os.rename(f'{save_dir}model_{save_name}.pt',
              f'{save_dir}model_{save_name}_acc_{acc}.pt')
    # Rename metrics
    os.rename(f'{save_dir}metrics_{save_name}.pt',
              f'{save_dir}metrics_{save_name}_acc_{acc}.pt')
