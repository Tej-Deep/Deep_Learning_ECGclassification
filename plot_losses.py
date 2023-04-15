import matplotlib.pyplot as plt
import sys
import torch

def plot_losses(path):
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

if __name__=="__main__":
    try:
        path = sys.argv[1]
    except:
        sys.exit("Please provide path to a metrics save file as an argument (python3 plot_losses.py <metrics_path) without quotes.")

    plot_losses(path)