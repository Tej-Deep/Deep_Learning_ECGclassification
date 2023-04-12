import torch
from preprocess.preprocess import load_dataset, compute_label_agg, select_data, get_data_loaders
from models.CNN import CNN, MMCNN_SUM, MMCNN_CAT, MMCNN_ATT
from utils.trainer import trainer
from torchinfo import summary

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
batch_size = 256
print("Start loading")    
data, raw_labels = load_dataset(path)
print("done loading")    
labels = compute_label_agg(raw_labels, path)

data, labels, Y = select_data(data, labels)

train_loader, valid_loader, test_loader = get_data_loaders(data, labels, Y, batch_size)
# models = [MMCNN_CAT, CNN, MMCNN_SUM]
# test_accuracies = {"CNN":0,"MMCNN_SUM":0, "MMCNN_CAT":0}
models = [MMCNN_ATT, MMCNN_CAT, CNN, MMCNN_SUM]
# test_accuracies = {i.name:0 for i in models}
for model_class in models:
    model = model_class().to(device)
    print(model)
    # summary(model, input_size=(batch_size, 12, 1000))

    lr = 0.0001
    epochs = 100

    train_accs, valid_accs, test_acc = trainer(model, train_loader, test_loader, valid_loader, lr = lr, num_epochs = epochs)
    # test_accuracies[model.name] = test_acc
    torch.save(model.state_dict(), f'./ckpt_use_best/{model.name}_epoch_{epochs}_best_lr_{lr}_test_acc_{test_acc:.4f}.pt')

# print(test_accuracies)