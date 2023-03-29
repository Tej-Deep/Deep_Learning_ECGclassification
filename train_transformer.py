import torch
from preprocess.preprocess import load_dataset, compute_label_agg, select_data, get_data_loaders
from models.transformer import MVMNet_Transformer, Classifier_CNN
# from models.transformer2 import Transformer
from utils.trainer import trainer
from torchinfo import summary

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 16


model = MVMNet_Transformer(d_model=120, vocab_size=250, num_layers=6, heads=12).to(device)
# model = Classifier_CNN().to(device)
print(model)
# summary(model, input_size=(BATCH_SIZE, 12, 250))

path = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
print("Start loading")    
data, raw_labels = load_dataset(path)
print("done loading")    
labels = compute_label_agg(raw_labels, path)

data, labels, Y = select_data(data, labels)

train_loader, valid_loader, test_loader = get_data_loaders(data, labels, Y, BATCH_SIZE)

lr = 0.0003
epochs = 3

train_accs, valid_accs, test_acc = trainer(model, train_loader, test_loader, valid_loader, num_epochs = epochs, lr = lr, eval_interval=100)

torch.save(model.state_dict(), f'./ckpt/{model.name}_epoch_{epochs}_lr_{lr}_test_acc_{test_acc:.4f}.pt')