import os
import shutil
import gradio as gr
import numpy as np
import wfdb
import torch
from wfdb.plot.plot import plot_wfdb
from wfdb.io.record import Record, rdrecord

from models.CNN import CNN, MMCNN_CAT
from models.RNN import MMRNN
from utils.helper_functions import predict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from langdetect import detect

# Define clinical models and tokenizers
en_clin_bert = 'emilyalsentzer/Bio_ClinicalBERT'
ger_clin_bert = 'smanjil/German-MedBERT'

en_tokenizer = AutoTokenizer.from_pretrained(en_clin_bert)
en_model = AutoModel.from_pretrained(en_clin_bert)

g_tokenizer = AutoTokenizer.from_pretrained(ger_clin_bert)
g_model = AutoModel.from_pretrained(ger_clin_bert)

def preprocess(data_file_path):
    data = [wfdb.rdsamp(data_file_path)]
    data = np.array([signal for signal, meta in data])
    return data

def embed(notes):
    if detect(notes) == 'en':
        tokens = en_tokenizer(notes, return_tensors='pt')
        outputs = en_model(**tokens)
    else:
        tokens = g_tokenizer(notes, return_tensors='pt')
        outputs = g_model(**tokens)
    
    embeddings = outputs.last_hidden_state
    embedding = torch.mean(embeddings, dim=1).squeeze(0)
    
    return embedding 
    # return torch.load(f'{"./data/embeddings/"}1.pt')
def plot_ecg(path):
    record100 = rdrecord(path)
    return plot_wfdb(record=record100, title='ECG Signal Graph', figsize=(12,10), return_fig=True)

def infer(model,data, notes):
    embed_notes = embed(notes).unsqueeze(0)
    data= torch.tensor(data)
    if model == "CNN":
        model = MMCNN_CAT()
        checkpoint = torch.load("./model_saves/model_MMCNN_CAT_epoch_30_acc_84.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        data = data.transpose(1,2).float()

    elif model == "RNN":
        model = MMRNN(device='cpu')
        model.load_state_dict(torch.load("./demo_data/model_MMRNN_undersampled_augmented_rn_epoch_20_acc_84.pt")['model_state_dict'])
        data = data.float()
    model.eval()
    outputs, predicted = predict(model, data, embed_notes, device='cpu')
    outputs = torch.sigmoid(outputs)[0]
    return {'CD':round(outputs[0].item(),2), 'HYP':round(outputs[1].item(),2), 'MI':round(outputs[2].item(),2), 'NORM':round(outputs[3].item(),2), 'STTC':round(outputs[4].item(),2)}

def run(model_name, header_file, data_file, notes):
    demo_dir = "C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/demo_data"
    hdr_dirname, hdr_basename = os.path.split(header_file.name)
    data_dirname, data_basename = os.path.split(data_file.name)
    shutil.copyfile(data_file.name, f"{demo_dir}/{data_basename}")
    shutil.copyfile(header_file.name, f"{demo_dir}/{hdr_basename}")
    data = preprocess(f"{demo_dir}/{hdr_basename.split('.')[0]}")
    ECG_graph = plot_ecg(f"{demo_dir}/{hdr_basename.split('.')[0]}")
    os.remove(f"{demo_dir}/{data_basename}")
    os.remove(f"{demo_dir}/{hdr_basename}")
    output = infer(model_name, data, notes)
    return output, ECG_graph

with gr.Blocks() as demo:
    with  gr.Row():
        model = gr.Radio(['CNN', 'RNN'])
    with gr.Row():
        with gr.Column(scale=1):
            header_file = gr.File(label = "header_file", file_types=[".hea"])
            data_file = gr.File(label = "data_file", file_types=[".dat"])
            notes = gr.Textbox(label = "Clinical Notes")
        with gr.Column(scale=1):
            output_prob = gr.Label({'NORM':0, 'MI':0, 'STTC':0, 'CD':0, 'HYP':0}, show_label=False)
    with gr.Row():
        ecg_graph = gr.Plot(label = "ECG Signal Visualisation")
    with gr.Row():    
        predict_btn = gr.Button("Predict Class")
        predict_btn.click(fn= run, inputs = [model, header_file, data_file, notes], outputs=[output_prob, ecg_graph])
    with gr.Row():    
        gr.Examples(examples=[["C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/data/test/00001_lr.hea", "C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/data/test/00001_lr.dat", "sinusrhythmus periphere niederspannung"]],
                    inputs = [header_file, data_file, notes])

if __name__ == "__main__":
    demo.launch()