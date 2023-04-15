import os
import gradio as gr
import numpy as np
import wfdb
import torch
from models.CNN import CNN, MMCNN_CAT
from utils.helper_functions import predict
from wfdb.plot.plot import plot_wfdb
from wfdb.io.record import Record, rdrecord

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

def infer(data, notes):
    embed_notes = embed(notes).unsqueeze(0)
    data= torch.tensor(data)
    model = MMCNN_CAT()
    model.load_state_dict(torch.load("./ckpt_use_best/MMCNN_CAT_epoch_50_best_lr_0.0001_test_acc_0.7779.pt"))
    model.eval()
    data = data.transpose(1,2).float()
    outputs, predicted = predict(model, data, embed_notes, device='cpu')
    outputs = torch.sigmoid(outputs)[0]
    return {'CD':round(outputs[0].item(),2), 'HYP':round(outputs[1].item(),2), 'MI':round(outputs[2].item(),2), 'NORM':round(outputs[3].item(),2), 'STTC':round(outputs[4].item(),2)}

def run(header_file, data_file, notes):
    demo_dir = "C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/demo_data"
    hdr_dirname, hdr_basename = os.path.split(header_file.name)
    data_dirname, data_basename = os.path.split(data_file.name)
    os.replace(data_file.name, f"{demo_dir}/{data_basename}")
    os.replace(header_file.name, f"{demo_dir}/{hdr_basename}")
    data = preprocess(f"{demo_dir}/{hdr_basename.split('.')[0]}")
    ECG_graph = plot_ecg(f"{demo_dir}/{hdr_basename.split('.')[0]}")
    os.remove(f"{demo_dir}/{data_basename}")
    os.remove(f"{demo_dir}/{hdr_basename}")
    output = infer(data, notes)
    return output, ECG_graph

with gr.Blocks() as demo:
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
        predict_btn.click(fn= run, inputs = [header_file, data_file, notes], outputs=[output_prob, ecg_graph])
    with gr.Row():    
        gr.Examples(examples=[["C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/data/test/00001_lr.hea", "C:/Users/ptejd/Documents/Deep_Learning/Deep_Learning_ECGclassification/data/test/00001_lr.dat", "sinusrhythmus periphere niederspannung"]],
                    inputs = [header_file, data_file, notes])

if __name__ == "__main__":
    demo.launch()