import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import torchvision

def get_clones(module, N):
    """
    Creates clones of N encoder and decoder layers.
    
    Args: 
        argument1 (str): Description of argument1.
        argument2 (int): Description of argument2. 
    
    Returns: 
        str: Optional description explaining the value returned by the function. 
    
    Notes: 
        Credits: 
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query, key, value : batch_size * heads * max_len * d_h
    used in multihead attention class
    
    Args: 
        argument1 (int): Description of argument1.
        argument2 (int): Description of argument2. 
    
    Returns: 
        str: batch_size * heads * max_len * d_h 
    
    Notes: 
        Credits: 
    """
    matmul = torch.matmul(query,key.transpose(-2,-1))
    scale = torch.tensor(query.shape[-1],dtype=float)
    logits = matmul / torch.sqrt(scale)
    if mask is not None:
        logits += (mask.float() * -1e9)
    
    attention_weights = F.softmax(logits,dim = -1)
    output = torch.matmul(attention_weights,value)
    return output

class Embedder(nn.Module):
    '''Input embedding layer of size vocab_size * dimensionality
    of word embedding'''
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self,x):
        return self.embed(x)
    

class PositionalEncoding(nn.Module):
    '''Transformers are not sequential so positional encoding
    gives some sequentiality to sentence'''

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        print(self.pe.shape, x.shape)
        y = self.pe.expand(x.size(0), self.pe.size(1), self.pe.size(2))[:,:,:x.size(-1)].cuda()
        # print(y.unsqueeze(1).shape)
        x +=  y.unsqueeze(1)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    '''Divides d_model into heads and
    applies attention to each layer with helper 
    function scaled_dot_product_attention'''

    def __init__(self, heads, d_model):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        print(d_model, self.heads)
        assert d_model % self.heads == 0

        self.d_h = self.d_model // self.heads

        self.q_dense = nn.Linear(d_model,d_model)
        self.k_dense = nn.Linear(d_model,d_model)
        self.v_dense = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)
    
    def forward(self, q, k, v, mask = None):
        
        # batch_size
        bs = q.size(0)

        k = self.k_dense(k).view(bs, -1, self.heads, self.d_h)
        q = self.q_dense(q).view(bs, -1, self.heads, self.d_h)
        v = self.v_dense(v).view(bs, -1, self.heads, self.d_h)

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = scaled_dot_product_attention(q,k,v,mask)
        
        # concat each heads
        concat = scores.transpose(1,2).contiguous()\
            .view(bs,-1,self.d_model)
        
        out = self.out(concat)

        return out
    
class FeedForward(nn.Module):
    '''Feed Forward neural network'''
    def __init__(self,d_model,d_ff = 1000,dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self,x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    '''Encoder layer of transformer 
    embedding -> positional_encoding -> attention
     -> Feed Forward with skip connection'''
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    
    def forward(self,x):
        x1 = self.norm_1(x)
        x1 = x + self.dropout_1(self.attn(x1,x1,x1))
        x2 = self.norm_2(x1)
        x3 = x1 + self.dropout_2(self.ff(x2))
        return x3

class Encoder(nn.Module):
    '''Cloning and making copies'''
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src[src < 0] = 0
        src[src > 249] = 249

        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x) 


class EncoderBlock(nn.Module):
      def __init__(self, vocab_size, d_model, num_layers, heads):
        super().__init__()
        self.encoder = Encoder(vocab_size,d_model,num_layers,heads)
        self.conven = nn.Conv2d(12,1,1)

      def forward(self, src):
          src[src < 0] = 0
          src[src > 249] = 249

        #   src = torch.reshape(src,(src.shape[0],250,12))
          src = torch.reshape(src, (src.shape[0],src.shape[1],src.shape[2], 1))
          print(src.shape)
        #   src = torch.transpose(src,0,2)
        #   print(src.shape)
          src.unsqueeze(-1)
          print(src.shape)
        #   input(":")
          src = self.conven(src.float())
          print(src.shape)

          src = torch.squeeze(src,3)
          print(src.shape)
        #   input(":")
          e_outputs = self.encoder(src.long())
          return e_outputs # 1x250x120
class DecoderLayer(nn.Module):
    '''Decoder layer - mha to notes embeddings -> add&norm -> mha to both notes embeddings and encoder output -> add&norm -> feed forward linear -> add&norm'''
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, encoder_out):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_out, encoder_out))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Decoder(nn.Module):
    '''Decoder module of transformer, running n sequential decoder segments - takes input of both encoder output and text embeddings'''
    def __init__(self, d_model, vocab_size, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out):
        for i in range(self.N):
            x = self.layers[i](x, encoder_out)
        return self.norm(x)
    

class CNN(nn.Module):
    '''Convolutional output network for classification task - takes in summed input sequence and decoder output'''
    def __init__(self):
        super(CNN, self).__init__()
        self.name = "cnn"
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 1, 3)
        self.conv3 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(377, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16,5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x
    

class Transformer(nn.Module):
    '''Overall Transformer architecture'''
    def __init__(self, vocab_size, d_model, num_layers, heads):
        super().__init__()
        self.name = "np_Transformer"
        self.encoderBlock = EncoderBlock(vocab_size,d_model,num_layers,heads)
        self.transform = torchvision.transforms.Resize((250,120))
        self.decoder = Decoder(d_model,vocab_size,num_layers,heads)
        self.conv_out = CNN()
        self.fc = nn.Linear(30000, 5)
        self.softmax = nn.Softmax(1)
        self.transform_cnn = torchvision.transforms.Resize((250,120))
        self.norm = nn.LayerNorm((1,250,120))

    def forward(self, src, txt=None):
        print(src.shape)
        src = ((src + 1.68)*100).long() #scale
        
        e_outputs = self.encoderBlock(src.cuda()) #encoder
        
        # text embeddings
        if txt == None:
            txt = torch.rand((250,120)).cuda()
        txt = torch.unsqueeze(txt, 0) 
        txt = self.transform(txt)
        
        d_output = self.decoder(txt, e_outputs.cuda()) #decoder
        d_output = torch.squeeze(d_output,0)

        src = self.transform_cnn(src.float())
        return self.conv_out(self.norm(src + d_output)) #output