import torch
import torch.nn as nn
import math
import json
from torch.nn.utils.rnn import pad_sequence

#构建此表
raw_data =[]
try:
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_data.append((item["english"], item["chinese"]))
except FileNotFoundError:                                                  #用来报错
    raw_data =[
        ("For greater sharpness, but with a slight increase in graininess, you can use a 1:1 dilution of this developer.", 
         "为了更好的锐度，但是附带的会多一些颗粒度，可以使用这个显影剂的1：1稀释液。")
    ]

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

eng_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}              #特殊符号
chn_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

for eng, chn in raw_data:
    for word in eng.lower().split():
        if word not in eng_vocab:
            eng_vocab[word] = len(eng_vocab)
    for char in chn:
        if char not in chn_vocab:
            chn_vocab[char] = len(chn_vocab)

chn_idx2word = {idx: word for word, idx in chn_vocab.items()}

class PositionalEncoding(nn.Module):                                       #位置编码
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):                                       #多头注意力核心部分
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        Q = self.fc_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.fc_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.fc_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e20'))
            
        attention = torch.softmax(scores, dim=-1)
        
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc_o(x)

class FeedForward(nn.Module):                                       #一些神经网络基本区块
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn))
        attn = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class SimpleTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=8, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_src = self.dropout(self.pos_enc(self.src_emb(src)))
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, src_mask)
            
        dec_tgt = self.dropout(self.pos_enc(self.tgt_emb(tgt)))
        for layer in self.decoder_layers:
            dec_tgt = layer(dec_tgt, enc_src, src_mask, tgt_mask)
            
        return self.fc_out(dec_tgt)

def make_src_mask(src, pad_idx):                              #掩码
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    return tgt_pad_mask & tgt_sub_mask

src_batch, tgt_batch = [], []
for eng, chn in raw_data:
    src_tokens =[eng_vocab.get(w, UNK_IDX) for w in eng.lower().split()]
    tgt_tokens = [SOS_IDX] +[chn_vocab.get(c, UNK_IDX) for c in chn] + [EOS_IDX]
    src_batch.append(torch.tensor(src_tokens))
    tgt_batch.append(torch.tensor(tgt_tokens))

src_data = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
tgt_data = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

model = SimpleTranslator(len(eng_vocab), len(chn_vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

model.train()
for epoch in range(150):
    optimizer.zero_grad()
    
    tgt_input = tgt_data[:, :-1]
    tgt_expected = tgt_data[:, 1:]
    
    src_mask = make_src_mask(src_data, PAD_IDX)
    tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)
    
    logits = model(src_data, tgt_input, src_mask, tgt_mask)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

def translate(sentence):
    model.eval()
    tokens =[eng_vocab.get(w, UNK_IDX) for w in sentence.lower().split()]
    src = torch.tensor(tokens).unsqueeze(0)
    src_mask = make_src_mask(src, PAD_IDX)
    
    tgt_tokens = [SOS_IDX]
    
    for _ in range(100):
        tgt = torch.tensor(tgt_tokens).unsqueeze(0)
        tgt_mask = make_tgt_mask(tgt, PAD_IDX)
        
        with torch.no_grad():
            logits = model(src, tgt, src_mask, tgt_mask)
            
        next_token = logits[0, -1, :].argmax().item()
        
        if next_token == EOS_IDX:
            break
        tgt_tokens.append(next_token)
        
    return "".join([chn_idx2word.get(idx, "") for idx in tgt_tokens if idx not in[SOS_IDX, PAD_IDX]])

print("\n")
test_sentences =[
    "For greater sharpness, but with a slight increase in graininess, you can use a 1:1 dilution of this developer."
]
for s in test_sentences:
    print(f"{s}\n -> {translate(s)}")