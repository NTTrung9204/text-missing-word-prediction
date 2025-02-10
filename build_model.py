import torch.nn as nn
import torch
import math

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)

        self.fcs = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        x, _ = self.lstm(x)

        x = self.fcs(x[:, -1, :])

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Tạo positional encoding để bổ sung thông tin thứ tự cho embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tạo ma trận positional encoding với kích thước (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Tính các hệ số chia theo công thức của Vaswani et al.
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # các vị trí chẵn
        pe[:, 1::2] = torch.cos(position * div_term)  # các vị trí lẻ
        pe = pe.unsqueeze(0)  # Kích thước cuối cùng: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor có kích thước (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, nhead=8):
        """
        vocab_size   : Số từ trong từ điển.
        embedding_dim: Kích thước vector nhúng (cũng là d_model của Transformer).
        hidden_dim   : Kích thước tầng ẩn của mạng feed-forward bên trong Transformer Encoder.
        num_layers   : Số lớp Transformer Encoder.
        num_classes  : Số lớp của bài toán phân loại.
        nhead        : Số đầu (heads) trong multi-head attention (lưu ý: embedding_dim phải chia hết cho nhead).
        """
        super(TransformerModel, self).__init__()
        
        # Lớp nhúng từ
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Tạo Transformer Encoder với num_layers lớp
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim,
            batch_first=True  # Sử dụng định dạng (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Lớp fully-connected để chuyển từ kích thước embedding_dim sang số lớp phân loại
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        x: Tensor có kích thước (batch_size, seq_len) chứa các chỉ số token.
        """
        # Nhúng các token thành vector
        x = self.embedding(x)  # Kích thước: (batch_size, seq_len, embedding_dim)
        
        # Thêm positional encoding
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)  # Kích thước vẫn là (batch_size, seq_len, embedding_dim)
        
        # Lấy vector của token cuối cùng (hoặc có thể sử dụng pooling) cho bài toán phân loại
        out = self.fc(x[:, -1, :])
        return out