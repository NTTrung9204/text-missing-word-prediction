import torch.nn as nn

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
