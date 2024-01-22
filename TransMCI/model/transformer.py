import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, num_decoder_layers, hidden_dim, dropout, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, src):
        embedded = self.embedding(src)
        decoder_input = torch.zeros_like(embedded)  # Adjust the batch size and sequence length
        output = self.transformer(embedded, decoder_input)
        output = output[:, -1, :]
        output = self.fc(output)
        return output
