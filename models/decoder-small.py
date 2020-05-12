import torch.nn.functional as F
from torch import nn
from .luongattention import LuongAttention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 n_layers, dropout):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=1)
        self.attention = LuongAttention(hidden_size)
        self.gru = nn.GRU(embedding_dim + hidden_size, hidden_size, n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embedding_dim)
        context, mask = self.attention(decoder_hidden[-1:], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2),
                                              decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, mask