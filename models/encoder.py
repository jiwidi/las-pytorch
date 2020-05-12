import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N
        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H
        """
        # Add total_length for supportting nn.DataParallel() later
        # see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True,enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        return output, hidden

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim,
                 n_layers, dropout):
        super(Decoder, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(target_vocab_size, embed_dim, padding_idx=hp.pad_idx)
        self.attention = LuongAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, output, encoder_out, decoder_hidden):
        """
        decodes one output frame
        """
        embedded = self.embed(output)  # (1, batch, embed_dim)
        context, mask = self.attention(decoder_hidden[-1:], encoder_out)  # 1, 1, 50 (seq, batch, hidden_dim)
        rnn_output, decoder_hidden = self.gru(torch.cat([embedded, context], dim=2),
                                              decoder_hidden)
        output = self.out(torch.cat([rnn_output, context], 2))
        return output, decoder_hidden, mask




