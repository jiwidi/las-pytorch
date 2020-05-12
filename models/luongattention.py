import torch.nn.functional as F
from torch import nn


class LuongAttention(nn.Module):
    """
    Luong Attention from Effective Approaches to Attention-based Neural Machine Translation
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, attention_dim):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)  # batch, seq, 1
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (2, 50, 15) @ (2, 15, 1)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        mask = mask.permute(2, 0, 1)  # (target, batch, source)
        return context, mask