import torch

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys

sys.path.append("..")
from utils.functions import CreateOnehotVariable, TimeDistributed
import numpy as np
import pdb

# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0


class LAS(nn.Module):
    def __init__(self, listener, speller):
        super(LAS, self).__init__()
        self.listener = listener
        self.speller = speller

    def forward(self, batch_data, batch_label, teacher_force_rate, is_training=True):
        listener_feature = self.listener(batch_data)
        if is_training:
            raw_pred_seq, attention_record = self.speller(
                listener_feature, ground_truth=batch_label, teacher_force_rate=teacher_force_rate
            )
        else:
            raw_pred_seq, attention_record = self.speller(
                listener_feature, ground_truth=None, teacher_force_rate=0
            )
        return raw_pred_seq, attention_record

    def serialize(self, optimizer, epoch, tr_loss, val_loss):
        package = {
            # encoder
            "einput": self.listener.input_feature_dim,
            "ehidden": self.listener.hidden_size,
            "elayer": self.listener.num_layers,
            "edropout": self.listener.dropout_rate,
            "etype": self.listener.rnn_unit,
            # decoder
            "dvocab_size": self.speller.label_dim,
            "dhidden": self.speller.hidden_size,
            "dlayer": self.speller.num_layers,
            "etype": self.speller.rnn_unit,
            # state
            "state_dict": self.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["val_loss"] = val_loss
        return package


class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, rnn_unit="LSTM", dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(
            input_feature_dim * 2,
            hidden_dim,
            1,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        # Reduce time resolution
        time_reduc = int(timestep / 2)
        input_xr = input_x.contiguous().view(batch_size, time_reduc, feature_dim * 2)
        # pdb.set_trace()
        # Bidirectional RNN
        output, hidden = self.BLSTM(input_xr)
        return output, hidden


# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(
        self,
        input_feature_dim,
        hidden_size,
        num_layers,
        rnn_unit,
        use_gpu,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.input_feature_dim = input_feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_unit = rnn_unit
        self.dropout_rate = dropout_rate
        assert self.num_layers >= 1, "Listener should have at least 1 layer"

        self.pLSTM_layer0 = pBLSTMLayer(
            input_feature_dim, hidden_size, rnn_unit=rnn_unit, dropout_rate=dropout_rate,
        )

        for i in range(1, self.num_layers):
            setattr(
                self,
                "pLSTM_layer" + str(i),
                pBLSTMLayer(
                    hidden_size * 2, hidden_size, rnn_unit=rnn_unit, dropout_rate=dropout_rate,
                ),
            )

    def forward(self, input_x):
        output, _ = self.pLSTM_layer0(input_x)
        for i in range(1, self.num_layers):
            output, _ = getattr(self, "pLSTM_layer" + str(i))(output)

        return output


# Speller specified in the paper
class Speller(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        rnn_unit,
        num_layers,
        max_label_len,
        use_mlp_in_attention,
        mlp_dim_in_attention,
        mlp_activate_in_attention,
        listener_hidden_size,
        multi_head,
        decode_mode,
        use_gpu=True,
        **kwargs,
    ):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = vocab_size
        self.rnn_layer = self.rnn_unit(
            vocab_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
        )
        self.attention = Attention(
            mlp_preprocess_input=use_mlp_in_attention,
            preprocess_mlp_dim=mlp_dim_in_attention,
            activate=mlp_activate_in_attention,
            input_feature_dim=2 * listener_hidden_size,
            multi_head=multi_head,
        )
        self.character_distribution = nn.Linear(hidden_size * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    # Stepwise operation of each sequence
    def forward_step(self, input_word, last_hidden_state, listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word, last_hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))

        return raw_pred, hidden_state, context, attention_score

    def forward(self, listener_feature, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]

        output_word = CreateOnehotVariable(
            self.float_type(np.zeros((batch_size, 1))), self.label_dim
        )
        if self.use_gpu:
            output_word = output_word.cuda()
        rnn_input = torch.cat([output_word, listener_feature[:, 0:1, :]], dim=-1)

        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []

        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]
        for step in range(max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(
                rnn_input, hidden_state, listener_feature
            )
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:, step : step + 1, :].type(self.float_type)
            else:
                # Case 0. raw output as input
                if self.decode_mode == 0:
                    output_word = raw_pred.unsqueeze(1)
                # Case 1. Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx, i in enumerate(raw_pred.topk(1)[1]):
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                # Case 2. Sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(raw_pred).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx, i in enumerate(sampled_word):
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)

            rnn_input = torch.cat([output_word, context.unsqueeze(1)], dim=-1)

        return raw_pred_seq, attention_record


# Attention mechanism
# Currently only 'dot' is implemented
# please refer to http://www.aclweb.org/anthology/D15-1166 section 3.1 for more details about Attention implementation
# Input : Decoder state                      with shape [batch size, 1, decoder hidden dimension]
#         Compressed feature from Listner    with shape [batch size, T, listener feature dimension]
# Output: Attention score                    with shape [batch size, T (attention score of each time step)]
#         Context vector                     with shape [batch size,  listener feature dimension]
#         (i.e. weighted (by attention score) sum of all timesteps T's feature)
class Attention(nn.Module):
    def __init__(
        self,
        mlp_preprocess_input,
        preprocess_mlp_dim,
        activate,
        mode="dot",
        input_feature_dim=512,
        multi_head=1,
    ):
        super(Attention, self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim, preprocess_mlp_dim * multi_head)
            self.psi = nn.Linear(input_feature_dim, preprocess_mlp_dim)
            if self.multi_head > 1:
                self.dim_reduce = nn.Linear(input_feature_dim * multi_head, input_feature_dim)
            if activate != "None":
                self.activate = getattr(F, activate)
            else:
                self.activate = None

    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input:
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.psi, listener_feature))
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_listener_feature = TimeDistributed(self.psi, listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature

        if self.mode == "dot":
            if self.multi_head == 1:
                energy = torch.bmm(
                    comp_decoder_state, comp_listener_feature.transpose(1, 2)
                ).squeeze(dim=1)
                attention_score = [self.softmax(energy)]
                context = torch.sum(
                    listener_feature
                    * attention_score[0].unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                    dim=1,
                )
            else:
                attention_score = [
                    self.softmax(
                        torch.bmm(att_querry, comp_listener_feature.transpose(1, 2)).squeeze(dim=1)
                    )
                    for att_querry in torch.split(
                        comp_decoder_state, self.preprocess_mlp_dim, dim=-1
                    )
                ]
                projected_src = [
                    torch.sum(
                        listener_feature * att_s.unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                        dim=1,
                    )
                    for att_s in attention_score
                ]
                context = self.dim_reduce(torch.cat(projected_src, dim=-1))
        else:
            # TODO: other attention implementations
            pass
        return attention_score, context
