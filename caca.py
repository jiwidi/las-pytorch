import torch
import torch.nn as nn
import torch.nn.functional as F
import np


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout=0.0, batch_first=False):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=True,
        )
        # self.layer_norm = nn.LayerNorm(rnn_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # x = self.layer_norm(x)
        # x = F.gelu(x)
        x, hidden = self.BiGRU(x)
        # x = self.dropout(x)
        return x, hidden


class Listener(nn.Module):
    def __init__(
        self, input_feature_dim_listener, hidden_size_listener, num_layers_listener
    ):
        super(Listener, self).__init__()
        assert num_layers_listener >= 1, "Listener should have at least 1 layer"
        self.hidden_size = hidden_size_listener
        self.gru_1 = BidirectionalGRU(
            rnn_dim=input_feature_dim_listener,
            hidden_size=hidden_size_listener,
            batch_first=True,
        )
        self.gru_2 = BidirectionalGRU(
            rnn_dim=hidden_size_listener * 2,
            hidden_size=hidden_size_listener,
            batch_first=True,
        )
        self.gru_3 = BidirectionalGRU(
            rnn_dim=hidden_size_listener * 2,
            hidden_size=hidden_size_listener,
            batch_first=True,
        )
        self.gru_4 = BidirectionalGRU(
            rnn_dim=hidden_size_listener * 2,
            hidden_size=hidden_size_listener,
            batch_first=True,
        )

    def initHidden(self):
        return torch.zeros([2, 8, 512])

    def forward(self, x):
        x = x.squeeze().permute(0, 2, 1)
        fake_hidden = self.initHidden()
        output, hidden = self.gru_1(x, fake_hidden)
        output, hidden = self.gru_2(output, hidden)
        output, hidden = self.gru_3(output, hidden)
        output, hidden = self.gru_4(output, hidden)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.rnn_layer = nn.LSTM(output_size * 2, hidden_size, num_layers=2)
        self.attention = Attention()
        self.character_distribution = nn.Linear(hidden_size * 2, 29)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward_step(self, input_word, last_hidden_state, listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word, last_hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))

        return raw_pred, hidden_state, context, attention_score

    def forward(self, listener_feature, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = (
            True if np.random.random_sample() < teacher_force_rate else False
        )

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
        mode="dot",
        input_feature_dim=512,
        multi_head=1,
    ):
        super(Attention, self).__init__()
        self.mode = mode.lower()
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_state, listener_feature):
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
                    * attention_score[0]
                    .unsqueeze(2)
                    .repeat(1, 1, listener_feature.size(2)),
                    dim=1,
                )
            else:
                attention_score = [
                    self.softmax(
                        torch.bmm(
                            att_querry, comp_listener_feature.transpose(1, 2)
                        ).squeeze(dim=1)
                    )
                    for att_querry in torch.split(
                        comp_decoder_state, self.preprocess_mlp_dim, dim=-1
                    )
                ]
                projected_src = [
                    torch.sum(
                        listener_feature
                        * att_s.unsqueeze(2).repeat(1, 1, listener_feature.size(2)),
                        dim=1,
                    )
                    for att_s in attention_score
                ]
                context = self.dim_reduce(torch.cat(projected_src, dim=-1))
        else:
            # TODO: other attention implementations
            pass

        return attention_score, context


class LAS(nn.Module):
    def __init__(
        self,
        input_feature_dim_listener,
        hidden_size_listener,
        num_layers_listener,
        hidden_size_speller,
    ):
        super(LAS, self).__init__()
        self.listener = Listener(
            input_feature_dim_listener, hidden_size_listener, num_layers_listener
        )
        self.speller = DecoderRNN(hidden_size_speller, hidden_size_listener)

    def forward(self, batch_data, batch_label, teacher_force_rate, is_training=True):
        listener_feature, hidden = self.listener(batch_data)
        if is_training:
            raw_pred_seq, attention_record = self.speller(
                listener_feature,
                hidden,
                # ground_truth=batch_label,
                # teacher_force_rate=teacher_force_rate,
            )
        else:
            raw_pred_seq, attention_record = self.speller(
                listener_feature, ground_truth=None, teacher_force_rate=0
            )
        return raw_pred_seq, attention_record


# listener = Listener(128, 512, 3)
# print(listener)
# speller = DecoderRNN(1024, 512)
spectrograms = torch.zeros([8, 1, 128, 671])
# output, hidden = listener(spectrograms)
# print("Done listener")
# # print(len(ls))
# speller(output, hidden)
# print("Done speller")
las = LAS(
    input_feature_dim_listener=128,
    hidden_size_listener=512,
    num_layers_listener=3,
    hidden_size_speller=100,
)
las(spectrograms, spectrograms, 0.5)
