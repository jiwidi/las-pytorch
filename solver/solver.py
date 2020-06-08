import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import editdistance as ed
import pdb


# LetterErrorRate function
# Merge the repeated prediction and calculate editdistance of prediction and ground truth
def LetterErrorRate(pred_y, true_y):
    ed_accumalate = []
    for p, t in zip(pred_y, true_y):
        compressed_t = [w for w in t if (w != 1 and w != 0)]

        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        ed_accumalate.append(ed.eval(compressed_p, compressed_t) / len(compressed_t))
    return ed_accumalate


def tensor2text(y, vocab):
    rounded_y = np.around(y).astype(int)
    rounded_y = np.array([vocab[str(n)] for n in rounded_y])
    return rounded_y


def label_smoothing_loss(pred_y, true_y, label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y, dim=-1), dim=-1, keepdim=True)

    # calculate smoothen label, last term ensures padding vector remains all zero
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0 - label_smoothing) * true_y + (label_smoothing / class_dim)) * torch.sum(true_y, dim=-1, keepdim=True)

    loss = -torch.mean(torch.sum((torch.sum(smooth_y * pred_y, dim=-1) / seq_len), dim=-1))

    return loss


def batch_iterator(
    batch_data,
    batch_label,
    las_model,
    optimizer,
    tf_rate,
    is_training,
    max_label_len,
    label_smoothing,
    use_gpu=True,
    vocab_dict=None,
):
    label_smoothing = label_smoothing
    max_label_len = min([batch_label.size()[1], max_label_len])
    criterion = nn.NLLLoss(ignore_index=0).cuda()
    optimizer.zero_grad()

    raw_pred_seq, _ = las_model(
        batch_data=batch_data, batch_label=batch_label, teacher_force_rate=tf_rate, is_training=is_training,
    )
    pred_y = (torch.cat([torch.unsqueeze(each_y, 1) for each_y in raw_pred_seq], 1)[:, :max_label_len, :]).contiguous()

    if label_smoothing == 0.0 or not (is_training):
        pred_y = pred_y.permute(0, 2, 1)  # pred_y.contiguous().view(-1,output_class_dim)
        true_y = torch.max(batch_label, dim=2)[1][:, :max_label_len].contiguous()  # .view(-1)

        loss = criterion(pred_y, true_y)
        # variable -> numpy before sending into LER calculator
        batch_ler = LetterErrorRate(
            torch.max(pred_y.permute(0, 2, 1), dim=2)[1].cpu().numpy(),  # .reshape(current_batch_size,max_label_len),
            true_y.cpu().data.numpy(),
        )  # .reshape(current_batch_size,max_label_len), data)

    else:
        true_y = batch_label[:, :max_label_len, :].contiguous()
        true_y = true_y.type(torch.cuda.FloatTensor) if use_gpu else true_y.type(torch.FloatTensor)
        loss = label_smoothing_loss(pred_y, true_y, label_smoothing=label_smoothing)
        # batch_ler = [1.0]
        # print(true_y)
        # print("vs")
        # print(pred_y)
        batch_ler = LetterErrorRate(
            torch.max(pred_y, dim=2)[1].cpu().numpy(),  # .reshape(current_batch_size,max_label_len),
            torch.max(true_y, dim=2)[1].cpu().data.numpy(),
        )  # .reshape(current_batch_size,max_label_len), data)

    if is_training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(las_model.parameters(), 1)
        optimizer.step()

    batch_loss = loss.cpu().data.numpy()

    return batch_loss, batch_ler
