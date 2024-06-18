import torch
import torch.nn.functional as F

# def recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5):
#     bs = logits_o.size(0)
#     loss_o = 0.0
#     loss1 = 0.0
#     loss2 = 0.0
#     for i in range(bs):
#         loss_o += F.cross_entropy(logits_o[i], labels_o[i])
#         loss1 += F.cross_entropy(logits1[i], labels1[i])
#         loss2 += F.cross_entropy(logits2[i], labels2[i])
#     rec_loss = (loss_o + omega * (loss1 + loss2)) / bs
#     return rec_loss

def recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5):
    loss_o, loss1, loss2, bs = 0.0, 0.0, 0.0, logits_o.size(0)
    for i in range(bs):
        loss_o += F.cross_entropy(logits_o[i], labels_o[i])
        loss1 += F.cross_entropy(logits1[i], labels1[i])
        loss2 += F.cross_entropy(logits2[i], labels2[i])
    rec_loss = (loss_o + omega * (loss1 + loss2)) / bs
    return rec_loss

def contrastive_loss(logits1, logits2, labels1, labels2, tau=2.0):
    def pair_loss(logit1, logit2, label1, label2):
        logit = torch.cat((logit1, logit2), dim=0)
        label = torch.cat((label1, label2), dim=0)
        loss_m = 0.0
        #TODO: 冗长
        for m in range(bs):
            pos_indices = torch.where((label == label[m]) & (torch.arange(label.size(0)) != m))[0]
            pos_logit = logit[pos_indices]
            if pos_logit.size(0) == 0:
                continue
            a_m = torch.where(torch.arange(label.size(0)) != m)[0]

            loss_p = 0.0
            for p in range(pos_logit.size(0)):
                loss_p -= torch.log(torch.exp(torch.dot(logit[m], logit[p]) / tau))
                for a in a_m:
                    loss_p += torch.log(torch.exp(torch.dot(logit[m], logit[a]) / tau))

            loss_m += loss_p / pos_logit.size(0)

        return loss_m

    bs = logits1.size(0)

    # 计算对比损失
    loss = 0.0
    for t in range(bs):
        loss += pair_loss(logits1[t], logits2[t], labels1[t], labels2[t])

    loss = loss / bs
    return loss


def contrastive_loss_g(logits1, logits2, labels1, labels2, tau=2.0):
    def pair_loss(logit, label):
        bs = logit.size(0)
        loss_m = 0.0
        for m in range(bs):
            pos_mask = (label == label[m]) & (torch.arange(bs) != m)
            pos_logit = logit[pos_mask]
            neg_logit = logit[~pos_mask]

            pos_loss = -torch.log(torch.exp(pos_logit @ logit[m] / tau).sum())
            neg_loss = torch.log(torch.exp(neg_logit @ logit[m] / tau).sum())

            loss_m += pos_loss + neg_loss
        return loss_m / bs

    logits = torch.cat((logits1, logits2), dim=0)
    labels = torch.cat((labels1, labels2), dim=0)

    loss = pair_loss(logits, labels)
    return loss

def total_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5, tau=2.0, lambda_=0.2):
    rec_loss = recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega)
    clr_loss = contrastive_loss(logits1, logits2, labels1, labels2, tau)
    total_loss = rec_loss + lambda_ * clr_loss
    return total_loss

# bs = 4
# ll = 10
# cc = 5
# test_logits = torch.randn(bs, ll, cc)
# test_labels = torch.randint(bs, (bs, ll))
# test_logits1 = torch.randn(bs, ll, cc)
# test_labels1 = torch.randint(bs, (bs, ll))
# test_logits2 = torch.randn(bs, ll, cc)
# test_labels2 = torch.randint(bs, (bs, ll))
# contrastive_loss(test_logits1, test_logits2, test_labels1, test_labels2)
# contrastive_loss_g(test_logits1, test_logits2, test_labels1, test_labels2)

