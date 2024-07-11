import time
import torch
import torch.nn.functional as F

# calculate recognition loss with iteration
def recognition_loss_o(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5):
    loss_o, loss1, loss2, bs = 0.0, 0.0, 0.0, logits_o.size(0)
    for i in range(bs):
        loss_o += F.cross_entropy(logits_o[i], labels_o[i], reduction='sum')
        loss1 += F.cross_entropy(logits1[i], labels1[i], reduction='sum')
        loss2 += F.cross_entropy(logits2[i], labels2[i], reduction='sum')
    rec_loss = (loss_o + omega * (loss1 + loss2)) / bs
    return rec_loss

# calculate recognition loss with cross_entropy directly
def recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5):
    return (F.cross_entropy(logits_o.transpose(1, 2), labels_o, reduction='sum') + omega * (F.cross_entropy(logits1.transpose(1, 2), labels1, reduction='sum')) + F.cross_entropy(logits2.transpose(1, 2), labels2, reduction='sum')) / logits_o.size(0)

# calculate contrastive loss with iteration
def contrastive_loss_o(logits1, logits2, labels1, labels2, tau=2.0):
    device = logits1.device
    def pair_loss(logit1, logit2, label1, label2):
        logit = torch.cat((logit1, logit2), dim=0)
        label = torch.cat((label1, label2), dim=0)
        loss_m = 0.0
        for m in range(label.size(0)):
            pos_indices = torch.where((label == label[m]) & (torch.arange(label.size(0), device=device) != m))[0]
            pos_logit = logit[pos_indices]
            if pos_logit.size(0) == 0:
                continue
            a_m = torch.where(torch.arange(label.size(0), device=device) != m)[0]
            similarity = (logit[m] @ logit[a_m].T) / tau
            b_max = similarity.max()
            loss_p = 0.0
            for p in pos_indices:
                loss_p -= torch.dot(logit[m], logit[p]) / tau - b_max
                esum = 0
                for a in a_m:
                    esum += torch.exp(torch.dot(logit[m], logit[a]) / tau - b_max)
                loss_p += torch.log(esum)
            loss_m += loss_p / pos_logit.size(0)
        return loss_m

    bs = logits1.size(0)
    loss = 0.0
    for t in range(bs):
        loss += pair_loss(logits1[t], logits2[t], labels1[t], labels2[t])

    loss = loss / bs
    return loss

# calculate contrastive loss with matrix operation
def contrastive_loss(logits1, logits2, labels1, labels2, tau=2.0):
    device = logits1.device
    bs = logits1.size(0)

    labels = torch.cat((labels1, labels2), dim=1)
    logits = torch.cat((logits1, logits2), dim=1)
    # calculate similarity
    sim = torch.bmm(logits, logits.transpose(2, 1)) / tau

    bs, seq_len = labels.shape
    
    labels_expanded = labels.unsqueeze(2).expand(bs, seq_len, seq_len)
    labels_expanded_T = labels.unsqueeze(1).expand(bs, seq_len, seq_len)

    am = ~torch.eye(seq_len, dtype=bool, device=device).unsqueeze(0).expand(bs, -1, -1)
    pm = (labels_expanded == labels_expanded_T) & am
    # num of positive pairs for m
    nnz = pm.sum(dim=1).unsqueeze(2)
    # mask m without P(m)
    am_nz = am & nnz.expand(bs, seq_len, seq_len).bool()
    sim = torch.masked_fill(sim, ~am_nz, 0)
    # prevent overflow
    sim = sim - sim.max(dim=2, keepdim=True)[0]

    sim_exp = torch.exp(sim)
    sim_p = torch.masked_fill(sim, ~pm, 0)  # similarity of positive pairs
    sim_a = torch.masked_fill(sim_exp, ~am, 0)  # similarity of all pairs (exp)

    # sum of positive pairs
    sum_p = torch.where(nnz != 0, sim_p / nnz, 0).sum()
    # sum of all pairs (sum(exp))
    sum_a_exp = torch.masked_fill(sim_a, ~am_nz, 0).sum(dim=2)
    # sum of all pairs (sum(log(sum(exp))))
    sum_a = torch.where(sum_a_exp != 0, torch.log(sum_a_exp), 0).sum()
    loss = (-sum_p + sum_a) / bs
    return loss

def total_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5, tau=2.0, lambda_=0.2):
    rec_loss = recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega)
    clr_loss = contrastive_loss(logits1, logits2, labels1, labels2, tau)
    total_loss = rec_loss + lambda_ * clr_loss
    return total_loss

# Measure execution time
if __name__ == '__main__':
    bs = 2
    ll = 4
    cc = 2
    test_logits_o = torch.randn(bs, int(ll/2), cc)
    test_labels_o = torch.randint(cc, (bs, int(ll/2)))
    test_logits1 = torch.randn(bs, ll, cc)
    test_labels1 = torch.randint(cc, (bs, ll))
    test_logits2 = torch.randn(bs, ll, cc)
    test_labels2 = torch.randint(cc, (bs, ll))
    
    start = time.time()
    rec_loss_o = recognition_loss_o(test_logits_o, test_logits1, test_logits2, test_labels_o, test_labels1, test_labels2)
    end = time.time()
    print(f'recognition_loss_o time used: {end - start}, loss: {rec_loss_o}')
    
    start = time.time()
    rec_loss = recognition_loss(test_logits_o, test_logits1, test_logits2, test_labels_o, test_labels1, test_labels2)
    end = time.time()
    print(f'recognition_loss time used: {end - start}, loss: {rec_loss}')
    assert rec_loss_o.item() == rec_loss.item()

    test_logits1 = torch.tensor([[[1.0, 2.0], [2.0, 3.0]], 
                            [[3.0, 4.0], [7.0, 5.0]]])
    test_logits2 = torch.tensor([[[2.0, 7.0], [1.0, 4.0]], 
                            [[3.0, 5.0], [3.0, 4.0]]])
    test_labels1 = torch.tensor([[0, 1], [4, 3]])
    test_labels2 = torch.tensor([[1, 5], [4, 4]])
    clr_loss_expected = 12.54502

    start = time.time()
    loss = contrastive_loss_o(test_logits1, test_logits2, test_labels1, test_labels2)
    end = time.time()
    print(f'contrastive_loss_o time used: {end - start}, loss: {loss}')
    assert abs(loss.item() - clr_loss_expected) < 5e-6

    start = time.time()
    loss = contrastive_loss(test_logits1, test_logits2, test_labels1, test_labels2)
    end = time.time()
    print(f'contrastive_loss time used: {end - start}, loss: {loss}')
    assert abs(loss.item() - clr_loss_expected) < 5e-6
