import time
import torch
import torch.nn.functional as F

def recognition_loss(logits_o, logits1, logits2, labels_o, labels1, labels2, omega=0.5):
    loss_o, loss1, loss2, bs = 0.0, 0.0, 0.0, logits_o.size(0)
    for i in range(bs):
        loss_o += F.cross_entropy(logits_o[i], labels_o[i])
        loss1 += F.cross_entropy(logits1[i], labels1[i])
        loss2 += F.cross_entropy(logits2[i], labels2[i])
    rec_loss = (loss_o + omega * (loss1 + loss2)) / bs
    return rec_loss

def contrastive_loss_o(logits1, logits2, labels1, labels2, tau=2.0):
    device = logits1.device
    def pair_loss(logit1, logit2, label1, label2):
        logit = torch.cat((logit1, logit2), dim=0)
        label = torch.cat((label1, label2), dim=0)
        loss_m = 0.0
        #TODO: 冗长
        for m in range(label.size(0)):
            pos_indices = torch.where((label == label[m]) & (torch.arange(label.size(0), device=device) != m))[0]
            pos_logit = logit[pos_indices]
            if pos_logit.size(0) == 0:
                continue
            a_m = torch.where(torch.arange(label.size(0), device=device) != m)[0]

            loss_p = 0.0
            for p in pos_indices:
                loss_p -= torch.log(torch.exp(torch.dot(logit[m], logit[p]) / tau))
                esum = 0
                for a in a_m:
                    esum += torch.exp(torch.dot(logit[m], logit[a]) / tau)
                    # loss_p += torch.log(torch.exp(torch.dot(logit[m], logit[a]) / tau))
                loss_p += torch.log(esum)
            loss_m += loss_p / pos_logit.size(0)

        return loss_m

    bs = logits1.size(0)

    loss = 0.0
    for t in range(bs):
        loss += pair_loss(logits1[t], logits2[t], labels1[t], labels2[t])

    loss = loss / bs
    return loss

def contrastive_loss_o1(logits1, logits2, labels1, labels2, tau=2.0):
    device = logits1.device
    bs = logits1.size(0)

    loss = 0.0
    labels = torch.cat((labels1, labels2), dim=1)
    logits = torch.cat((logits1, logits2), dim=1)
    similarity = torch.bmm(logits, logits.transpose(2, 1))
    assert similarity.shape == (bs, logits.size(1), logits.size(1))
    mat = torch.log(torch.exp(similarity / tau))
    a1, a2 = 0.0, 0.0
    for t in range(bs):
        label = labels[t]
        for m in range(label.size(0)):
            pos_indices = torch.where((label == label[m]) & (torch.arange(label.size(0), device=device) != m))[0]
            if pos_indices.size(0) == 0:
                continue
            a_m = torch.where(torch.arange(label.size(0), device=device) != m)[0]

            a1 -= mat[t][m][pos_indices].sum() / pos_indices.size(0)
            a2 += mat[t][m][a_m].sum()

            loss += (- mat[t][m][pos_indices].sum() + mat[t][m][a_m].sum() * pos_indices.size(0)) / pos_indices.size(0)

    loss = loss / bs
    return loss

def contrastive_loss(logits1, logits2, labels1, labels2, tau=2.0):
    def standardize(tensor):
        return (tensor - tensor.mean()) / tensor.std()

    logits1 = standardize(logits1)
    logits2 = standardize(logits2)

    device = logits1.device
    bs = logits1.size(0)

    labels = torch.cat((labels1, labels2), dim=1)
    logits = torch.cat((logits1, logits2), dim=1)

    bs, seq_len = labels.shape
    
    labels_expanded = labels.unsqueeze(2).expand(bs, seq_len, seq_len)
    labels_expanded_T = labels.unsqueeze(1).expand(bs, seq_len, seq_len)

    am = ~torch.eye(seq_len, dtype=bool, device=device).unsqueeze(0).expand(bs, -1, -1)
    pm = (labels_expanded == labels_expanded_T) & am
    
    similarity = torch.bmm(logits, logits.transpose(2, 1))
    m1 = similarity / tau

    #TODO: overflow
    # m1 = m1 / 100

    me = torch.exp(m1)

    nnz = pm.sum(dim=1).unsqueeze(2)
    p = torch.masked_fill(m1, ~pm, 0)
    a = torch.masked_fill(me, ~am, 0)
    a1 = torch.where(nnz != 0, p / nnz, torch.zeros_like(p)).sum()
    # TODO: sum may be zero
    a2 = torch.log(torch.masked_fill(a, ~nnz.expand(bs, seq_len, seq_len).bool(), 0).sum(dim=1)).sum()
    return (-a1 + a2) / bs

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
    test_labels = torch.randint(bs, (bs, ll))
    test_logits1 = torch.randn(bs, ll, cc)
    test_labels1 = torch.randint(bs, (bs, ll))
    test_logits2 = torch.randn(bs, ll, cc)
    test_labels2 = torch.randint(bs, (bs, ll))

    test_labels = torch.randint(bs, (bs, ll))
    test_logits1 = torch.ones(bs, ll, cc) / 2
    test_labels1 = torch.randint(bs, (bs, ll))
    test_logits2 = torch.ones(bs, ll, cc) / 2
    test_labels2 = torch.randint(bs, (bs, ll))


    test_logits1 = torch.tensor([[[1.0, 2.0], [15.0, 14.0]], 
                            [[3.0, 4.0], [7.0, 8.0]]])
    test_logits2 = torch.tensor([[[15.0, 16.0], [11.0, 12.0]], 
                            [[13.0, 14.0], [30.0, 4.0]]])
    test_labels1 = torch.tensor([[0, 1], [4, 3]])
    test_labels2 = torch.tensor([[1, 5], [6, 4]])


    start = time.time()
    loss = contrastive_loss_o(test_logits1, test_logits2, test_labels1, test_labels2)
    end = time.time()
    print(f'contrastive_loss_o time used: {end - start}, loss: {loss}')

    # start = time.time()
    # loss = contrastive_loss_o1(test_logits1, test_logits2, test_labels1, test_labels2)
    # end = time.time()
    # print(f'contrastive_loss_o1 time used: {end - start}, loss: {loss}')

    start = time.time()
    loss = contrastive_loss(test_logits1, test_logits2, test_labels1, test_labels2)
    end = time.time()
    print(f'contrastive_loss time used: {end - start}, loss: {loss}')
