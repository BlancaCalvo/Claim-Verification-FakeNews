import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU


class SelfAttentionLayer(nn.Module):
    def __init__(self, nhid, nins, nclaim=1):
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid
        self.nins = nins
        self.nclaim = nclaim
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ReLU(True),
            Linear(64, 1)
        )

    def forward(self, inputs, index, claims):
        tmp = None
        if index > -1: # comes here when inside AttentionLayer
            idx = torch.LongTensor([index]).cuda()
            own = torch.index_select(inputs, 1, idx)
            own = own.repeat(1, self.nins, 1)
            tmp = torch.cat((own, inputs), 2)
        else: # comes here in the aggregation part
            print(claims.shape) #[batch_size, 5, 768]
            print(inputs.shape) #[batch_size, 25, 768]
            if self.nclaim == 1:
                #claims = claims.unsqueeze(1) #adds a dimension of 1 to the claims (it needed it before, I think not anymore)
                claims = claims.repeat(1, self.nins, 1)
            else:
                claims = claims.repeat(1, int(self.nins/self.nclaim), 1) # repeats the claim vector as many time as evidences there are, so that claims and inputs can be concatenated
                #each argument is the repetitions of each axis, we now repeat axis 1 5 times because we want it to have [256, 25, 768]
            print(claims.shape)
            tmp = torch.cat((claims, inputs), 2)
            print(tmp.shape)
        # before
        attention = self.project(tmp) # problema cpu aquí quan fa servir claims_attention
        weights = F.softmax(attention.squeeze(-1), dim=1)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        print(outputs.shape)
        return outputs


class AttentionLayer(nn.Module):
    def __init__(self, nins, nhid):
        super(AttentionLayer, self).__init__()
        self.nins = nins
        self.attentions = [SelfAttentionLayer(nhid=nhid * 2, nins=nins) for _ in range(nins)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, inputs):
        # outputs = torch.cat([att(inputs) for att in self.attentions], dim=1)
        outputs = torch.cat([self.attentions[i](inputs, i, None) for i in range(self.nins)], dim=1)
        outputs = outputs.view(inputs.shape) # reshape to input shape
        return outputs


class GEAR(nn.Module):
    def __init__(self, nfeat, nins, nclaim, nclass, nlayer, pool): #nins és número de evidències
        super(GEAR, self).__init__()
        self.nlayer = nlayer
        self.nclaim = nclaim

        self.attentions = [AttentionLayer(nins, nfeat) for _ in range(nlayer)]
        self.claim_attentions = [AttentionLayer(nclaim, nfeat) for _ in range(nlayer)]
        self.batch_norms = [BatchNorm1d(nins) for _ in range(nlayer)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.pool = pool
        if pool == 'att':
            self.aggregate = SelfAttentionLayer(nfeat * 2, nins, nclaim)
        self.index = torch.LongTensor([0]).cuda()

        self.weight = nn.Parameter(torch.FloatTensor(nfeat, nclass))
        self.bias = nn.Parameter(torch.FloatTensor(nclass))

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, claims):
        #for i in range(self.nlayer): # maybe we could try to do an ablation test removing this part?
        #    inputs = self.attentions[i](inputs) # between evidences attention coefficients

        # if we add claim a graph features we have to add here too the attention coefficients for claims
        #if self.nclaim > 1:
        #    for i in range(self.nlayer):
        #        claims = self.claim_attentions[i](claims)

        if self.pool == 'att':
            inputs = self.aggregate(inputs, -1, claims) #attention coefficient of evidence in relation to claim
        # if self.pool == 'max': #maybe try this in the future, but it's not really the point of the thesis
        #     inputs = torch.max(inputs, dim=1)[0]
        # if self.pool == 'mean':
        #     inputs = torch.mean(inputs, dim=1)
        # if self.pool == 'top':
        #     inputs = torch.index_select(inputs, 1, self.index).squeeze()
        # if self.pool == 'sum':
        #     inputs = inputs.sum(dim=1)

        inputs = F.relu(torch.mm(inputs, self.weight) + self.bias)
        print(inputs.shape)
        return F.log_softmax(inputs, dim=1)
