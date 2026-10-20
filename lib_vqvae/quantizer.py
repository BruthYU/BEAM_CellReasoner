import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n : the number of codebooks an embedding is split into
    - n_e : the number of codebooks
    - e_dim : the dim of each codebook
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n, n_e, e_dim, beta, gamma, distance='cosine', 
                 anchor='probrandom', first_batch=False, contras_loss=False, training=True):
        super(VectorQuantizer, self).__init__()
        self.n = n
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gamma = gamma
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False
        self.training = training
    
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.pool = FeaturePool(self.n_e, self.e_dim)
        self.register_buffer("embed_prob", torch.zeros(self.n_e))


    def forward(self, z, mode, device, training=True):
        """
        the size of z is  (batch_size, n*e_dim)
        """
        if mode=="cvqvae":
            # breakpoint()
            z = z.view(z.size(0), self.n, self.e_dim).to(device)
            # flatten z
            z_flattened = z.view(-1, self.e_dim).to(device)

            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            # d[i,j] is the square of the distance between z_flattened[i] and the j-th embedding
            if self.distance == 'l2':
                d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                    torch.sum(self.embedding.weight ** 2, dim=1) + \
                    2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
            elif self.distance == 'cosine':
                # cosine distances from z to embeddings e_j 
                normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
                normed_codebook = F.normalize(self.embedding.weight, dim=1)
                d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

            # find closest encodings
            # the min indice of each row
            sort_distance, indices = d.sort(dim=1)
            min_encoding_indices = indices[:,-1]
            # one-hot, if z_flattened[i] is the closest embedding to the j-th embedding, then min_encodings[i,j] is 1 and the other positions are 0
            min_encodings = torch.zeros(min_encoding_indices.unsqueeze(1).shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)

            # get quantized latent vectors
            # matrix multiplication of min_encodings with all embedding vectors
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

            # compute loss for embedding
            loss1 = torch.mean((z_q.detach()-z)**2) # e_latent_loss
            loss2 = torch.mean((z_q-z.detach())**2) # q_latent_loss
            loss = self.gamma * loss1 + self.beta * loss2

            # preserve gradients
            # z_q size (batch_size, n, e_dim)
            z_q = z + (z_q - z).detach()

            # perplexity
            # obtained by the entropy of the cluster distribution obtained by min_encodings, and its function is to measure the degree of dispersion of the model
            e_mean = torch.mean(min_encodings, dim=0)
            # print(e_mean)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            # reshape for decoder
            z_output = z_q.view(z_q.size(0), -1)

            # online clustered reinitialisation for unoptimized points
            if training:
                # calculate the average usage of code entries
                self.embed_prob.mul_(self.decay).add_(e_mean, alpha= 1 - self.decay)
                # running average updates
                if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                    # closest sampling
                    if self.anchor == 'closest':
                        sort_distance, indices = d.sort(dim=0)
                        random_feat = z_flattened.detach()[indices[-1,:]]
                    # feature pool based random sampling
                    elif self.anchor == 'random':
                        random_feat = self.pool.query(z_flattened.detach())
                    # probabilitical based random sampling
                    elif self.anchor == 'probrandom':
                        norm_distance = F.softmax(d.t(), dim=1)
                        prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                        random_feat = z_flattened.detach()[prob]
                    # decay parameter based on the average usage
                    decay = torch.exp(-(self.embed_prob*self.n_e*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.e_dim)
                    self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                    if self.first_batch:
                        self.init = True
                # contrastive loss
                if self.contras_loss:
                    sort_distance, indices = d.sort(dim=0)
                    dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.n_e)):,:].mean(dim=0, keepdim=True)
                    dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                    dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                    contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=device))
                    loss +=  contra_loss

            return loss, loss1, loss2, z_q, z_output, perplexity, min_encodings, min_encoding_indices
        
        
        elif mode=="vqvae":
            # breakpoint()
            z = z.view(z.size(0), self.n, self.e_dim)
            # flatten z
            z_flattened = z.view(-1, self.e_dim)

            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            # d[i,j] is the square of the distance between z_flattened[i] and the j-th embedding
            d = torch.cdist(z_flattened, self.embedding.weight) # 欧氏距离来找

            # find closest encodings
            # the min indice of each row
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
            # one-hot, if z_flattened[i] is the closest embedding to the j-th embedding, then min_encodings[i,j] is 1 and the other positions are 0
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(device)
            min_encodings.scatter_(1, min_encoding_indices, 1)

            # get quantized latent vectors
            # matrix multiplication of min_encodings with all embedding vectors
            z_q = self.embedding(min_encoding_indices.squeeze(1)).view(z.shape)

            # compute loss for embedding
            loss1 = torch.mean((z_q.detach() - z)**2)  
            loss2 = torch.mean((z_q - z.detach()) ** 2)  
            loss = self.gamma * loss1 + self.beta * loss2

            # preserve gradients
            z_q = z + (z_q - z).detach()

            # perplexity
            # obtained by the entropy of the cluster distribution obtained by min_encodings, and its function is to measure the degree of dispersion of the model
            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            # reshape for decoder
            z_output = z_q.view(z_q.size(0), -1)

            return loss, loss1, loss2, z_q, z_output, perplexity, min_encodings, min_encoding_indices
    
    
    
class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features