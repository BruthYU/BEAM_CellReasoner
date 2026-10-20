import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import lib_vqvae.utils as utils
from lib_vqvae.model import Encoder, Decoder
from lib_vqvae.dataset import *
from lib_vqvae.quantizer import VectorQuantizer

import wandb


class VQVAE(nn.Module):
    def __init__(self, data_dim, n=4, codebook_dim=16, n_codebooks=512, encoder_hidden_dim=[1600,1024,800], decoder_hidden_dim=[800,1024,1600], beta=1, gamma=1, dropout_rate=0.2):
        super(VQVAE, self).__init__()
        self.data_dim = data_dim
        self.n = n
        self.codebook_dim = codebook_dim
        self.n_codebooks = n_codebooks
        self.dropout_rate=dropout_rate
        
        self.encoder = Encoder(self.data_dim, self.n, self.codebook_dim, encoder_hidden_dim, self.dropout_rate)
        self.vector_quantization = VectorQuantizer(self.n, self.n_codebooks, self.codebook_dim, beta, gamma)
        self.decoder = Decoder(self.data_dim, self.n, self.codebook_dim, decoder_hidden_dim, self.dropout_rate)
        
        
    def forward(self, x, mode, device, verbose=False):
        z_e = self.encoder(x)
        embedding_loss, loss1, loss2, z_q, z_output, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(z_e, mode, device)
        x_hat = self.decoder(z_output)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('vq data shape:', z_q.shape)
            print('before decoded data shape', z_output.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices
    
    
    def eval_step(self, eval_loader, mode, device):
        # - eval step
        # self.to(device)
        self.eval()
        step_total_loss = 0
        
        with torch.no_grad():
            for i, x in enumerate(eval_loader):
                x = x.float().to(device)
                
                embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices=self(x, mode, device)
                       
                recon_loss = F.mse_loss(x_hat, x)
                loss = recon_loss + embedding_loss
                
                step_total_loss += loss.item()
                break
        
        return step_total_loss

    
    def fit(self, args, dataloader, eval_adata, eval_loader, mode, device, max_iter=300, lr=3e-4, weight_decay=5e-4, log_val=10, save=True):
        print("device:", device)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
        
        wandb.watch(self, log="all")
        
        results = {
        'n_updates': [],
        'recon_loss': [],
        'embedding_loss': [],
        'train_loss': [],
        'perplexities': [],
        'eval_loss': [],
                    }
        
        totallen = len(dataloader)
        
        for epoch in range(max_iter):
            # self.train()
            
            epoch_reconloss, epoch_embeddloss, epoch_perplexities = 0, 0, 0
            
            tk0 = tqdm(enumerate(dataloader), total=totallen, leave=False, desc='training epoch: {}'.format(epoch+1))
            for i, x in tk0:
                x = x.float().to(device)
                # print("x", x)
                optimizer.zero_grad()

                embedding_loss, loss1, loss2, x_hat, z_q, z_output, perplexity, min_encodings, min_encoding_indices=self(x, mode, device)
                recon_loss = F.mse_loss(x_hat, x)
                
                loss = recon_loss + embedding_loss

                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip
                optimizer.step()
                
                epoch_reconloss += recon_loss.item()
                epoch_embeddloss += embedding_loss.item()
                epoch_perplexities += perplexity.item()
                
                # tk0.set_postfix_str('recon_loss={:.8f} embedding_loss={:.8f} perplexities={:.8f}'.format(
                #        recon_loss.cpu().detach().numpy(), embedding_loss.cpu().detach().numpy(), perplexity.cpu().detach().numpy()))
                tk0.update(1)
                
                if (i+1) % 100 == 0:
                    eval_loss = self.eval_step(eval_loader, mode, device)
                    results["eval_loss"].append(eval_loss)
                    wandb.log({"eval_loss": eval_loss})
                    self.train()
    

            if (epoch+1) % args.log_val == 0:
                """
                save model and print values
                """
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    self, eval_adata, eval_loader, results, hyperparameters, epoch+1, args)
                self.train()

            print('epoch:' + str(epoch+1))
            print('recon_loss={:.10f} embedding_loss={:.10f} perplexities={:.10f}'.format(epoch_reconloss/totallen, epoch_embeddloss/totallen, epoch_perplexities/totallen))
            
            results["recon_loss"].append(epoch_reconloss/totallen)
            results["embedding_loss"].append(epoch_embeddloss/totallen)
            results["perplexities"].append(epoch_perplexities/totallen)
            results["train_loss"].append((epoch_reconloss+epoch_embeddloss)/totallen)
            results["n_updates"].append(epoch+1)
            
            wandb.log(
                {
                    "epoch": epoch+1,
                    "recon_loss": recon_loss,
                    "embedding_loss": embedding_loss,
                    "loss_1": loss1,
                    "loss_2": loss2,
                    "perplexities": perplexity
                })
            
            scheduler.step(results["eval_loss"][-1])

        return results
                                       

    def encode_batch(self, dataloader, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()
            
        zq = []
        zoutput = []
        reconx = []
        for x in dataloader:
            x = x.float().to(device)
            _, _, _, x_hat, z_q, z_output, _, _, _ = self(x, mode, device)
            zq.append(z_q.detach().cpu())
            zoutput.append(z_output.detach().cpu())
            reconx.append(x_hat.cpu().detach().data)

        zq = torch.cat(zq)
        zoutput = torch.cat(zoutput)
        reconx = torch.cat(reconx)

        return zq, zoutput, reconx    
    
    
    def get_codebook_index(self, dataloader, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()
            
        indices = []
        for x in dataloader:
            x = x.float().to(device)
            _, _, _, _, _, _, _, min_encodings, min_encoding_indices = self(x, mode, device)
            
            min_encoding_indices = min_encoding_indices.view(x.shape[0], -1)
            indices.append(min_encoding_indices.detach().cpu())

        indices = torch.cat(indices)

        return indices   
    

    def get_adata_codebook_index(self, adata, mode, device='cuda', eval=True, transforms=None):
        if eval:
            self.eval()
        else:
            self.train()
        
        batch_size = 5000
        _, dataloader = create_dataset_loader(
            adata=adata,
            batch_size=batch_size)
        
        indices = []
        zoutput = []
        reconx = []
        
        for x in dataloader:
            x = x.float().to(device)
            _, _, _, x_hat, _, z_output, _, min_encodings, min_encoding_indices = self(x, mode, device)
            
            min_encoding_indices = min_encoding_indices.view(x.shape[0], -1)
            indices.append(min_encoding_indices.detach().cpu())
            zoutput.append(z_output.detach().cpu())
            reconx.append(x_hat.cpu().detach().data)

        indices = torch.cat(indices)
        zoutput = torch.cat(zoutput)
        reconx = torch.cat(reconx)
        
        adata.obsm['code_index'] = indices.numpy()
        adata.obsm['latent'] = zoutput.numpy()
        adata.obsm['recon_x'] = reconx.numpy()

        return adata