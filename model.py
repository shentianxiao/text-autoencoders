import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noise import noisy

class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class AE(TextModel):
    """Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
        self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg == 'greedy' or alg == 'sample'
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            else:
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
        return torch.cat(sents)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = AE.reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='sum') / targets.size(1)

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets):
        _, _, _, logits = self(inputs)
        return {'rec': self.loss_rec(logits, targets)}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()


class VAE(AE):
    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    @staticmethod
    def loss_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    def autoenc(self, inputs, targets):
        mu, logvar, _, logits = self(inputs)
        return {'rec': self.loss_rec(logits, targets),
                'kl': VAE.loss_kl(mu, logvar)}

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']


class AAE(AE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), nn.Sigmoid())
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy(self.D(zn), ones)
        loss_g = F.binary_cross_entropy(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
            self.args.lambda_p * losses['|lvar|']

    def autoenc(self, inputs, targets):
        noisy_inputs = noisy(self.vocab, inputs, *self.args.noise)
        mu, logvar = self.encode(noisy_inputs)
        z = AE.reparameterize(mu, logvar)
        logits, _ = self.decode(z, inputs)
        loss_d, adv = self.loss_adv(z)
        return {'rec': self.loss_rec(logits, targets),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, losses):
        super().step(losses)

        self.optD.zero_grad()
        losses['loss_d'].backward()
        self.optD.step()
