import argparse
import os
import torch
import numpy as np

from vocab import Vocab
from model import DAE, VAE, AAE
from utils import *
from batchify import get_batches
from train import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--output', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
                    help='path to data file')

parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
                    help='max sequence length')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate on data file')
parser.add_argument('--reconstruct', action='store_true',
                    help='reconstruct data file')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences from prior')
parser.add_argument('--arithmetic', action='store_true',
                    help='compute vector offset avg(b)-avg(a) and apply to c')
parser.add_argument('--interpolate', action='store_true',
                    help='interpolate between pairs of sentences')
parser.add_argument('--n', type=int, default=5, metavar='N',
                    help='num of sentences to generate for sample/interpolate')
parser.add_argument('--k', type=float, default=1, metavar='R',
                    help='k * offset for vector arithmetic')

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
args = parser.parse_args()

def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if args.enc == 'mu':
            zi = mu
        else:
            zi = AE.reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

if __name__ == '__main__':
    vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
    set_seed(args.seed)
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = get_model(os.path.join(args.checkpoint, 'model.pt'))

    if args.evaluate:
        sents = load_sent(args.data)
        batches, _ = get_batches(sents, vocab, args.batch_size, device)
        meters = evaluate(model, batches)
        print(' '.join(['{} {:.2f},'.format(k, meter.avg)
            for k, meter in meters.items()]))

    if args.sample:
        z = np.random.normal(size=(args.n, model.args.dim_z)).astype('f')
        sents = decode(z)
        write_sent(sents, os.path.join(args.checkpoint, args.output))

    if args.reconstruct:
        sents = load_sent(args.data)
        z = encode(sents)
        sents_rec = decode(z)
        write_sent(sents_rec, os.path.join(args.checkpoint, args.output))

    if args.arithmetic:
        fa, fb, fc = args.data.split(',')
        sa, sb, sc = load_sent(fa), load_sent(fb), load_sent(fc)
        za, zb, zc = encode(sa), encode(sb), encode(sc)
        zd = zc + args.k * (zb.mean(axis=0) - za.mean(axis=0))
        sd = decode(zd)
        write_sent(sd, os.path.join(args.checkpoint, args.output))

    if args.interpolate:
        f1, f2 = args.data.split(',')
        s1, s2 = load_sent(f1), load_sent(f2)
        z1, z2 = encode(s1), encode(s2)
        zi = [interpolate(z1_, z2_, args.n) for z1_, z2_ in zip(z1, z2)]
        zi = np.concatenate(zi, axis=0)
        si = decode(zi)
        si = list(zip(*[iter(si)]*(args.n)))
        write_doc(si, os.path.join(args.checkpoint, args.output))
