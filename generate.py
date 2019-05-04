import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from model_pytorch import LMModel, load_openai_pretrained_model
from text_utils import TextEncoder

encoded_words = []


def create_dictionary(encoder):
    words = ['go', 'to', 'agent', 'red', 'green', 'blue', 'landmark',
             'circle', 'triangle', 'continue', 'next', 'ahead', 'done',
             'good', 'stay', 'there']
    for w in words:
        encoded_words.append(encoder.encode([w,]))


def make_batch(X):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch

def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--gen_len', type=int, default=20) #how long the sentance Should\can be
    parser.add_argument('--topk', type=int, default=10) ## changed to 5 from 10


    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    submit = args.submit
    dataset = args.dataset
    n_ctx = args.n_ctx
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir #I think this is the location of the vocablery?
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    n_special = 0   # XD: useless for language modeling task
    vocab = n_vocab + n_special + n_ctx #the size of the vocabalery - in this case it is letters - so I don;t think its what we need

    lm_model = LMModel(args, vocab, n_ctx, return_probs=True)
    load_openai_pretrained_model(lm_model.transformer, n_ctx=n_ctx, n_special=n_special)
    lm_model.to(device)

    lm_model.eval()
    #till now it loaded the previuos model and the vocabalery that will be used
    text = input('Input some beginning words:') #why we need this?

    create_dictionary(text_encoder)

    while text != 'q':
        X = text_encoder.encode([text,])
        XMB = make_batch(X)

        for _ in range(args.gen_len):
            lm_probs = lm_model(XMB) #the porbability of each word in the vocabalry?
            if args.topk == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                # prob = 0
                # choosen_word = 0
                tmp = []
                for index in encoded_words:
                    tmp += [lm_probs[:, -1, :][:, index].item()] #ToDo
                    # if tmp >= prob:
                    #     prob = tmp
                    #     choosen_word = index
                next_idx = torch.Tensor([tmp])
                values, indices = next_idx.topk(args.topk)
                # values, indices = lm_probs[:, -1, :].topk(args.topk)
                next_idx = indices.gather(-1, torch.multinomial(values, 1)) #random from the top KW. here is where the new word is set. form the letters that were set form the indices above
            next_token = text_encoder.decoder[encoded_words[next_idx.item()][0][0]].replace('</w>', '') #here its wehre it is translated to a word
            # next_token = text_encoder.decoder[next_idx.item()].replace('</w>', '') #here its wehre it is translated to a word
            print(next_token, end=' ')
            XMB = append_batch(XMB, next_idx)

        print()
