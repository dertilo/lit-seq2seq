# -*- coding: utf-8 -*-

"""
Chatbot Tutorial
================
stolen from: https://github.com/pytorch/tutorials/blob/master/beginner_source/chatbot_tutorial.py
**Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple

import torch
import torch.nn as nn
from torch import optim
import random
import os
import itertools

from load_preprocess_data import (
    write_formatted_data,
    loadPrepareData,
    trimRareWords,
    PAD_token,
    EOS_token,
    MAX_LENGTH,
    SOS_token,
)
from models import EncoderRNN, LuongAttnDecoderRNN

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell movie-dialogs corpus"
data_dir = os.environ.get("DATA_DIR", "data")
corpus = os.path.join(data_dir, corpus_name)
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
write_formatted_data(corpus, datafile)

# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)


# Trim voc and pairs
MIN_COUNT = 3  # Minimum word count threshold for trimming
pairs = trimRareWords(voc, pairs, MIN_COUNT)


######################################################################
# Prepare Data for Models
# -----------------------


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(
    input_variable,
    lengths,
    target_variable,
    mask,
    max_target_len,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    batch_size,
    clip,
    teacher_forcing_ratio=1.0,
):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[: decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(
    model_name,
    voc,
    pairs,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    save_dir,
    n_iteration,
    batch_size,
    print_every,
    save_every,
    clip,
    corpus_name,
    checkpoint=None,
):

    training_batches = [
        batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iteration)
    ]

    start_iteration = 1
    print_loss = 0
    if checkpoint:
        start_iteration = checkpoint.iteration + 1  # TODO(tilo): WTF!

    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            batch_size,
            clip,
        )
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg
                )
            )
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(
                {
                    "iteration": iteration,
                    "en": encoder.state_dict(),
                    "de": decoder.state_dict(),
                    "en_opt": encoder_optimizer.state_dict(),
                    "de_opt": decoder_optimizer.state_dict(),
                    "loss": loss,
                    "voc_dict": voc.__dict__,
                    "embedding": embedding.state_dict(),
                },
                os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")),
            )


def main():

    # Configure models
    model_name = "cb_model"
    attn_model = "dot"
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None

    decoder, embedding, encoder = build_model(
        attn_model, decoder_n_layers, dropout, encoder_n_layers, hidden_size
    )

    assert decoder.n_layers == decoder_n_layers

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    clip = 50.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 400
    print_every = 1
    save_every = 500

    encoder.train()
    decoder.train()

    print("Building optimizers ...")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * decoder_learning_ratio
    )

    Checkpoint = namedtuple(
        "Checkpoint", "en de en_opt de_opt embedding voc_dict iteration"
    )

    if loadFilename is not None:
        checkpoint = Checkpoint(**torch.load(loadFilename))

        encoder.load_state_dict(checkpoint.en)
        decoder.load_state_dict(checkpoint.de)
        embedding.load_state_dict(checkpoint.embedding)

        encoder_optimizer.load_state_dict(checkpoint.en_opt)
        decoder_optimizer.load_state_dict(checkpoint.de_opt)
        voc.__dict__ = checkpoint.voc_dict
    else:
        checkpoint = None

    optimizers_to_cuda(decoder_optimizer, encoder_optimizer)

    trainIters(
        model_name,
        voc,
        pairs,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        embedding,
        save_dir,
        n_iteration,
        batch_size,
        print_every,
        save_every,
        clip,
        corpus_name,
        checkpoint,
    )


def optimizers_to_cuda(decoder_optimizer, encoder_optimizer):
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def build_model(attn_model, decoder_n_layers, dropout, encoder_n_layers, hidden_size):
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(
        attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
    )
    return decoder, embedding, encoder


if __name__ == "__main__":
    main()
