"""Seq2seq model for language models
"""

import numpy as np
import torch

import src.constants as C

import tensorflow as tf

from src.DecoderRNN import Decoder
from src.EncoderRNN import EncoderRNN, Encoder


class Seq2Seq(tf.keras.Model):

    def __init__(self, vocab_size, h_sizes, params):
        super(Seq2Seq, self).__init__()

        e_size, max_len, n_layers, dropout_p, batch_size, model_name, use_attention, attn_architecture = [params[i] for
                                                                                                          i in
                                                                                                          [
                                                                                                              C.EMBEDDING_DIM,
                                                                                                              C.OUTPUT_MAX_LEN,
                                                                                                              C.H_LAYERS,
                                                                                                              C.DROPOUT,
                                                                                                              C.BATCH_SIZE,
                                                                                                              C.MODEL_NAME,
                                                                                                              C.USE_ATTENTION,
                                                                                                              C.ATTENTION_ARCHITECTURE]]
        r_hsize, q_hsize, a_hsize = h_sizes

        self.use_attention = use_attention
        self.model_name = model_name
        # self.decoder = DecoderRNN(vocab_size=vocab_size, max_len=max_len, embedding_size=e_size, hidden_size=a_hsize,
        #                           n_layers=n_layers, dropout_p=dropout_p,
        #                           sos_id=C.SOS_INDEX, eos_id=C.EOS_INDEX, model_name=model_name,
        #                           use_attention=self.use_attention)
        self.tf_decoder = Decoder(vocab_size, e_size, a_hsize, max_len, batch_size, self.use_attention, model_name,
                                  C.EOS_INDEX, C.SOS_INDEX, attn_architecture)

        if model_name == C.LM_ANSWERS:
            self.question_encoder = None
        else:

            ################## Tensorflow ###################

            self.question_tf_encoder = Encoder(vocab_size, e_size, q_hsize, batch_size, dropout_p)
            # self.tf_decoder.set_embedding_weights(self.question_tf_encoder.embedding)

        if model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            self.reviews_encoder = EncoderRNN(vocab_size=vocab_size, max_len=max_len, embedding_size=e_size,
                                              hidden_size=r_hsize, n_layers=n_layers, dropout_p=dropout_p)
            self.decoder.embedding.weight = self.reviews_encoder.embedding.weight
        else:
            self.reviews_encoder = None

        if self.model_name == C.LM_QUESTION_ANSWERS:
            assert q_hsize == a_hsize
        if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            # TODO Fix this workaround
            if self.use_attention:
                assert a_hsize == q_hsize == r_hsize
            else:
                assert a_hsize == q_hsize + r_hsize

    def call(self,
             question_seqs,
             review_seqs,
             answer_seqs,
             target_seqs,
             teacher_forcing_ratio
             ):
        # print(question_seqs, review_seqs, answer_seqs, target_seqs)
        if self.model_name == C.LM_ANSWERS:
            d_hidden = None
            question_out = None
            review_outs = None
        elif self.model_name == C.LM_QUESTION_ANSWERS:
            # question_out, d_hidden = self.question_encoder(question_seqs)
            question_out, h_hidden, c_hidden = self.question_tf_encoder(question_seqs)
            d_hidden = (h_hidden, c_hidden)
            review_outs = None
        elif self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            question_out, question_hidden = self.question_encoder(question_seqs)
            reviews_encoder_outs = [self.reviews_encoder(seq) for seq in review_seqs]
            review_outs, review_hiddens = map(list, zip(*reviews_encoder_outs))

            # TODO Fix this workaround
            if self.use_attention:
                d_hidden = question_hidden
            else:
                reviews_hidden = list(map(_mean, zip(*review_hiddens)))
                # d_hidden = tuple(torch.cat([q_h, r_h], 2) for q_h, r_h in zip(question_hidden, reviews_hidden))
                d_hidden = tuple(tf.concat([q_h, r_h], 2) for q_h, r_h in zip(question_hidden, reviews_hidden))
        else:
            raise 'Unimplemented model: %s' % self.model_name

        if self.use_attention:
            d_out = (question_out, review_outs)
        else:
            d_out = None
        # return self.decoder(inputs=target_seqs, encoder_hidden=d_hidden,
        #                     encoder_outputs=d_out, teacher_forcing_ratio=teacher_forcing_ratio)

        return self.tf_decoder(inputs=target_seqs, encoder_hidden=d_hidden,
                               encoder_outputs=d_out, teacher_forcing_ratio=teacher_forcing_ratio)

    def trainable_variables(self):
        return self.question_tf_encoder.trainable_weights + self.tf_decoder.trainable_weights


def _mean(vars):
    return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)

# def _cat_hidden(h1, h2):
#     return (torch.cat([h])
