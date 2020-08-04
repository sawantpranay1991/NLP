"""Class for loss computation
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import tensorflow as tf
import src.constants as C


class Loss:

    def __init__(self):
        self.loss_type = C.WORD_LOSS
        # self.batch_size = batch_size

        if self.loss_type not in {C.WORD_LOSS, C.SENTENCE_LOSS}:
            raise 'Unimplemented Loss Type: %s' % self.loss_type

        self.criterion = nn.NLLLoss(ignore_index=C.PAD_INDEX, size_average=False)
        if C.USE_CUDA:
            self.criterion.cuda()

    def reset(self):
        self.total_num_tokens = 0
        self.total_num_sentences = 0
        self.total_num_batches = 0
        self.total_loss = 0

    def loss_func(self, logits, targets):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # mask = tf.math.logical_not(tf.math.equal(targets, 0))
        # mask = tf.cast(mask, dtype=tf.int64)
        # loss = crossentropy(targets, logits, sample_weight=mask)
        loss = crossentropy(targets, logits)

        return loss

    # [256, 101, 6215]
    def eval_batch_loss_for_tf(self, logits, targets):
        batch_num_sentences = targets.shape[0]
        batch_num_tokens = np.sum((tf.math.not_equal(targets, C.PAD_INDEX)).numpy())
        logits = tf.stack(logits, 1)
        targets = targets[:, 1:]
        batch_loss = self.loss_func(logits, targets)

        print("Batch loss", batch_loss)

        self.total_num_sentences += batch_num_sentences
        self.total_num_tokens += batch_num_tokens
        self.total_num_batches += 1

        self.total_loss += batch_loss.numpy()

        if self.loss_type == C.WORD_LOSS:
            loss = batch_loss / float(batch_num_tokens)
        elif self.loss_type == C.SENTENCE_LOSS:
            loss = batch_loss / float(batch_num_tokens)

        return loss, _perplexity(batch_loss, batch_num_tokens)


    def epoch_loss(self):
        """NLL loss per sentence since the last reset
        """
        if self.loss_type == C.WORD_LOSS:
            epoch_loss = self.total_loss / float(self.total_num_tokens)
        elif self.loss_type == C.SENTENCE_LOSS:
            epoch_loss = self.total_loss / float(self.total_num_sentences)

        return epoch_loss

    def epoch_perplexity(self):
        """Corpus perplexity per token since the last reset
        """
        return _perplexity(self.total_loss, self.total_num_tokens)


def _perplexity(loss, num_tokens):
    return np.exp(loss / float(num_tokens)) if num_tokens > 0 else np.nan
