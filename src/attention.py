# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import tensorflow as tf

import src.constants as C


# class Attention(nn.Module):
#     r"""
#     Applies an attention mechanism on the output features from the decoder.
#
#     .. math::
#             \begin{array}{ll}
#             x = context*output \\
#             attn = exp(x_i) / sum_j exp(x_j) \\
#             output = \tanh(w * (attn * context) + b * output)
#             \end{array}
#
#     Args:
#         dim(int): The number of expected features in the output
#
#     Inputs: output, context
#         - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
#         - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
#
#     Outputs: output, attn
#         - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
#         - **attn** (batch, output_len, input_len): tensor containing attention weights.
#
#     Attributes:
#         linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
#         mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
#
#     Examples::
#
#          >>> attention = seq2seq.models.Attention(256)
#          >>> context = Variable(torch.randn(5, 3, 256))
#          >>> output = Variable(torch.randn(5, 5, 256))
#          >>> output, attn = attention(output, context)
#
#     """
#
#     def __init__(self, dim, model_name):
#         super(Attention, self).__init__()
#
#         self.model_name = model_name
#         if self.model_name == C.LM_QUESTION_ANSWERS:
#             self.dim_factor = 2
#         elif self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
#             self.dim_factor = 3
#         else:
#             raise Exception('Unexpected')
#         self.linear_out = nn.Linear(dim * self.dim_factor, dim)
#         self.mask = None
#
#     def set_mask(self, mask):
#         """
#         Sets indices to be masked
#
#         Args:
#             mask (torch.Tensor): tensor containing indices to be masked
#         """
#         self.mask = mask
#
#     def get_mix(self, output, context):
#         input_size = context.size(1)
#         # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
#         attn = torch.bmm(output, context.transpose(1, 2))
#         if self.mask is not None:
#             attn.data.masked_fill_(self.mask, -float('inf'))
#         attn = F.softmax(attn.view(-1, input_size)).view(self.batch_size, -1, input_size)
#
#         # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
#         mix = torch.bmm(attn, context)
#         return attn, mix
#
#     def forward(self, output, context):
#         self.batch_size = output.size(0)
#         self.hidden_size = output.size(2)
#         self.wc = tf.keras.layers.Dense(self.hidden_size, activation='tanh')
#
#         (question_out, review_outs) = context
#         attn, question_mix = self.get_mix(output, question_out)
#
#         if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
#             review_mixs = [self.get_mix(output, review_out)[1] for review_out in review_outs]
#             review_mix = _mean(review_mixs)
#             # concat -> (batch, out_len, 2*dim)
#             combined = torch.cat((question_mix, review_mix, output), dim=2)
#         else:
#             combined = torch.cat((question_mix, output), dim=2)
#
#         # output -> (batch, out_len, dim)
#         output = F.tanh(self.linear_out(combined.view(-1, self.dim_factor * self.hidden_size))).view(self.batch_size,
#                                                                                                      -1,
#                                                                                                      self.hidden_size)
#         return output, attn


# def _mean(vars):
#     return torch.mean(torch.cat([i.unsqueeze(0) for i in vars], 0), 0)


########################## Tensorflow ######################


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, model_name, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func
        self.model_name = model_name

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(rnn_size)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(rnn_size, activation='tanh')
            self.va = tf.keras.layers.Dense(1)

    def call(self, decoder_output, encoder_output):

        self.batch_size = decoder_output.shape[0]
        self.hidden_size = decoder_output.shape[2]

        self.wc = tf.keras.layers.Dense(self.hidden_size, activation='tanh')

        (question_out, review_outs) = encoder_output

        attn, context = self.attention_mechanism(decoder_output, question_out)

        if self.model_name == C.LM_QUESTION_ANSWERS_REVIEWS:
            review_mixs = [self.get_mix(decoder_output, review_out)[1] for review_out in review_outs]
            ####*****review_mix = _mean(review_mixs)
            # concat -> (batch, out_len, 2*dim)
            # combined = /.cat((context, review_mix, decoder_output), dim=2)
        else:
            combined = tf.concat([context, decoder_output], 2)  ## confirm dim here.

        output = self.wc(combined)

        return attn, output

        ## alignment 256, 101, 128
        ## encoder_outputs 256,52,128
        ## context 256,101,128
        ## atten 256,101,52
        ## lstm_out 256,101,128

    def attention_mechanism(self, decoder_output, encoder_output):

        batch_sz = encoder_output[0].shape
        input_size = encoder_output[1].shape
        out_len = decoder_output[1].shape

        if self.attention_func == 'dot':
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':

            # (256, 52, 128) Encoder output
            # (256, 102, 128) Decoder output

            encoder_output = tf.tile(encoder_output, [1, 2, 1])

            # Concat => Wa => va
            # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
            score = self.va(
                self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))

            # Transpose score vector to have the same shape as other two above
            # (batch_size, max_len, 1) => (batch_size, 1, max_len)
            score = tf.transpose(score, [0, 2, 1])

        # alignment a_t = softmax(score)
        score__ = tf.reshape(score, [-1, 52])
        score__0 = tf.nn.softmax(score__, axis=0)
        score__1 = tf.nn.softmax(score__, axis=1)
        # alignment = tf.nn.softmax(score, axis=1)

        alignment = tf.reshape(score__1, [256, 52, 101])
        #
        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return alignment, context
