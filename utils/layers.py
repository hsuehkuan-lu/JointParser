import math
import tensorflow as tf


def concat_bidirectional_rnn_states(states):
    state_fw, state_bw = states
    final_state_c = tf.concat((state_fw.c, state_bw.c), axis=1)
    final_state_h = tf.concat((state_fw.h, state_bw.h), axis=1)
    return tf.contrib.rnn.LSTMStateTuple(c=final_state_c, h=final_state_h)


def birnn_encoder_layer(cells, w, sent_len, dropout_rate, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cells[0], output_keep_prob=dropout_rate)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cells[1], output_keep_prob=dropout_rate)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, w, sequence_length=sent_len, dtype=tf.float32)
    return outputs, states

def stack_birnn_encoder_layer(cells_fw, cells_bw, w, sent_len, dropout_rate, name, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        cfw = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate) for cell in cells_fw]
        cbw = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate) for cell in cells_bw]
        
        outputs, states_fw, states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cfw, cbw, w, sequence_length=sent_len, dtype=tf.float32)
        final_state_c = tf.concat((states_fw[-1].c, states_bw[-1].c), axis=1)
        final_state_h = tf.concat((states_fw[-1].h, states_bw[-1].h), axis=1)
        final_state = tf.contrib.rnn.LSTMStateTuple(c=final_state_c, h=final_state_h)
    return outputs, final_state


def birnn_att_encoder_layer(cells, w, sent_len, dropout_rate, hidden_size, name):
    with tf.variable_scope(name):
        for idx, cell in enumerate(cells):
            cells[idx] = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cells[0], cells[1], w, sent_len, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        attn_output = tf.contrib.seq2seq.LuongAttention(hidden_size, outputs, memory_sequence_length=sent_len)
    return attn_output, states

def rnn_decoder_layer(cell, inputs, sent_len, dropout_rate, project_layer, name):
    with tf.variable_scope(name):
        zero_states = cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)

        # project_layer = tf.layers.Dense(vocab_size, name='projection_layer')
        # cell
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)
        # helper
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, sent_len)
        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, zero_states, output_layer=project_layer)
        # dynamic decoding
        final_outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        # output
        # logits = tf.reshape(final_outputs.rnn_output, shape=[-1, vocab_size])
        logits = final_outputs.rnn_output

#         current_ts = tf.to_int32(tf.minimum(tf.shape(labels)[1], tf.shape(logits)[1]))
#         labels = tf.slice(labels, begin=[0, 0], size=[-1, current_ts])
#         mask_ = tf.sequence_mask(sent_len, current_ts, dtype=logits.dtype)
#         logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, current_ts, -1])
#         # labels = tf.boolean_mask(labels, tf.cast(final_outputs.sample_id, tf.bool))
#         # mask = tf.cast(tf.not_equal(labels, 0), dtype=tf.float32)
#         # loss
#         # loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels, weights=mask)
#         # loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, mask)
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return logits

def rnn_inference_layer(cell, inputs, sent_len, dropout_rate, vocab_size, name):
    with tf.variable_scope(name, reuse=True):
        zero_states = cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)

        project_layer = tf.layers.Dense(vocab_size, name='projection_layer')
        # cell
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_rate)
        # helper
        helper = tf.contrib.seq2seq.TrainingHelper(inputs, sent_len)
        # decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, zero_states, output_layer=project_layer)
        # dynamic decoding
        final_outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        # output
        # logits = tf.reshape(final_outputs.rnn_output, shape=[-1, vocab_size])
        logits = final_outputs.rnn_output

#         current_ts = tf.to_int32(tf.minimum(tf.shape(labels)[1], tf.shape(logits)[1]))
#         labels = tf.slice(labels, begin=[0, 0], size=[-1, current_ts])
#         mask_ = tf.sequence_mask(sent_len, current_ts, dtype=logits.dtype)
#         logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, current_ts, -1])
#         # labels = tf.boolean_mask(labels, tf.cast(final_outputs.sample_id, tf.bool))
#         # mask = tf.cast(tf.not_equal(labels, 0), dtype=tf.float32)
#         # loss
#         # loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels, weights=mask)
#         # loss = tf.losses.sparse_softmax_cross_entropy(labels, logits, mask)
#         loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return logits