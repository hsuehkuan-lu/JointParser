import os
import re
from datetime import *
import time
import tensorflow as tf
import pickle
import argparse
from os.path import join as pjoin

from nltk.tokenize import word_tokenize
from model import Model
from utils.config import Config
from utils.general_utils import Progbar
from utils.parser_utils_las import minibatches, load_and_preprocess_data, load_and_preprocess_evaluate_data, load_and_preprocess_inference_data
from utils.layers import concat_bidirectional_rnn_states, birnn_encoder_layer, birnn_att_encoder_layer, rnn_decoder_layer

import numpy as np

class ParserModel(Model):
    """
        Parser model using transition-based algorithm inherit from Model class
    """
    
    def add_placeholders(self):  
        """
            Placeholders for training data
        """
        self.sent_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len], name='sent_input')
        self.char_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len, self.config.n_char], name='char_input')
        self.n_gram_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len, self.config.n_char], name='n_gram_input')
        
        self.idx_features_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.seq_len, self.config.n_features], name='idx_features')

        self.sent_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='sent_len')
        self.char_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len], name='char_len')
        self.n_gram_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len], name='n_gram_len')
        self.seq_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='seq_len')

        self.arc_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.seq_len], name='arc_labels')
        self.pos_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sent_len], name='pos_labels')
        self.dep_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.seq_len], name='dep_labels')

        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name='dropout')

    def add_inf_placeholders(self):
        """
            Placeholders for inferencing data.
        """
        self.inf_sent_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inf_sent_input')
        self.inf_char_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None, self.config.n_char], name='inf_char_input')
        self.inf_n_gram_input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None, self.config.n_char], name='inf_n_gram_input')
        
        self.inf_idx_features_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.config.n_features], name='inf_idx_features')

        self.inf_sent_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='inf_sent_len')
        self.inf_char_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inf_char_len')
        self.inf_n_gram_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inf_n_gram_len')
        
        self.inf_pos_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name='inf_pos_labels')
        self.inf_text_rep_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, 2 * self.config.rnn_size], name='inf_text_rep')

    def create_feed_dict(self, input_batch, dropout=1):
        """
            Creates the feed_dict for the dependency parser.
        """
        feed_dict = { 
            self.sent_input_placeholder: input_batch["sent"],
            self.char_input_placeholder: input_batch["char"],
            self.n_gram_input_placeholder: input_batch["n_gram"],
            self.idx_features_placeholder: input_batch["features"],
            
            self.char_len_placeholder: input_batch["char_len"],
            self.n_gram_len_placeholder: input_batch["n_gram_len"],
            self.sent_len_placeholder: input_batch["sent_len"],
            self.seq_len_placeholder: input_batch["seq_len"],
            
            self.arc_labels_placeholder: input_batch["arc_y"],
            self.pos_labels_placeholder: input_batch["pos_y"],
            self.dep_labels_placeholder: input_batch["dep_y"],
            self.dropout_placeholder: dropout 
        }
        return feed_dict

    def create_inf_feed_dict(self, input_batch):
        """
            Creates the feed_dict for the dependency parser in inferencing stage.
        """
        feed_dict = { 
            self.inf_sent_input_placeholder: input_batch["sent"],
            self.inf_char_input_placeholder: input_batch["char"],
            self.inf_n_gram_input_placeholder: input_batch["n_gram"],
            
            self.inf_sent_len_placeholder: input_batch["sent_len"],
            self.inf_char_len_placeholder: input_batch["char_len"],
            self.inf_n_gram_len_placeholder: input_batch["n_gram_len"],
            
            self.inf_idx_features_placeholder: input_batch["features"],
            self.inf_pos_labels_placeholder: input_batch["pos_y"],
        }
        return feed_dict

    def add_embedding(self):
        """
            Adds an embedding layer that maps from input tokens (integers) to vectors.
        """
        embeddings = {}
        with tf.device("/cpu:0"):
            emb = tf.Variable(self.pretrained_embeddings)
            mask_padding_zero_op = tf.scatter_update(emb, self.config.pad_id, tf.zeros([self.config.embed_size], dtype=tf.float32))
            # zero padding
            with tf.control_dependencies([mask_padding_zero_op]):
                sent_emb = tf.nn.embedding_lookup(emb, self.sent_input_placeholder)
                char_emb = tf.nn.embedding_lookup(emb, self.char_input_placeholder)
                n_gram_emb = tf.nn.embedding_lookup(emb, self.n_gram_input_placeholder)

            embeddings["sent"] = sent_emb
            embeddings["char"] = tf.reshape(char_emb, shape=[-1, self.config.n_char, self.config.embed_size])
            embeddings["n_gram"] = tf.reshape(n_gram_emb, shape=[-1, self.config.n_char, self.config.embed_size])
            
            self.emb = emb
            self.mask_padding_op = mask_padding_zero_op
        return embeddings

    def extract_features_by_idx(self, idx, o):
        """
            Extract features from current configuration.
        """
        # idx = [None, seq_len, sent_len]
        true_mask = tf.not_equal(idx, -1)
        idx = tf.where(true_mask, idx, tf.zeros_like(idx))
        true_mask = tf.expand_dims(tf.cast(true_mask, dtype=tf.float32), axis=-1)
        
        initial_output = tf.TensorArray(dtype=tf.float32, size=tf.shape(o)[0])
        def cond(o, idx, i, output):
            return tf.less(i, tf.shape(o)[0])
        def body(o, idx, i, output):
            output = output.write(i, tf.gather(o[i], idx[i]))
            i = tf.add(i, 1)
            return o, idx, i, output
        _, _, _, output = tf.while_loop(cond, body, [o, idx, tf.constant(0), initial_output])
        
        features = output.stack()
        
        features = tf.multiply(features, true_mask)
        return features

    def add_prediction_op(self):
        """
            Neural model architecture.
            Return:
                - pos_pred
                - arc_pred
                - dep_pred
        """

        x = self.add_embedding()
        
        char_len = self.char_len_placeholder
        sent_len = self.sent_len_placeholder
        seq_len = self.seq_len_placeholder
        n_gram_len = self.n_gram_len_placeholder
        
        char_len = tf.reshape(char_len, shape=[-1])
        n_gram_len = tf.reshape(n_gram_len, shape=[-1])

        with tf.variable_scope("enc"):
            char_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.char_hidden_size) for _ in range(2)]
            char_outputs, char_states = birnn_encoder_layer(char_enc_cells, x["char"], char_len, 1. - self.dropout_placeholder, name="char")
            ch_concat_state = concat_bidirectional_rnn_states(char_states)
            c = tf.reshape(ch_concat_state.h, shape=[-1, self.config.sent_len, 2 * self.config.char_hidden_size])
            
            n_gram_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.char_hidden_size) for _ in range(2)]
            n_gram_outputs, n_gram_states = birnn_encoder_layer(n_gram_enc_cells, x["n_gram"], n_gram_len, 1. - self.dropout_placeholder, name="n_gram")
            ng_concat_state = concat_bidirectional_rnn_states(n_gram_states)
            ng = tf.reshape(ng_concat_state.h, shape=[-1, self.config.sent_len, 2 * self.config.char_hidden_size])
            
            context_input = tf.concat([ng, x["sent"]], axis=-1)
            sent_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.rnn_size) for _ in range(2)]
            sent_outputs, sent_states = birnn_encoder_layer(sent_enc_cells, context_input, sent_len, 1. - self.dropout_placeholder, name="sent")
       
            o = tf.concat(sent_outputs, axis=-1)     
            self.tensor_dict["o"] = o

        with tf.variable_scope("pos"):
            pos_h = tf.layers.dense(o, self.config.hidden_size, activation=tf.nn.relu, name="pos_mlp_1")
            pos_h = tf.layers.dropout(pos_h, self.dropout_placeholder)
            pos_pred = tf.layers.dense(pos_h, self.config.n_pos_classes, name="pos_mlp_2")
            self.tensor_dict["pos_pred"] = pos_pred
            
            pos_sequence = tf.argmax(pos_pred, axis=-1) + self.config.pos_offset
            self.tensor_dict["pos_seq"] = pos_sequence
            with tf.device("/cpu:0"):
                with tf.control_dependencies([self.mask_padding_op]):
                    pos_emb = tf.nn.embedding_lookup(self.emb, pos_sequence)

        with tf.variable_scope("features"):
            joint_pos_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.rnn_size) for _ in range(2)]
            joint_outputs, joint_states = birnn_encoder_layer(joint_pos_enc_cells, tf.concat([o, pos_emb], axis=-1), sent_len, 1. - self.dropout_placeholder, name="joint")
            joint_o = tf.concat(joint_outputs, axis=-1)
            
            idx = self.idx_features_placeholder
            features = self.extract_features_by_idx(idx, joint_o)

            total_features = tf.reshape(features, shape=[-1, self.config.seq_len, 2 * self.config.n_features * self.config.rnn_size])
            
            # stack 1st, 2nd features
            arc_pairwise_head_features = features[:, :, 2]
            arc_pairwise_tail_features = features[:, :, 1]

        with tf.variable_scope("arc_dec"):            
            arc_h = tf.layers.dense(total_features, self.config.hidden_size, activation=tf.nn.relu, name="arc_mlp_1")
            arc_h = tf.layers.dropout(arc_h, self.dropout_placeholder)
            arc_pred = tf.layers.dense(arc_h, 3, name="arc_mlp_2")
            self.tensor_dict["arc_pred"] = arc_pred
            
        with tf.variable_scope("dep_dec"):
            pairwise_features = tf.concat([arc_pairwise_head_features, arc_pairwise_tail_features, tf.multiply(arc_pairwise_head_features, arc_pairwise_tail_features), tf.abs(arc_pairwise_head_features - arc_pairwise_tail_features)], axis=-1)
            dep_h = tf.layers.dense(pairwise_features, self.config.hidden_size, activation=tf.nn.relu, name="dep_mlp_1")
            dep_h = tf.layers.dropout(dep_h, self.dropout_placeholder)
            dep_pred = tf.layers.dense(dep_h, self.config.n_dep_classes, name="dep_mlp_2")

        outputs = {}
        outputs["pos_pred"] = pos_pred
        outputs["arc_pred"] = arc_pred
        outputs["dep_pred"] = dep_pred
        return outputs

    def add_inference_op(self):
        """
            Neural model architecture for inference graph (not including POS).
            Return:
                - arc_pred
                - dep_pred
        """
        sent_len = self.inf_sent_len_placeholder
        n_gram_len = tf.reshape(self.inf_n_gram_len_placeholder, shape=[-1])
        char_len = tf.reshape(self.inf_char_len_placeholder, shape=[-1])
        
        with tf.device("/cpu:0"):
            emb = self.emb
            with tf.control_dependencies([self.mask_padding_op]):
                sent_emb = tf.nn.embedding_lookup(self.emb, self.inf_sent_input_placeholder)
                char_emb = tf.nn.embedding_lookup(self.emb, self.inf_char_input_placeholder)
                char_emb = tf.reshape(char_emb, shape=[-1, self.config.n_char, self.config.embed_size])
                
                n_gram_emb = tf.nn.embedding_lookup(self.emb, self.inf_n_gram_input_placeholder)
                n_gram_emb = tf.reshape(n_gram_emb, shape=[-1, self.config.n_char, self.config.embed_size])
            
        with tf.variable_scope("enc", reuse=True):
            char_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.char_hidden_size, reuse=True) for _ in range(2)]
            char_outputs, char_states = birnn_encoder_layer(char_enc_cells, char_emb, char_len, 1., name="char", reuse=True)
            ch_concat_state = concat_bidirectional_rnn_states(char_states)
#             c = ch_concat_state.h
            c = tf.reshape(ch_concat_state.h, shape=[-1, tf.shape(self.inf_sent_input_placeholder)[1], 2 * self.config.char_hidden_size])
            
            n_gram_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.char_hidden_size) for _ in range(2)]
            n_gram_outputs, n_gram_states = birnn_encoder_layer(n_gram_enc_cells, n_gram_emb, n_gram_len, 1., name="n_gram", reuse=True)
            ng_concat_state = concat_bidirectional_rnn_states(n_gram_states)
#             ng = ng_concat_state.h
            ng = tf.reshape(ng_concat_state.h, shape=[-1, tf.shape(self.inf_sent_input_placeholder)[1], 2 * self.config.char_hidden_size])
            
            context_input = tf.concat([ng, sent_emb], axis=-1)
            sent_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, reuse=True) for _ in range(2)]
            sent_outputs, sent_states = birnn_encoder_layer(sent_enc_cells, context_input, sent_len, 1., name="sent", reuse=True)
            
            o = tf.concat(sent_outputs, axis=-1)
            
        with tf.variable_scope("pos", reuse=True):
            pos_h = tf.layers.dense(o, self.config.hidden_size, activation=tf.nn.relu, name="pos_mlp_1", reuse=True)
            pos_pred = tf.layers.dense(pos_h, self.config.n_pos_classes, name="pos_mlp_2", reuse=True)
            
            # gold-POS
#             pos_sequence = self.inf_pos_labels_placeholder
            # auto-POS
            pos_sequence = tf.argmax(pos_pred, axis=-1)
            with tf.device("/cpu:0"):
                with tf.control_dependencies([self.mask_padding_op]):
                    pos_emb = tf.nn.embedding_lookup(self.emb, pos_sequence + self.config.pos_offset)
        
        with tf.variable_scope("features", reuse=True):
            joint_pos_enc_cells = [tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, reuse=True) for _ in range(2)]
            joint_outputs, joint_states = birnn_encoder_layer(joint_pos_enc_cells, tf.concat([o, pos_emb], axis=-1), sent_len, 1., name="joint", reuse=True)
            joint_o = tf.concat(joint_outputs, axis=-1)
            
#             idx = self.inf_idx_features_placeholder
#             features = self.extract_features_by_idx(idx, joint_o)
#             total_features = tf.reshape(features, shape=[-1, 2 * self.config.n_features * self.config.rnn_size])
            
#             # stack 1st and 2nd elements
#             head_features = features[:, 2]
#             tail_features = features[:, 1]

#         with tf.variable_scope("arc_dec", reuse=True):
#             arc_h = tf.layers.dense(total_features, self.config.hidden_size, activation=tf.nn.relu, name="arc_mlp_1", reuse=True)
#             arc_pred = tf.layers.dense(arc_h, 3, name="arc_mlp_2", reuse=True)
            
#         with tf.variable_scope("dep_dec"):            
#             pairwise_features = tf.concat([head_features, tail_features, tf.multiply(head_features, tail_features), tf.abs(head_features - tail_features)], axis=-1)
#             dep_h = tf.layers.dense(pairwise_features, self.config.hidden_size, activation=tf.nn.relu, name="dep_mlp_1", reuse=True)
#             dep_pred = tf.layers.dense(dep_h, self.config.n_dep_classes, name="dep_mlp_2", reuse=True)
        
        outputs = {}
#         outputs["arc_pred"] = arc_pred
#         outputs["dep_pred"] = dep_pred
        outputs["pos_sequence"] = pos_sequence
        outputs["joint_rep"] = joint_o
        return outputs
    
    def add_inference_op_with_rep(self):
        joint_o = self.inf_text_rep_placeholder
        idx = self.inf_idx_features_placeholder
        
        features = self.extract_features_by_idx(idx, joint_o)
        total_features = tf.reshape(features, shape=[-1, 2 * self.config.n_features * self.config.rnn_size])

        # stack 1st and 2nd elements
        head_features = features[:, 2]
        tail_features = features[:, 1]

        with tf.variable_scope("arc_dec", reuse=True):
            arc_h = tf.layers.dense(total_features, self.config.hidden_size, activation=tf.nn.relu, name="arc_mlp_1", reuse=True)
            arc_pred = tf.layers.dense(arc_h, 3, name="arc_mlp_2", reuse=True)
            
        with tf.variable_scope("dep_dec", reuse=True):            
            pairwise_features = tf.concat([head_features, tail_features, tf.multiply(head_features, tail_features), tf.abs(head_features - tail_features)], axis=-1)
            dep_h = tf.layers.dense(pairwise_features, self.config.hidden_size, activation=tf.nn.relu, name="dep_mlp_1", reuse=True)
            dep_pred = tf.layers.dense(dep_h, self.config.n_dep_classes, name="dep_mlp_2", reuse=True)
        
        outputs = {}
        outputs["arc_pred"] = arc_pred
        outputs["dep_pred"] = dep_pred
        return outputs

    def add_loss_op(self, pred):
        """
            Adds Ops for the loss function to the computational graph.
        """
        arc_labels = self.arc_labels_placeholder
        dep_labels = self.dep_labels_placeholder
        pos_labels = self.pos_labels_placeholder
        
        sent_mask = tf.sequence_mask(self.sent_len_placeholder, self.config.sent_len, dtype=tf.bool)
        pos_loss = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos_labels, logits=pred["pos_pred"]), sent_mask))

        seq_mask = tf.sequence_mask(self.seq_len_placeholder, self.config.seq_len, dtype=tf.bool)
        arc_loss = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=arc_labels, logits=pred["arc_pred"]), seq_mask))

        dep_mask = tf.logical_and(tf.not_equal(arc_labels, 2), seq_mask)
        dep_loss = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dep_labels, logits=pred["dep_pred"]), dep_mask))
        
        loss = arc_loss + pos_loss + dep_loss

        loss_dict = {
            "pos": pos_loss,
            "arc": arc_loss,
            "dep": dep_loss
        }
        return loss, loss_dict
    
    def add_summary_op(self):
        """
            Add summary Ops for all loss functions.
        """
        loss = self.loss_dict
        for key, val in loss.items():
            tf.summary.scalar(key, val)
        tf.summary.scalar('total_loss', self.loss)
        merged = tf.summary.merge_all()
        return merged

    def add_training_op(self, loss):
        """
            Sets up the training Ops.
        """
        tv = tf.trainable_variables()
        regularization_cost = self.config.l2_alpha * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name or 'Variable:0' not in v.name])
        cost = loss + regularization_cost
        
        lr = tf.train.exponential_decay(learning_rate=self.config.lr, global_step=self.batch_cnt, decay_steps=100, decay_rate=0.9, staircase=False)
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=cost)
        return train_op
    
    def predict_pos_on_batch(self, sess, input_batch):
        """
            Make POS predictions for the provided batch of data
        """
        feed = {
            self.inf_sent_input_placeholder: input_batch["sent_list"],
            self.inf_sent_len_placeholder: input_batch["sent_len_list"],
            self.inf_char_input_placeholder: input_batch["char_list"],
            self.inf_char_len_placeholder: input_batch["char_len_list"],
            self.inf_n_gram_input_placeholder: input_batch["n_gram_list"],
            self.inf_n_gram_len_placeholder: input_batch["n_gram_len_list"]
        }
        pos_sequence = sess.run(self.pos_sequence, feed_dict=feed)

        return pos_sequence
    
    def predict_joint_rep_on_batch(self, sess, input_batch):
        """
            Make POS predictions for the provided batch of data
        """
        feed = {
            self.inf_sent_input_placeholder: input_batch["sent_list"],
            self.inf_sent_len_placeholder: input_batch["sent_len_list"],
            self.inf_char_input_placeholder: input_batch["char_list"],
            self.inf_char_len_placeholder: input_batch["char_len_list"],
            self.inf_n_gram_input_placeholder: input_batch["n_gram_list"],
            self.inf_n_gram_len_placeholder: input_batch["n_gram_len_list"]
        }
        pos_sequence, joint_rep = sess.run([self.pos_sequence, self.joint_rep], feed_dict=feed)

        return pos_sequence, joint_rep

    def predict_on_batch(self, sess, inputs_batch):
        """
            Make transition predictions for the provided batch of data
        """
#         feed = self.create_inf_feed_dict(inputs_batch)
        feed = {
            self.inf_text_rep_placeholder: inputs_batch["joint_rep"],
            self.inf_idx_features_placeholder: inputs_batch["features"]
        }
        inf_outputs = sess.run(self.inf_dp_outputs, feed_dict=feed)

        return inf_outputs

    def train_on_batch(self, sess, train_ex):
        feed = self.create_feed_dict(train_ex, dropout=self.config.dropout)
        _, loss, summary = sess.run([self.train_op, self.loss, self.sum_op], feed_dict=feed)
        return loss, summary

    def run_epoch(self, sess, parser, train_examples, dev_set):
        total_loss = []
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, train_ex in enumerate(minibatches(train_examples, self.config.batch_size, parser.pad_instance, self.config)):
            loss, summary = self.train_on_batch(sess, train_ex)
            self.writer.add_summary(summary, self.batch_cnt)
            prog.update(i + 1, [("train loss", loss)])
            total_loss += [loss]

        print()
        print("Evaluating on dev set")
        dev_UAS, dev_LAS, POS_acc, _ = parser.parse(dev_set, pjoin(self.args.model_path, "dev.result"))
        print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        print("- dev LAS: {:.2f}".format(dev_LAS * 100.0))
        print("- dev POS acc: {:.2f}".format(POS_acc * 100.0))
        return dev_UAS, dev_LAS, POS_acc, np.mean(total_loss)

    def fit(self, sess, saver, parser, train_examples, dev_set):
        train_loss = float("inf")
        best_dev_LAS = 0
        self.sum_op = self.add_summary_op()
        
        self.writer = tf.summary.FileWriter(self.args.model_path + '/train')
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS, dev_LAS, POS_acc, loss = self.run_epoch(sess, parser, train_examples, dev_set)
            with open(pjoin(self.args.model_path, "dev.eval"), "a") as fout:
                fout.write("Epoch {}\n".format(epoch + 1))
                fout.write("Dev UAS: {}\tLAS: {}\tPOS-acc: {}\tloss: {}\n".format(dev_UAS * 100.0, dev_LAS * 100.0, POS_acc * 100.0, loss))
            if dev_LAS > best_dev_LAS:
                best_dev_LAS = dev_LAS
                if saver:
                    print("New best dev LAS! Saving model in {}".format(pjoin(self.args.model_path, "model.weights")))
                    saver.save(sess, pjoin(self.args.model_path, "model.weights"))
            if loss < train_loss:
                print("Old loss: {}\t New loss: {}".format(train_loss, loss))
                train_loss = loss
            print()
            
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss, self.loss_dict = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, args, config, pretrained_embeddings):
        self.batch_cnt = 0
        self.tensor_dict = {}
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.args = args
        self.build()

        self.add_inf_placeholders()        
        self.inf_outputs = self.add_inference_op()
        self.pos_sequence = self.inf_outputs["pos_sequence"]
        self.joint_rep = self.inf_outputs["joint_rep"]
        self.inf_dp_outputs = self.add_inference_op_with_rep()


def main(args):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    debug = args.debug

    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(args, debug)
    config = parser.config
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    

    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = ParserModel(args, config, embeddings)
        parser.model = model
        print("took {:.2f} seconds\n".format(time.time() - start))

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        tf_config = tf.ConfigProto()
#         tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as session:
            parser.session = session
            session.run(init)

            print("Parameters:")
            print(tf.trainable_variables())
            print(80 * "=")
            print("TRAINING")
            print(80 * "=")
            model.fit(session, saver, parser, train_examples, dev_set)

            if not debug:
                print(80 * "=")
                print("TESTING")
                print(80 * "=")
                print("Restoring the best model weights found on the dev set")
                saver.restore(session, pjoin(args.model_path, "model.weights"))
                print("Final evaluation on test set",)
                UAS, LAS, POS_acc, dependencies = parser.parse(test_set, pjoin(args.model_path, "test.result"))
                print("- test UAS: {:.2f}".format(UAS * 100.0))
                print("- test LAS: {:.2f}".format(LAS * 100.0))
                print("- test POS acc: {:.2f}".format(POS_acc * 100.0))
                print("Writing predictions")
                with open(pjoin(args.model_path, "test.eval"), 'w') as f:
                    f.write("Test UAS: {}\tLAS: {}\n\tPOS-acc: {}".format(UAS * 100.0, LAS * 100.0, POS_acc * 100.0))
                with open(pjoin(args.model_path, "test.predicted.pkl"), 'wb') as f:
                    pickle.dump(dependencies, f, -1)
                with open(pjoin(args.model_path, "config.pkl"), 'wb') as f:
                    pickle.dump(config, f)
                with open(pjoin(args.model_path, "tran2id.pkl"), 'wb') as f:
                    pickle.dump(parser.tran2id, f)
                with open(pjoin(args.model_path, "tok2id.pkl"), 'wb') as f:
                    pickle.dump(parser.tok2id, f)
                print("Done!")
                print("Done!")
                
def evaluate(args):
    import copy
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    with open(pjoin(args.model_path, "config.pkl"), 'rb') as f:
        config = pickle.load(f)
    with open(pjoin(args.model_path, "tok2id.pkl"), 'rb') as f:
        tok2id = pickle.load(f)
    with open(pjoin(args.model_path, "tran2id.pkl"), 'rb') as f:
        tran2id = pickle.load(f)
    
    origin_config = copy.deepcopy(config)
    parser, embeddings, test_set = load_and_preprocess_evaluate_data(args, config, tok2id, tran2id)
    if not os.path.exists(pjoin(args.model_path, args.output_path)):
        os.makedirs(pjoin(args.model_path, args.output_path))
    
    
    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = ParserModel(args, origin_config, embeddings)
        parser.model = model
        print("took {:.2f} seconds\n".format(time.time() - start))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as session:
            parser.session = session
            # session.run(init)

            print("Parameters:")
            print(tf.trainable_variables())
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, pjoin(args.model_path, "model.weights"))

            print("Final evaluation on test set",)
            UAS, LAS, POS_acc, dependencies = parser.parse(test_set, pjoin(args.output_path, "test.result"))
            print("- test UAS: {:.2f}".format(UAS * 100.0))
            print("- test LAS: {:.2f}".format(LAS * 100.0))
            print("- test POS acc: {:.2f}".format(POS_acc * 100.0))
            
            print("Writing predictions")
            with open(pjoin(args.output_path, "test.eval"), 'w') as f:
                f.write("Test UAS: {}\tLAS: {}\tPOS-acc: {}\n".format(UAS * 100.0, LAS * 100.0, POS_acc * 100.0))
            with open(pjoin(args.output_path, "test.predicted.pkl"), 'wb') as f:
                pickle.dump(dependencies, f, -1)
            print("Done!")
            print("Done!")
            
def inference(args):
    import copy
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    with open(pjoin(args.model_path, "config.pkl"), 'rb') as f:
        config = pickle.load(f)
    with open(pjoin(args.model_path, "tok2id.pkl"), 'rb') as f:
        tok2id = pickle.load(f)
    with open(pjoin(args.model_path, "tran2id.pkl"), 'rb') as f:
        tran2id = pickle.load(f)
    
    origin_config = copy.deepcopy(config)
    parser, embeddings = load_and_preprocess_inference_data(args, config, tok2id, tran2id)
    
    with tf.Graph().as_default():
        print("Building model...")
        start = time.time()
        model = ParserModel(args, origin_config, embeddings)
        parser.model = model
        print("took {:.2f} seconds\n".format(time.time() - start))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as session:
            parser.session = session
            # session.run(init)

            print("Parameters:")
            print(tf.trainable_variables())
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, pjoin(args.model_path, "model.weights"))
            
            # dependency parsing and POS tagging
            s1 = "今天 天气 真好"
            s2 = "关于 三 年 建设 长春 汽车厂 的 指示"
            pos_seq, dependencies = parser.parse_sents([s1.split(), s2.split()])
            print(pos_seq)
            print(dependencies)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests an BilingualAutoencoder model')
    
    subparsers = parser.add_subparsers()
    
    # train
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-g', '--gpu', default='1', help="GPU no.")
    command_parser.add_argument('-m', '--model_path', help="Model path.")
    command_parser.add_argument('--emb_file', help="Emb path.")
    command_parser.add_argument('-d', '--debug', default=0, type=int, help="Model path.")
    command_parser.add_argument('--train_file', default='/data/lxk/DependencyParsing/train.txt', help="Train file.")
    command_parser.add_argument('--dev_file', default='/data/lxk/DependencyParsing/dev.txt', help="Dev file.")
    command_parser.add_argument('--test_file', default='/data/lxk/DependencyParsing/test.txt', help="Test file.")
    command_parser.set_defaults(func=main)
    
    # evaluate
    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-g', '--gpu', default='1', help="GPU no.")
    command_parser.add_argument('-m', '--model_path', help="Model path.")
    command_parser.add_argument('-o', '--output_path', help="Output path.")
    command_parser.add_argument('-d', '--debug', default=0, type=int, help="Model path.")
    command_parser.add_argument('--test_file', default='/data/lxk/DependencyParsing/test.txt', help="Test file.")
    command_parser.set_defaults(func=evaluate)
    
    # infernece
    command_parser = subparsers.add_parser('inference', help='')
    command_parser.add_argument('-g', '--gpu', default='1', help="GPU no.")
    command_parser.add_argument('-m', '--model_path', help="Model path.")
    command_parser.set_defaults(func=inference)
    
    args = parser.parse_args()
    args.func(args)