# Utilities for training the dependency parser.

import time
import os
import logging
from collections import Counter
from utils.config import Config
from utils.general_utils import logged_loop, get_minibatches
from parser_transitions_las import PartialParse, minibatch_parse

import numpy as np
from os.path import join as pjoin

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
A_PREFIX = '<a>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'
PAD = '<PAD>'

n_trans = 3


class Parser(object):
    """
        Contains everything needed for transition-based dependency parsing except for the model
    """
    def __init__(self):
        pass

    def init_with_dataset(self, dataset, config):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel =  [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))

        self.dep_offset = 0
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        self.config = config

        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.arc_offset = len(tok2id)        

        global n_trans
        n_trans = len(trans)
        self.config.n_classes = 3
        self.config.n_trans = n_trans
        self.config.n_dep_classes = len(deprel)
        self.n_trans = n_trans
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.')
        self.pos_offset = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']], tok2id, offset=len(tok2id)))
        self.config.n_pos_classes = len(tok2id) - self.pos_offset
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)

        # logging.info('Build dictionary for words.')

        #tok2id.update(build_dict([w for ex in dataset for w in ex['word']] + [c for ex in dataset for w in ex['ch'] for c in w], offset=len(tok2id)))
        
        print("Loading pretrained embeddings...",)
        start = time.time()
        word_vectors = {}
        if config.embedding_file:
            for line in open(config.embedding_file).readlines():
                sp = line.strip().split()
                word_vectors[sp[0]] = [float(x) for x in sp[1:]]
                
        self.word_offset = len(tok2id)
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']], tok2id, offset=len(tok2id)))
        self.word_end_offset = len(tok2id)
        tok2id.update(build_dict([g for ex in dataset for w in ex['word'] for g in get_n_gram(w, config.n_gram)] + [c for ex in dataset for w in ex['ch'] for c in w] + list(word_vectors.keys()), tok2id, offset=len(tok2id)))
        
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)
        tok2id[PAD] = self.PAD = len(tok2id)

        self.config.pad_id = self.PAD
        self.config.dep_offset = self.dep_offset
        self.config.arc_offset = self.arc_offset
        self.config.pos_offset = self.pos_offset
        self.config.word_offset = self.word_offset
        self.config.word_end_offset = self.word_end_offset
        
        embeddings_matrix = np.asarray(np.random.uniform(-1., 1., [len(tok2id), self.config.embed_size]), dtype='float32')

        for token in tok2id:
            i = tok2id[token]
            if token in word_vectors:
                embeddings_matrix[i] = word_vectors[token]
            elif token.lower() in word_vectors:
                embeddings_matrix[i] = word_vectors[token.lower()]
        print("took {:.2f} seconds".format(time.time() - start))

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}
        
        self.embeddings_matrix = embeddings_matrix
        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)
        self.n_tokens = len(tok2id)

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            n_gram = [[self.ROOT]] + [[self.tok2id[g] if g in self.tok2id else self.UNK for g in get_n_gram(w, self.config.n_gram)] for w in ex['word']]
            ch = [[self.ROOT]] + [[self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in c] for c in ex['ch']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else self.UNK for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label, 'ch': ch, 'n_gram': n_gram, 'sp': ex['sp']})
        return vec_examples
    
    def extract_features_idx(self, stack, buf, arcs):
        """
            Features: 
                first 3 elements from stack, first 1 element from buffer
        """
        if stack[0] == "ROOT":
            stack[0] = 0

        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        idx = [x for x in stack[-3:]]
        if idx[0] == 0:
            idx = idx[1:]
        idx = [-1] * (3 - len(idx)) + idx
        idx += [x for x in buf[:1]] + [-1] * (1 - len(buf))

        return idx

    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:
            return self.n_trans - 1

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            if (i1 > 0) and (h1 == i0):
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        fail = 0
        for id, ex in enumerate(logged_loop(examples)):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            for i in range(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    fail += 1
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                features = self.extract_features_idx(stack, buf, arcs)
                
                head = tail = -1
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                elif gold_t < self.n_deprel:
                    arcs.append((stack[-1], stack[-2], gold_t))
                    head = stack[-1]
                    tail = stack[-2]
                    stack = stack[:-2] + [stack[-1]]
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    head = stack[-2]
                    tail = stack[-1]
                    stack = stack[:-1]
                instances.append((features, legal_labels, gold_t, [head, tail]))
            else:
                succ += 1
                ex_dict = {}
                
                ex_dict["word"] = ex["word"]
                ex_dict["pos"] = ex["pos"]
                ex_dict["ch"] = ex["ch"]
                ex_dict["n_gram"] = ex["n_gram"]

                all_instances += [
                    {
                        "instances": instances,
                        "ex": ex_dict
                    }
                ]
                
        print("Fail {}, Success {}, Total {}".format(fail, succ, fail + succ))

        self.pad_instance = (self.extract_features_idx([0], [], []), self.legal_labels([0], []), 0)

        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels
    
    def generate_ex(self, dataset):
        sent_list, char_list, n_gram_list = [], [], []
        sent_len_list, char_len_list, n_gram_len_list = [], [], []
        
        max_sent_len = max([len(i['word']) for i in dataset])
        
        sentences = []
        sentence_id_to_idx = {}
        
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i
            
            n_words += 1
            sent_list.append(example['word'][:max_sent_len] + [self.config.pad_id] * (max_sent_len - n_words))
            char_list.append([i[:self.config.n_char] + [self.config.pad_id] * (self.config.n_char - len(i)) for i in example["ch"][:max_sent_len]] + [[self.config.pad_id] * self.config.n_char] *  (max_sent_len - n_words))
            char_len_list.append([min(len(i), self.config.n_char) for i in example["ch"][:max_sent_len]] + [0] * (max_sent_len - n_words))
            sent_len_list.append(min(n_words, max_sent_len))
            n_gram_list.append([i[:self.config.n_char] + [self.config.pad_id] * (self.config.n_char - len(i)) for i in example["n_gram"][:max_sent_len]] + [[self.config.pad_id] * self.config.n_char] *  (max_sent_len - n_words))
            n_gram_len_list.append([min(len(i), self.config.n_char) for i in example["n_gram"][:max_sent_len]] + [0] * (max_sent_len - n_words))
        
        sent_list = np.array(sent_list).astype('int32')
        sent_len_list = np.array(sent_len_list).astype('int32')
        char_list = np.array(char_list).astype('int32') 
        char_len_list = np.array(char_len_list).astype('int32') 
        n_gram_list = np.array(n_gram_list).astype('int32') 
        n_gram_len_list = np.array(n_gram_len_list).astype('int32')
        
        data = {
            "sent_list": sent_list,
            "sent_len_list": sent_len_list,
            "char_list": char_list,
            "char_len_list": char_len_list,
            "n_gram_list": n_gram_list,
            "n_gram_len_list": n_gram_len_list,
            "sentence_id_to_idx": sentence_id_to_idx,
            "sentences": sentences
        }
        
        return data

    def parse(self, dataset, output_path, eval_batch_size=1000):
        data = self.generate_ex(dataset)
        input_batch = {}
    
        # predict POS here
        print()
        print("Eval POS tagger")
        
        joint_rep_list = []
        pos_sequence_list = []
        for i in range(int(len(data["sent_list"]) / eval_batch_size) + 1):
            input_batch = {
                "sent_list": data["sent_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "sent_len_list": data["sent_len_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "char_list": data["char_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "char_len_list": data["char_len_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "n_gram_list": data["n_gram_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "n_gram_len_list": data["n_gram_len_list"][i * eval_batch_size: (i+1)*eval_batch_size]
            }
            pos_sequence, joint_rep = self.model.predict_joint_rep_on_batch(self.session, input_batch)
            pos_sequence_list += pos_sequence.tolist()
            joint_rep_list += joint_rep.tolist()
        
        data["joint_rep"] = joint_rep_list
        model = ModelWrapper(self, data)
        dependencies = minibatch_parse(data["sentences"], model, eval_batch_size, self.config)
        
        print()
        print("Eval dependency parser")
        
        conll_out = open(output_path+".conll", "w")
        with open(output_path, "w") as fout:
            fout.write("{},{},{},{},{},{},{}\n".format("id", "word", "pred_pos", "gold_pos", "pred_l", "gold_l", "pred_h", "gold_h"))
            
            POS_acc = 0.0
            UAS = LAS = all_tokens = 0.0
            # confus[gold][pred]
            confus = [[0.0 for _ in range(self.config.n_dep_classes)] for _ in range(self.config.n_dep_classes)]
            pos_confus = [[0.0 for _ in range(self.config.n_pos_classes + 1)] for _ in range(self.config.n_pos_classes + 1)]
            for i, ex in enumerate(dataset):
                pred_pos_list = pos_sequence_list[i]
                
                n_words = len(ex['word']) - 1
                head = [-1] * len(ex['word'])
                head_l = [-1] * len(ex['word'])
                for h, t, l in dependencies[i]:
                    head[t] = h
                    head_l[t] = l

                for pred_h, pred_l, gold_h, gold_l, pos, pred_pos, w, sp in \
                        zip(head[1:], head_l[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:], pred_pos_list[1:n_words+1], ex['word'][1:], ex['sp']):
                        
                        assert self.id2tok[pos].startswith(P_PREFIX)
                        pos_str = self.id2tok[pos][len(P_PREFIX):]
                        pred_pos_str = self.id2tok[pred_pos + self.config.pos_offset][len(P_PREFIX):]
                        pred_l_str = self.id2tran[pred_l][2:] if pred_l in self.id2tran else "<P-NULL>"
                        gold_l_str = self.id2tran[gold_l][2:] if gold_l in self.id2tran else "<G-NULL>"
                        
                        sp[3] = pred_pos_str
                        sp[4] = "_"
                        sp[6] = str(pred_h)
                        sp[7] = pred_l_str
                        conll_out.write("{}\n".format("\t".join(sp)))

                        fout.write("{},{},{},{},{},{},{}\n".format(self.id2tok[w], pred_pos_str, pos_str, pred_l_str, gold_l_str, pred_h, gold_h))
                        if (self.with_punct) or (not punct(self.language, pos_str)):
                            pos_idx = pos - self.config.pos_offset
                            if pos_idx == pred_pos:
                                POS_acc += 1
#                             pl = pred_l % self.n_deprel
#                             gl = gold_l % self.n_deprel
                            if pred_h == gold_h:
                                UAS += 1
                                if pred_l == gold_l:
                                    LAS += 1
                            try:
                                confus[gold_l][pred_l] += 1
                            except IndexError:
                                print(gold_l, pred_l)
                            pos_confus[pos_idx][pred_pos] += 1
                            all_tokens += 1
                fout.write("\n")
                conll_out.write("\n")        
        conll_out.close()
                

        with open(output_path+".pred", "w") as f:
            f.write("{},{},{},{}\n".format("class", "p", "r", "f1"))
            p_mat = np.sum(confus, axis=0)
            r_mat = np.sum(confus, axis=1)
            for c in range(self.n_deprel):
                c_str = self.id2tok[c][len(L_PREFIX):]
                p = 0. if confus[c][c] == 0.0 else confus[c][c] / p_mat[c]
                r = 0. if confus[c][c] == 0.0 else confus[c][c] / r_mat[c]
                f1 = 0. if (p+r) == 0. else 2 * (p * r) / (p + r)
                f.write("{},{:.4f},{:.4f},{:.4f}\n".format(c_str, p, r, f1))
                
        with open(output_path+".pos.pred", "w") as f:
            f.write("{},{},{},{}\n".format("class", "p", "r", "f1"))
            p_mat = np.sum(pos_confus, axis=0)
            r_mat = np.sum(pos_confus, axis=1)
            for c in range(self.config.n_pos_classes):
                c_str = self.id2tok[c + self.config.pos_offset][len(P_PREFIX):]
                p = 0. if pos_confus[c][c] == 0.0 else pos_confus[c][c] / p_mat[c]
                r = 0. if pos_confus[c][c] == 0.0 else pos_confus[c][c] / r_mat[c]
                f1 = 0. if (p+r) == 0. else 2 * (p * r) / (p + r)
                f.write("{},{:.4f},{:.4f},{:.4f}\n".format(c_str, p, r, f1))

        UAS /= all_tokens
        LAS /= all_tokens
        POS_acc /= all_tokens
        return UAS, LAS, POS_acc, dependencies
    
    def parse_sents(self, sents, eval_batch_size=1000):
        examples = []
        
        for sent in sents:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id else self.UNK for w in sent]
            n_gram = [[self.ROOT]] + [[self.tok2id[g] if g in self.tok2id else self.UNK for g in get_n_gram(w, self.config.n_gram)] for w in sent]
            ch = [[self.ROOT]] + [[self.tok2id[w] if w in self.tok2id else self.UNK for w in c] for c in [i for i in sent]]
            examples.append({"word": word, "ch": ch, "n_gram": n_gram})
        
        data = self.generate_ex(examples)
        
        input_batch = {}
    
        # predict POS here
        print()
        print("Eval POS tagger")
        
        joint_rep_list = []
        pos_sequence_list = []
        for i in range(int(len(data["sent_list"]) / eval_batch_size) + 1):
            input_batch = {
                "sent_list": data["sent_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "sent_len_list": data["sent_len_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "char_list": data["char_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "char_len_list": data["char_len_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "n_gram_list": data["n_gram_list"][i * eval_batch_size: (i+1)*eval_batch_size],
                "n_gram_len_list": data["n_gram_len_list"][i * eval_batch_size: (i+1)*eval_batch_size]
            }
            pos_sequence, joint_rep = self.model.predict_joint_rep_on_batch(self.session, input_batch)
            pos_sequence_list += pos_sequence.tolist()
            joint_rep_list += joint_rep.tolist()
        
        pos_result = []
        for i in range(len(pos_sequence_list)):
            pos_result += [[self.id2tok[j + self.config.pos_offset][len(P_PREFIX):] for j in pos_sequence_list[i][1: len(sents[i]) + 1]]]
        
        data["joint_rep"] = joint_rep_list
        model = ModelWrapper(self, data)
        dependencies = minibatch_parse(data["sentences"], model, eval_batch_size, self.config)
        dep_results = []
        for i in range(len(dependencies)):
            sent = ["ROOT"] + sents[i]
            d = dependencies[i]
            dp = []
            for head, tail, label in d:
                dp += [(sent[head], sent[tail], self.id2tok[label][len(L_PREFIX):])]
            dep_results += [dp]
        return pos_result, dep_results


class ModelWrapper(object):
    def __init__(self, parser, data):
        self.parser = parser
        self.data = data
        self.sentence_id_to_idx = data["sentence_id_to_idx"]

    def predict(self, partial_parses, config):
        input_dict = {}
        
        features = []
        joint_o_list = []
        for p in partial_parses:
            joint_o = self.data["joint_rep"][self.sentence_id_to_idx[id(p.sentence)]]
            n_words = self.data["sent_len_list"][self.sentence_id_to_idx[id(p.sentence)]]
            
            x = self.parser.extract_features_idx(p.stack, p.buffer, p.dependencies)
            features += [x]
            joint_o_list += [joint_o]
        
        input_dict["joint_rep"] = np.asarray(joint_o_list, dtype=np.float32)
        input_dict["features"] = np.asarray(features, dtype=np.float32)
    
#         for key, val in input_dict.items():
#             input_dict[key] = np.array(val).astype('int32')
            

        # mb_x = np.array(mb_x).astype('int32')
        slices = [0, config.n_dep_classes, 2 * config.n_dep_classes]
        mb_l = [[self.parser.legal_labels(p.stack, p.buffer)[i] for i in slices] for p in partial_parses]

        inf_outputs = self.parser.model.predict_on_batch(self.parser.session, input_dict)
        arc_pred = inf_outputs["arc_pred"]
        dep_pred = np.argmax(inf_outputs["dep_pred"], -1)

        arc_pred = np.argmax(arc_pred + 10000 * np.array(mb_l).astype('float32'), -1)
        tran_pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in arc_pred]
        return list(zip(tran_pred, dep_pred))


def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        is_valid = True
        sp_list = []
        word, pos, xpos, head, label, ch = [], [], [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    ch.append([i.lower() if lowercase else i for i in sp[1]])
                    pos.append(sp[3])
                    xpos.append(sp[4])
                    try:
                        head.append(int(sp[6]))
                    except ValueError:
                        head.append(0)
                        is_valid = False
                    label.append(sp[7])
                    sp_list.append(sp)
            elif len(word) > 0:
                if is_valid:
                    examples.append({'word': word, 'pos': pos, 'xpos': xpos, 'head': head, 'label': label, 'ch': ch, 'sp': sp_list})
                is_valid = True
                sp_list = []
                word, pos, xpos, head, label, ch = [], [], [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            if is_valid:
                examples.append({'word': word, 'pos': pos, 'xpos': xpos, 'head': head, 'label': label, 'ch': ch, 'sp': sp_list})
    return examples

def get_n_gram(word, n_gram):
    n_gram_list = []
    word = "<" + word + ">"
    for i in range(len(word) - n_gram + 1):
        n_gram_list += [word[i: i+n_gram]]
    return n_gram_list


def build_dict(keys, table, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        if key in table: continue
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)
        

def minibatches(data, batch_size, pad_instance, config):
    data_dict = {
        "features": []
    }
    sent = []
    ch = []
    ch_len = []
    n_gram = []
    n_gram_len = []
    sent_len = []
    seq_len = []
    arc_pair_x = []
    arc_y = []
    pos_y = []
    dep_y = []

    for d in data:
        instances = d["instances"][:config.seq_len]
        ex = d["ex"]
        word_list = ex["word"][:config.sent_len]
        pos_list = ex["pos"][:config.sent_len]
        ch_list = ex["ch"][:config.sent_len]
        n_gram_list = ex["n_gram"][:config.sent_len]
        
        n_words = min(len(ex["word"]), config.sent_len)
        data_dict["features"].append([i[0] for i in instances])
        
        sent += [word_list + [config.pad_id] * (config.sent_len - n_words)]
        ch += [[i[:config.n_char] + [config.pad_id] * (config.n_char - len(i)) for i in ch_list] + [[config.pad_id] * config.n_char] *  (config.sent_len - n_words)]
        n_gram += [[i[:config.n_char] + [config.pad_id] * (config.n_char - len(i)) for i in n_gram_list] + [[config.pad_id] * config.n_char] *  (config.sent_len - n_words)]

        arc_y.append([int(i[2] / config.n_dep_classes) for i in instances])
        dep_y.append([int(i[2] % config.n_dep_classes) for i in instances])
        pos_y.append([i - config.pos_offset for i in pos_list] + [0] * (config.sent_len - n_words))

        seq_len += [n_words * 2]
        sent_len += [n_words]
        ch_len += [[min(len(i), config.n_char) for i in ch_list] + [0] * (config.sent_len - n_words)]
        n_gram_len += [[min(len(i), config.n_char) for i in n_gram_list] + [0] * (config.sent_len - n_words)]
    
    data_dict["sent"] = sent
    data_dict["char"] = ch
    data_dict["n_gram"] = n_gram
    data_dict["char_len"] = ch_len
    data_dict["n_gram_len"] = n_gram_len
    data_dict["seq_len"] = seq_len
    data_dict["sent_len"] = sent_len
    data_dict["arc_y"] = arc_y
    data_dict["dep_y"] = dep_y
    data_dict["pos_y"] = pos_y

    return get_minibatches(data_dict, batch_size, pad_instance, config)


def load_and_preprocess_data(args, reduced=True):
    config = Config()
    
    if hasattr(args, "emb_file"):
        config.embedding_file = args.emb_file
        
    print("Loading data...",)
    start = time.time()
#     train_set = read_conll(os.path.join(config.data_path, config.train_file),
#                            lowercase=config.lowercase)
#     dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
#                          lowercase=config.lowercase)
#     test_set = read_conll(os.path.join(config.data_path, config.test_file),
#                           lowercase=config.lowercase)
    
    train_set = read_conll(args.train_file,
                           lowercase=config.lowercase)
    if os.path.exists(args.dev_file):
        dev_set = read_conll(args.dev_file, lowercase=config.lowercase)
    else:
        dev_set = read_conll(args.test_file, lowercase=config.lowercase)
    test_set = read_conll(args.test_file,
                          lowercase=config.lowercase)
    
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Building parser...",)
    start = time.time()
    parser = Parser()
    parser.init_with_dataset(train_set, config)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Vectorizing data...",)
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))

    print("Preprocessing training data...")
    train_examples = parser.create_instances(train_set)

    return parser, parser.embeddings_matrix, train_examples, dev_set, test_set,

def load_and_preprocess_evaluate_data(args, config, tok2id, tran2id):
    import copy
    print("Loading data...",)
    start = time.time()
    test_set = read_conll(args.test_file,
                          lowercase=config.lowercase)
    print("took {:.2f} seconds".format(time.time() - start))
    if args.debug:
        test_set = test_set[:500]
    
    origin_config = copy.deepcopy(config)
    
    parser = Parser()
    parser.init_with_dataset(test_set, config)
    parser.config = origin_config
    parser.tok2id = tok2id
    parser.id2tok = {v: k for (k, v) in tok2id.items()}
    parser.tran2id = tran2id
    parser.id2tran = {v: k for (k, v) in tran2id.items()}
    parser.n_deprel = origin_config.n_dep_classes
    parser.UNK = tok2id[UNK]
    parser.NULL = tok2id[NULL]
    parser.ROOT = tok2id[ROOT]
    parser.PAD = tok2id[PAD]

    print("Vectorizing data...",)
    start = time.time()
    test_set = parser.vectorize(test_set)
    print("took {:.2f} seconds".format(time.time() - start))
    
    embeddings_matrix = np.asarray(np.random.uniform(-1., 1., [len(tok2id), origin_config.embed_size]), dtype='float32')

    return parser, embeddings_matrix, test_set,

def load_and_preprocess_inference_data(args, config, tok2id, tran2id):
#     import copy
#     origin_config = copy.deepcopy(config)
    
    parser = Parser()
    parser.config = config
    parser.tok2id = tok2id
    parser.id2tok = {v: k for (k, v) in tok2id.items()}
    parser.tran2id = tran2id
    parser.id2tran = {v: k for (k, v) in tran2id.items()}
    parser.n_deprel = config.n_dep_classes
    parser.UNK = tok2id[UNK]
    parser.NULL = tok2id[NULL]
    parser.ROOT = tok2id[ROOT]
    parser.PAD = tok2id[PAD]
    
    embeddings_matrix = np.asarray(np.random.uniform(-1., 1., [len(tok2id), config.embed_size]), dtype='float32')
    
    return parser, embeddings_matrix

if __name__ == '__main__':
    pass
