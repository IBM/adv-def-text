# import numpy as np
# arr = [0.008, 0.032, 0.038, 0.036, 0.035, 0.018, 0.023, 0.026, 0.040, 0.036, 0.040, 0.037, 0.016, 0.019, 0.063, 0.066, 0.056, 0.053, 0.043, 0.045, 0.043, 0.034, 0.032, 0.024, 0.031, 0.030, 0.038, 0.022, 0.008, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
# arr = np.array(arr)
#
# max = np.max(arr)
# min = np.min(arr)
#
# arr = (arr - min) / (max - min)
# print(arr)

# in_s, out_s = [], []
# for line in open('/Users/yxu132/pub-repos/nmt/data/para/yelp-para-80/paraphrase_dev.in', 'r'):
#     in_s.append(line.strip())
# for line in open('/Users/yxu132/pub-repos/nmt/data/para/yelp-para-80/paraphrase_dev.out', 'r'):
#     out_s.append(line.strip())
#
# import numpy as np
# shuffled_ids = np.arange(len(in_s))
# np.random.shuffle(shuffled_ids)
# in_shuffled = np.array(in_s)[shuffled_ids]
# out_shuffled = np.array(out_s)[shuffled_ids]
#
# shuffled_ids = np.arange(len(in_s))
# np.random.shuffle(shuffled_ids)
# in_shuffled = np.array(in_shuffled)[shuffled_ids]
# out_shuffled = np.array(out_shuffled)[shuffled_ids]
#
#
# with open('/Users/yxu132/pub-repos/nmt/data/para/yelp-para-80/paraphrase_dev.shuffled.in', 'w') as output_file:
#     for line in in_shuffled:
#         output_file.write(line+'\n')
#
# with open('/Users/yxu132/pub-repos/nmt/data/para/yelp-para-80/paraphrase_dev.shuffled.out', 'w') as output_file:
#     for line in out_shuffled:
#         output_file.write(line+'\n')
#
# import numpy as np
#
# sizes = [0, 0, 0, 0, 0]
# labels = []
# for line in open('/Users/yxu132/pub-repos/nmt/data/yelp_clss/yelp_train.label.multi.out', 'r'):
#     label = 0
#     if line.strip() == '0.0 0.0 0.0 0.0 1.0':
#         label = 4
#     elif line.strip() == '0.0 0.0 0.0 1.0 0.0':
#         label = 3
#     elif line.strip() == '0.0 0.0 1.0 0.0 0.0':
#         label = 2
#     elif line.strip() == '0.0 1.0 0.0 0.0 0.0':
#         label = 1
#     sizes[label] += 1
#     labels.append(label)
# print(sizes)
#
# sents = []
# for line in open('/Users/yxu132/pub-repos/nmt/data/yelp_clss/yelp_train.in', 'r'):
#     sents.append(line.strip())
#
# balance_number = 100000
#
# shuffle_ids = np.arange(len(labels))
# np.random.shuffle(shuffle_ids)
#
# sents = np.array(sents)[shuffle_ids]
# labels = np.array(labels)[shuffle_ids]
#
# filtered_sents = []
# fitlered_labels = []
#
# filtered_size = [0, 0, 0, 0, 0]
#
# for ind, sent in enumerate(sents):
#     if filtered_size[labels[ind]] < balance_number:
#         filtered_sents.append(sent)
#         fitlered_labels.append(labels[ind])
#         filtered_size[labels[ind]] += 1
#
# shuffle_ids = np.arange(len(fitlered_labels))
# np.random.shuffle(shuffle_ids)
#
# sents = np.array(filtered_sents)[shuffle_ids]
# labels = np.array(fitlered_labels)[shuffle_ids]
#
# sizes = [0, 0, 0, 0, 0]
# for label in labels:
#     sizes[label] += 1
# print(sizes)
#
# with open('/Users/yxu132/pub-repos/nmt/data/yelp_clss/yelp_train.balanced.multi.in', 'w') as output_file:
#     output_file.write('\n'.join(sents))
#
#
# with open('/Users/yxu132/pub-repos/nmt/data/yelp_clss/yelp_train.balanced.label.multi.out', 'w') as output_file:
#     for label in labels:
#         if label == 0:
#             output_file.write('1.0 0.0 0.0 0.0 0.0\n')
#         elif label == 1:
#             output_file.write('0.0 1.0 0.0 0.0 0.0\n')
#         elif label == 2:
#             output_file.write('0.0 0.0 1.0 0.0 0.0\n')
#         elif label == 3:
#             output_file.write('0.0 0.0 0.0 1.0 0.0\n')
#         elif label == 4:
#             output_file.write('0.0 0.0 0.0 0.0 1.0\n')

# import numpy as np
#
# outputs = np.zeros((32, 5), dtype=np.float)
# outputs[np.arange(32), np.array([1]*32)] = 1.0
# print(outputs)

# negs, poss = 0, 0
# for line in open('/Users/yxu132/pub-repos/nmt/data/yelp_clss/yelp_dev.tiny.label.bin.out', 'r'):
#     if line.strip() == '1.0 0.0':
#         negs += 1
#     else:
#         poss += 1
# print('pos num: '+str(poss))
# print('neg num: '+str(negs))

# import os
# def rec_count_lines(dir):
#     num = 0
#     for file in os.listdir(dir):
#         if file.strip().endswith('.py'):
#             for line in open(dir+'/'+file, 'r'):
#                 num += 1
#         elif os.path.isdir(dir+'/'+file):
#             num += rec_count_lines(dir+'/'+file)
#     return num
#
# print(rec_count_lines('.'))

# import utils
# import json, codecs
# import numpy as np
# vocab = utils.readlines('/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/vocab.in')
# vocab = ["<unk>", "<s>", "</s>"] + vocab
# # emb_matrix = np.array(json.load(codecs.open('/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/emb.json', "r", "utf-8")))
# emb_matrix = np.array(json.load(codecs.open('emb.refined.json', "r", "utf-8")))
#
# np.random.seed(0)
# emb_mat_var = np.random.rand(3, 300)
# emb_mat_var = np.array(emb_mat_var, dtype=np.float32)
# emb_matrix = np.concatenate([emb_mat_var, emb_matrix], axis=0)
# # emb_matrix = np.array(json.load(codecs.open('emb.refined.json', "r", "utf-8")))
#
def parse(input_line):

    sent = []
    att = []
    comps = input_line.split()
    for comp in comps:
        word = comp[:-7]
        att_score = float(comp[-6:-1])
        sent.append(word)
        att.append(att_score)
    return sent, att
#
# def avg_emb(sent1, att1):
#     weighted_embs = []
#     for ind, word in enumerate(sent1):
#         word_index = vocab.index(word)
#         emb = emb_matrix[word_index]
#         weighted_emb = emb * att1[ind]
#         weighted_embs.append(weighted_emb)
#     avg_weighted_embs = sum(weighted_embs) / len(weighted_embs)
#     return avg_weighted_embs
#
# def minmax_norm(arr):
#     return (arr - min(arr)) / (max(arr)-min(arr))
#
# import evaluate
# from scipy.spatial.distance import cosine
# def sim_score(input_line1, input_line2):
#     sent1, att1 = parse(input_line1)
#     sent2, att2 = parse(input_line2)
#     # att1 = minmax_norm(np.array(att1))
#     # att2 = minmax_norm(np.array(att2))
#     emb1 = avg_emb(sent1, att1)
#     emb2 = avg_emb(sent2, att2)
#     print(cosine([emb1], [emb2]))
#     # print(evaluate.cosine_similarity([emb1], [emb2]))
#
# from numpy import linalg as LA
# def cos_dist_a(emb1, emb2):
#     emb1_norm = LA.norm(emb1, ord=2)
#     emb2_norm = LA.norm(emb2, ord=2)
#     cos_sim = np.sum(np.multiply(emb1_norm, emb2_norm))
#     return 1-cos_sim
#
# import tensorflow as tf
# a = tf.placeholder(tf.float32, shape=[None, None], name="input_placeholder_a")
# b = tf.placeholder(tf.float32, shape=[None, None], name="input_placeholder_b")
# normalize_a = tf.nn.l2_normalize(a,1)
# normalize_b = tf.nn.l2_normalize(b,1)
# cos_distance=1-tf.reduce_sum(tf.multiply(normalize_a,normalize_b), axis=-1)
# sess=tf.Session()
# # cos_dist=sess.run(cos_distance,feed_dict={a:[0.4, 0.8, 0.1, 0.9],b:[-0.4, 0.8, -0.1, -0.9]})
# # print(cos_dist)
#
# # from scipy.spatial.distance import cosine
# # print(cosine([[0.4, 0.8, 0.1, 0.9]], [[-0.4, 0.8, -0.1, -0.9]]))
#
# def validate_():
#
#     output_dir = 'learned_models'
#     dec_tgt_seqs = np.load(output_dir + '/dec_tgt_seq.npy')
#     dec_tgt_embs = np.load(output_dir + '/dec_tgt_emb.npy')
#     dec_tgt_avgs = np.load(output_dir + '/dec_tgt_avg.npy')
#     dec_tgt_alphas = np.load(output_dir + '/dec_tgt_alpha.npy')
#     sample_seqs = np.load(output_dir + '/sample_seq.npy')
#     sample_embs = np.load(output_dir + '/sample_emb.npy')
#     sample_avgs = np.load(output_dir + '/sample_avg.npy')
#     sample_alphas = np.load(output_dir + '/sample_alpha.npy')
#     dist_losss = np.load(output_dir + '/dist_loss.npy')
#
#     attention_weights = np.load(output_dir + '/attention_weights.npy')[0]
#
#     sample_seqs_argmax = np.argmax(sample_seqs, axis=-1)
#
#     sims = []
#
#     for ind, dec_tgt_seq in enumerate(dec_tgt_seqs):
#
#         print(' '.join([vocab[aa] for aa in dec_tgt_seq]))
#         print(' '.join([vocab[aa] for aa in sample_seqs_argmax[ind]]))
#
#         dec_tgt_emb = dec_tgt_embs[ind]
#         dec_tgt_avg = dec_tgt_avgs[ind]
#         dec_tgt_alpha = dec_tgt_alphas[ind]
#         embs = []
#         for j, word_index in enumerate(dec_tgt_seq):
#             # word_index = vocab.index(word)
#             emb = emb_matrix[word_index]
#             embs.append(emb * attention_weights[ind][j])
#             # emb_ = dec_tgt_emb[j]
#             print()
#         avg_embs = sum(embs) / len(embs)
#
#         embs1 = []
#         for j, word_index in enumerate(sample_seqs_argmax[ind]):
#             # word_index = vocab.index(word)
#             emb = emb_matrix[word_index]
#             embs1.append(emb * sample_alphas[ind][j])
#             # emb_ = dec_tgt_emb[j]
#             print()
#         avg_embs_2 = sum(embs1) / len(embs1)
#
#         sample_avg = sample_avgs[ind]
#         score_0 = cos_dist_a(dec_tgt_avg, avg_embs_2)
#         # score_0 = sess.run(cos_distance,feed_dict={a:dec_tgt_avg,b:avg_embs_2})
#         score_1 = cosine(dec_tgt_avg, sample_avgs[ind])
#         sims.append(score_1)
#         print()
#         print()
#
#     score_0 = sess.run(cos_distance, feed_dict={a: dec_tgt_avgs, b: sample_avgs})
#
#     print(sum(sims)/len(sims))
#     print()




# line1 = 'the(0.012) worst(0.073) pizza(0.072) ever(0.074) .(0.064) pizza(0.036) dough(0.019) is(0.009) nt(0.010) supposed(0.024) to(0.013) taste(0.017) like(0.016) wonder(0.026) bread(0.016) ,(0.019) all(0.009) doughy(0.025) &(0.016) sweet(0.003) .(0.013) this(0.007) place(0.004) is(0.006) an(0.010) insult(0.013) to(0.013) italian(0.028) pizza(0.021) .(0.025) these(0.004) people(0.010) are(0.001) nt(0.043) even(0.037) italian(0.035) .(0.028) just(0.007) sad(0.042) .(0.042)'
# line2 = 'the(0.014) awesome(0.061) pizza(0.057) ever(0.057) .(0.046) pizza(0.030) dough(0.015) is(0.005) supposed(0.024) to(0.012) go(0.014) everything(0.016) is(0.004) similar(0.011) bread(0.015) ,(0.021) all(0.014) doughy(0.032) &(0.016) sweet(0.004) .(0.014) this(0.014) place(0.017) and(0.021) an(0.006) dolmades(0.017) liquor(0.017) split(0.047) parmesan(0.039) .(0.040) keeps(0.035) th3(0.021) still(0.014) also(0.027) loose(0.020) .(0.009) do(0.009) great(0.056) rewards(0.037) .(0.040)'
# sim_score(line1, line2)

# validate_()

# with open('../nmt/data/res_para-80-parse.txt', 'w') as output:
#     for line in open('../nmt/data/res_para-80.txt', 'r'):
#         if line.strip() != '':
#             comps = line.strip().split('\t')
#             sent, att = parse(comps[1])
#             line = comps[0]+'\t'+' '.join(sent) +'\t'+comps[2]
#         output.write(line+'\n')

# import nltk
# nltk.download('sentiwordnet')

# from nltk.corpus import sentiwordnet as swn
# excellent = swn.senti_synsets('disgusting', 'a')
# for a in excellent:
#     print(a.synset._lemma_names)
#     print(a)
# all = swn.all_senti_synsets()
# negative_words = []
# for a in all:
#     if (a.synset._pos == 'a' or a.synset._pos=='r' or a.synset._pos=='s') and a.neg_score() > 0.6 and a.pos_score()<=0.25:
#         negative_words.extend(a.synset._lemma_names)
#         # print(a.synset._lemma_names)
#         # print(a)
# print('\n'.join(set(negative_words)))
# print(all)

# for line in open('test.txt', 'r'):
#     print(len(line.strip().split()))

# import tensorflow as tf
#
# a = tf.constant(1.3, name='const_a')
# b = tf.Variable(3.1, name='variable_b')
# c = tf.add(a, b, name='addition')
# d = tf.multiply(c, a, name='multiply')
#
# for op in tf.get_default_graph().get_operations():
#     print(str(op.name))


# import urllib.request
#
# print('Beginning file download with urllib2...')
#
# url = 'https://www.kaggle.com/c/16880/datadownload/dfdc_train_part_00.zip'
# urllib.request.urlretrieve(url, '0.zip')

# import json
# def _create_pretrained_emb_from_counter(embed_file):
#
#     emb_mat = []
#     for line in open(embed_file, 'r'):
#         comps = line.strip().split()
#         emb = [float(t) for t in comps[1:]]
#         emb_mat.append(emb)
#     # np.save('/Users/yxu132/pub-repos/TextFooler1/counter-fitted-emb.npy', np.array(emb_mat))
#     with open('/Users/yxu132/pub-repos/TextFooler1/counter-fitted-emb.json', 'w') as outfile:
#         json.dump(emb_mat, outfile)
#
# _create_pretrained_emb_from_counter('/Users/yxu132/pub-repos/TextFooler1/counter-fitted-vectors.txt')

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import rc
# #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
# plt.rcParams['font.size'] = 12
#
# M = np.array([[0.681, 0.290],
# [0.854, 0.450],
# [1.073, 0.333],
# [1.051, 0.409],
# [0.379, 0.859],
# [0.369, 0.930],
# [0.317, 0.956],
# [0.454, 0.961]])

# M_p = []
# for wei in M:
#     normed = wei / np.sum(wei)
#     M_p.append(normed)
# M_p = np.array(M_p)
# print(M_p)
#
# print_categories=['cross_stitch_unit(A*)', 'cross_stitich_unit(B*)']
# yprint_categories=['fc1', 'conv3', 'conv2', 'conv1', 'fc1', 'conv3', 'conv2', 'conv1', ]
#
#
# plt.figure(figsize=(12, 6))
# sns.heatmap(M_p,cmap="Blues", annot=True, xticklabels=print_categories, yticklabels=yprint_categories, fmt='.3f')
# plt.ylabel('cross_stitch_unit(*B)                cross_stitch_unit(*A)')
# plt.yticks(rotation=0)
# # plt.show()
# plt.savefig('corr.png', dpi=300)

# import tensorflow as tf
# import numpy as np
# indices = [-1, 1, 3, 7, -1]
# depth = 9
# arr = tf.constant([[1],[0], [1], [1]], tf.int32)
# b = tf.tile(arr, [1, 2])
# a = tf.reduce_sum(tf.one_hot(indices, depth), axis=0)
#
# vocab_size=10
# time = 0
# _encoder_input_sim_ids = tf.constant([[[3, 5, 1, -1, -1], [4, 8, 7, -1, -1]],
#                           [[4, 6, 7, 2, -1], [5, 1, -1, -1, -1]],
#                           [[5, -1, -1, -1, -1], [5, 8, 9, 1, -1]]
#                           ], tf.int32)
# _copy_mask=tf.constant([[0, 1],
#             [1, 0],
#             [1, 1]
#             ], tf.float32)
#
# _encoder_input_ids = tf.constant([[5, 7],
#                       [4, 1],
#                       [5, 9]], tf.int32)
#
# ids_cur = _encoder_input_ids[:, time]
# copy_mask_cur = _copy_mask[:, time] # [batch_size, ?]
# keep_mask = tf.tile(tf.expand_dims(1 - copy_mask_cur, -1), [1, vocab_size])
# prob_one_hot = tf.one_hot(ids_cur, vocab_size)
# prob_c_one_hot = prob_one_hot * tf.expand_dims(copy_mask_cur, -1)
# copy_mask = prob_c_one_hot + keep_mask
# cur_lex_mask = tf.reduce_sum(tf.one_hot(_encoder_input_sim_ids[:, time], vocab_size, axis=-1), axis=1)
#
# mask = copy_mask * cur_lex_mask
#
# with tf.Session() as sess:
#     aa = sess.run(mask)
#     print(aa)

# import numpy as np
# a = np.array([[1, 2, 3, 4, 5]])
# a[:, :10] = -1
# print(a)


# import torch
# from transformers import *
#
# # PyTorch-Transformers has a unified API
# # for 7 transformer architectures and 30 pretrained weights.
# #          Model          | Tokenizer          | Pretrained weights shortcut
# MODELS = [(XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
#           ]
#
# # Let's encode some text in a sequence of hidden-states using each model:
# for model_class, tokenizer_class, pretrained_weights in MODELS:
#     # Load pretrained model/tokenizer
#     tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#     model = model_class.from_pretrained(pretrained_weights)


import tensorflow as tf

val = 3
m = tf.placeholder(tf.int32)
m_feed = [[0  ,   0, val,   0, val],
          [val,   0, val, val,   0],
          [0  , val,   0,   0,   0]]

tmp_indices = tf.where(tf.equal(m, val))
result = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])

with tf.Session() as sess:
    print(sess.run(result, feed_dict={m: m_feed})) # [2, 0, 1]