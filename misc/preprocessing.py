import os
import spacy
from misc import utils

nlp = spacy.load('en')
tokenizer = lambda x: nlp(x)

glove_emb_path = '/dccstor/ddig/ying/data/nmtree'

def parse_files(input_dir, output_name):
    sents = []
    for file in os.listdir(input_dir):
        for line in open(input_dir+'/'+file, 'r'):
            par = line.strip().lower()
            par = par.replace('<br />', ' ')
            par = par.replace('(', ' (').replace(')', ') ')
            par = par.replace('"', ' " ')
            par = par.replace(',', ', ').replace('?', '? ').replace('!', '! ').replace('.', '. ')
            par = par.replace('*', ' ').replace('-', ' ').replace('~', ' ')
            par = par.strip()
            par = ' '.join(par.split())
            par = nlp(par)
            par = [tok.text for tok in par]
            sents.append(' '.join(par))
    with open('../nmt/data/imdb/'+output_name+'.txt', 'w') as output_file:
        output_file.write('\n'.join(sents))

import numpy as np
def stats(input_dir):
    all_sents = []
    all_labels = []
    for line in open(input_dir+'/positives', 'r'):
        all_sents.append(line.strip().split())
        all_labels.append(1)
    for line in open(input_dir+'/negatives', 'r'):
        all_sents.append(line.strip().split())
        all_labels.append(0)
    print('all: '+str(len(all_sents)))
    sizes = []
    thresholds = [50, 200, 400, 300, 600, 1000]
    thre_nums = [0, 0, 0, 0, 0, 0]
    for sent in all_sents:
        sizes.append(len(sent))
        for ind, thre in enumerate(thresholds):
            if len(sent) < thre:
                thre_nums[ind] += 1
    for ind, thre in enumerate(thresholds):
        print(str(thre)+': '+str(thre_nums[ind]))
    sel_sents, sel_labels = [], []
    for ind, sent in enumerate(all_sents):
        if len(sent) < 50:
            sel_sents.append(sent)
            sel_labels.append(all_labels[ind])
    with open(input_dir+'/imdb_test.len50.in', 'w') as output:
        for sent in sel_sents:
            output.write(' '.join(sent)+'\n')
    with open(input_dir+'/imdb_test.len50.bin.out', 'w') as output:
        for label in sel_labels:
            if label == 1:
                output.write('0.0 1.0\n')
            else:
                output.write('1.0 0.0\n')

def filter_tiny(input_dir, dataset):
    orig_inputs, orig_outputs = [], []
    tiny_inputs, tiny_outputs = [], []
    for line in open(input_dir+'/yelp_'+dataset+'.in', 'r'):
        orig_inputs.append(line.strip())
    for line in open(input_dir+'/yelp_'+dataset+'.label.multi.out', 'r'):
        orig_outputs.append(line.strip())
    for line in open(input_dir+'/yelp_'+dataset+'.tiny.in', 'r'):
        tiny_inputs.append(line.strip())
    for ind, tiny_input in enumerate(tiny_inputs):
        index = orig_inputs.index(tiny_input)
        tiny_outputs.append(orig_outputs[index])
    with open(input_dir+'/yelp_'+dataset+'.tiny.label.multi.out', 'w') as output_file:
        output_file.write('\n'.join(tiny_outputs))


def stats_label_yelp(input_file, input_label_file):
    ratings = [0, 0, 0, 0, 0]
    # lines = utils.readlines(input_file)
    for ind, line in enumerate(open(input_label_file, 'r')):
        if line.strip() == '1.0 0.0 0.0 0.0 0.0':
            ratings[0] += 1
        elif line.strip() == '0.0 1.0 0.0 0.0 0.0':
            ratings[1] += 1
        elif line.strip() == '0.0 0.0 1.0 0.0 0.0':
            # print(lines[ind])
            ratings[2] += 1
        elif line.strip() == '0.0 0.0 0.0 1.0 0.0':
            ratings[3] += 1
        else:
            ratings[4] += 1
    print(ratings)
    print(sum(ratings[:3]))
    print(sum(ratings[3:]))


def balance_label_yelp(input_file, input_label_file):
    ratings = [0, 0, 0, 0, 0]
    sentences = utils.readlines(input_file)
    pos_sents, neg_sents = [], []
    for ind, line in enumerate(open(input_label_file, 'r')):
        if line.strip() == '1.0 0.0 0.0 0.0 0.0':
            ratings[0] += 1
            neg_sents.append(sentences[ind])
        elif line.strip() == '0.0 1.0 0.0 0.0 0.0':
            ratings[1] += 1
            neg_sents.append(sentences[ind])
        elif line.strip() == '0.0 0.0 1.0 0.0 0.0':
            ratings[2] += 1
        elif line.strip() == '0.0 0.0 0.0 1.0 0.0':
            ratings[3] += 1
            pos_sents.append(sentences[ind])
        else:
            ratings[4] += 1
            pos_sents.append(sentences[ind])

    shuffled_ids = np.arange(len(pos_sents))
    np.random.shuffle(shuffled_ids)
    pos_sents = np.array(pos_sents)[shuffled_ids]

    sents = neg_sents + pos_sents.tolist()[:len(neg_sents)]
    labels = ['1.0 0.0'] * len(neg_sents) + ['0.0 1.0'] * len(neg_sents)

    shuffled_ids = np.arange(len(sents))
    np.random.shuffle(shuffled_ids)
    sents = np.array(sents)[shuffled_ids]
    labels = np.array(labels)[shuffled_ids]

    with open('../nmt/data/yelp_clss_x3/yelp_dev.balanced.in', 'w') as output_file:
        for line in sents:
            output_file.write(line+'\n')
    with open('../nmt/data/yelp_clss_x3/yelp_dev.balanced.label.bin.out', 'w') as output_file:
        for line in labels:
            output_file.write(line+'\n')

def glove_vocab(emb_file):
    vocab = []
    for line in tqdm(open(emb_file, 'r'), total=2196017):
        comps = line.strip().split()
        word = ''.join(comps[0:-300])
        vocab.append(word)
    with open('/Users/yxu132/pub-repos/decaNLP/embeddings/vocab.txt', 'w') as output_file:
        output_file.write('\n'.join(vocab))

import json
from tqdm import tqdm
def filter_emb(vocab_file, emb_file):

    vocab = utils.readlines(vocab_file)
    emb_map = dict()
    for line in tqdm(open(emb_file, 'r'), total=2196017):
        comps = line.strip().split()
        word = ''.join(comps[0:-300])
        vec = comps[-300:]
        if word in vocab:
            emb_map[word] = [float(a) for a in vec]

    emb_matrix = []
    for word in vocab:
        if word in emb_map:
            emb_matrix.append(emb_map[word])
        else:
            # emb_matrix.append([0.0]*300)
            print('word '+word+' not found')
            raise OSError
    json.dump(emb_matrix, open('/Users/yxu132/pub-repos/nmt/data/rt-short/emb.json', 'w'))


def gen_emb_for_vocab(vocab_file, emb_file, output_json_file):

    vocab = utils.readlines(vocab_file)
    emb_map = dict()
    for line in tqdm(open(emb_file, 'r')):
        comps = line.strip().split()
        word = ''.join(comps[0:-300])
        vec = comps[-300:]
        emb_map[word] = [float(a) for a in vec]

    emb_matrix = []
    for i in range(3):
        emb_matrix.append([0.0]*300)
    count = 0
    for word in vocab:
        if word in emb_map:
            emb_matrix.append(emb_map[word])
            count+=1
        else:
            emb_matrix.append([0.0]*300)
    print(count)
    json.dump(emb_matrix, open(output_json_file, 'w'))


import codecs
def transform_cf_emb(vocab_file, emb_file, cf_vocab_file, cf_emb_file, output_vocab_file, output_json_file):

    vocab = utils.readlines(vocab_file)
    cf_vocab = utils.readlines(cf_vocab_file)

    print('vocab_size: '+str(len(vocab)))
    print('cf_vocab_size: ' + str(len(cf_vocab)))

    with codecs.open(emb_file, "r", "utf-8") as fh:
        emb = json.load(fh)

    with codecs.open(cf_emb_file, "r", "utf-8") as fh:
        cf_emb = json.load(fh)

    vocab_diff = []
    vocab_diff_ind = []
    for ind, word in enumerate(vocab):
        if word not in cf_vocab:
            vocab_diff.append(word)
            vocab_diff_ind.append(ind)

    print('gen_vocab_size: '+str(len(cf_vocab)))
    print('copy_vocab_size: ' + str(len(vocab_diff_ind)))


    new_cf_vocab = cf_vocab + vocab_diff
    new_emb = cf_emb
    for ind, word in enumerate(vocab_diff):
        new_emb.append(emb[vocab_diff_ind[ind]])

    print('copy_emb_size: ' + str(len(new_emb)))

    utils.write_lines(new_cf_vocab, output_vocab_file)
    json.dump(new_emb, open(output_json_file, 'w'))


def combine_vocab(vocab_f_1, vocab_f_2):
    vocab1 = set(utils.readlines(vocab_f_1))
    vocab2 = set(utils.readlines(vocab_f_2))
    print(len(vocab1))
    print(len(vocab2))
    vocab = vocab1.union(vocab2)
    print(len(vocab))
    with open('../nmt/data/imdb/vocab.in', 'w') as output:
        for line in vocab:
            output.write(line+'\n')
    return

def combine_all_vocab(input_dir, size=10):
    all_vocab = set([])
    for i in range(size):
        vocab = set(utils.readlines(input_dir+'/vocab_'+str(i)+'.txt'))
        all_vocab = all_vocab.union(vocab)

    with open('/Users/yxu132/pub-repos/tf-seq2seq/vocab-90.in', 'w') as output:
        output.write('\n'.join(all_vocab))

def split_dataset(input_file, output_file, output_dir):

    div = 20

    input_sents = utils.readlines(input_file)
    labels = utils.readlines(output_file)

    num_per_div = len(input_sents) // div

    for i in range(div+1):
        utils.write_lines(input_sents[i*num_per_div: (i+1)* num_per_div],
                          output_path=output_dir +os.path.basename(input_file)+'_'+str(i))
        utils.write_lines(labels[i*num_per_div: (i+1)* num_per_div],
                          output_path=output_dir +os.path.basename(output_file)+'_'+str(i))

def combine_splits(input_file, output_file):

    all_sents = []
    for i in range(21):
        input_sents = utils.readlines(input_file+str(i)+'.txt')
        all_sents.extend(input_sents)
    utils.write_lines(all_sents, output_file)

def split_pos_neg(dir_path, input_file, label_file):

    input_sents = utils.readlines(input_file)
    labels = utils.readlines(label_file)

    pos_out_file = open(dir_path+'/imdb_train.pos.in', 'w')
    neg_out_file = open(dir_path+'/imdb_train.neg.in', 'w')
    pos_lab_file = open(dir_path+'/imdb_train.pos.out', 'w')
    neg_lab_file = open(dir_path+'/imdb_train.neg.out', 'w')

    for ind, sent in enumerate(input_sents):
        label = labels[ind]
        if label == '1.0 0.0':
            neg_out_file.write(sent+'\n')
            neg_lab_file.write(label+'\n')
        elif label == '0.0 1.0':
            pos_out_file.write(sent + '\n')
            pos_lab_file.write(label + '\n')

def removeBrace(sim_word):
    sim_word = sim_word[:sim_word.index('(')]
    return sim_word

def rt_long(input_dir):

    pos_sents, neg_sents = [], []
    posd = input_dir+'/pos'
    for file in os.listdir(posd):
        if file.endswith('.txt'):
            with codecs.open(posd+'/'+file, 'r', 'utf-8', errors='replace') as f:
                pos_sents = f.readlines()
    negd = input_dir+'/neg'
    for file in os.listdir(negd):
        if file.endswith('.txt'):
            with codecs.open(negd + '/' + file, 'r', 'utf-8', errors='replace') as f:
                neg_sents = f.readlines()
    pos_sents = [sent.strip() for sent in pos_sents]
    neg_sents = [sent.strip() for sent in neg_sents]

    pos_tokens = [sent.split() for sent in pos_sents]
    neg_tokens = [sent.split() for sent in neg_sents]

    tokens = []
    for sent in pos_tokens:
        tokens.extend(sent)
    for sent in neg_tokens:
        tokens.extend(sent)

    vocabs = set(tokens)
    with open(input_dir+'/vocab.in', 'w') as output_file:
        output_file.write('\n'.join(vocabs))

    pos_lengths = [len(token) for token in pos_tokens]
    neg_lengths = [len(token) for token in neg_tokens]
    avg_len = sum(pos_lengths+neg_lengths)/(len(pos_lengths)+len(neg_lengths))
    print(avg_len)

    train_size = 500
    val_size = 100

    train_set = pos_sents[:train_size] + neg_sents[:train_size]
    train_label = [['0', '1']] * train_size + [['1', '0']] * train_size
    val_set = pos_sents[train_size:train_size+val_size] + neg_sents[train_size:train_size+val_size]
    val_label = [['0', '1']] * (val_size) + [['1', '0']] * val_size
    test_set = pos_sents[train_size+val_size:] + neg_sents[train_size+val_size:]
    test_label = [['0', '1']] * (len(pos_sents)-train_size-val_size) + [['1', '0']] * (len(neg_sents)-train_size-val_size)

    np.random.seed(0)
    shuffled_ids = np.arange(len(train_set))
    np.random.shuffle(shuffled_ids)
    train_set = np.array(train_set)[shuffled_ids]
    train_label = np.array(train_label)[shuffled_ids]

    np.random.seed(0)
    shuffled_ids = np.arange(len(val_set))
    np.random.shuffle(shuffled_ids)
    val_set = np.array(val_set)[shuffled_ids]
    val_label = np.array(val_label)[shuffled_ids]

    np.random.seed(0)
    shuffled_ids = np.arange(len(test_set))
    np.random.shuffle(shuffled_ids)
    test_set = np.array(test_set)[shuffled_ids]
    test_label = np.array(test_label)[shuffled_ids]

    with open(input_dir+'/train.in', 'w') as output_file:
        output_file.write('\n'.join(train_set))
    with open(input_dir+'/train.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in train_label]))

    with open(input_dir+'/dev.in', 'w') as output_file:
        output_file.write('\n'.join(val_set))
    with open(input_dir+'/dev.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in val_label]))

    with open(input_dir+'/test.in', 'w') as output_file:
        output_file.write('\n'.join(test_set))
    with open(input_dir+'/test.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in test_label]))

def write_to_vocab(input_dir, vocabs):

    with codecs.open(glove_emb_path+'/vocab.txt', 'r') as f:
        glove_emb = f.readlines()

    glove_emb = [word.strip() for word in glove_emb]
    glove_emb = set(glove_emb)


    print('total vocabs: '+str(len(vocabs)) )
    words = []
    for token in vocabs:
        if token in glove_emb:
            words.append(token)
    vocabs = words
    print('found vocabs: ' + str(len(vocabs)))

    with open(input_dir+'/vocab.in', 'w') as output_file:
        output_file.write('\n'.join(vocabs))

def write_to_files(input_dir, train_set, train_label, val_set, val_label, test_set, test_label):

    with open(input_dir+'/train.in', 'w') as output_file:
        output_file.write('\n'.join(train_set))
    with open(input_dir+'/train.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in train_label]))

    with open(input_dir+'/dev.in', 'w') as output_file:
        output_file.write('\n'.join(val_set))
    with open(input_dir+'/dev.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in val_label]))

    with open(input_dir+'/test.in', 'w') as output_file:
        output_file.write('\n'.join(test_set))
    with open(input_dir+'/test.out', 'w') as output_file:
        output_file.write('\n'.join([' '.join(aa) for aa in test_label]))


def rt_short(input_dir):

    with codecs.open(input_dir+'/rt-polaritydata/rt-polarity.pos', 'r', 'utf-8', errors='ignore') as f:
        pos_sents = f.readlines()
    with codecs.open(input_dir+'/rt-polaritydata/rt-polarity.neg', 'r', 'utf-8', errors='ignore') as f:
        neg_sents = f.readlines()
    pos_sents = [sent.strip() for sent in pos_sents]
    neg_sents = [sent.strip() for sent in neg_sents]

    def parse_sent(sent):
        sent = sent.replace(' \'', ' \' ') if ' \'' in sent else sent
        sent = sent.replace('\' ', ' \' ') if '\' ' in sent else sent
        sent = sent.replace('\'', '\' ') if sent.startswith('\'') else sent
        sent = sent.replace('\'', ' \'') if sent.endswith('\'') else sent

        sent = sent.replace('n\'t', ' n\'t') if 'n\'t' in sent else sent
        sent = sent.replace('\'s', ' \'s') if '\'s' in sent else sent
        sent = sent.replace('\'ve', ' \'ve') if '\'ve' in sent else sent
        sent = sent.replace('\'ll', ' \'ll') if '\'ll' in sent else sent
        sent = sent.replace('\'m', ' \'m') if '\'m' in sent else sent
        sent = sent.replace('\'re', ' \'re') if '\'re' in sent else sent
        sent = sent.replace('\'d', ' \'d') if '\'d' in sent else sent

        sent = sent.replace('[', ' [ ') if '[' in sent else sent
        sent = sent.replace(']', ' ] ') if ']' in sent else sent
        sent = sent.replace('-', ' - ') if '-' in sent else sent
        sent = sent.replace('<', ' <') if '<' in sent else sent
        sent = sent.replace('>', '> ') if '>' in sent else sent
        sent = sent.replace('/', ' / ') if '/' in sent else sent
        sent = ' '.join(sent.split()).strip()
        return sent

    possents, negsents = [], []
    for sent in pos_sents:
        sent = parse_sent(sent)
        possents.append(sent)
    for sent in neg_sents:
        sent = parse_sent(sent)
        negsents.append(sent)
    pos_sents = possents
    neg_sents = negsents

    pos_tokens = [sent.split() for sent in pos_sents]
    neg_tokens = [sent.split() for sent in neg_sents]

    pos_lengths = [len(token) for token in pos_tokens]
    neg_lengths = [len(token) for token in neg_tokens]
    max_len = max(pos_lengths+neg_lengths)
    print(max_len)

    tokens = []
    for sent in pos_tokens:
        tokens.extend(sent)
    for sent in neg_tokens:
        tokens.extend(sent)

    vocabs = set(tokens)

    write_to_vocab(input_dir, vocabs)


    pos_lengths = [len(token) for token in pos_tokens]
    neg_lengths = [len(token) for token in neg_tokens]
    avg_len = sum(pos_lengths+neg_lengths)/(len(pos_lengths)+len(neg_lengths))
    print(avg_len)

    train_size = 4331
    val_size = 500

    train_set = pos_sents[:train_size] + neg_sents[:train_size]
    train_label = [['0', '1']] * train_size + [['1', '0']] * train_size
    val_set = pos_sents[train_size:train_size+val_size] + neg_sents[train_size:train_size+val_size]
    val_label = [['0', '1']] * (val_size) + [['1', '0']] * val_size
    test_set = pos_sents[train_size+val_size:] + neg_sents[train_size+val_size:]
    test_label = [['0', '1']] * (len(pos_sents)-train_size-val_size) + [['1', '0']] * (len(neg_sents)-train_size-val_size)

    def shuffle_data(input_set, output_set):
        np.random.seed(0)
        shuffled_ids = np.arange(len(input_set))
        np.random.shuffle(shuffled_ids)
        input_set = np.array(input_set)[shuffled_ids]
        output_set = np.array(output_set)[shuffled_ids]
        return input_set, output_set

    train_set, train_label = shuffle_data(train_set, train_label)
    val_set, val_label = shuffle_data(val_set, val_label)
    test_set, test_label = shuffle_data(test_set, test_label)

    write_to_files(input_dir, train_set, train_label, val_set, val_label, test_set, test_label)



def snli_extract_(lines):
    sentences, labels = [], []
    for ind, line in enumerate(lines):
        if ind == 0:
            continue
        comps = line.strip().split('\t')
        sentences.append(comps[5].lower() + ' ' + comps[6].lower())
        labels.append(comps[0])
    return sentences, labels


def snli_tokenize(lines):
    sents, tokens = [], []
    for line in tqdm(lines, total=len(lines)):
        par = nlp(line)
        par = [tok.text for tok in par]
        tokens.extend(par)
        sents.append(' '.join(par))
    return sents, tokens


def snli(data_dir):
    train_lines = utils.readlines(data_dir + '/snli_1.0/snli_1.0_train.txt')
    dev_lines = utils.readlines(data_dir + '/snli_1.0/snli_1.0_dev.txt')
    test_lines = utils.readlines(data_dir + '/snli_1.0/snli_1.0_test.txt')

    train_sentences, train_labels = snli_extract_(train_lines)
    dev_sentences, dev_labels = snli_extract_(dev_lines)
    test_sentences, test_labels = snli_extract_(test_lines)

    train_sentences, train_tokens = snli_tokenize(train_sentences)
    dev_sentences, dev_tokens = snli_tokenize(dev_sentences)
    test_sentences, test_tokens = snli_tokenize(test_sentences)

    label_set = list(set(dev_labels))
    utils.write_lines(label_set, data_dir + '/labels.txt')
    label_size = len(label_set)
    label2ind = {label: ind for ind, label in enumerate(label_set)}

    def one_hot(labels):
        new_labels = []
        for label in labels:
            new_label = [0] * label_size
            new_label[label2ind[label]] = 1
            new_labels.append([str(a) for a in new_label])
        return new_labels

    train_labels = one_hot(train_labels)
    dev_labels = one_hot(dev_labels)
    test_labels = one_hot(test_labels)

    write_to_files(data_dir, train_sentences, train_labels, dev_sentences, dev_labels, test_sentences, test_labels)

    vocabs = set(train_tokens + dev_tokens + test_tokens)
    write_to_vocab(data_dir, vocabs)

def split_snli_input(input_file, output_path):

    sents = utils.readlines(input_file)
    texts, hypos = [], []
    for sent in sents:
        comps = sent.split(' . ')
        texts.append(comps[0])
        hypos.append(comps[1])

    utils.write_lines(texts, output_path+'.prem.in')
    utils.write_lines(hypos, output_path+'.hypo.in')


if __name__ == '__main__':
    # parse_files('/Users/yxu132/pub-repos/text_classification/data/aclImdb/test/pos', 'test_pos')
    # stats('../nmt/data/imdb')
    # filter_tiny('../nmt/data/yelp_clss', 'test')
    # stats_label_yelp('../nmt/data/yelp_clss/yelp_dev.in', '../nmt/data/yelp_clss/yelp_train.label.multi.out')
    # balance_label_yelp('../nmt/data/yelp_clss/yelp_dev.in', '../nmt/data/yelp_clss/yelp_dev.label.multi.out')

    # combine_vocab('../nmt/data/yelp_clss/vocab.in', '../nmt/data/imdb/imdb-400/vocab.txt')
    # combine_all_vocab('/Volumes/YING_backup/datasets/para7090/para90', size=12)
    #
    # glove_vocab('/Users/yxu132/pub-repos/decaNLP/embeddings/glove.840B.300d.txt')
    # filter_emb('/Users/yxu132/pub-repos/nmt/data/rt-short/vocab.in', '/Users/yxu132/pub-repos/decaNLP/embeddings/glove.840B.300d.txt')
    # gen_emb_for_vocab('/Users/yxu132/pub-repos/nmt/data/rt-short/vocab.in',
    #                   '/Users/yxu132/pub-repos/usif/vectors/czeng.txt',
    #                   '/Users/yxu132/pub-repos/nmt/data/rt-short/rt-vocab-paranmt-emb.json')
    # transform_cf_emb('/Users/yxu132/pub-repos/nmt/data/rt-short/vocab.in',
    #                  '/Users/yxu132/pub-repos/nmt/data/rt-short/emb.json',
    #                  '/Users/yxu132/pub-repos/TextFooler1/vocab.txt',
    #                  '/Users/yxu132/pub-repos/TextFooler1/counter-fitted-emb.json',
    #                  '/Users/yxu132/pub-repos/nmt/data/rt-short/copy_vocab.in',
    #                  '/Users/yxu132/pub-repos/nmt/data/rt-short/copy_emb.json')
    # split_dataset('/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.in',
    #               '/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.label.bin.out',
    #               '/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/train_splits/')

    # combine_splits('/Users/yxu132/pub-repos/tf-seq2seq/los_analysis/adv-yelp50-rnn-tsai-train-T2-dev',
    #                '/Users/yxu132/pub-repos/nmt/data/yelp_clss_x3/yelp_train.balanced.tsai.in')

    # split_pos_neg('/Users/yxu132/pub-repos/nmt/data/imdb-inf/',
    #               '/Users/yxu132/pub-repos/nmt/data/imdb-inf/imdb_train.in',
    #               '/Users/yxu132/pub-repos/nmt/data/imdb-inf/imdb_train.out')


    # rt_long('/Users/yxu132/pub-repos/nmt/data/rt-long')
    # rt_short('/Users/yxu132/pub-repos/nmt/data/rt-short')
    snli('/dccstor/ddig/ying/data/snli')
    split_snli_input('/dccstor/ddig/ying/data/snli/train.in', '/dccstor/ddig/ying/data/snli/train')
    split_snli_input('/dccstor/ddig/ying/data/snli/dev.in', '/dccstor/ddig/ying/data/snli/dev')
    split_snli_input('/dccstor/ddig/ying/data/snli/test.in', '/dccstor/ddig/ying/data/snli/test')


