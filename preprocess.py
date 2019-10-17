import argparse
import pickle
from dict_helper import Dict
import random

parser = argparse.ArgumentParser(description='preprocess.py')
parser.add_argument('-load_data', default='./origin_data/',
                    help="input file for the data")
parser.add_argument('-save_data', default='./origin_data/',
                    help="Output file for the prepared data")
parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_filter', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-src_trun', type=int, default=0,
                    help="Truncate source sequence length")
parser.add_argument('-src_char', action='store_true', help='character based encoding')
parser.add_argument('-src_suf', default='tgt',
                    help="the suffix of the source filename")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-sample_num', type=int, default=1200000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

PAD_WORD = "<blank>"
UNK_WORD = "<unk>"

def makeVocabulary(filename, trun_length, filter_length, char, vocab, size):

    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding='utf8') as f:
        for sent in f.readlines():
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            max_length = max(max_length, len(tokens))
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))

    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def sample_data(vacab_size):
    # if opt.src_char: # 句子长度为8-30
    len = random.randint(8, 30)
    sentence = []
    for i in range(len):
        w = random.randint(1, vacab_size-1)
        sentence.append(str(w))
    sen = ' '.join(sentence)

    return sen

def makeData(srcFile, srcDicts, save_srcFile, lim=0):
    sizes = 0
    count, limit_ignored = 0, 0

    print('Processing %s ...' % srcFile)
    srcF = open(srcFile, encoding='utf8')
    srcIdF = open(save_srcFile + '.id', 'w')


    while True:
        sline = srcF.readline()
        # normal end of file
        if sline == "" :
            break
        sline = sline.strip().lower()
        srcWords = sline.split() if not opt.src_char else list(sline)
        # if len(srcWords) > max:
        #     max = len(srcWords)
        # if len(srcWords) < min:
        #     min = len(srcWords)
        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter):

            if opt.src_trun > 0:
                srcWords = srcWords[:opt.src_trun]

            srcIds = srcDicts.convertToIdx(srcWords, UNK_WORD)
            srcIds_sam = sample_data(srcDicts.size())
            srcIdF.write(" ".join(list(map(str, srcIds)))+'\t'+'1'+'\n')
            srcIdF.write(srcIds_sam+'\t'+'0'+'\n')
            sizes += 2
        else:
            limit_ignored += 1

        count += 2

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()

    srcIdF.close()

    print('Prepared %d sentences (%d ignored due to length == 0 or > )' %
          (sizes, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'length': sizes}


def main():
    dicts = {}

    train_src = opt.load_data + 'train.' + opt.src_suf
    valid_src = opt.load_data + 'valid.' + opt.src_suf
    test_src = opt.load_data + 'test.' + opt.src_suf

    save_train_src = opt.save_data + 'train.' + opt.src_suf
    save_valid_src = opt.save_data + 'valid.' + opt.src_suf
    save_test_src = opt.save_data + 'test.' + opt.src_suf

    src_dict = opt.save_data + 'src.dict'

    dicts['src'] = Dict([PAD_WORD, UNK_WORD])
    print('Building source vocabulary...')
    dicts['src'] = makeVocabulary(train_src, opt.src_trun, opt.src_filter, opt.src_char, dicts['src'], opt.src_vocab_size)


    print('Preparing training ...')
    train = makeData(train_src, dicts['src'], save_train_src)

    print('Preparing validation ...')
    valid = makeData(valid_src, dicts['src'], save_valid_src)

    print('Preparing test ...')
    test = makeData(test_src, dicts['src'], save_test_src)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    data = {'train': train, 'valid': valid,
             'test': test, 'dict': dicts}
    pickle.dump(data, open(opt.save_data+'data.pkl', 'wb'))


if __name__ == "__main__":
    main()
