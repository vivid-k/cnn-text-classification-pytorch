import linecache
import torch
import torch.utils.data as torch_data
from random import Random

class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False):

        self.srcF = infos['srcF']
        self.length = infos['length']
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        data = linecache.getline(self.srcF, index+1).strip().split('\t')
        src = list(map(int, data[0].split()))
        tgt = list(map(int, data[1]))

        return src, tgt

    def __len__(self):
        return len(self.indexes)


def padding(data):
    src, tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[:end])

    return src_pad, tgt
