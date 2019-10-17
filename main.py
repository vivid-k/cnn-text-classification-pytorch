#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import model
import train
import pickle

import data_helper as utils


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=512, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

# device
parser.add_argument('-device', type=int, default=1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default="./snapshot/2019-10-17_14-40-13/best_steps_200.pt", help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default="predict", help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-eval_size', action='store_true', default=False, help='train or test')
args = parser.parse_args()



# load data
print("\nLoading data...")
data = pickle.load(open('./origin_data/data.pkl', 'rb'))
vocab = data['dict']['src']
vocab_size = vocab.size()
args.embed_num = vocab_size
trainset = utils.BiDataset(data['train'])
validset = utils.BiDataset(data['valid'])
testset = utils.BiDataset(data['test'])
args.eval_size = data['valid']['length']
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=0,
                                            collate_fn=utils.padding)
validloader = torch.utils.data.DataLoader(dataset=validset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=utils.padding)
testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            collate_fn=utils.padding)

args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        
# train or predict
if args.predict is not None:
    text = list("今年年前我一定要发一篇论文")
    text = vocab.convertToIdx(text, '<unk>')
    label = train.predict(text, cnn, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(testloader, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(trainloader, validloader, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')