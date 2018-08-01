#!/usr/bin/env python
"""Sample script of word embedding model.
This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections

import numpy as np
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions
import proceeding as proc

# pretrainingのモデルから抽出
def get_model_data():

    text_name = 'model/pre_train.model'

    with open(text_name, 'r') as f:
        ss = f.readline().split()
        n_vocab, n_units = int(ss[0]), int(ss[1])
        word2index = {}
        index2word = {}
        vocabulary = set()
        w = np.empty((n_vocab, n_units), dtype=np.float32)
        for i, line in enumerate(f):
            ss = line.split()
            assert len(ss) == n_units + 1
            word = ss[0]
            vocabulary.add(word)
            word2index[word] = i
            index2word[i] = word
            w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

    return w, word2index

# CBOWです。こっちはいじった。
class ContinuousBoW(chainer.Chain):
    """Definition of Continuous Bag of Words Model"""

    def __init__(self, n_vocab, n_units, loss_func, ori_con_data, vocab):
        super(ContinuousBoW, self).__init__()

        with self.init_scope():

            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            vector, word2index= get_model_data()

            # initialized
            for element in vocab.keys():
                try:            
                    self.embed.W.data[vocab[element]] = vector[word2index[element]]
                except:
                    pass

            for i in range(len(ori_con_data)):
                self.embed.W.data[ori_con_data[i][0]] = self.embed.W.data[ori_con_data[i][1]]

            self.loss_func = loss_func



    def __call__(self, x, contexts, flg_save, epoch, index2word):

        e = self.embed(contexts)
        h = F.sum(e, axis=1) * (1. / contexts.shape[1])
        x = x.astype(np.int32)
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)



        return loss



class SoftmaxCrossEntropyLoss(chainer.Chain):
    """Softmax cross entropy loss function preceded by linear transformation.
    """

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        with self.init_scope():
            self.out = L.Linear(n_in, n_out, initialW=0)

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


class WindowIterator(chainer.dataset.Iterator):
    """Dataset iterator to create a batch of sequences at different positions.
    This iterator returns a pair of the current words and the context words.
    """

    def __init__(self, dataset, window, batch_size, original_index, line_num_fin_data, index2word, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window  # size of context window
        self.batch_size = batch_size
        self._repeat = repeat


        self.order = np.random.permutation(len(dataset) - len(original_index) - window * 2).astype(np.int32)
        self.order += window

        self.current_position = 0
        self.flg = 1
        self.flg_fin = 0

        self.epoch = 0
        self.line_num_fin_data = line_num_fin_data

        self.is_new_epoch = False
        self.index2word = index2word

    def __next__(self):
        """This iterator returns a list representing a mini-batch.
        Each item indicates a different position in the original sequence.
        """



        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])


        last_num = self.dataset[-1]
        contexts = []
        position = []



        text_name = 'context_word/index_data/{}.txt'.format(int(self.flg - 1))
        data_mi = []
        with open(text_name, 'r', encoding = 'utf-8',errors = 'ignore') as f:
            for line in f:
                line = list(map(int, line.strip().split('\t')))
                data_mi.append(line)

        # print(self.flg, self.line_num_fin_data)
        if self.flg == self.line_num_fin_data+1:
            self.flg_fin = 1





        for i in range(len(data_mi)):
            hasei_num = data_mi[i][0]
            for j in range(1, len(data_mi[i])):
                replace_num = data_mi[i][j]

                for k in range(len(offset)):
                    position.append(replace_num+offset[k])
                    contexts_kari = [last_num for kk in range(len(offset))]
                    contexts_kari[-1-k] = hasei_num
                    contexts.append(contexts_kari)






        position = np.array(position)

        if contexts == []:
            center = np.array([0])
            contexts = np.array([[0 for i in range(2*w)]])
            # center = self.dataset.take(position)
            # contexts = np.array(contexts)
        else:
            center = self.dataset.take(position)
            contexts = np.array(contexts)



        if self.flg_fin:
            # print('New')
            print('epoch finish')
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
            self.flg_fin = 0
            self.flg = 1

            flg_save = 1
        else:
            self.is_new_epoch = False
            self.flg += 1
            flg_save = 0

        return center, contexts, flg_save, self.epoch, self.index2word

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    center, contexts, flg_save, epoch, index2word = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        contexts = cuda.to_gpu(contexts)
    return center, contexts, flg_save, epoch, index2word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=300, type=int,
                        help='number of units')
    parser.add_argument('--window', '-w', default=5, type=int,
                        help='window size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--negative-size', default=5, type=int,
                        help='number of negative samples')
    parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                        default='hsm',
                        help='output model type ("hsm": hierarchical softmax, '
                        '"ns": negative sampling, "original": '
                        'no approximation)')
    parser.add_argument('--out', default='model',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Window: {}'.format(args.window))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Output type: {}'.format(args.out_type))
    print('')

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    # Load the dataset
    train, val, _ = chainer.datasets.get_ptb_words()
    vocab = chainer.datasets.get_ptb_words_vocabulary()
    train, vocab, original_index, ori_con_data, line_num_fin_data = proc.get_pair(train, vocab)
    counts = collections.Counter(train)
    n_vocab = max(train) + 1

    if args.test:
        train = train[:100]


    index2word = {wid: word for word, wid in six.iteritems(vocab)}


    if args.out_type == 'hsm':
        HSM = L.BinaryHierarchicalSoftmax
        tree = HSM.create_huffman_tree(counts)
        loss_func = HSM(args.unit, tree)
        loss_func.W.data[...] = 0
    elif args.out_type == 'ns':
        cs = [counts[w] for w in range(len(counts))]
        loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
        loss_func.W.data[...] = 0
    elif args.out_type == 'original':
        loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))

    model = ContinuousBoW(n_vocab, args.unit, loss_func, ori_con_data, vocab)


    if args.gpu >= 0:
        model.to_gpu()

    # Set up an optimizer
    optimizer = O.Adam()
    optimizer.setup(model)

    # Set up an iterator
    train_iter = WindowIterator(train, args.window, args.batchsize, original_index, line_num_fin_data, index2word)
    # val_iter = WindowIterator(val, args.window, args.batchsize, original_index, repeat=False)

    # Set up an updater
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)

    # Set up a trainer
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    print('run_start')
    trainer.run()

    # Save the word2vec model
    with open('model/post_train.model'.format(args.epoch), 'w') as f:
        f.write('%d %d\n' % (len(index2word)-1, args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            if i == len(index2word)-1:
                print(i)
                continue
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))


if __name__ == '__main__':
    main()
