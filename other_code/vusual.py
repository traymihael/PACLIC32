
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_model_data(w2v_model_name):
    text_name = w2v_model_name

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

    s = np.sqrt((w * w).sum(1))
    w /= s.reshape((s.shape[0], 1))  # normalize

    return w, index2word, word2index, vocabulary



def draw_word_scatter(word, topn=30):
    """ 入力されたwordに似ている単語の分布図を描くためのメソッド """

    # 似ている単語を求めるためにはGensim word2vecの以下の機能を利用
    # model.most_similar(word, topn=topn)
    words = [x[0] for x in sorted(model.most_similar(word, topn=topn))]
    words.append(word)

    # 各単語のベクトル表現を求めます。Gensimのmost_similarをベースとして
    # 単語のベクトルを返すメソッド(model.calc_vec)を定義しています
    # 長くなるので実装は本稿の末尾に記載しました。
    vecs = [model.calc_vec(word) for word in words]

    # 分布図
    draw_scatter_plot(vecs, words)

def draw_scatter_plot(vecs, tags):
    """ 入力されたベクトルに基づき散布図(ラベル付き)を描くためのメソッド """

    # Scikit-learnのPCAによる次元削減とその可視化
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)

    # matplotlibによる可視化
    fig, ax = plt.subplots()

    # x = [v[0] for v in coords]
    # y = [v[1] for v in coords]

    x1 = [v[0] for v in coords[:5]]
    y1 = [v[1] for v in coords[:5]]
    x2 = [v[0] for v in coords[5:]]
    y2 = [v[1] for v in coords[5:]]


    # ax.scatter(x, y)

    for i, txt in enumerate(tags):
        ax.annotate(txt, (coords[i][0], coords[i][1]), color='r')

    plt.scatter(x1, y1, c='b', s=50)
    plt.scatter(x2, y2, c='g', marker='^', s=50)


    plt.xlim(xmax=1)
    plt.xlim(xmin=-0.8)
    plt.ylim(ymax=0.9)
    plt.ylim(ymin=-0.8)
    ax.tick_params(labelbottom="off", bottom="off")  # x軸の削除
    ax.tick_params(labelleft="off", left="off")  # y軸の削除

    plt.show()
    plt.savefig('figure.png')

if __name__ == '__main__':
    w2v_model_name = '../semeval2007/model/epoch_sem_pre_nopos/epoch_20.model'
    vector, index2word, word2index, vocabulary = get_model_data(w2v_model_name)

    word_list = ['hard', 'badly', 'heavily', 'carefully', 'intensively', \
                 'hard_listen', 'hard_hit', 'carefully_listen', 'badly_hit',\
                 'intensively_listen', 'heavily_hit',\
                  'heavily_listen']

    for word in word_list:
        print(word, word in vocabulary)
    vect_list = [vector[word2index[word]] for word in word_list]



    draw_scatter_plot(vect_list, word_list)
