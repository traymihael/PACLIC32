import numpy as np

def get_pair(train, vocab):

    text_name = 'context_word/context_words_data.txt'

    original_index = []
    ori_con_data = []
    data_mi = []
    train = list(train)
    start = len(vocab)

    with open(text_name, 'r', encoding = 'utf-8',errors = 'ignore') as f:
        for line in f:
            data = line.strip().split('\t')
            data_mi.append([start])
            for i in range(len(data)-1):
                train.append(start)
                original_index.append(int(data[i+1]))
                data_mi[-1].append(data[i+1])
            ori_con_data.append([start, int(train[int(data[1])])])
            vocab.update({data[0]: start})
            start += 1


    with open('context_word/context_word_index.txt', 'w') as out:
        for i in range(len(data_mi)):
            if len(data_mi[i]) >= 25000:
                for j in range(int(len(data_mi[i])/25000)+1):
                    if j == 0:
                        data_mi_kari = list(map(str, data_mi[i][j*25000:(j+1)*25000]))
                    else:
                        data_mi_kari = list(map(str, data_mi[i][j*25000 - 1:(j+1)*25000]))
                        data_mi_kari[0] = str(data_mi[i][0])
                    out.write('\t'.join(data_mi_kari))
                    out.write('\n')
            else:
                data_mi_kari = list(map(str, data_mi[i]))
                out.write('\t'.join(data_mi_kari))
                out.write('\n')

    with open('context_word/context_word_index.txt', 'r', encoding='utf-8', errors='ignore') as f:
        data_mi = []
        count = 0
        num_data = 0
        for line in f:
            line = line.strip().split('\t')
            count += len(line) - 1
            data_mi.append(line)

            if count >= int((num_data+1) * 25000):
                text_name = 'context_word/index_data/{}.txt'.format(num_data)
                with open(text_name, 'w') as out:
                    for line_data_mi in data_mi:
                        out.write('{}\n'.format('\t'.join(line_data_mi)))
                num_data += 1
                data_mi = []

        text_name = 'context_word/index_data/{}.txt'.format(num_data)
        with open(text_name, 'w') as out:
            for line_data_mi in data_mi:
                out.write('{}\n'.format('\t'.join(line_data_mi)))




    train.append(start)
    vocab.update({'xxxxxxxx': start})

    train = np.array(train)

    return  train, vocab, original_index, ori_con_data, num_data
