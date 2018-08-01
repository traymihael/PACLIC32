import numpy as np

def get_model_data(w2v_model_name):
    text_name = '../model/{}'.format(w2v_model_name)

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

def get_eval(text_name, pos_yes_no):
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        if pos_yes_no == 'nopos':
            lines = f.read().strip().split('\n')
            target = [line.split('\t')[2] for line in lines]
            pos = [line.split('\t')[3] for line in lines]
            candidate = [[can.split('_')[0] for can in line.split('\t')[4:]] for line in lines]
            text = [line.split('\t')[1] for line in lines]
        elif pos_yes_no == 'pos':
            lines = f.read().strip().split('\n')
            target = ['{}_{}'.format(line.split('\t')[2], ex_pos(line.split('\t')[3]))
                      for line in lines]
            pos = [ex_pos(line.split('\t')[3]) for line in lines]
            candidate = [['{}_{}'.format(can.split('_')[0], ex_pos(line.split('\t')[3]))
                          for can in line.split('\t')[4:]] for line in lines]
            text = [line.split('\t')[1] for line in lines]
        ans = []
        for line in lines:
            line = line.strip().split('\t')
            ans_kari = {element.split('_')[0]:element.split('_')[1] for element in line[4:]}
            ans.append(ans_kari)

    candidate_list = []
    for i in range(len(candidate)):
        candidate_list += candidate[i]

    return target, candidate_list

def check_list_num(data_list, vocabulary):
    count = 0
    for i in range(len(data_list)):
        if data_list[i] not in vocabulary:
            count += 1
            print(data_list[i])
    return count

def check(target_list, candidate_list, vocabulary):
    # out_num_target = check_list_num(target_list, vocabulary)
    # out_num_cand = check_list_num(candidate_list, vocabulary)
    # print('target\t{}/{}({})'
    #       .format(out_num_target, len(target_list), out_num_target/len(target_list)))
    # print('candidate\t{}/{}({})'
    #       .format(out_num_cand, len(candidate_list), out_num_cand / len(candidate_list)))
    # print()

    target_list = list(set(target_list))
    candidate_list = list(set(candidate_list))
    # out_num_target = check_list_num(target_list, vocabulary)
    out_num_cand = check_list_num(candidate_list, vocabulary)
    print('target\t{}/{}({})'
          .format(out_num_target, len(target_list), out_num_target / len(target_list)))
    print('candidate\t{}/{}({})'
          .format(out_num_cand, len(candidate_list), out_num_cand / len(candidate_list)))

if __name__ == '__main__':
    pos_yes_no = 'nopos'
    w2v_model_name = 'epoch_sem_pre_nopos/epoch_0.model'
    text_name = '../eval_data/eval_data3.txt'

    print(text_name)
    target_list, candidate_list = get_eval(text_name, pos_yes_no)
    w, index2word, word2index, vocabulary = get_model_data(w2v_model_name)


    check(target_list, candidate_list, vocabulary)