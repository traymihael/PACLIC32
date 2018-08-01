
import numpy as np
from pprint import pprint
from collections import defaultdict

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

def ex_pos(pos):
    # if pos == 'verb':
    #     cefr_pos = 'v'
    # elif pos == 'adverb':
    #     cefr_pos = 'r'
    # elif pos == 'adjective':
    #     cefr_pos = 'j'
    # elif pos == 'noun':
    #     cefr_pos = 'n'
    # else:
    #     cefr_pos = 'x'
    cefr_pos = pos

    return cefr_pos

def pos_exchange_cefr(pos):

    if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VB' or pos == 'VBP' or pos == 'VBZ':
        cefr_pos = 'v'
    elif pos == 'RB' or pos == 'RBR' or pos == 'RBS':
        cefr_pos = 'r'
    elif pos == 'JJ' or pos == 'JJR' or pos == 'JJS':
        cefr_pos = 'j'
    elif pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
        cefr_pos = 'n'
    else:
        cefr_pos = 'x'

    return cefr_pos

def get_text(pos_yes_no):
    text_name = '../parse_data/sentences_parse.txt'
    text = []
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        lines = f.read().split('\n\n')[:-1]
        for line in lines:
            text_kari = []
            line = line.split('\n')
            line_text = line[1].split()
            line_pos = line[2].split()
            for i in range(len(line_text)):
                pos = pos_exchange_cefr(line_pos[i])
                if pos == 'x':
                    element = line_text[i]
                
                elif pos_yes_no == '_nopos':
                    element = line_text[i]
                elif pos_yes_no == '_pos':
                    element = '{}_{}'.format(line_text[i], pos)
                text_kari.append(element)
            text.append(text_kari)

    # print(len(text))
    # print(text[0])
    return text


def get_eval_data(pos_yes_no, eval_name):


    with open(eval_name, 'r') as f:
        if pos_yes_no == '_nopos':
            lines = f.read().strip().split('\n')
            target = [line.split('\t')[2] for line in lines]
            pos = [line.split('\t')[3] for line in lines]
            candidate = [[can.split('_')[0] for can in line.split('\t')[4:]] for line in lines]
            text = [line.split('\t')[1] for line in lines]
        elif pos_yes_no == '_pos':
            lines = f.read().strip().split('\n')
            pos = [ex_pos(line.split('\t')[3]) for line in lines]
            text = [line.split('\t')[1] for line in lines]
            candidate = []
            target= []

            for line in lines:
                candidate.append([])
                for can in line.split('\t')[4:]:
                    can = can.split('_')
                    if can [1] == 'x':
                        candidate[-1].append(can[0])
                    else:
                        candidate[-1].append('{}_{}'.format(can[0], can[1]))

                if line.split('\t')[3] == 'x':
                    target.append(line.split('\t')[2])
                else:
                    target.append('{}_{}'.format(line.split('\t')[2], line.split('\t')[3]))


        ans = []
        for line in lines:
            line = line.strip().split('\t')
            ans_kari = {element.split('_')[0]:element.split('_')[-1] for element in line[4:]}
            ans.append(ans_kari)
            # ans.append([{element.split('_')[0]:element.split('_')[1]} for element in line[4:]])
            # ans.append([])
            # for i in range(4, len(line)):
            #     elem = line[i].split('_')
            #     if elem[1] == '1':
            #         ans[-1].append(elem[0])

        # print(candidate[10])
        # print(ans[0]['intelligent'])
        # a = input()


    # print(target[0])
    # print(candidate[0])
    # print(pos[0])
    # print(text[0])
    # a = input()

    return target, candidate, pos, text,ans

def cos_sim(vec_a, vec_b):
    norm_ab = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm_ab != 0:
        return np.dot(vec_a, vec_b) / norm_ab
    else:
        # ベクトルのノルムが0だと似ているかどうかの判断すらできないので最低値
        return -1

def calc_cos_value(vector, target, candidate, word2index, text, cos_mode, vocabulary):
    # print(target)
    if cos_mode == 'normal' or cos_mode == 'minMax':
        cos_simu_value = cos_sim(vector[word2index[target]],vector[word2index[candidate]])
    elif cos_mode == 'addCos':
        # print(text)
        # cos_simu_value = cos_sim(vector[word2index[target]], vector[word2index[candidate]])
        cos_simu_value = 0
        for i in range(len(text)):
            if text[i] in vocabulary:
                cos_simu_value += cos_sim(vector[word2index[text[i]]], vector[word2index[candidate]])
        #         print(cos_sim(vector[word2index[text[i]]], vector[word2index[candidate]]))
        # a = input()
    elif cos_mode == 'balAddCos':
        cos_simu_value = (len(text)-1) * cos_sim(vector[word2index[target]], vector[word2index[candidate]])
        # cos_simu_value = 0
        for i in range(len(text)):
            if text[i] in vocabulary:
                cos_simu_value += cos_sim(vector[word2index[text[i]]], vector[word2index[candidate]])

    # a = input()

    return cos_simu_value

def calc_cos_value_avg(vector, target, candidate, word2index, text, cos_mode, vocabulary):
    if cos_mode == 'normal' or cos_mode == 'minMax':
        cos_simu_value = 0
        for i in range(len(target)):
            cos_simu_value += cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])

    elif cos_mode == 'addCos':

        cos_simu_value = 0
        for i in range(len(target)):
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])

    elif cos_mode == 'balAddCos':
        cos_simu_value = 0
        for i in range(len(target)):

            cos_simu_value += (len(text) - 1) * cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])

    if len(target) == 0:
        cos_simu_value = -1.000
    else:
        cos_simu_value = cos_simu_value / len(target)

    return cos_simu_value

def calc_cos_value_max(vector, target, candidate, word2index, text, cos_mode, vocabulary):
    cos_simu_value = -1.00

    if cos_mode == 'normal' or cos_mode == 'minMax':
        for i in range(len(target)):
            cos_simu_value_kari = cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari > cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    elif cos_mode == 'addCos':
        for i in range(len(target)):
            cos_simu_value_kari = 0
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value_kari += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari > cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    elif cos_mode == 'balAddCos':
        for i in range(len(target)):
            cos_simu_value_kari = (len(text) - 1) * cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value_kari += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari > cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    if len(target) == 0:
        cos_simu_value = -1.000

    return cos_simu_value

def calc_cos_value_min(vector, target, candidate, word2index, text, cos_mode, vocabulary):
    cos_simu_value = 1.00

    if cos_mode == 'normal' or cos_mode == 'minMax':
        for i in range(len(target)):
            cos_simu_value_kari = cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari < cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    elif cos_mode == 'addCos':
        for i in range(len(target)):
            cos_simu_value_kari = 0
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value_kari += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari < cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    elif cos_mode == 'balAddCos':
        for i in range(len(target)):
            cos_simu_value_kari = (len(text) - 1) * cos_sim(vector[word2index[target[i]]], vector[word2index[candidate[i]]])
            for j in range(len(text)):
                if text[j] in vocabulary:
                    cos_simu_value_kari += cos_sim(vector[word2index[text[j]]], vector[word2index[candidate[i]]])
            if cos_simu_value_kari < cos_simu_value:
                cos_simu_value = cos_simu_value_kari

    if len(target) == 0:
        cos_simu_value = -1.000

    return cos_simu_value


def make_eval_data(vector, target, candidate,word2index, ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode):
    output_data = []
    count = 0

    if output_mode == 'normal':
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])


            if cos_mode == 'minMax':
                pass
                # dic_kari = {}
                # for i in range(len(out_kari)):
                #     cand = out_kari[i][0].split('_')[0]
                #     if cand not in dic_kari.keys():
                #         dic_kari.update({cand: out_kari[i][1]})
                #     elif dic_kari[cand] > out_kari[i][1]:
                #         dic_kari[cand] = out_kari[i][1]
                # dic_kari = list(dic_kari.items())
                # dic_kari.sort(key=lambda x: -x[1])
                # output_data.append(dic_kari[:ans_num])
            else:
                out_kari.sort(key=lambda x: -x[1])
                output_data.append(out_kari)
            # print(out_kari)

    elif output_mode == 'pw':

        for i in range(len(target)):
            # 候補を全てとったかチェック。とったやつから入れていく。
            get_cand_list = []
            # 重複ないように[単語, 類似度]入れていく
            out_kari_nolap = []
            out_kari = []
            for k in range(len(pw_list[i])):
                for j in range(len(candidate[i])):
                    target_word = '{}_{}'.format(target[i], pw_list[i][k])
                    candidate_word = '{}_{}'.format(candidate[i][j], pw_list[i][k])

                    if target_word not in vocabulary or candidate_word not in vocabulary:
                        continue

                    cos_value = calc_cos_value(vector, target_word, candidate_word, word2index, text_pos[i], cos_mode, vocabulary)
                    # out_kari.append([candidate[i][j], cos_value])
                    out_kari.append([candidate_word, cos_value])
            out_kari.sort(key=lambda x: -x[1])
            for kari_num in out_kari:
                if kari_num[0].split('_')[0] not in get_cand_list:
                    get_cand_list.append(kari_num[0].split('_')[0])
                    out_kari_nolap.append(kari_num)

            out_kari = []
            for j in range(len(candidate[i])):
                if candidate[i][j] in get_cand_list:
                    continue
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])

            # if len(out_kari) == 0:
            #     # for j in range(len(candidate[i])):
            #     #     if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
            #     #         continue
            #     #     cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
            #     #     out_kari.append([candidate[i][j], cos_value])
            #
            #     out_kari.append(['XXX', 0.00])
            # else:
            #     count += 1



            # if len(out_kari) == 0:
            #     out_kari.append(['XXX', 0.00])

            if cos_mode == 'minMax':
                pass
                # dic_kari = {}
                # for i in range(len(out_kari)):
                #     # print(out_kari[i])
                #     cand = out_kari[i][0].split('_')[0]
                #     if cand not in dic_kari.keys():
                #         dic_kari.update({cand:out_kari[i][1]})
                #     elif dic_kari[cand] > out_kari[i][1]:
                #         dic_kari[cand] = out_kari[i][1]
                # dic_kari = list(dic_kari.items())
                # dic_kari.sort(key=lambda x: -x[1])
                # output_data.append(dic_kari[:ans_num])
            else:
                out_kari_nolap.sort(key=lambda x: -x[1])
                out_kari.sort(key=lambda x: -x[1])
                output_data.append(out_kari_nolap + out_kari)

    elif output_mode == 'miorder':

        for i in range(len(target)):
            output_data.append([])
            # 候補を全てとったかチェック。とったやつから入れていく。
            get_cand_list = []
            # 重複ないように[単語, 類似度]入れていく
            out_kari_nolap = []

            for k in range(len(pw_list[i])):
                out_kari = []
                for j in range(len(candidate[i])):
                    if candidate[i][j] in get_cand_list:
                        continue
                    target_word = '{}_{}'.format(target[i], pw_list[i][k])
                    candidate_word = '{}_{}'.format(candidate[i][j], pw_list[i][k])

                    if target_word not in vocabulary or candidate_word not in vocabulary:
                        continue

                    cos_value = calc_cos_value(vector, target_word, candidate_word, word2index, text_pos[i], cos_mode, vocabulary)
                    # out_kari.append([candidate[i][j], cos_value])
                    out_kari.append([candidate_word, cos_value])
                out_kari.sort(key=lambda x: -x[1])
                output_data[-1] += out_kari
                for kari_num in out_kari:
                    if kari_num[0].split('_')[0] not in get_cand_list:
                        get_cand_list.append(kari_num[0].split('_')[0])
                        # out_kari_nolap.append(kari_num)

            out_kari = []
            for j in range(len(candidate[i])):
                if candidate[i][j] in get_cand_list:
                    continue
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])


            out_kari.sort(key=lambda x: -x[1])
            output_data[-1] += out_kari



    print('output {} {} = {}'.format(output_mode, cos_mode, count))
    return output_data

def make_eval_data_avg(vector, target, candidate,word2index, ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode):
    output_data = []
    count = 0

    if output_mode == 'normal':
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])

            else:
                out_kari.sort(key=lambda x: -x[1])
                output_data.append(out_kari)
            # print(out_kari)

    elif output_mode == 'pw':

        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                tar_lis, can_lis = [], []
                for k in range(len(pw_list[i])):
                    for l in range(len(pw_list[i])):
                        target_word = '{}_{}'.format(target[i], pw_list[i][l])
                        candidate_word = '{}_{}'.format(candidate[i][j], pw_list[i][k])

                        if target_word not in vocabulary or candidate_word not in vocabulary:
                            continue

                        tar_lis.append(target_word)
                        can_lis.append(candidate_word)

                if len(tar_lis) == 0:
                    if target[i] in vocabulary and candidate[i][j] in vocabulary:
                        tar_lis.append(target[i])
                        can_lis.append(candidate[i][j])

                cos_value = calc_cos_value_avg(vector, tar_lis, can_lis, word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])


            out_kari.sort(key=lambda x: -x[1])
            output_data.append(out_kari)


    print('output {} {} = {}'.format(output_mode, cos_mode, count))
    return output_data

def make_eval_data_max(vector, target, candidate,word2index, ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode):
    output_data = []
    count = 0

    if output_mode == 'normal':
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])

            else:
                out_kari.sort(key=lambda x: -x[1])
                output_data.append(out_kari)
            # print(out_kari)


    elif output_mode == 'pw':
        count_ok, count_ng = [], []
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                tar_lis, can_lis = [], []
                for k in range(len(pw_list[i])):
                    # for l in range(len(pw_list[i])):
                    for l in range(1):
                        target_word = '{}_{}'.format(target[i], pw_list[i][k])
                        candidate_word = '{}_{}'.format(candidate[i][j], pw_list[i][k])

                        if target_word in vocabulary:
                            count_ok.append(target_word)
                        else:
                            count_ng.append(target_word)

                        if candidate_word in vocabulary:
                            count_ok.append(candidate_word)
                        else:
                            count_ng.append(candidate_word)


                        if target_word not in vocabulary or candidate_word not in vocabulary:
                            continue

                        tar_lis.append(target_word)
                        can_lis.append(candidate_word)

                if len(tar_lis) == 0:
                    if target[i] in vocabulary and candidate[i][j] in vocabulary:
                        tar_lis.append(target[i])
                        can_lis.append(candidate[i][j])

                cos_value = calc_cos_value_max(vector, tar_lis, can_lis, word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])


            out_kari.sort(key=lambda x: -x[1])
            output_data.append(out_kari)


    print('output {} {} = {}'.format(output_mode, cos_mode, count))

    print('OK {} ({}種類)'.format(len(count_ok), len(set(count_ok))))
    print('NG {} ({}種類)'.format(len(count_ng), len(set(count_ng))))

    return output_data

def make_eval_data_min(vector, target, candidate,word2index, ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode):
    output_data = []
    count = 0

    if output_mode == 'normal':
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                if target[i] not in vocabulary or candidate[i][j] not in vocabulary:
                    out_kari.append([candidate[i][j], -1.0])
                    continue

                cos_value = calc_cos_value(vector, target[i], candidate[i][j], word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])

            else:
                out_kari.sort(key=lambda x: -x[1])
                output_data.append(out_kari)
            # print(out_kari)


    elif output_mode == 'pw':
        count_ok, count_ng = [], []
        for i in range(len(target)):
            out_kari = []
            for j in range(len(candidate[i])):
                tar_lis, can_lis = [], []
                for k in range(len(pw_list[i])):
                    # for l in range(len(pw_list[i])):
                    for l in range(1):
                        target_word = '{}_{}'.format(target[i], pw_list[i][k])
                        candidate_word = '{}_{}'.format(candidate[i][j], pw_list[i][k])

                        if target_word in vocabulary:
                            count_ok.append(target_word)
                        else:
                            count_ng.append(target_word)

                        if candidate_word in vocabulary:
                            count_ok.append(candidate_word)
                        else:
                            count_ng.append(candidate_word)


                        if target_word not in vocabulary or candidate_word not in vocabulary:
                            continue

                        tar_lis.append(target_word)
                        can_lis.append(candidate_word)

                if len(tar_lis) == 0:
                    if target[i] in vocabulary and candidate[i][j] in vocabulary:
                        tar_lis.append(target[i])
                        can_lis.append(candidate[i][j])

                cos_value = calc_cos_value_min(vector, tar_lis, can_lis, word2index, text_pos[i], cos_mode, vocabulary)
                out_kari.append([candidate[i][j], cos_value])


            out_kari.sort(key=lambda x: -x[1])
            output_data.append(out_kari)


    print('output {} {} = {}'.format(output_mode, cos_mode, count))

    print('OK {} ({}種類)'.format(len(count_ok), len(set(count_ok))))
    print('NG {} ({}種類)'.format(len(count_ng), len(set(count_ng))))

    return output_data

def get_mi_valus(word, pair_word, mi_list):
    word_pw = '{}_{}'.format(word, pair_word)
    if word_pw in mi_list.keys():
        mi_valus = mi_list[word_pw]
    else:
        mi_valus = -100
    return mi_valus

def get_pw_list(target_list, pos_yes_no, mi_list):

    parse_name = '../parse_data/sentences_parse.txt'
    pattern = ["?", "!", "_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "`", "/", ":"]

    pw_list = []

    with open(parse_name, 'r') as f:
        lines = f.read()
        lines = lines.split('\n\n')
        lines.pop(-1)

        for i in range(len(lines)):
            line1, line2, line3, line4 = lines[i].split('\n')
            line2 = line2.split()
            line3 = line3.split()
            line4 = line4.split('\t')
            # print(line3)
            kari_word = []
            for j in range(1, len(line4)):
                if len(line4[j].split()) != 3:
                    continue
                other, dep1, dep2 = line4[j].split()
                word_data1, num1 = dep1.rsplit('-', 1)
                word_data2, num2 = dep2.rsplit('-', 1)
                num1 = int(num1.replace("'", ''))
                num2 = int(num2.replace("'", ''))

                pos1 = pos_exchange_cefr(line3[num1 - 1])
                pos2 = pos_exchange_cefr(line3[num2 - 1])

                try:
                    word1, word2 = line2[num1 - 1], line2[num2 - 1]
                except:
                    # print('out_depend')
                    continue


                if pos1 == 'x' or pos2 == 'x':
                    continue

                if word1[0].isupper() or word2[0].isupper():
                    continue
                flg = 0
                for fig in pattern:
                    if fig in word1 or fig in word2:
                        flg = 1
                        break
                if flg:
                    continue

                if pos_yes_no == '_nopos':
                    word_left = word1
                    word_right = word2
                    if word_left == target_list[i]:
                        mi_valus = get_mi_valus(word_left, word_right, mi_list)
                        kari_word.append([word_right, mi_valus])
                    if word_right == target_list[i]:
                        mi_valus = get_mi_valus(word_right, word_left, mi_list)
                        kari_word.append([word_left, mi_valus])

                elif pos_yes_no == '_pos':
                    word_left = word1 + '_' + pos1
                    word_right = word2 + '_' + pos2
                    if word_left == target_list[i]:
                        mi_valus = get_mi_valus(word_left, word_right, mi_list)
                        kari_word.append([word_right, mi_valus])
                    if word_right == target_list[i]:
                        mi_valus = get_mi_valus(word_right, word_left, mi_list)
                        kari_word.append([word_left, mi_valus])

            kari_word.sort(key=lambda x: -x[1])
            kari_word2 = [element[0] for element in kari_word]
            pw_list.append(kari_word2)

    return pw_list

def write_data(out_name, out_data, ans_num, text, pos, target, ans):

    # ans_num = [0, 0, 0, 0]
    # all_num = [0, 0, 0, 0]
    # dic = {'n':0, 'v':1, 'r':2, 'j':3}

    # print(pos)

    #
    # for i in range(len(target)):
    #     all_num[dic[pos[i]]] += 1
    #     # print(ans[i])
    #     print(result_data[0][i][0][0])
    #     if result_data[0][i][0][0] in ans[i]:
    #         ans_num[dic[pos[i]]] += 1
    #
    # print(ans_num)
    # print(all_num)
    #
    # a=input()

    with open(out_name, 'w') as out:
        for i in range(len(target)):
            out.write('{}\t{}\t{}\t{}\t'.format(i + 1, text[i], target[i], pos[i]))
            # print(text[i])
            # print(target[i])
            # print(ans[i])
            # for kkk in range(3):
            #     print(result_data[kkk][i])

            for j in range(len(out_data[i])):
                cand = out_data[i][j][0]
                try:
                    tag_ans = ans[i][cand.split('_')[0]]
                except:
                    print(i, j)
                    print(cand)
                    print(ans[i])
                    print(out_data[i])
                    a = input()

                simu = out_data[i][j][1]
                # print(cand, tag_ans, simu)
                out.write('{}_{}\t{:.3}'.format(cand, tag_ans, simu))

                if j == len(out_data[i]) - 1:
                    out.write('\n')
                else:
                    out.write('\t')

            # print()


# def make_eval_data_max(output_data_normal, output_data_pw, ans_num):
#     output_data_max = []
#
#     for i in range(len(output_data_normal)):
#         kari = output_data_normal[i] + output_data_pw[i]
#         kari.sort(key=lambda x: -x[1])
#         output_data_max.append(kari[:ans_num])
#
#     return output_data_max

def get_mi_data(pos_yes_no):
    text_name = '../eval_data/depend_mi.txt'
    mi_list = {}
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.strip().split('\t')
            if pos_yes_no == '_nopos':
                mi_list.update({'{}_{}'.format(line[0], line[2]):float(line[5])})
            elif pos_yes_no == '_pos':
                mi_list.update({'{}_{}_{}_{}'.format(line[0], line[1], line[2], line[3]): float(line[5])})
    return mi_list

if __name__ == '__main__':

    # print(out_name)
    # 上位何件出力か
    ans_num = 1

    pos_yes_no_list = ['_nopos', '_pos']
    train_method_list = ['_pre']
    output_mode_list = ['pw', 'normal']
    # cos_mode_list = ['normal']
    cos_mode_list = ['normal', 'addCos', 'balAddCos']

    eval_name = '../eval_data/eval_data3.txt'

    for epoch_num in range(20, 21):
        # epoch_num = 11
        for pos_yes_no in pos_yes_no_list:
            # pos_yes_no = '_pos'
            target, candidate, pos, text, ans = get_eval_data(pos_yes_no, eval_name)
            # print(text)
            mi_list = get_mi_data(pos_yes_no)
            # mi_list = {'a':0}
            pw_list = get_pw_list(target, pos_yes_no, mi_list)

            text_pos = get_text(pos_yes_no)


            for train_method in train_method_list:
                result_data = []
                w2v_model_name = 'epoch_sem{}{}/epoch_{}.model'.format(train_method, pos_yes_no, epoch_num)
                # w2v_model_name = 'word2vec_ori_nopos.model'
                print(w2v_model_name)

                vector, index2word, word2index, vocabulary = get_model_data(w2v_model_name)

                # cos_simu_value = cos_sim(vector[word2index['car']], vector[word2index['automobile']])
                # print(cos_simu_value)
                # print(len(vocabulary))
                # a = input()

                for output_mode in output_mode_list:
                    if output_mode == 'normal':
                        continue
                    # output_mode = 'pw'

                    for cos_mode in cos_mode_list:
                        if cos_mode == 'addCos':
                            continue
                        # cos_mode = 'balAddCos'

                        # output_data = make_eval_data(vector, target, candidate,word2index,
                        #                              ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode)
                        # out_name = 'result/epoch_sem{}{}/epoch_{}.txt'.format(train_method, pos_yes_no, epoch_num)


                        output_data = make_eval_data_max(vector, target, candidate, word2index,
                                                     ans_num, vocabulary, pw_list, text_pos, output_mode, cos_mode)
                        out_name = 'result/{}_{}_{}.txt'.format(pos_yes_no, output_mode, cos_mode)

                        print(out_name)
                        write_data(out_name, output_data, ans_num, text, pos, target, ans)

                        # a = input()



        # break