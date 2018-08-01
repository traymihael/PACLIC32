import numpy as np
from pprint import pprint
from collections import defaultdict


def ex_pos(pos):
    if pos == 'verb':
        cefr_pos = 'v'
    elif pos == 'adverb':
        cefr_pos = 'r'
    elif pos == 'adjective':
        cefr_pos = 'j'
    elif pos == 'noun':
        cefr_pos = 'n'
    else:
        cefr_pos = 'x'

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


def get_eval_data(pos_yes_no):
    eval_name = 'eval_data/eval_data3.txt'

    with open(eval_name, 'r') as f:
        if pos_yes_no == '_nopos':
            lines = f.read().strip().split('\n')
            target = [line.split('\t')[2] for line in lines]
            pos = [ex_pos(line.split('\t')[3]) for line in lines]
            candidate = [[can.split('_')[0] for can in line.split('\t')[4:]] for line in lines]
            text = [line.split('\t')[1] for line in lines]
        elif pos_yes_no == '_pos' or pos_yes_no == '_pos_nopos':
            lines = f.read().strip().split('\n')
            target = ['{}_{}'.format(line.split('\t')[2], line.split('\t')[3])
                      for line in lines]
            pos = [ex_pos(line.split('\t')[3]) for line in lines]
            candidate = [['{}_{}'.format(can.split('_')[0], can.split('_')[1])
                          for can in line.split('\t')[4:]] for line in lines]
            text = [line.split('\t')[1] for line in lines]

        ans = []
        for line in lines:
            line = line.strip().split('\t')
            ans.append([])
            for i in range(4, len(line)):
                elem = line[i].split('_')
                if len(elem) != 3:
                    print('???')
                    continue
                if elem[-1] == '0':
                    continue
                ans[-1].append(elem[0])

    # return target, candidate, pos, text, ans
    return candidate, pos, text, ans, target





def get_pw_list(target_list, pos_yes_no):
    parse_name = 'parse_data/sentences_parse.txt'
    pattern = ["?", "!", "_", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "`", "/", ":"]

    pw_list = []

    with open(parse_name, 'r') as f:
        lines = f.read()
        lines = lines.split('\n\n')
        lines.pop(-1)

        print(len(lines))

        for i in range(len(lines)):
            line1, line2, line3, line4 = lines[i].split('\n')
            line2 = line2.split()
            line3 = line3.split()
            line4 = line4.split('\t')

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
                    print(num1, num2)
                    # print(line2)
                    # print('out')
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
                        kari_word.append(word_right)
                    if word_right == target_list[i]:
                        kari_word.append(word_left)
                elif pos_yes_no == '_pos':
                    word_left = word1 + '_' + pos1
                    word_right = word2 + '_' + pos2
                    if word_left == target_list[i]:
                        kari_word.append(word_right)
                    if word_right == target_list[i]:
                        kari_word.append(word_left)
                if pos_yes_no == '_nopos':
                    word_left = word1
                    word_right = word2
                    if word_left == target_list[i]:
                        kari_word.append(word_right)
                    if word_right == target_list[i]:
                        kari_word.append(word_left)
                elif pos_yes_no == '_pos_nopos':
                    word_left = word1 + '_' + pos1
                    word_right = word2 + '_' + pos2
                    if word_left == target_list[i]:
                        kari_word.append(word2)
                    if word_right == target_list[i]:
                        kari_word.append(word1)

            pw_list.append(list(set(kari_word)))

    return pw_list

def get_target():
    text_name = 'eval_data/target.txt'
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        target = [line.split('\t')[0] for line in f.read().split('\n')[:-1]]
    # print(len(target))
    return target

def get_elemnt(target, pw_list, pos, pos_yes_no):
    elements = []
    for i in range(len(pw_list)):
        if pos_yes_no == '_nopos':
            kari1 = '{}_{}'.format(target, pw_list[i])
            kari2 = '{}_{}'.format(pw_list[i], target)
        elif pos_yes_no == '_pos' or pos_yes_no == '_pos_nopos':
            kari1 = '{}_{}'.format(target, pw_list[i])
            kari2 = '{}_{}'.format(pw_list[i], target)
        elements.append(kari1)
        elements.append(kari2)
    return elements

def write_pw_list(target, candidate, pw_list, pos_yes_no, pos):
    out_name = 'eval_data/depend_list_pos_nopos.txt'
    elemnts_list = set()
    for i in range(len(target)):
        elemnts = get_elemnt(target[i], pw_list[i], pos[i], pos_yes_no)
        for elemnt in elemnts:
            elemnts_list.add(elemnt)
        for j in range(len(candidate[i])):
            elemnts = get_elemnt(candidate[i][j], pw_list[i], pos[i], pos_yes_no)
            for elemnt in elemnts:
                elemnts_list.add(elemnt)
    elemnts_list = list(elemnts_list)
    with open(out_name, 'w') as out:
        for i in range(len(elemnts_list)):
            out.write('{}\n'.format(elemnts_list[i]))



if __name__ == '__main__':

    pos_yes_no = '_pos_nopos'


    # target = get_target()
    candidate, pos, text, ans, target = get_eval_data(pos_yes_no)
    pw_list = get_pw_list(target, pos_yes_no)

    write_pw_list(target, candidate, pw_list, pos_yes_no, pos)