import xml.etree.ElementTree as ET
import re

def get_data():
    text_name = 'data/trial/lexsub_trial.xml'
    # context_list = []
    target_list, pos_list = [], []
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            # context = re.search('<context>(.*?)</context>', line)
            word = re.search('<lexelt item="(.*?)">', line)
            # if context:
            #     context = context.group(1)
            #     context = context.replace('<head>', '')
            #     context = context.replace('</head>', '')
            #     context_list.append(context)
            if word:
                word = word.group(1).split('.')
                if word[1] == 'a':
                    word[1] = 'j'
                target_list.append(word[0])
                pos_list.append(word[1])

    # print(len(target_list))
    # print(len(context_list))
    # print(len(pos_list))
    # print(target_list)


    # return context_list, target_list, pos_list
    return target_list, pos_list


def get_can():
    text_name2 = 'data/trial/BLoutof10.out'
    candidate_list = [[] for i in range(300)]

    with open(text_name2, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            candidate_list_kari = set()
            line = line.strip().split(' ::: ')
            num = int(line[0].split()[1]) - 1
            cands = line[1].split(';')
            for i in range(len(cands)):
                if ' ' in cands[i]:
                    continue
                cands[i] = re.sub('\((.*?)\)', '', cands[i])
                candidate_list_kari.add(cands[i])
            candidate_list_kari = list(candidate_list_kari)
            candidate_list[num] = candidate_list_kari

    # print(candidate_list)

    return candidate_list


def get_ans():
    text_name2 = 'data/trial/gold.trial'
    ans_list = [[] for i in range(300)]

    with open(text_name2, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            ans_list_kari = set()
            line = line.split(' :: ')
            if len(line) != 2:
                continue

            num = int(line[0].split()[1]) - 1
            anses = line[1].split(';')[:-1]

            for i in range(len(anses)):
                element = anses[i].split()
                if len(element) >= 3:
                    continue
                ans_list_kari.add(element[0])
            ans_list[num] = list(ans_list_kari)

    # print(ans_list)
    return ans_list


def write_one_line(context_list):
    write_name = 'parse_data/one_line.txt'
    with open(write_name, 'w') as out:
        for context in context_list:
            out.write('{}\n'.format(context))

def write_data(context_list, target_list, pos_list, candidate_list, ans_list):
    text_name = 'eval_data/eval_data.txt'
    print(len(context_list))
    with open(text_name, 'w') as out:
        for i in range(len(context_list)):
            out.write('{}\t{}\t{}\t{}\t'
                      .format(i+1, context_list[i], target_list[int(i/10)], pos_list[int(i/10)]))
            for j in range(len(ans_list[i])):
                out.write('{}_1\t'.format(ans_list[i][j]))

            cand_kari = []
            for j in range(len(candidate_list[i])):
                if candidate_list[i][j] not in ans_list[i]:
                    # print(candidate_list[i][j])
                    cand_kari.append(candidate_list[i][j])

            for j in range(len(cand_kari)):
                out.write('{}_0'.format(cand_kari[j]))
                if j == len(cand_kari)-1:
                    out.write('\n')
                else:
                    out.write('\t')
            if len(cand_kari) == 0:
                out.write('\n')

def get_sent():
    text_name = 'parse_data/one_line2.txt'
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        lines = f.read().split('\n')[:-1]
        context_list = [line for line in lines]

    # print(len(context_list))
    # a = input()
    return context_list

if __name__ == '__main__':


    context_list = get_sent()
    target_list, pos_list = get_data()
    # with open('eval_data/target_list.txt', 'w') as out:
    #     for i in target_list:
    #         for j in range(10):
    #             out.write('{}\n'.format(i))
    candidate_list = get_can()
    ans_list = get_ans()

    write_one_line(context_list)
    write_data(context_list, target_list, pos_list, candidate_list, ans_list)
