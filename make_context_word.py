from collections import defaultdict

def ex_pos(pos):

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

def get_context_word_data(text_name):

    context_word_list = defaultdict(int)
    count = 0

    with open(text_name, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n\n')[:-1]
        for line in lines:
            line1, line2, line3, line4 = line.split('\n')
            line2 = line2.strip().split()
            line3 = line3.strip().split()
            line4 = line4.strip().split('\t')

            for i in range(1, len(line4)):

                data = line4[i].split()
                word_data1, word_data2 = data[1].rsplit('-', 1), data[2].rsplit('-', 1)
                num1, num2 = int(word_data1[1].replace("'", ''))-1, int(word_data2[1].replace("'", ''))-1
                pos1, pos2 = ex_pos(line3[num1]), ex_pos(line3[num2])
                word1, word2 = line2[num1], line2[num2]

                element = word1 + '_' + word2
                element_rev = word2 + '_' + word1

                if pos1 == 'x' or pos2 == 'x':
                    continue

                if element in context_word_list:
                    context_word_list[element].append(str(count + num1))
                else:
                    context_word_list[element] = [str(count + num1)]

                if element_rev in context_word_list:
                    context_word_list[element_rev].append(str(count + num2))
                else:
                    context_word_list[element_rev] = [str(count + num2)]
            count += len(line2) + 1

    return context_word_list


def write_mi_data(mi_data_list, out_name):
    with open(out_name, 'w') as out:
        for word, number in mi_data_list.items():
            out.write('{}\t{}\n'.format(word, '\t'.join(number[:10000])))


def main():
    text_name = 'corpus/corpus_parse.txt'
    out_name = 'context_word/context_words_data.txt'

    context_word_list = get_context_word_data(text_name)
    write_mi_data(context_word_list, out_name)

if __name__ == '__main__':
    main()





