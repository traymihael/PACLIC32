
import itertools

def get_out(output_name):

    out_data = []
    with open(output_name, 'r') as f:
        lines = f.read().split('\n')[:-1]
        for line in lines:
            line = line.split('\t')
            out_kari = []
            for i in range(int((len(line) - 4) / 2)):
                out_kari.append([line[4+2*i].split('_')[0]])
            out_data.append(out_kari)

        out_data = list(map(list, zip(*out_data)))

    return out_data

def get_and_data(eval_name):
    ans_data = []
    with open(eval_name, 'r') as f:
        for line in f:
            ans_kari = []
            line = line.strip().split('\t')
            for element in line[4:]:
                if element.split('_')[1] == '1':
                    ans_kari.append(element.split('_')[0])
            ans_data.append(ans_kari)

    return ans_data

def calc_data(ans_data, out_data, model_list):
    count_ans_num = [0 for i in range(len(out_data))]
    # print(ans_data[:5])
    # print(out_data[0][:5])
    out_num = 263
    for i in range(len(ans_data)):
        for j in range(len(out_data)):
            if out_data[j][i][0] in ans_data[i]:
                count_ans_num[j] += 1

    for i in range(len(model_list)):
        print('{} {} is\t{}/{}({:.3}%)'
              .format(model_list[i][0], model_list[i][1], count_ans_num[i], out_num,
                      count_ans_num[i]/out_num))


if __name__ == '__main__':

    pos_yes_no_list = ['_nopos']
    train_method_list = ['_pre']
    output_mode_list = ['normal', 'pw']
    cos_mode_list = ['normal', 'addCos', 'balAddCos', 'minMax']


    model_list = list(itertools.product(output_mode_list, cos_mode_list))

    for pos_yes_no in pos_yes_no_list:
        for train_method in train_method_list:

            # if pos_yes_no == '_nopos' and train_method == '_ori':
            #     pos_yes_no ='_nopos2'
            # else:
            #     pos_yes_no = '_nopos'
            pos_yes_no = '_pos'

            print('word2vec_semEval{}{}.model'.format(train_method, pos_yes_no))

            output_name = 'result/output{}{}.txt'.format(train_method, pos_yes_no)
            eval_name = '../eval_data/eval_data.txt'

            out_data = get_out(output_name)
            ans_data = get_and_data(eval_name)

            calc_data(ans_data, out_data, model_list)

            print()
            # break