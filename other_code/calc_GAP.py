
def get_data(text_name, flg_out_voca):
    ans_num_data = []
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            line1 = line.strip().split('\t')[4::2]
            line2 = line.strip().split('\t')[5::2]

            if flg_out_voca == 0:
                line_num_data = [int(num.split('_')[-1]) for num in line1]
            if flg_out_voca == 1:
                line_num_data = []
                if len(line1) != len(line2):
                    print('out')
                for i in range(len(line1)):

                    if line2[i] == '-1.0':
                        line_num_data.append(0)
                        # pass
                        # break
                    else:
                        line_num_data.append(int(line1[i].split('_')[-1]))



            ans_num_data.append(line_num_data)

        # print(ans_num_data[0])
        # a = input()

    return ans_num_data

def calc_gap_value(data):
    value = 0
    accumulation = 0
    for i in range(len(data)):
        if data[i] == 0:
            continue
        accumulation += data[i]
        value += accumulation/(i+1)

    return value

def calc_gap(ans_num_data):
    calc_relust, ideal_value = [], []
    for i in range(len(ans_num_data)):
        num_data = ans_num_data[i].copy()
        calc_relust.append(calc_gap_value(num_data))
        num_data = sorted(num_data, key=lambda x: -x)
        ideal_value.append(calc_gap_value(num_data))

    return calc_relust, ideal_value

def calc_final(calc_relust, ideal_value):
    count = 0
    final_value = 0
    for i in range(len(calc_relust)):
        if int(ideal_value[i]) == 0:
            continue
        count += 1
        final_value += calc_relust[i]/ideal_value[i]

    final_value = final_value/count

    print(final_value*100)
    print(count)

    return final_value*100

def out_data(result_list, flg_out_voca):
    text_name = 'result{}.txt'.format(flg_out_voca)
    result_list = map(str, result_list)
    with open(text_name, 'w') as out:
        out.write('\t'.join(result_list))

if __name__ == '__main__':
    # 語彙外の単語をどうするか。無視するなら1、無視しないなら0
    flg_out_voca = 0
    result_list = []

    # for i in range(21):
    #     # text_name = 'result/sem_pre_nopos_balAddCos.txt'
    #     # text_name = 'result/epoch_sem_pre_pos/epoch_{}.txt'.format(i)
    #     print(text_name.split('/')[-1])
    #     ans_num_data = get_data(text_name, flg_out_voca)
    #     # ans_num_data = [[4,0,2,6], [0,1,5,3]]
    #     calc_relust, ideal_value = calc_gap(ans_num_data)
    #     final_value = calc_final(calc_relust, ideal_value)
    #     result_list.append(final_value)
    #     print()
    #     break
    #
    # # out_data(result_list, flg_out_voca)

    pos_yes_no_list = ['_nopos', '_pos']
    output_mode_list = ['pw']
    cos_mode_list = ['normal', 'addCos', 'balAddCos']


    for pos_yes_no in pos_yes_no_list:
        for output_mode in output_mode_list:
            for cos_mode in cos_mode_list:
                text_name = 'result/{}_{}_{}.txt'.format(pos_yes_no, output_mode, cos_mode)
                print(text_name)

                ans_num_data = get_data(text_name, flg_out_voca)
                calc_relust, ideal_value = calc_gap(ans_num_data)
                final_value = calc_final(calc_relust, ideal_value)
                result_list.append(final_value)
                print()
                # break

