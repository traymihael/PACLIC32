
def get_data(text_name):
    data_list = []
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.strip().split('\t')
            data_list.append(line)
    return data_list


def get_out(text_name):
    out_data_list = []
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.strip().split()
            out_data_list.append(line[7])
    return out_data_list


if __name__ == '__main__':
    text_name = '../data/ratings.txt'
    data_list = get_data(text_name)

    out_name = 'result/scws_nopos.txt'
    out_data_list = get_out(out_name)

    data = []
    for i in range(len(data_list)):
        ans_value = float(data_list[i][7]) / 10
        out_value = float(out_data_list[i])
        if out_value <= 0.01:
            continue
        if out_value >= 0.99:
            continue
        if data_list[i][2] != data_list[i][4]:
            continue
        if data_list[i][1] == data_list[i][3]:
            continue
        if ans_value < 0.8:
            continue
        if out_value > 0.3:
            continue
        if abs(ans_value - out_value) > 0:
        # if out_value - ans_value > 0.3:
            # print(data_list[i][0], data_list[i][1], data_list[i][3], data_list[i][7], out_data_list[i])
            data.append([[data_list[i][0], data_list[i][1], data_list[i][3], data_list[i][7], out_data_list[i]], abs(ans_value - out_value)])

    data.sort(key=lambda x: -x[1])

    for i in range(len(data)):
        print(data[i][0])
