
from scipy.stats import pearsonr

def get_ans():
    text_name = '../data/ratings.txt'
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        lines = f.read().strip().split('\n')
        ans_data = [float(line.split('\t')[7]) for line in lines]
    return ans_data

def get_outdata(pos_yes_no):
    text_name = '../output/result/scws{}.txt'.format(pos_yes_no)
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        lines = f.read().strip().split('\n')
        output_data = [float(line.split('\t')[7]) for line in lines]
    return output_data

def main(pos_yes_no):
    ans_data = get_ans()
    output_data = get_outdata(pos_yes_no)
    r, p = pearsonr(ans_data, output_data)

    print('相関係数 r: {r}'.format(r=r))
    print('有意確率 p: {p}'.format(p=p))
    print('有意確率 p > 0.05: {result}'.format(result=(p < 0.05)))

if __name__ == '__main__':
    pos_yes_no = '_pos'
    main(pos_yes_no)