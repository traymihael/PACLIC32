import numpy as np

def make_score(depend_list, mode):
    out_name = 'score{}.txt'.format(mode)
    with open(out_name, 'w') as out:
        for i in range(26):
            text_name = '../../../make_corpus/mi_depend/mi_depend_score1/{}.txt'.format(chr(97 + i))
            print(text_name.split('/')[-1])
            with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip().split('\t')
                    if mode == '_pos':
                        words = '_'.join(line[0:4])
                    elif mode == '_nopos':
                        words = '{}_{}'.format(line[0], line[2])
                    # print(words)
                    if words in depend_list:
                        out.write('{}\n'.format('\t'.join(line)))
            # break

def get_model_data(w2v_model_name):
    text_name = '{}'.format(w2v_model_name)

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


def make_depend_list(pos_yes_no):
	text_name = 'epoch_sem_pre{}/epoch_20.model'.format(pos_yes_no)
	w, index2word, word2index, vocabulary = get_model_data(text_name)
	# print(len(vocabulary))
	# a = input()
	return vocabulary


def make_5(mode, vector, word2index, vocabulary, pw_list_5):
	out_name = 'sem{}_5'.format(mode)
	with open(out_name, 'w') as out:
		count = 0
		for word in vocabulary:
			if word.count('_') <= 1:
				count += 1
			elif word in pw_list_5:
				count += 1
			# print(word)
		out.write('%d %d\n' % (count, 300))
		# print(w2v_model_name1)
		for word in vocabulary:
			if word.count('_') <= 1 or word in pw_list_5:
				vec = ' '.join(map(str, vector[word2index[word]]))
				out.write('%s %s\n' % (word, vec))


def get_5(mode):
	text_name = out_name = 'score{}.txt'.format(mode)
	new_depend_list = set()

	with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
		for line in f:
			line = line.strip().split('\t')
			if float(line[5]) < -100 or int(line[4]) > 5:
				continue
			# print(line[4])
			# if int(line[4]) <= 5:
			#     continue
			if mode == '_pos':
			    words = '_'.join(line[0:4])
			elif mode == '_nopos':
			    words = '{}_{}'.format(line[0], line[2])
			new_depend_list.add(words)

	return new_depend_list

def make_5_100(mode, vector, word2index, vocabulary, pw_list_5_100):
	out_name = 'sem{}_5_100'.format(mode)
	with open(out_name, 'w') as out:
		count = 0
		
		for word in vocabulary:
			if word.count('_') <= 1:
				count += 1
			elif word in pw_list_5_100:
				count += 1
		out.write('%d %d\n' % (count, 300))
		# print(w2v_model_name1)
		for word in vocabulary:
			if word.count('_') <= 1 or word in pw_list_5_100:
				vec = ' '.join(map(str, vector[word2index[word]]))
				out.write('%s %s\n' % (word, vec))


def get_5_100(mode):
	text_name = out_name = 'score{}.txt'.format(mode)
	new_depend_list = set()

	with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
		for line in f:
			line = line.strip().split('\t')
			if float(line[5]) < -100 or int(line[4]) > 100:
				continue
			# print(line[4])
			if int(line[4]) <= 5:
			    continue
			if mode == '_pos':
			    words = '_'.join(line[0:4])
			elif mode == '_nopos':
			    words = '{}_{}'.format(line[0], line[2])
			new_depend_list.add(words)

	return new_depend_list


def make_100(mode, vector, word2index, vocabulary, pw_list_100):
	out_name = 'sem{}_100'.format(mode)
	with open(out_name, 'w') as out:
		count = 0
		
		for word in vocabulary:
			if word.count('_') <= 1:
				count += 1
			elif word in pw_list_100:
				count += 1
		out.write('%d %d\n' % (count, 300))
		# print(w2v_model_name1)
		for word in vocabulary:
			if word.count('_') <= 1 or word in pw_list_100:
				vec = ' '.join(map(str, vector[word2index[word]]))
				out.write('%s %s\n' % (word, vec))


def get_100(mode):
	text_name = out_name = 'score{}.txt'.format(mode)
	new_depend_list = set()

	with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
		for line in f:
			line = line.strip().split('\t')
			if float(line[5]) < -100 or int(line[4]) < 100:
				continue
			# print(line[4])
			# if int(line[4]) <= 5:
			#     continue
			if mode == '_pos':
			    words = '_'.join(line[0:4])
			elif mode == '_nopos':
			    words = '{}_{}'.format(line[0], line[2])
			new_depend_list.add(words)

	return new_depend_list

def make_dep_lis3(mode):
    text_name = 'score{}_pre.txt'.format(mode)
    out_name = 'score{}.txt'.format(mode)
    dic = {}
    with open(text_name, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            line = line.strip().split('\t')
            word = '{}_{}'.format(line[0], line[2])
            if word in dic.keys():
                if int(dic[word][4]) < int(line[4]):
                    dic[word][4] = line[4]
                continue
            dic.update({word:line})

    with open(out_name, 'w') as out:
        for element in dic.values():
            out.write('{}\n'.format('\t'.join(element)))

if __name__ == '__main__':
	mode = '_pos'

	# pw_list = make_depend_list(mode)
	# make_score(pw_list, mode)

	# make_dep_lis3(mode)
	
	model_name = 'epoch_sem_pre{}/epoch_20.model'.format(mode)
	vector, index2word, word2index, vocabulary = get_model_data(model_name)
	
	pw_list_5 = get_5(mode)
	print(len(pw_list_5))
	make_5(mode, vector, word2index, vocabulary, pw_list_5)

	pw_list_5_100 = get_5_100(mode)
	print(len(pw_list_5_100))
	make_5_100(mode, vector, word2index, vocabulary, pw_list_5_100)

	pw_list_100 = get_100(mode)
	print(len(pw_list_100))
	make_100(mode, vector, word2index, vocabulary, pw_list_100)

