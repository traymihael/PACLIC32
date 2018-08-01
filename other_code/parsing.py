import corenlp


corenlp_dir = "../../../stanford/stanford-corenlp-full-2015-01-29/"
properties_file = "../../../stanford/stanford-corenlp-full-2015-01-29/user.properties"
parser = corenlp.StanfordCoreNLP(corenlp_path=corenlp_dir, properties=properties_file)


def parse(text_name, out_name):
    with open(text_name, 'r', encoding='utf-8', errors='ignore') as text, open(out_name, 'w') as out:
        for line in text:
            for sentence in parser.raw_parse(line)["sentences"]:
                words, lemma, pos = [], [], []

                for word_element in sentence['words']:
                    words.append(word_element[0])
                    lemma.append(str(word_element[1]["Lemma"]).lower() )
                    pos.append(word_element[1]["PartOfSpeech"])

                depend = sentence["indexeddependencies"]
                out.write(' '.join(words) + '\n')
                out.write(' '.join(lemma) + '\n')
                out.write(' '.join(pos) + '\n')

                for depend_num in range(len(depend)):
                    out.write(' '.join(depend[depend_num]))
                    if depend_num != len(depend) - 1:
                        out.write('\t')
                out.write('\n\n')

def split_sent(text_name, out_name):
    with open(text_name, 'r', encoding='utf-8', errors='ignore') as text, open(out_name, 'w') as out:
        for line in text:
            for sentence in parser.raw_parse(line)["sentences"]:
                out.write('{}\n'.format(sentence['text']))



def main():
    text_name = 'nlp.txt'
    out_sent_name = 'corpus.txt'
    out_parse_name = 'copus_parse.txt'

    print(text_name)
    print(out_sent_name)
    print(out_parse_name)

    split_sent(text_name, out_sent_name)
    parse(text_name, out_parse_name)



if __name__ == '__main__':
    main()



