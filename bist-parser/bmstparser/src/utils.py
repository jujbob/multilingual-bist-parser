import csv
from collections import Counter
import re


class ConllEntry:
    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None, language=None):
#    def __init__(self, id, form, lemma, pos, xpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.language = language

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.xpos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


class LanguageEntry:
    def __init__(self, lang_name, lang_num, lang_code, lang_vec):

        self.lang_name = lang_name
        self.lang_num = int(lang_num)
        self.lang_code = lang_code
        self.lang_vec = [int(x) for x in lang_vec.split(",")]



def readFileList(file_list_loc):
    fileAndLangList = []
    with open(file_list_loc, 'r') as file_list:
        for a_file in file_list:
            if a_file.startswith('#'): continue
            else:
                fileAndLangList.append(a_file.strip().split('|||'))

    if len(fileAndLangList) >= 1:
        return fileAndLangList
    else:
        print "There is no file, format should be (filelocation|||languagecode)"


def vocab(conll_path, language):
    wordsCount = Counter()
    posCount = Counter()
    xposCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP, language):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            xposCount.update([node.xpos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), xposCount.keys(), relCount.keys())

def vocab_multilingual(file_path):
    wordsCount = Counter()
    posCount = Counter()
    xposCount = Counter()
    relCount = Counter()

    fileAndLangList = readFileList(file_path)
    for a_file in fileAndLangList:
        print a_file[0]
        with open(a_file[0], "r") as conllFP:
            for sentence in read_conll(conllFP, a_file[1]):
                wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
                xposCount.update([node.xpos for node in sentence if isinstance(node, ConllEntry)])
                relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), xposCount.keys(), relCount.keys())


def read_conll(fh, language):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-XPOS', '_', -1, 'rroot', '_', '_', language)
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9], language))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


def read_languageVec(lang_vec_file):
    csv_file = open(lang_vec_file, 'rb')
    reader = csv.reader(csv_file)
    languageVec_dic = {row[4]: LanguageEntry(row[0], row[3], row[4], row[5]) for row in reader}
    csv_file.close()

    return languageVec_dic


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

