import os
import pickle
from dynet import *
from utils import read_conll, write_conll, read_languageVec
from operator import itemgetter
import utils, time, random, decoder
import numpy as np


class MSTParserLSTM:
    def __init__(self, new_options):

        print '1. Init Parser'
        print '1-1. Preparing vocab'
        if not new_options.predictFlag:
            if new_options.train_multilingual :
                vocab, w2i, pos, xpos, rels = utils.vocab_multilingual(new_options.conll_train)
#                new_options.xpembedding_dims = 0
            else:
                vocab, w2i, pos, xpos, rels = utils.vocab(new_options.conll_train, new_options.conll_train_language)
                new_options.xpembedding_dims = 0 if len(xpos) < 5 else new_options.xpembedding_dims
            options = new_options
        else:
            with open(new_options.params, 'r') as paramsfp:
                vocab, w2i, pos, xpos, rels, ex_trn, exc_trnd, stored_opt = pickle.load(paramsfp)
                self.extrnd = ex_trn
                self.exctrnd = exc_trnd

            stored_opt.conll_test = new_options.conll_test
            stored_opt.conll_test_language = new_options.conll_test_language
            stored_opt.predictFlag = new_options.predictFlag
            stored_opt.lang_vec_file = new_options.lang_vec_file

            options = stored_opt
        print "     ls it using multilingual embedding?:", options.multilingual_emb
        print '1-1. End of preparing vocab'

        self.model = Model()
        random.seed(1)
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag
        self.extConcateFlag = options.extConcateFlag

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.xpdims = options.xpembedding_dims

        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.xpos = {word: ind + 3 for ind, word in enumerate(xpos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.train_multilingual = options.train_multilingual
        self.lang_vec_file = options.lang_vec_file
        self.multilingual_emb = options.multilingual_emb
        self.add_lang_vec = options.add_lang_vec
        self.languageVec_dic = read_languageVec(self.lang_vec_file) ## read language_vec.csv file
        self.landims = len(self.languageVec_dic.values()[0].lang_vec)
        self.conll_test_language = options.conll_test_language
        self.conll_train_language = options.conll_train_language if options.conll_train_language is not None else "Unknown"
        print "     Training language: ", self.conll_train_language
        print "     Load Language vector, Dimensions: ", self.landims

        print "1-2. Load external embedding"
        self.external_embedding, self.edim = None, 0
        if options.predictFlag:
            self.elookup = self.model.add_lookup_parameters((3, 1))  # set temporal variable for model, later it will be resetted by model.load automatically.
            if options.external_embedding is not None:
                self.external_embedding = options.external_embedding
                self.edim = options.edim
        else:
            if options.external_embedding is not None:
                external_embedding_fp = open(options.external_embedding,'r')
                external_embedding_fp.readline()
                self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
                external_embedding_fp.close()

                self.edim = len(self.external_embedding.values()[0])
                self.noextrn = [0.0 for _ in xrange(self.edim)]
                self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
                self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
                for word, i in self.extrnd.iteritems():
                    self.elookup.init_row(i, self.external_embedding[word])
                self.extrnd['*PAD*'] = 1
                self.extrnd['*INITIAL*'] = 2

                if options.extConcateFlag :
                    print '     Load external embedding. It will be used for an additional embedding', self.edim
                else:
                    self.wdims = self.edim
                    print '     Load external embedding. It will be used for the word dimension vector', self.edim
            else:
                self.elookup = self.model.add_lookup_parameters((3, 1)) #set temporal variable for model, later it will be resetted by model.load automatically.
                self.extrnd = {}
        print "1-2. End of loading external embedding"

        print "1-3. Load external cluster embedding"
        self.external_cluster_embedding, self.ecdim = None, 0
        if options.predictFlag:
            self.eclookup = self.model.add_lookup_parameters((3, 1))
            if options.external_cluster_embedding is not None:
                self.external_cluster_embedding = options.external_cluster_embedding
                self.ecdim = options.ecdim
        else:
            if options.external_cluster_embedding is not None:
                external_cluster_embedding_fp = open(options.external_cluster_embedding,'r')
                external_cluster_embedding_fp.readline()
                self.external_cluster_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_cluster_embedding_fp}
                external_cluster_embedding_fp.close()

                self.ecdim = len(self.external_cluster_embedding.values()[0])
                self.noexctrn = [0.0 for _ in xrange(self.ecdim)]
                self.exctrnd = {word: i + 3 for i, word in enumerate(self.external_cluster_embedding)}
                self.eclookup = self.model.add_lookup_parameters((len(self.external_cluster_embedding) + 3, self.ecdim))
                for word, i in self.exctrnd.iteritems():
                    self.eclookup.init_row(i, self.external_cluster_embedding[word])
                self.exctrnd['*PAD*'] = 1
                self.exctrnd['*INITIAL*'] = 2
                print '     Load external cluster embedding. It will be used for an additional embedding', self.ecdim
            else:
                self.eclookup = self.model.add_lookup_parameters((3, 1))
                self.exctrnd = {}
        print "1-3 End of loading external cluster embedding"

        ### Add language embedding
        if self.add_lang_vec:
            print "Add Language Vector", "language dims: ", self.landims
            self.llookup = self.model.add_lookup_parameters((self.landims + 3, self.landims))
            for key in self.languageVec_dic.keys():
                self.llookup.init_row(self.languageVec_dic.get(key).lang_num, self.languageVec_dic.get(key).lang_vec)
        ## Finish language embedding

        self.dims = self.wdims + self.pdims + self.xpdims + (self.landims if self.add_lang_vec else 0) + (self.edim if options.extConcateFlag else 0) + (self.ecdim if self.external_cluster_embedding else 0)
        print "Total dims: ", self.dims, "word dims: ", self.wdims

        if self.bibiFlag:
            self.builders = [VanillaLSTMBuilder(1, self.dims, self.ldims, self.model),
                             VanillaLSTMBuilder(1, self.dims, self.ldims, self.model)]
            self.bbuilders = [VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                              VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        elif self.layers > 0:
            self.builders = [VanillaLSTMBuilder(self.layers, self.dims, self.ldims, self.model),
                             VanillaLSTMBuilder(self.layers, self.dims, self.ldims, self.model)]
        else:
            self.builders = [SimpleRNNBuilder(1, self.dims, self.ldims, self.model),
                             SimpleRNNBuilder(1, self.dims, self.ldims, self.model)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.xpos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        self.xpos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.xplookup = self.model.add_lookup_parameters((len(xpos) + 3, self.xpdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))

        self.hid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias = self.model.add_parameters((self.hidden2_units))

        self.outLayer = self.model.add_parameters((1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))

        if self.labelsFlag:
            self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
            self.rhidBias = self.model.add_parameters((self.hidden_units))

            self.rhid2Layer = self.model.add_parameters((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = self.model.add_parameters((self.hidden2_units))

            self.routLayer = self.model.add_parameters((len(self.irels), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
            self.routBias = self.model.add_parameters((len(self.irels)))

        if not new_options.predictFlag:
            options.edim = self.edim
            options.ecdim = self.ecdim
            with open(os.path.join(new_options.output, new_options.params), 'w') as paramsfp:
                    pickle.dump((vocab, w2i, pos, xpos, rels, self.extrnd, self.exctrnd, options), paramsfp)
            print 'Finished collecting vocab'



    def  __getExpr(self, sentence, i, j, train):
        if sentence[i].headfov is None:
            sentence[i].headfov = self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].modfov is None:
            sentence[j].modfov  = self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])
        if self.hidden2_units > 0:
            output = self.outLayer.expr() * self.activation(self.hid2Bias.expr() + self.hid2Layer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr())) # + self.outBias
        else:
            output = self.outLayer.expr() * self.activation(sentence[i].headfov + sentence[j].modfov + self.hidBias.expr()) # + self.outBias
        return output


    def __evaluate(self, sentence, train):
        exprs = [ [self.__getExpr(sentence, i, j, train) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])
        return scores, exprs


    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = self.rhidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]])
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov  = self.rhidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]])

        if self.hidden2_units > 0:
            output = self.routLayer.expr() * self.activation(self.rhid2Bias.expr() + self.rhid2Layer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr())) + self.routBias.expr()
        else:
            output = self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

        return output.value(), output


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.load(filename)



    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, self.conll_test_language)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

                for entry in conll_sentence:
                    posID = self.pos[entry.pos] if self.pos.has_key(entry.pos) else 0
                    posvec = self.plookup[int(posID)] if self.pdims > 0 else None
                    xposID = self.xpos[entry.xpos] if self.xpos.has_key(entry.xpos) else 0
                    xposvec = self.xplookup[int(xposID)] if self.xpdims > 0 else None
                    evec = None
                    ecevc = None
                    lang_code = entry.language.split('_')[0]+':' if self.multilingual_emb else ""

                    if self.external_embedding is not None:
                        if self.extConcateFlag:
                            wordvec = self.wlookup[int(self.vocab.get(entry.norm.lower(), 0))] if self.wdims > 0 else None
                            evec = self.elookup[self.extrnd.get(lang_code+entry.form.lower(), self.extrnd.get(lang_code+entry.norm.lower(), 0))]
                        else:
                            wordvec = self.elookup[self.extrnd.get(lang_code+entry.form.lower(), self.extrnd.get(lang_code+entry.norm.lower(), 0))]
                    else:
                        wordvec = self.wlookup[int(self.vocab.get(entry.norm.lower(), 0))] if self.wdims > 0 else None

                    if self.external_cluster_embedding is not None:
                        ecevc = self.eclookup[self.exctrnd.get(lang_code+entry.form.lower(), self.exctrnd.get(lang_code+entry.norm.lower(), 0))]

                    # Add language embedding
                    langvec = self.llookup[self.languageVec_dic[entry.language].lang_num] if self.add_lang_vec else None

                    entry.vec = concatenate(filter(None, [wordvec, posvec, xposvec, evec, ecevc, langvec]))
                    entry.lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence, True)
                heads = decoder.parse_proj(scores)

                ## ADD for handling multi-roots problem
                rootHead = [head for head in heads if head==0]
                if len(rootHead) != 1:
                    print "it has multi-root, changing it for heading first root for other roots"
                    rootHead = [seq for seq, head in enumerate(heads) if head == 0]
                    for seq in rootHead[1:]:heads[seq] = rootHead[0]
                ## finish to multi-roots

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                dump = False

                if self.labelsFlag:
                    for modifier, head in enumerate(heads[1:]):
                        scores, exprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                        conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump: yield sentence


    def Train(self, conll_path):
        errors = 0
        batch = 0
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        shuffledData = []
        if self.train_multilingual:
            file_list = utils.readFileList(conll_path)
            for fileAndLang in file_list:
                print "Reading Training file:",fileAndLang[0], " language: ",fileAndLang[1]
                with open(fileAndLang[0], "r") as conllFP:
                    shuffledData = shuffledData + list(read_conll(conllFP, fileAndLang[1]))
        else:
            with open(conll_path, 'r') as conllFP:
                shuffledData = list(read_conll(conllFP, self.conll_train_language))
        random.shuffle(shuffledData)

        errs = []
        lerrs = []
        eeloss = 0.0
        undefined_term = 0

        for iSentence, sentence in enumerate(shuffledData):
            if iSentence % 100 == 0 and iSentence != 0:
                print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Time', time.time()-start
                start = time.time()
                eerrors = 0
                eloss = 0.0
                etotal = 0
                lerrors = 0
                ltotal = 0

            conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]

            for entry in conll_sentence:
                c = float(self.wordsCount.get(entry.norm, 0))
                dropFlag = (random.random() < (c/(0.25+c)))
                posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
                xposvec = self.xplookup[int(self.xpos[entry.xpos])] if self.xpdims > 0 else None

                evec = None
                ecevc = None
                lang_code = entry.language.split('_')[0]+':' if self.multilingual_emb else ""

                # Add word and external embedding
                if self.external_embedding is not None:
                    if self.extConcateFlag:
                        wordvec = self.wlookup[int(self.vocab.get(entry.norm.lower(), 0)) if dropFlag else 0] if self.wdims > 0 else None
                        evec = self.elookup[self.extrnd.get(lang_code+entry.form.lower(), self.extrnd.get(lang_code+entry.norm.lower(), 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                    else:
                        wordvec = self.elookup[self.extrnd.get(lang_code+entry.form.lower(), self.extrnd.get(lang_code+entry.norm.lower(), 0)) if (dropFlag or (random.random() < 0.5)) else 0]
                        if self.extrnd.get(lang_code + entry.form.lower(), self.extrnd.get(lang_code + entry.norm.lower(), 0)) == 0:
                            undefined_term = undefined_term + 1
                else:
                    wordvec = self.wlookup[int(self.vocab.get(entry.norm.lower(), 0)) if dropFlag else 0] if self.wdims > 0 else None

                # Add external cluster embedding
                if self.external_cluster_embedding is not None: ecevc = self.eclookup[self.exctrnd.get(lang_code+entry.form.lower(), self.exctrnd.get(lang_code+entry.norm.lower(), 0)) if (dropFlag or (random.random() < 0.5)) else 0]

                # Add language embedding
                langvec = self.llookup[self.languageVec_dic[entry.language].lang_num] if self.add_lang_vec else None

#                    print langvec.value()

                entry.vec = concatenate(filter(None, [wordvec, posvec, xposvec, evec, ecevc, langvec]))

                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

            if self.blstmFlag:
                lstm_forward = self.builders[0].initial_state()
                lstm_backward = self.builders[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = lstm_forward.output()
                    rentry.lstms[0] = lstm_backward.output()
                if self.bibiFlag:
                    for entry in conll_sentence:
                        entry.vec = concatenate(entry.lstms)
                    blstm_forward = self.bbuilders[0].initial_state()
                    blstm_backward = self.bbuilders[1].initial_state()
                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        blstm_forward = blstm_forward.add_input(entry.vec)
                        blstm_backward = blstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = blstm_forward.output()
                        rentry.lstms[0] = blstm_backward.output()

            scores, exprs = self.__evaluate(conll_sentence, True)
            gold = [entry.parent_id for entry in conll_sentence]
            heads = decoder.parse_proj(scores, gold if self.costaugFlag else None)

            if self.labelsFlag:
                for modifier, head in enumerate(gold[1:]):
                    rscores, rexprs = self.__evaluateLabel(conll_sentence, head, modifier+1)
                    goldLabelInd = self.rels[conll_sentence[modifier+1].relation]
                    wrongLabelInd = max(((l, scr) for l, scr in enumerate(rscores) if l != goldLabelInd), key=itemgetter(1))[0]
                    if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                        lerrs.append(rexprs[wrongLabelInd] - rexprs[goldLabelInd])

            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            eerrors += e
            if e > 0:
                loss = [(exprs[h][i] - exprs[g][i]) for i, (h,g) in enumerate(zip(heads, gold)) if h != g] # * (1.0/float(e))
                eloss += (e)
                mloss += (e)
                errs.extend(loss)

            etotal += len(conll_sentence)

            if iSentence % 1 == 0 or len(errs) > 0 or len(lerrs) > 0:
                eeloss = 0.0

                if len(errs) > 0 or len(lerrs) > 0:
                    eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
                    eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    lerrs = []

                renew_cg()

        if len(errs) > 0:
            eerrs = (esum(errs + lerrs)) #* (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []
            eeloss = 0.0

            renew_cg()
        print " # of uddefined_term= ", undefined_term
        self.trainer.update_epoch()
        print "Loss: ", mloss/iSentence
