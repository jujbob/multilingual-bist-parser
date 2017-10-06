from optparse import OptionParser
import pickle, utils, mstlstm, os, os.path, time
from utils import read_languageVec

import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--disablelabels", action="store_false", dest="labelsFlag", default=True)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--disablecostaug", action="store_false", dest="costaugFlag", default=True)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)

    parser.add_option("--multilingual_emb", action="store_true", dest="multilingual_emb", help="Is it using multilingual embedding?", default=False)
    parser.add_option("--extConcateFlag", action="store_false", dest="extConcateFlag", help="concatenate external embedding? or it will be used as a word vector", default=True)
    parser.add_option("--lang_vec_file", dest="lang_vec_file", help="language vector file", metavar="FILE", default="./language_vec.csv" )
    parser.add_option("--extcluster", dest="external_cluster_embedding", help="External Cluster embeddings", metavar="FILE")
    parser.add_option("--add_lang_vec", action="store_true", dest="add_lang_vec", help="add language vector?", default=False)
    parser.add_option("--test_lang", type="string", dest="conll_test_language", help="Test language code ex)English", default="Unknown")
    parser.add_option("--train_lang", type="string", dest="conll_train_language", help="Test language code ex)English", default="Unknown")
    parser.add_option("--ext_dim", type="int", dest="edim", default=0)
    parser.add_option("--ext_cluster_dim", type="int", dest="ecdim", default=0)
    parser.add_option("--xpembedding", type="int", dest="xpembedding_dims", default=0)
    parser.add_option("--train_multilingual", action="store_true", dest="train_multilingual", help="Train it based on multilingual source?", default=False)
    parser.add_option("--out_file_name", type="string", dest="out_file_name", default="unknown.conllu")

    (options, args) = parser.parse_args()

    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(options)

        print "Start loading a model"
        parser.Load(options.model)
        print "Finished loading a model"
#        languageVec_dic = read_languageVec(options.lang_vec_file) ## read language_vec.csv file
#        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
#        tespath = os.path.join(options.output, languageVec_dic[options.conll_test_language].lang_code+'.conll' if not conllu else languageVec_dic[options.conll_test_language].lang_code+'.conllu')
        testpath = options.output+options.out_file_name

        ts = time.time()
        test_res = list(parser.Predict(options.conll_test))
        te = time.time()
        print 'Finished predicting test.', te-ts, 'seconds.'
        utils.write_conll(testpath, test_res)

#        if not conllu:
#            os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt')
#        else:
        os.system('python ./bist-parser/bmstparser/src/utils/evaluation_script/conll17_ud_eval.py -v ' + options.conll_test + ' ' + devpath + ' > ' + devpath +str(te)+ '.txt')
    else:

        if options.train_multilingual:
            print "########## Multilingual-source training ########## You may need to have --multilingual_emb and --add_language_vec"
        elif options.conll_train is None or options.conll_train_language=="Unknown":
            print "########## You must set --train_multilingual or --train with --train_lang"
            sys.exit()
        if options.multilingual_emb:
            print "########## Using multilingual embedding ##########"
            if options.external_embedding is None and options.external_cluster_embedding is None:
                print 'You must use --extr or(and) --extcluster for training it as a multilingual parser or if you delete --multilingual_emb to train it as a monolingual parser'
                sys.exit()
            elif not options.add_lang_vec:
                print "You may need to put on language vectors to distinguash languages, but it's not a mandatory!"

        print "Training language: ", options.conll_train_language , " external word embedding concatenate Flag: ", options.extConcateFlag

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(options)

        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            parser.Train(options.conll_train)
            conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch+1) + ('.conll' if not conllu else '.conllu'))
            parser.Save(os.path.join(options.output, os.path.basename(options.model) + str(epoch+1)))
            utils.write_conll(devpath, parser.Predict(options.conll_dev))

            if not conllu:
                os.system('perl src/utils/eval.pl -g ' + options.conll_test  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
            else:
#                os.system('python /home/ktlim/git/bist-parser/bist-parser/bmstparser/src/utils/evaluation_script/conll17_ud_eval.py -v ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                os.system('python ./bist-parser/bmstparser/src/utils/evaluation_script/conll17_ud_eval.py -v ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')

