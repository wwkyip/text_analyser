#
# nlphelper
# Description: class to do NLP processing for feature extraction and predict model
#
# 20170303 feature extraction functions created

import sys
import re
import json
#import enchant
import nltk
from nltk.tag import pos_tag, map_tag
from nltk import sentiment
from nltk import PorterStemmer
from nltk import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.corpus import sentiwordnet
from nltk.corpus import names, wordnet
from nltk.sentiment import util
from datalab_nlp import datahelper
import pickle
import numpy as np
#from pyaspeller import Word
import pandas as pd


###########################################################################
# function definition


#########################################################
# variable
#

global modelfile, pth
pth = "datalab_nlp/"
modelfile = pth + "modelconstructive" # for predicting the constructiveness probability [0,1]


######################################################### 
# nlphelper class
# 
class nlphelper(object):
    #description of class

    def __init__(self):
        #intializer
        #global loaded_model, topfeature # for constructiveness model
        self.loaded_model     = pickle.load(open(modelfile+"/model.pickle", 'rb'))
        self.topfeature       = pickle.load(open(modelfile+"/topfeatures.pickle", 'rb'))


        #global classes, keywords_all        
        self.classes = []
        self.classes = datahelper.query("themekeyword", "distinct class", "1=1", self.classes) 
        
        self.keywords_all = []
        for c in self.classes:
            self.keywords = []
            datahelper.query("themekeyword","keyword,class","class='" + c + "'",self.keywords)
            self.keywords.append(c)  # add the class name as keyword as well
            class_synonyms = self.synonyms(c) # add synonyms
            if len(class_synonyms) > 0:
               self.keywords.append(" ".join(class_synonyms))
            self.keywords_all.append(self.keywords) 
            
        #global english_words
        self.english_words = pd.read_csv(pth+"english_words.txt",header=None)[0].tolist()

    # count_syllables
    # Find the number of syllables in word input
    #
    def count_syllables(self, word):
        # taken from http://stackoverflow.com/questions/405161/detecting-syllables-in-a-word
        vowels = "aeiouy"
        numVowels = 0
        lastWasVowel = False
        for wc in word:
            foundVowel = False
            for v in vowels:
                if v == wc:
                    if not lastWasVowel: #don't count diphthongs
                        numVowels += 1
                        foundVowel = True
                        lastWasVowel = True
                        break
            if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
                lastWasVowel = False
        if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
            numVowels-=1
        elif len(word) > 1 and word[-1:] == "e":    #remove silent e
            numVowels-=1
        return numVowels
       
    # flesch_read
    # Find the Flesch readability metric of a input sentence
    #
    def flesch_read(self, sentence):
        words = sentence.replace("."," ").replace(";"," ").replace(","," ").split()
        total_words = len(words)
        total_syllables = 0
        for w in words:
            total_syllables += self.count_syllables(w)
        sentences = sentence.split(".")
        total_sentences = 0
        for i in range(0,len(sentences)):
            if len(sentences[i]) > 0:
                total_sentences += 1
                   
        return(206.835-1.015*(total_words/total_sentences)-84.6*(total_syllables/total_words))

    
    def stemmer(self, stemtype, words):

        if stemtype == "Snowball":
            stemmer = SnowballStemmer("english")
        elif stemtype == "Lancaster":
            stemmer = LancasterStemmer()
        else:
            stemmer = PorterStemmer()

        stemwords = [stemmer.stem(word) for word in words]

        return(stemwords)

    def remstopwords(self, words):
        stop = set(stopwords.words('english'))
        nostopwords = [word for word in words if word not in stop]

        return(nostopwords)

    def tokenize(self, sentence):
        words = nltk.word_tokenize(sentence);
        return(words)

    def tokenize_sentence(self, sentence):
        sentencelist = nltk.sent_tokenize(sentence);
        return(sentencelist)
    
    def spellcheck(self, words):
        wordsuggestlist = []

        for word in words:
            if (Word(word).correct == False) and (not word.istitle()) and (not word.isupper()) and (len(word)>1): #exclude all capital/title cases
                wordsuggestlist.append({"word": word, "wordsuggest": ",".join(Word(word)).variants})

        return(wordsuggestlist)

    '''   
    # p_correctspell
    # percentage of correctly spelt words using spell checker
    def p_correctspell(self, words):
        validcount = 0
        count = 0
        for w in words:
            if w not in (".",","):
                if (Word(w).correct and not(w.isupper())) or w.isupper() or w.istitle():
                    validcount += 1
                count += 1
        return(validcount*100.0/count)
    '''
    
    # p_correctspell
    # percentage of correctly spelt words
    def p_correctspell(self, words):
        words = set(w.lower() for w in words)
        non_english = words.difference(self.english_words)

        if len(words) > 0:
            return (100-(len(non_english)*100.0 / len(words)))
        return (0)

    
    def checkprofanity(self, sentence):
        profanities = []
        profanitydict = []
        datahelper.query("generalkeyword", "phrase", "category='profanity'", profanitydict)
        for profanityword in profanitydict:
            if re.search( r'(^|\s)(' + profanityword + ')($|\s|\.)', sentence.lower()):
                profanities.append(profanityword)
        return(profanities)

    # p_profanity
    # percentage of profanity words
    def p_profanity(self, sentence, words):
        profanities = self.checkprofanity(sentence)
        return(len(profanities)*100.0/len(words))

    def checkjudgemental(self, sentence):
        judgementallist = []
        judgementaldict = []
        datahelper.query("generalkeyword", "phrase", "category='judgemental'", judgementaldict)
        for jw in judgementaldict:
            if re.search( r'(^|\s)(' + jw.lower() + ')($|\s|\.)', sentence.lower()):
                judgementallist.append(jw)
        return(judgementallist)

    # p_judgemental
    # percentage of judgemental phrases
    def p_judgemental(self, sentence, words):
        judgemental = self.checkjudgemental(sentence)
        return(len(judgemental)*100.0/len(words))

    # p_pos
    # percentage of POS types
    def p_pos(self, pos):
        prp =0
        adj =0
        adv =0
        noun =0
        pnoun =0
        verb =0
        for i in pos:
            if i[1] in ("PRP"):
                prp += 1
            if i[1] in ("JJ","JJS","JJR"):
                adj += 1
            if i[1] in ("RB","RBR","RBS"):
                adv += 1
            if i[1] in ("NN","NNS"):
                noun += 1
            if i[1] in ("VB","VBD","VBG","VBN","VBP","VBZ"):
                verb += 1
            if i[1] in ("NNP","NNPS"):
                pnoun += 1
            poscount = [prp, adj, adv, noun, pnoun, verb]
        return([p*100.0/len(pos) for p in poscount])

    def personalityattack(self, words):
        personattackwords = []
        wherewords = ",".join(["'%s'" % (w.replace("'","''")) for w in words])
        whereclause = "phrase IN (" + wherewords + ") and subjecttype='strongsubj' and polarity='negative'"
        datahelper.query("sentimentkeyword", "phrase,polarityn", whereclause, personattackwords)
        return(personattackwords)

    # p_personalityattack
    # percentage of personality attack phrases
    def p_personalityattack(self, words):
        attackwords = self.personalityattack(words)
        return(len(attackwords)*100.0/len(words))

    # extract skill/duties/actions phrases
    def skillduties(self, pos):

        # extract action, must have verb followed by what ever in between (numbers, adjectives) * and end with noun
        # or Modal followed by determinant, etc and end with noun
        actions = ""
        actionreg = """
                    Action: {<V.*>*<RB>*<V.*><R.*>*<DT>*<JJ>*<PRP.*>*<IN>*<CD>*<N.*>+}
                            {<MD><DT>*<R.*>*<N.*>+}
                    """
        action = nltk.RegexpParser(actionreg)
        tree = action.parse(pos)
        for subtree in tree.subtrees():
             if subtree.label() == 'Action':
                 actions = actions + " ".join(re.findall(r"([A-Za-z0-9-/]+)/[A-Z$#]{2,4}", str(subtree))) + ";"

        # extract bigram and above terms (skills)
        terms = ""
        bigramreg = """
                    Skills: {<JJ>+<N.*>+}
                            {<N.*><N.*>+}
                            {<R.*><J.*>}
                    """
        bigram = nltk.RegexpParser(bigramreg)
        tree = bigram.parse(pos)
        for subtree in tree.subtrees():
            if subtree.label() == 'Skills':
                #print(str(subtree))
                skill = " ".join(re.findall(r"([A-Za-z0-9#-/]+)/[A-Z$#]{2,4}", str(subtree)))
                found = False
                # make sure the noun didn't appear in any of the actions
                for a in actions.split(";"):
                    if re.search("\\b"+ skill.replace("*","").replace("+","") +"\\b",a,re.M|re.I):
                        found = True
                        break
                if found == False:
                    terms = terms + skill + ";"


        matchphrases = []
        phrases = terms[:-1].split(";") + actions[:-1].split(";")
        for p in phrases:
            matched = []
            wd = [w for w in p.lower().split(" ")]
            stem_p = " ".join(self.stemmer("Snowball",wd))
            whereclause = "phrase='" + stem_p.replace("'","''") + "' and category='jobngram'"
            datahelper.query("generalkeyword", "phrase", whereclause, matched)
            if len(matched) > 0:
                matchphrases.append(p)
                
        # unigram skills (must be matched from the knowledge base)
        for w in pos:
            matched = []
            if w[1] in ("NN","NNS","NNP","NNPS"):
                uniterm = w[0].lower().replace("'","''")
                whereclause = "phrase='" + uniterm + "' and category='skillunigram'"
                datahelper.query("generalkeyword", "phrase", whereclause, matched)
            if len(matched) > 0:
                matchphrases.append(w[0])
                
        return([phrases,matchphrases])

    # p_skillduties
    # Find percentage of phrases extracted by POS regex and those that matched knowledge base
    def p_skillduties(self, sentence):
        [phrases,matchphrases] = self.skillduties(sentence)
        return([len(phrases),len(matchphrases)])

    # find the sentiment score
    def sentimentscore(self, sentence):

        words = nltk.word_tokenize(sentence)
        postag = pos_tag(words)
        neg = sentiment.util.mark_negation(words)
        negflg = [i.endswith("_NEG") for i in neg]

        # we want to only process those words that are valid POS (exclude all the NNP ones and punctuations)
        validPOS = ['JJ','JJR','JJS','RB','RBR','RBS','NN','NNS','VB','VBD','VBG','VBN','VBP','VBZ']
        senti_index = []
        for pi in range(0,len(postag)):
            if postag[pi][1] in validPOS:
                senti_index.append(pi)

        # initialise
        abstotal = 0
        total = 0 
        score = 0
        poswords = []
        negwords = []
        sentimentwords = {}
        
        if len(senti_index)>0: # only search if there are found possible sentiment words
            wherewords = ",".join(["'%s'" % (words[i].replace("'","''")) for i in senti_index])
            whereclause = "phrase IN (" + wherewords + ")"
            datahelper.query("sentimentkeyword", "phrase,polarityn", whereclause, sentimentwords) 

        # do sentiment scoring
        for i in senti_index: 

            if words[i] in sentimentwords:
                if int(sentimentwords[words[i]]) > 0:
                    if negflg[i] == False:
                        poswords.append(words[i])
                        total = total + int(sentimentwords[words[i]])
                    else:
                        negwords.append("NEG_"+words[i])
                        total = total - int(sentimentwords[words[i]])
                if int(sentimentwords[words[i]]) < 0:
                    if negflg[i] == False:
                        negwords.append(words[i])
                        total = total + int(sentimentwords[words[i]])
                    else:
                        poswords.append("NEG_"+words[i])
                        total = total - int(sentimentwords[words[i]])
                abstotal = abstotal + abs(int(sentimentwords[words[i]]))

            if abstotal > 0:
                score = "%.2f" % (total*1.0/abstotal)  
            else:
                score = 0

        #return ({"score":float(score), "poswords": poswords, "negwords": negwords})
        return ({"score":total, "poswords": poswords, "negwords": negwords})

    # find synonyms of a input text
    def synonyms(self, text):
        words = text.split(' ')
        synm = []
        for w in words:
            syn_sets = wordnet.synsets(w, pos=wordnet.NOUN)
            for syn_set in syn_sets:
                synm += syn_set.lemma_names()
        return(list(set(synm)))
    
    
    # check for any subset of n words (stemmed) that match theme look up keywords
    def anysubset(self,s,k):
        N = 3
        
        words = nltk.word_tokenize(s)
        keys = k.split(' ')
        if len(keys) <= 5 or len(words) <= 5:
            return(False)

        posTagged = pos_tag(words)
        validpos = ['NN','NNS','VB','VBD','VBG','VBP','VBZ','VBN','JJ','JJR','JJS']
        validwords = []
        
        for i in range(0,len(posTagged)):
            if posTagged[i][1] in validpos:
                validwords.append(posTagged[i][0])
        validwords = self.remstopwords(validwords)
        keys = self.remstopwords(keys)
        stemwords = list(set(self.stemmer("Snowball", validwords))) # add list(set( to deduplicate
        stemkeys = list(set(self.stemmer("Snowball", keys)))

        matched = 0
        for w in stemwords:
            if w in stemkeys:
                matched += 1
            if matched > N: # break when found matched words
                #increase the threshold above when you want stricter matching
                return(True)
        
        return(False) 

    # classify comments into themes
    # using keyword search
    def classifykeyword(self,s,k):
        
        msg =""
        k = k.replace("*","[a-zA-Z]*")
        
        # search two words existing in sentence (don't care locations)
        if re.search("(AND)", k):
            ks = k.split(" (AND) ")
            if (re.search("\\b"+ks[0]+"\\b",s,re.M|re.I) and re.search("\\b"+ks[1]+"\\b",s,re.M|re.I)):
                k0str = re.search("\\b"+ks[0]+"\\b",s,re.M|re.I)
                k1str = re.search("\\b"+ks[1]+"\\b",s,re.M|re.I)
                msg = s[k0str.span(0)[0]:k0str.span(0)[1]] + " AND " + s[k1str.span(0)[0]:k1str.span(0)[1]] + ";"
                return(msg)
        
        # search two words near each other
        k0found = -9
        k1found = -9
        if re.search("(NEAR)", k):
            ks = k.split(" (NEAR) ")
            tokens = s.split(" ")
            for t in range(0,len(tokens)):
                if re.search("\\b"+ks[0]+"\\b",tokens[t],re.M|re.I):
                    k0found = t
                if re.search("\\b"+ks[1]+"\\b",tokens[t],re.M|re.I):
                    k1found = t
            if k0found == -9 or k1found == -9:
                return("")
            if k0found != -9 and k1found != -9:
                if abs(k0found-k1found) > 9:
                    return("")
                else:
                    msg = tokens[k0found] + " NEAR " + tokens[k1found] + ";"
                    return(msg)
        
        # search by phrase
        if re.search("\\b"+ k + "\\b", s, re.M|re.I):
            x = re.search("\\b"+ k + "\\b", s, re.M|re.I)
            msg = s[x.span(0)[0]:x.span(0)[1]]+";"
            return(msg)

        else:
            # search by subset of words overlapping
            if self.anysubset(s,k):
                msg = "subset("+ k + ");"
                return(msg)
            
        return("")

    # tag themes using keyword matching by calling classifykeyword
    def tagtheme(self,sentence):
        result = ""
        sentence = sentence.lower()
            
        for c in range(0,len(self.classes)):
           for k in self.keywords_all[c]:
               msg = self.classifykeyword(sentence,k)
               if msg != "":
                  result += (self.classes[c] + "[" + msg + "];")
                  break
        return(result)
 
    #  how many http mentions?
    def http_count(self, sentence):
        return len(re.findall(r'http',sentence))
    
    #  how many http mentions?
    def book_count(self, sentence):
        return len(re.compile(r"\b(book)\b").findall(sentence))
    
    # find constructive verbs
    def constructive_verb(self, sentence):
        constructivewords = {}
        datahelper.query("constructivekeyword", "phrase,score", "category='word'", constructivewords)
        startwith = {}
        datahelper.query("constructivekeyword", "phrase,score", "category='startwith'", startwith)
        sentencelist = self.tokenize_sentence(sentence)
        n=0
        for s in sentencelist:
            words = self.tokenize(s.lower())
            found_start = re.compile("^(" + "|".join(startwith) + ")").findall(s.lower())
            if len(found_start)>0:
                found = found_start[0]
                n += startwith[found]
            for w in words:
                if w in constructivewords:
                    n += constructivewords[w]
        return(n)
    
    
    # predict constructiveness
    def constructiveness(self, sentence):
        features = self.extractfeatures(sentence)
        #constructiveness = loaded_model.predict_proba([np.array(features)[topfeature]])[0][1]
        constructiveness = self.loaded_model.predict_proba([np.array(features)])[0][1]
        return(constructiveness)
    
    # extract features for machine learning training
    def extractfeatures(self, sentence):
        words = self.tokenize(sentence) 
        correctspell = self.p_correctspell(words)
        profanity = self.p_profanity(sentence, words)
        judgemental = self.p_judgemental(sentence, words)
        personattack = self.p_personalityattack(words)
        nowords = len(words)
        postag = pos_tag(words)
        sentimentobj = self.sentimentscore(sentence)
        sentiment = sentimentobj["score"]
        flesch = self.flesch_read(sentence)
        skillduties = self.p_skillduties(postag)
        p_pos = self.p_pos(postag)
        http_cnt = self.http_count(sentence)
        book_cnt = self.book_count(sentence)
        construct_verb = self.constructive_verb(sentence)
        return([nowords,correctspell,profanity,judgemental,personattack,sentiment,flesch]+skillduties+p_pos+[http_cnt,book_cnt,construct_verb])

    def textanalyzer(self, sentence):

        words = self.tokenize(sentence) #nltk tokenizer (Winny has undone the nopunct because we need the punct to do POS properly)
        #spellchecked = self.spellcheck(nswords) #check for spelling
        pfwords = self.checkprofanity(sentence)
        postag = pos_tag(words)
        [skillduties,matches] = self.skillduties(postag)
        themes = self.tagtheme(sentence)
        sentiment,poswords,negwords = self.sentimentscore(sentence)

        # features extraction
        features = self.extractfeatures(sentence)

        # predict from model
        constructiveness = self.loaded_model.predict_proba([np.array(features)[self.topfeature]])[0][1]
        
        result = {"raw":sentence, 
                  "profanity":",".join(pfwords),
                  #"spellcheck":",".join(map(str, spellchecked)),
                  #"postag": postag,
                  "poswords":poswords,
                  "negwords":negwords,
                  "actions/keyphrase": ",".join(skillduties),
                  "matchedskillduties": ",".join(matches),
                  "themes": themes,
                  "constructiveness":constructiveness, # probability [0,1] of constructiveness from decision tree
                  "nowords": features[0], # number of words
                  "p_correctspell": features[1], # - percentage of correctly spelt words (excluding named nouns),
                  "p_profanity": features[2], #percentage of profanity words, 0-1 real number,
                  "p_judgemental": features[3], # - percentage of judgemental words,
                  "p_personalityattack": features[4], # - percentage of personality attack words,
                  "sentiscore": features[5], # sentiment score (scaled to 1)
                  "flesch": features[6],  # flesch readability metric
                  "p_actions": features[7], # - percentage of detected skills/duties / action phrases
                  "p_matches": features[8], # - percentage of detected skills/duties from knowledge base
                  "p_prp": features[9], # - percentage of pronoun,
                  "p_adj": features[10], # - percentage of adjectives,
                  "p_adv": features[11], # - percentage of adverbs,
                  "p_noun": features[12], # - percentage of nouns,
                  "p_pnoun": features[3], # - percentage of named nouns,
                  "p_verb": features[14] # - percentage of verbs,
                  }
    
        return(json.dumps(result, sort_keys=True, indent=4))


