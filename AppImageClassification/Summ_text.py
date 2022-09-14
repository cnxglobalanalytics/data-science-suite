import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pandas as pd
#from ImageClassification.settings import Stopwords

stopWords = set(stopwords.words("english"))

def read_csv_data(csv_filename):
    try:
        csv_data = pd.read_csv(csv_filename)
    except UnicodeDecodeError:
        csv_data = pd.read_csv(csv_filename, encoding = 'latin-1')

    if csv_data.shape[0] == 1:
        return csv_data['text'].iloc[0]
    elif csv_data.shape[0] > 1:
        return '.'.join(csv_data['text'].tolist())

def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if (tag == "NN" or tag == "NNP" or tag == "NNS") & (tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ"):
             pos_tagged_noun_verb.append(word)
    return '.'.join(pos_tagged_noun_verb)
def summ(csvfile):

    d={}

    text = read_csv_data(csvfile)
    words = len(text.split())
    d['Language'] = "English"
    d["Words"] = words
    d["String"] = text
    text = re.sub(r"\(.*?\)|\{.*?\}|\[.*?\]", "", text)
    text = re.sub(r'[^\.\,A-z0-9\s]',"",text)
    text = re.sub(r'[^\.\,A-z0-9\s]', "", text)
    #text = pos_tagging(text)
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    mention_regex = '@[\w\-]+'
    #text = re.sub(space_pattern, ' ', match)
    text = re.sub(giant_url_regex, '', text)
    text = re.sub(mention_regex, '', text)
    text = re.sub('[-=/!%@#$;():~]', '', text)
    text = re.sub('http[s]?://t.co/.*?', '', text)
    text = re.sub('#[A-z]+', '', text)
    text = re.sub('@[A-z]+', '', text)

    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.

    summary = ''
    if len(text.split(" ")) < 50:
        par = 0.7

    elif (len(text.split(" ")) >= 50)&(len(text.split(" ")) <= 100):
        par = 0.7

    elif (len(text.split(" ")) > 100)&(len(text.split(" ")) <= 200):
        par = 1.0
    elif (len(text.split(" ")) > 200)&(len(text.split(" ")) <= 400):
        par = 1.1
    elif (len(text.split(" ")) > 400)&(len(text.split(" ")) <= 1000):
        par = 1.2
    elif (len(text.split(" ")) > 1000) & (len(text.split(" ")) <= 10000):
        par = 1.5
    else:
        par = 2.4


    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (par * average)):
            summary += " " + sentence
    d['Result'] = summary
    return d