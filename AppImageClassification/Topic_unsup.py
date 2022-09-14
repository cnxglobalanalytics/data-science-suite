import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from nltk import everygrams,word_tokenize
#from ImageClassification.settings import Stopwords

stop_words = stopwords.words('english')
newStopWords = ['lol', 'LOL', 'said', 'this', 'year', 'please', 'know', 'find', 'look', 'back', 'today', 'also', 'read',
                'reading', 'want', 'much', 'emi', 'nocost', 'no', 'not', 'yes',
                'address', 'every', 'news', 'article', 'feel', 'call', 'auto', 'think', 'going', 'cancel', 'rate',
                'month', 'called', 'does',"p's",'http','Http','https','Https',
                'recieved', 'offered', 'able', 'new', 'make', 'save', 'saved', 'article', 'really', 'need', 'offer',
                'change', 'even', 'paper', 'further','covid','rt',
                'page', 'take', 'feel', 'right', 'many', 'thing', 'comment', 'section',
                'articles', 'full', 'todays', 'latest', 'share', 'comments', 'last', 'night', 'months', 'several',
                'centre', 'times', 'himself', 'nokia', 'honor', "that's",'these', 'those',
                'answer', 'question', 'questions', 'told', 'anymore', 'towards',
                'feels', 'require', 'chosen', 'yesterday', 'today', 'ready', 'Verified', 'Purchase', 'Color', 'Size:',
                'Style', 'verified', 'purchase', 'color', 'size', 'style', 'Colour', 'i', 's', 't', 'n', 'p',
                'br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'oh', 'ourselves', 'you', "you're","you've",
                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', '-', 'lg', 'g', 'his',
                'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                'theirs', 'themselves', 'would', "i've", 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                'am', 'is', 'are', 'was', 'were', 'be', "i'm", 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'etc', 'between', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', "'s", 'oh', 'iphone',
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'colour', 'now', 'd', 'gb',
                'll', 'm', 'o', 're', 'got', 'wont', 'led', 'going', 'let', 'due',
                'y', 'ain', 'aren', 'hadn', 'will', 'yes', 'one', 'still', 'dont', 'im', 'bad', 'ive', 'doesnt',
                'everything', 'actually', 'though', 'isnt', 'anything', 'either', 'something', 'never', 'fully',
                'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'fhnjm','needn', 'shan', "shan't", 'shouldn', 'wasn', 'weren',
                'won', 'I', 'wouldn','Size:', 'size', 'style', 'Colour', 'i', 's', 't', 'n', 'p','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

stop_words.extend(newStopWords)
stopadd = ['no', 'not']
stop_words.remove('no')
stop_words.remove('not')
#stop_words_final = set(Stopwords)
stop_words=set(stop_words)

n_topics = 5
top_tokens_per_topic = 6
cls=['green','orange','grey','blue','brown']

def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    return colorstr

def read_csv_data(csv_filename):
    try:
        csv_data = pd.read_csv(csv_filename)['text']
        return csv_data
    except UnicodeDecodeError:
        csv_data = pd.read_csv(csv_filename, encoding='latin-1')['text']
        return csv_data

def stopword_removal(match):
    filtered_sentence = []
    word_tokens = word_tokenize(match)
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)


def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
        docs[idx] = stopword_removal(docs[idx])
        # docs[idx] = tokenizer.tokenize(docs[idx])
        docs[idx] = ['_'.join(list(i)) for i in [k for k in list(everygrams(docs[idx].split(' '), 3, 3))]]

    return docs


def text_cleanup(match):
    match = match.lower()
    match = match.replace("won't", 'will not')
    match = match.replace("shouldn't", 'should not')
    match = match.replace("aren't", 'are not')
    match = match.replace("couldn't", 'could not')
    match = match.replace("doesn't", 'does not')
    match = match.replace(r"isn't", "is not")
    match = match.replace(r"weren't", "were not")
    match = match.replace(r"hasn't", "has not")
    match = match.replace(r"hadn't", "had not")
    match = ''.join([i for i in match if not i.isdigit()])
    # Remove additional white spaces
    match = re.sub('[\s]+', ' ', match)
    # Remove '.' in between a sentence
    match = re.sub('\s[\\.]\s', ' ', match)
    # remove individual characters
    shortword = re.compile(r'\W*\b\w{1}\b')
    match = shortword.sub('', match)
    match = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', match)
    match = re.sub('[\s]+', ' ', match)
    match = match.strip('\'"')
    match = re.sub('^Renewal Email\s?[A-z]+', "", match)
    match = re.sub('[-=/!%@#$;():~]', '', match)
    match = re.sub('[\s]+', ' ', match)
    match = match.strip('\'"')
    match = stopword_removal(match)
    match = re.sub('[^A-Za-z0-9]+', ' ', match)
    return match


def lda_model(dictionary, corpus):

    global n_topics
    num_topics = n_topics
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha='auto',random_state=1)

    return model


def topic_model(data):

    #docs = array(data)
    docs = docs_preprocessor(data)

    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    bigram = Phrases(docs, min_count=10, delimiter=' ')
    trigram = Phrases(bigram[docs], delimiter=' ')
    dictionary = Dictionary(docs)

    #dictionary.filter_extremes(no_below=2, no_above=0.1)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    model = lda_model(dictionary=dictionary, corpus=corpus)
    return model, model.print_topics()


def topic_extract_unsup(csv_filename):

    d = {}
    d['Language'] = "English"
    d["Topic_Count"] =n_topics
    d["Top_tokens"] = top_tokens_per_topic
    tt_0 = read_csv_data(csv_filename)
    tt_0 = tt_0.astype('str')
    tt0_cleaned = list(map(lambda x: text_cleanup(x), tt_0))
    model, model_topics = topic_model(tt0_cleaned)
    list1 = []
    for topic_id in range(len(model_topics)):
        # print(ls[topic_id][1].split("+")[:6])
        p1 = model_topics[topic_id][1].split("+")[:6]
        # print(p1)
        p2 = '|'.join(p1)
        # p2 =p2.split("*")[1].replace("_", " ")
        list1.append(p2)

    list12 = '$$$$$$$'.join(list1)
    str2 = re.sub('\d*\.\d*\*', '', list12)
    stt = str2.replace('|', ',').replace('$$$$$$$', ' | ')

    stt1 = re.sub("_", " ", stt)
    stt1 = re.sub('\"', "", stt1)
    stt2 = ' '.join([i.capitalize() for i in stt1.split(" ")])
    lstt1 = []
    for i in range(1, 6):

        lstt1.append('{  * Topic ' + f'{i}'+" *     } ")
        #lstt1.append("\n")
    lstt3 = []
    for i, j in zip(lstt1,stt2.split('|')):
        lstt3.append(i)
        lstt3.append(j)
    lstt4 = []
    for i in range(len(lstt3)):
        if i % 2 == 0:
            lstt4.append(str(lstt3[i].rstrip()) + ' = ' + str(lstt3[i + 1]))

    st3 = ' | '.join(lstt4)
    ls = []
    for i, j in zip(st3.split('|'), cls):
        ls.append(printmd(i+'\n', j))
    d['Result'] = '  '.join(ls)
    return d
