import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from ImageClassification.settings import BASE_DIR
import datetime

#nltk.download('omw-1.4')
stop_words = stopwords.words('english')

def read_csv_data(csv_filename):

    try:
        csv_data = pd.read_csv(csv_filename)['text']
    except UnicodeDecodeError:
        csv_data = pd.read_csv(csv_filename, encoding = 'latin-1')['text']

    return csv_data

def stopword_removal(sent):
    filtered_tokens = []
    for words in sent.split(" "):
        if words not in stop_words:
            filtered_tokens.append(words)
        else:
            continue
    return " ".join(filtered_tokens)
def lemma(sent):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(words) for words in sent.split(" ")])


def verb_contraction(sent):
    sent = re.sub(r"n't", ' not', sent)
    sent = re.sub(r"'ll", ' will', sent)
    sent = re.sub(r"'ve", ' have', sent)
    sent = re.sub(r"'s", ' is', sent)
    sent = re.sub(r"'re", ' are', sent)
    return sent

def preprocess(sent):

    sent = sent.lower()
    verb_contraction(sent)
    sent = re.sub(r'[0-9]+','',sent)
    sent = re.sub(r'[-:~^@!#$\'\'""``?;%\^&\*\(\)\[\]\/]+','',sent)
    shortword = re.compile(r'\W*\b\w{1}\b')
    sent=shortword.sub('', sent)
    sent = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',sent)
    sent = re.sub('[\s]+', ' ', sent)
    sent = sent.strip('\'"')
    sent=stopword_removal(sent)
    sent=lemma(sent)
    sent = ''.join([words for words in sent if len(sent.split(" "))>=3])
    return sent

def preprocessing_main(csv_filename):

    tt_0 = read_csv_data(csv_filename)
    data = pd.DataFrame()
    tt_0 = tt_0.astype('str')
    csv_file_name = str(csv_filename).split("/")
    #data["Data_source"] = (csv_file_name[-1])

    data['Input_Text'] = tt_0

    data['Text_preprocessed'] = list(map(lambda x: preprocess(x), tt_0))

    excel_filename = "Preprocessed_text_" + str(csv_file_name[-1]) + "_" + str(datetime.datetime.today().date()) + ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename
    return True, df_filepath