import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
#from textblob import TextBlob, Word
import joblib
import re
#from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import datetime
from ImageClassification.settings import BASE_DIR,MODELS_PATH

df = pd.DataFrame(columns=['Text_sample','Result'])
#nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
other_exclusions = ["#ff", "ff", "rt"]
stop_words.extend(other_exclusions)
stop_words=set(stop_words)
#nltk.download('wordnet')
#nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
clf= joblib.load(MODELS_PATH + "/Hatespeech_offensivelang_detection/logreg_clf")
tf_idf_vector = joblib.load(MODELS_PATH + "/Hatespeech_offensivelang_detection/tfidf_vectorizer")

def read_csv_data(csv_filename):

    try:
        csv_data = pd.read_csv(csv_filename)['text']
        return csv_data
    except UnicodeDecodeError:
        csv_data = pd.read_csv(csv_filename, encoding = 'latin-1')['text']
        return csv_data

def lemmatization(sentence):
    return " ".join([lemmatizer.lemmatize(w) for w in sentence.split(" ")])

def stopword_removal(match):
    filtered_sentence = []
    word_tokens = word_tokenize(match)
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return ' '.join(filtered_sentence)


def text_cleanup(match):

    match = match.lower()
    match = match.replace("n't", ' not')
    match = match.replace("'m", " am")
    match = match.replace("'ve", " have")
    match = match.replace("nigger","nigga")
    match = re.sub('[\s]+', ' ', match)
    # Remove '.' in between a sentence
    match = re.sub('\s[\\.]\s', ' ', match)
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', match)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    match = re.sub(mention_regex, '', parsed_text)
    match = re.sub('[-=/!%@#$;():~]', '', match)
    match = re.sub('[\s]+', ' ', match)
    match = match.strip('\'"')
    match = stopword_removal(match)
    match = lemmatization(match)

    return match

def action(prob):
    if prob <= 0.65 :
        return "Informative, Like/Retweet"
    else :
        return "Can be Deleted"

def detector_from_csvfile(csv_filename):

    tt_0 = read_csv_data(csv_filename)
    mapping_classes = {0 :'Hate_speech',1:"Offensive language",2:"Neither"}
    data = pd.DataFrame()

    tt_0 = tt_0.astype('str')
    tt0_cleaned = list(map(lambda x: text_cleanup(x), tt_0))
    tfidf_testdata = tf_idf_vector.transform(tt0_cleaned)
    arr = clf.predict(tfidf_testdata)

    csv_file_name = str(csv_filename).split("/")
    #data["Data_source"] = (csv_file_name[-1])
    arr_probs = clf.predict_proba(tfidf_testdata).tolist()
    data['Input_Text'] = tt_0
    data['Result'] = pd.Series(arr).map(mapping_classes).tolist()
    data['Probability'] = np.max(arr_probs,axis =1).tolist()
    data["Action"] = data['Probability'].apply(lambda x :action(x))
    data["Recommended_Action"] = np.where(data['Result'] == "Neither","Informative, Like/Retweet","Can be Deleted")
    excel_filename = "Hatespeech_offensivelanguage_detection_"+ ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename


    return True, df_filepath

def detector_from_string(string):

    d = {}
    tt_0 =[string]
    tt0_cleaned = list(map(lambda x: text_cleanup(x),tt_0))
    tfidf_testdata = tf_idf_vector.transform(list(tt0_cleaned))
    arr = clf.predict(tfidf_testdata)
    if arr[0] == 0:
        d['Result'] = "It's a hate speech"
    elif arr[0] ==1:
        d['Result'] = "It contains offensive language"
    else:
        d['Result'] = "It's neither a hate speech nor containing offensive language"

    excel_filename = "Hatespeech_detection_" + "_" + str(datetime.datetime.today().date()) + ".xlsx"

    d["file_name"] = '/media/' + excel_filename

    return True, d


