import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import joblib
import re
from nltk.stem import WordNetLemmatizer
from ImageClassification.settings import MODELS_PATH

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
clf= joblib.load(MODELS_PATH + "/Profane_lang_detect_realtime/logreg_clf")
tf_idf_vector = joblib.load(MODELS_PATH + "/Profane_lang_detect_realtime/tfidf_vectorizer")

def read_json_data(json_filename):

    if json_filename.split(".")[1] == 'csv':
        try:
            text_data = pd.read_csv(json_filename)['text'].iloc[0]
            return text_data
        except UnicodeDecodeError:
            csv_data = pd.read_csv(csv_filename, encoding='latin-1')['text'].iloc[0]
            return csv_data

    else:
        data = pd.read_json(json_filename)
        text_data = data.results[1][0]['transcript']
        return text_data

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

# def action(prob):
#     if prob <= 0.65 :
#         return "Informative, Like/Retweet"
#     else :
#         return "Can be Deleted"



def detector_from_string(json_filename):

    d = {}
    string = read_json_data(json_filename)
    #mapping_classes = {0: 'Hateful language', 1: "Profane language", 2: "Non Profane Language"}
    tt_0 =[string]
    tt0_cleaned = list(map(lambda x: text_cleanup(x),tt_0))
    tfidf_testdata = tf_idf_vector.transform(list(tt0_cleaned))
    arr = clf.predict(tfidf_testdata)
    if json_filename.split(".")[0].split("_")[1] == "1":
        d['Language'] = "French"
    elif json_filename.split(".")[0].split("_")[1] == "2":
        d['Language'] = "Spanish"
    elif json_filename.split(".")[0].split("_")[1] == "3":
        d['Language'] = "English"
    elif json_filename.split(".")[0].split("_")[1] == "4":
        d['Language'] = "English"
    else:
        d['Language'] = "English"


    if arr[0] == 0:
        d['Result'] = "No"
        d["Sentiment"] = "Negative"
        d['action'] ="Remove the content"
    elif arr[0] ==1:
        d['Result'] = "Yes"
        d["Sentiment"] = "Negative"
        d['action'] = "Remove the Content"
    else:
        d['Result'] = "No"
        d["Sentiment"] = "Positive"
        d['action'] = "Like/Comment"
    excel_filename = "Profanity_detection_" + ".xlsx"

    d["file_name"] = '/media/' + excel_filename

    return  d


