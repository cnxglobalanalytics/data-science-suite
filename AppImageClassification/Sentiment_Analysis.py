import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import joblib
import re
from nltk.stem import WordNetLemmatizer
from ImageClassification.settings import BASE_DIR,MODELS_PATH


df = pd.DataFrame(columns=['Text_sample','Result'])
#nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
stop_words=set(stop_words)
#nltk.download('wordnet')
#nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
tf_idf_vector = joblib.load(MODELS_PATH + "/Sentiment_Analysis/tfidf_vector_sentiment_an")
clf= joblib.load(MODELS_PATH + "/Sentiment_Analysis/logreg_classifier_sentiment_an")
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

def action(result):
    if result == 0 :
        return "Neutral"
    elif result == 1 :
        return "Positive"
    elif result == 2:
        return "Negative"

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


def prediction_from_csvfile(csv_filename):

    tt_0 = read_csv_data(csv_filename)
    #mapping_classes = {0 :'Non Racist/Sexist tweets',1:"Racist/Sexist tweets"}
    mapping_classes = {'neutral':0,'positive':1,'negative':2}
    data = pd.DataFrame()

    tt_0 = tt_0.astype('str')
    tt0_cleaned = list(map(lambda x: text_cleanup(x), tt_0))
    tfidf_testdata = tf_idf_vector.transform(tt0_cleaned)
    arr = clf.predict(tfidf_testdata)

    arr_probs = clf.predict_proba(tfidf_testdata).tolist()
    csv_file_name = str(csv_filename).split("/")
    #data["Data_source"] = (csv_file_name[-1])
    #print(pd.Series(arr))#.map(mapping_classes)
    data['Input_Text'] = tt_0
    data['Sentiment_Result'] = pd.Series(arr).tolist()
    data['Sentiment_Result'] = data['Sentiment_Result'].apply(lambda x: action(x))
    #data['Probability'] = np.max(arr_probs, axis=1).tolist()
    #data["Action"] = data['Twitter_Sentiment_Result'].apply(lambda x: action(x))

    excel_filename = "Sentiment_Analysis_"+ ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename


    return True, df_filepath




