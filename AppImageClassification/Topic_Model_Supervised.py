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
other_exclusions = ["#ff", "ff", "rt"]
stop_words.extend(other_exclusions)
stop_words=set(stop_words)
#nltk.download('wordnet')
#nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
clf= joblib.load(MODELS_PATH + "/Topic_Model_Supervised/logreg_clf_topic_extract")
tf_idf_vector = joblib.load(MODELS_PATH + "/Topic_Model_Supervised/tfidf_vector_topic_extract")

def read_csv_data(csv_filename):

    data = pd.read_csv(csv_filename)['text']
    return data




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

def action(arr_0):

    if arr_0 == 0:
        return "Agents,Metrics,Team"

    elif arr_0 == 1:

        return "Client,Process,Agent"

    elif arr_0 == 2:
        return "Company,Salary,Work_Home"

    elif arr_0 == 3:
        return "Customer,Employees,Tool"

    elif arr_0 == 4:
        return "Option,Customers,Help"

    elif arr_0 == 5:
        return "Policy,Service,Time"

    elif arr_0 == 6:
        return "Process,Reduce_cost"

    elif arr_0 == 7:
        return "Repeat_calls,Refund,Reduce_repeat"

    elif arr_0 == 8:
        return "Simplify,Escalation Happening"

    elif arr_0 == 9:
        return "Team,Agents,Real_time"

    elif arr_0 == 10:
        return "Team,Company,Culture,Employee"

    elif arr_0 == 11:
        return "Work,Week,Leave"


def detector_from_string(csv_filename):

    data = pd.DataFrame()
    tt_0 = read_csv_data(csv_filename)
    #mapping_classes = {0: 'Hateful language', 1: "Profane language", 2: "Non Profane Language"}

    tt_0 = tt_0.astype('str')
    tt0_cleaned = list(map(lambda x: text_cleanup(x), tt_0))
    tfidf_testdata = tf_idf_vector.transform(list(tt0_cleaned))
    arr = clf.predict(tfidf_testdata)

    arr_probs = clf.predict_proba(tfidf_testdata).tolist()

    data['Input_Text'] = tt_0
    data['Topic_Predicted'] = pd.Series(arr).tolist()
    data['Topic_Predicted'] = data['Topic_Predicted'].apply(lambda x: action(x))
    # data['Probability'] = np.max(arr_probs, axis=1).tolist()
    # data["Action"] = data['Twitter_Sentiment_Result'].apply(lambda x: action(x))

    excel_filename = "Supervised_Topics_Extraction_" +  ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath

    #print(tt_0)



