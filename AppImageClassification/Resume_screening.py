import pandas as pd
import re
import joblib
from ImageClassification.settings import BASE_DIR,MODELS_PATH

clf = joblib.load(MODELS_PATH + '/Resume_screening/resume_screening_clf')
word_vectorizer = joblib.load(MODELS_PATH + '/Resume_screening/resume_screening_word_vect')

def read_csv_data(csv_filename):


    if csv_filename.split(".")[-1] == 'csv':
        data = pd.read_csv(csv_filename)['text']
        return data
    elif csv_filename.split(".")[-1] == 'xlsx':
        data = pd.read_excel(csv_filename)['text']
        return data

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def screen_main(csv_file):

    tt_0 = read_csv_data(csv_file)
    data = pd.DataFrame()
    data['Resume'] = tt_0
    tt_0 = tt_0.astype('str')
    tt0_cleaned= tt_0.apply(lambda x: cleanResume(x))
    word_vector = word_vectorizer.transform(tt0_cleaned)
    prediction = clf.predict(word_vector)
    data['Predicted_category'] = prediction

    #arr_probs = clf.predict_proba(tfidf_testdata).tolist()

    excel_filename = "Resume_Screening_" + ".xlsx"
    data.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath