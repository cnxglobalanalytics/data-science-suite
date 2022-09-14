import spacy
import subprocess
#from string import punctuation

def read_from_textfiles(datafile):
    with open(datafile, 'r') as f:
        text = f.read()
    return text


subprocess.call("python -m spacy download en_core_web_sm",shell=True)

  # load the small english language model,
nlp = spacy.load("en_core_web_sm")

def ner_extraction(data_text):

    d = {}
    data = read_from_textfiles(data_text)
    doc = nlp(data)
    ls = []
    for entity in doc.ents:
        ls.append(entity.label_ + ' | ' + entity.text)
    d['language'] = "English"
    d['length'] = len(data)
    d['unique_words'] = len(set(data.split(" ")))
    d['ent'] =  ' , '.join(ls)

    return d

