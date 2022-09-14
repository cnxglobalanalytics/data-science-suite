import spacy
import subprocess
from string import punctuation

def read_from_textfiles(datafile):
    with open(datafile, 'r') as f:
        text = f.read()
    return text

def extract_keywords(nlp, sequence, special_tags: list = None):

    result = []

    # custom list of part of speech tags we are interested in
    # we are interested in proper nouns, nouns, and adjectives
    # edit this list of POS tags according to your needs.
    pos_tag = ['PROPN', 'NOUN', 'ADJ']

    # create a spacy doc object by calling the nlp object on the input sequence
    doc = nlp(sequence.lower())

    # if special tags are given and exist in the input sequence
    # add them to results by default
    if special_tags:
        tags = [tag.lower() for tag in special_tags]
        for token in doc:
            if token.text in tags:
                result.append(token.text)

    for chunk in doc.noun_chunks:
        final_chunk = ""
        for token in chunk:
            if (token.pos_ in pos_tag):
                final_chunk = final_chunk + token.text + " "
        if final_chunk:
            result.append(final_chunk.strip())

    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return list(set(result))

subprocess.call("python -m spacy download en_core_web_sm",shell=True)

  # load the small english language model,
nlp = spacy.load("en_core_web_sm")

def theme_extraction(data_text):

    d = {}

    d['language'] = "English"

    data = read_from_textfiles(data_text)
    d['length'] = len(data)
    d['unique_words'] = len(set(data.split(" ")))

    d['extracted_themes'] = ','.join(extract_keywords(nlp,data))

    return d