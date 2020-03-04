import spacy
import pickle
from os import path

from sklearn.linear_model import LogisticRegression

DATA_FOLDER = "DATA_FOLDER"


nlp_sm = spacy.load(path.join(DATA_FOLDER, 'en_core_web_sm-2.0.0'))
nlp_md = spacy.load(path.join(DATA_FOLDER, 'en_core_web_md-2.0.0'))
nlp_lg = spacy.load(path.join(DATA_FOLDER, 'en_core_web_lg-2.0.0'))

logreg = pickle.load(open('./DATA_FOLDER/logreg.model', 'rb'))
rf = pickle.load(open('./DATA_FOLDER/rf-scikit-learn.model', 'rb'))