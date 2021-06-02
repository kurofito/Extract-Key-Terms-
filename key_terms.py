import string
import nltk
from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Extract:
    def __init__(self):
        self.root = None
        self.head = []
        self.words = []
        self.puncts = None
        self.stopwords = None

    def read_xml(self, xml):
        tree = etree.parse(xml)
        root = tree.getroot()
        self.root = root[0]
        self.process()

    def extract_head(self):
        for title in self.root:
            self.head.append(title[0].text + ':')

    def extract_words(self):
        processed_doc = []
        for text in self.root:
            data = text[1].text
            key_terms = self.key_terms(data)
            processed_doc.append(' '.join(key_terms))

        self.vectorize(processed_doc)

    def clean(self, data):
        strings = []
        for key, value in data.items():
            strings.append(key)
        self.words.append(' '.join(strings))
        strings = []

    def vectorize(self, data):
        for i in range(len(data)):
            tfidf_vectorizer = TfidfVectorizer(use_idf=True)
            tfidf_vectorizer_vector = tfidf_vectorizer.fit_transform(data)
            terms = tfidf_vectorizer.get_feature_names()
            vector = tfidf_vectorizer_vector[i]

            sorted_items = self.sort_coo(vector.tocoo())
            key_words = self.extract_top5(terms, sorted_items)
            self.clean(key_words)

    def sort_coo(self, data):
        tuples = zip(data.col, data.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_top5(self, names, items, topn=5):

        # use only topn items from vector
        items_sorted = items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in items_sorted:
            score_vals.append(round(score, 3))
            feature_vals.append(names[idx])

        result = {}
        for idx in range(len(feature_vals)):
            result[feature_vals[idx]] = score_vals[idx]

        return result

    def key_terms(self, text):
        text = word_tokenize(text.lower())
        text = self.lemmatize(text)
        text = self.stop_words(text)
        text = self.punctuation(text)
        text = self.nouns(text)
        return text

    def lemmatize(self, data):
        lemma = WordNetLemmatizer()
        lemmatized_sentence = []
        for word in data:
            lemmatized_sentence.append(lemma.lemmatize(word))

        return lemmatized_sentence

    def nouns(self, data):
        noun_list = []
        for word in data:
            word_tagged = nltk.pos_tag([word])
            if word_tagged[0][1] == 'NN':
                noun_list.append(word)
        return noun_list

    def punctuation(self, data):
        punct_dict = self.puncts
        for p in data:
            if p in punct_dict:
                data.remove(p)
        return data

    def punct_generate(self):
        punct_dict = dict()
        for index, word in enumerate(list(string.punctuation)):
            punct_dict[word] = index
        self.puncts = punct_dict

    def stop_words(self, datum):
        w = []
        for word in datum:
            if word not in self.stopwords:
                w.append(word)
        return w

    def generate(self):
        words = dict()
        for index, word in enumerate(stopwords.words('english')):
            words[word] = index
        self.stopwords = words

    def process(self):
        self.punct_generate()
        self.generate()
        self.extract_head()
        self.extract_words()
        self.display()

    def display(self):
        for i in range(len(self.head)):
            print(f'{self.head[i]}\n{self.words[i]}\n')


xml_path = 'news.xml'
extract = Extract()
extract.read_xml(xml_path)

























