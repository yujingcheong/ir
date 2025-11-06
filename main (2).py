# ==============================================================
# KD34103 Text Mining Project
# Task 1: Game Type Classification (based on title and review)
# ==============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
import re
import nltk

# --------------------------------------------------------------
# 1. DOWNLOAD REQUIRED NLTK DATA
# --------------------------------------------------------------
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

# --------------------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------------------
file_path = "Video_Games.txt"  # <-- change to your actual file name

with open(file_path, "r", encoding="utf-8") as f:
    data = f.read()

# Split into individual reviews
reviews = data.split("product/productId:")

titles = []
texts = []
for r in reviews[1:]:
    title_match = re.search(r"product/title:\s*(.*)", r)
    text_match = re.search(r"review/text:\s*(.*)", r, re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    text = text_match.group(1).strip() if text_match else ""
    titles.append(title)
    texts.append(text)

print("Total reviews loaded:", len(texts))

# --------------------------------------------------------------
# 3. PREPROCESS TEXT
# --------------------------------------------------------------
def preprocess(texts):
    for n in range(0, len(texts)):
        text = texts[n]
        text = word_tokenize(text)
        text = [word.lower() for word in text]
        text = [word for word in text if word not in stop_words]
        text = [word for word in text if word.isalpha()]
        text = [porter.stem(word) for word in text]
        texts[n] = " ".join(text)
    return texts

texts = preprocess(texts)

# --------------------------------------------------------------
# 4. ASSIGN GAME TYPE BASED ON TITLE (MANUAL CATEGORIES)
# --------------------------------------------------------------
def get_game_type(title):
    title = title.lower()
    if any(w in title for w in ["soccer", "football", "fifa", "nba", "golf", "baseball"]):
        return 0  # sports
    elif any(w in title for w in ["race", "racing", "speed", "kart", "drive", "car"]):
        return 1  # racing
    elif any(w in title for w in ["shoot", "battle", "war", "sniper", "duty", "combat"]):
        return 2  # shooter
    elif any(w in title for w in ["puzzle", "logic", "brain"]):
        return 3  # puzzle
    elif any(w in title for w in ["adventure", "quest", "hero", "fantasy"]):
        return 4  # adventure
    elif any(w in title for w in ["kids", "learning", "educational", "fisher", "price"]):
        return 5  # kids
    else:
        return 6  # other

Y = [get_game_type(t) for t in titles]
labels = ["sports", "racing", "shooter", "puzzle", "adventure", "kids", "other"]

# --------------------------------------------------------------
# 5. FEATURE EXTRACTION
# --------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)

# Split manually (60% training, 40% testing)
n = round(X.shape[0] * 0.6)
trainX = X[0:n]
trainY = Y[0:n]
testX = X[n:]
testY = Y[n:]

# Select top 1000 features using chi-square
ch2 = SelectKBest(chi2, k=1000)
trainX = ch2.fit_transform(trainX, trainY)
testX = ch2.transform(testX)

# --------------------------------------------------------------
# 6. CLASSIFICATION
# --------------------------------------------------------------
clf = MultinomialNB()
clf.fit(trainX, trainY)
predY = clf.predict(testX)

# --------------------------------------------------------------
# 7. RESULTS
# --------------------------------------------------------------
print("Accuracy: %0.3f" % metrics.accuracy_score(testY, predY))
print(metrics.confusion_matrix(testY, predY))
print(metrics.classification_report(testY, predY, target_names=labels))
