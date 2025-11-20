import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- FIX SECTION ---
# NLTK recently updated and requires 'punkt_tab' for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab') # <--- THIS WAS MISSING

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# -------------------

ps = PorterStemmer()

def transform_text(text):
    """
    1. Lowercase
    2. Tokenize (split into list)
    3. Remove special chars & punctuation
    4. Remove stopwords
    5. Stemming (running -> run)
    """
    # Handle non-string inputs (just in case)
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum(): # Keep only alpha-numeric
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i)) # Stemming
    
    return " ".join(y)