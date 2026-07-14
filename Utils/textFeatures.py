"""
Creates a 12 dimensional vector that includes the following:
[0] VADER neg
[1] VADER neu
[2] VADER pos
[3] VADER compound
[4] character length (log1p-scaled)
[5] word count (log1p-scaled)
[6] average word length
[7] exclamation mark count (capped)
[8] question mark count (capped)
[9] capital-letter ratio
[10] type-token ratio (lexical diversity)
[11] punctuation density

Used for VADER sentiment and lexical + stylistic analysis
"""

#Required imports
import math
import re
import string

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_WORD_RE = re.compile(r"[A-Za-z']+")

def extractFeatures(text):
    if text is None or (isinstance(text, float) and math.isnan(text)):
        text = ""
    text = str(text)

    vs = _analyzer.polarity_scores(text)

    words = _WORD_RE.findall(text)
    word_count = len(words)
    char_len = len(text)
 
    #gets the average characters per word
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
    
    #gets the amount of exclamation and question signs, fake or sensational posts overuse these
    exclam = min(text.count("!"), 10)
    question = min(text.count("?"), 10)
    
    #gets all caps letters
    letters = [c for c in text if c.isalpha()]
    #gets the fraction of capital letters from the whole text
    capital_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
 
    lower_words = [w.lower() for w in words]
    #this is the diversity of the language and is calculated using unique words divided by the total
    #amount of words
    type_token_ratio = (len(set(lower_words)) / word_count) if word_count > 0 else 0.0
 
    punct_count = sum(1 for c in text if c in string.punctuation)
    #how many punctuation characters are inside the provided text
    punct_density = (punct_count / char_len) if char_len > 0 else 0.0
 
    features = [
        vs["neg"],
        vs["neu"],
        vs["pos"],
        vs["compound"],
        math.log1p(char_len),
        math.log1p(word_count),
        avg_word_len,
        float(exclam),
        float(question),
        capital_ratio,
        type_token_ratio,
        punct_density,
    ]
    return features