from .factcg_score import FactCGScore
import nltk

try:
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
