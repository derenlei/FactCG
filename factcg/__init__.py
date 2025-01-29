from .grounding_score import GroundingScore
import nltk

try:
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')