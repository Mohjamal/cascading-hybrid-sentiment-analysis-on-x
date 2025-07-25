from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader_analyzer = SentimentIntensityAnalyzer()

def get_vader_compound_score(text):
    """
    Applies VADER to get the compound sentiment score for a given text.
    The compound score is a normalized, weighted composite score which
    ranges from -1 (most extreme negative) to +1 (most extreme positive).
    """
    vs = vader_analyzer.polarity_scores(text)
    return vs['compound']

def get_vader_binary_prediction(compound_score, pos_thresh=0.05, neg_thresh=-0.05):
    """
    Converts VADER compound scores to a binary (0/1) prediction.
    1 for positive, 0 for negative. Neutral scores are typically mapped to 0 for binary.
    """
    if compound_score >= pos_thresh:
        return 1 # Positive
    elif compound_score <= neg_thresh:
        return 0 # Negative
    else:
        return 0 # Defaulting neutral to negative for binary classification comparison

