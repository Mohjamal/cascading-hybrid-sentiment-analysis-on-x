import re
import html

# Basic emoticon lexicon
emoticon_lexicon = {
    ':)': 'happy', ':-)': 'happy', ';)': 'wink', ':-D': 'laugh', 'XD': 'laugh',
    ':(': 'sad', ':-(': 'sad', ';(': 'cry', ':|': 'neutral', ':-|': 'neutral',
    '<3': 'love', ':-P': 'tongue', ':P': 'tongue'
}

def preprocess_tweet(text):
    # 1. Lowercasing
    text = text.lower()

    # 2. URL Removal
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. User Mention Removal
    text = re.sub(r'@\w+', '', text)

    # 4. Hashtag Processing (remove #, keep text)
    text = re.sub(r'#', '', text)

    # 5. HTML Entity Removal
    text = html.unescape(text) # Decodes HTML entities like &amp; to &

    # 6. Emoticon Replacement (using a simple lexicon)
    for emo, word in emoticon_lexicon.items():
        text = text.replace(emo, word)

    # 7. Punctuation Removal (keep letters and spaces)
    # This regex keeps alphanumeric characters and spaces.
    # if you want to keep specific punctuation for sentiment (e.g., multiple '!!!')
    # but for simplicity, we remove most as per the methodology.
    text = re.sub(r'[^\w\s]', '', text)

    # 8. Removing Digits
    text = re.sub(r'\d+', '', text)

    # 9. Whitespace Normalization (remove extra spaces and trim)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

