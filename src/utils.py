import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer


class WordTools:
    """
    WordTools is a utility class for text preprocessing tasks such as stopword filtering, 
    case folding (lowercasing), and stemming using Porter or Snowball stemmers.

    It provides methods to perform these common text processing operations on a list of tokens.

    Usage:
    word_tools = WordTools()
    filtered_tokens = word_tools.stopword_filter(tokens)
    lowercased_tokens = word_tools.case_folding(tokens)
    stemmed_tokens = word_tools.porter_stemmer(tokens)

    Attributes:
    - None

    Methods:
    - stopword_filter(tokens: list[str]) -> list[str]:
        Removes common English stopwords from a list of tokens.
        :param tokens: A list of input tokens.
        :return: A list of tokens with stopwords removed.

    - case_folding(tokens: list[str]) -> list[str]:
        Converts all tokens to lowercase (case folding).
        :param tokens: A list of input tokens.
        :return: A list of lowercase tokens.

    - porter_stemmer(tokens: list[str]) -> list[str]:
        Applies the Porter stemming algorithm to reduce tokens to their root form.
        :param tokens: A list of input tokens.
        :return: A list of stemmed tokens.

    - snowball_stemmer(tokens: list[str]) -> list[str]:
        Applies the Snowball stemming algorithm for English to reduce tokens to their root form.
        :param tokens: A list of input tokens.
        :return: A list of stemmed tokens.

    """
    @staticmethod
    def stopword_filter(tokens:list[str]) -> list[str]:
        """
        Removes common English stopwords from a list of tokens.
        :param tokens: A list of input tokens.
        :return: A list of tokens with stopwords removed.
        """
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in tokens if token not in stop_words]

    @staticmethod
    def case_folding(tokens: list[str]) -> list[str]:
        """
        Converts all tokens to lowercase (case folding).
        :param tokens: A list of input tokens.
        :return: A list of lowercase tokens.
        """
        return [token.lower() for token in tokens]

    @staticmethod
    def punct_remover(tokens: list[str]) -> list[str]:
        """
        Removes some punctuation signs from the token list: period, comma,
        quotation, question, exclamation, parenthesis, colon, semicolon
        :param tokens: A list of input tokens.
        :return: A filtered list of tokens.
        """
        punct_signs = ['.', ',', '"', "'", "''", '?', '!', '(', ')', ';', ':']
        return [token for token in tokens if token not in punct_signs]
    
    @staticmethod
    def porter_stemmer(tokens: list[str]) -> list[str]:
        """
        Applies the Porter stemming algorithm to reduce tokens to their root form.
        :param tokens: A list of input tokens.
        :return: A list of stemmed tokens.
        """
        stemmer = PorterStemmer()
        return [stemmer.stem(token).strip("'") for token in tokens]
    
    @staticmethod
    def snowball_stemmer(tokens: list[str]) -> list[str]:
        """
        Applies the Snowball stemming algorithm for English to reduce tokens to their root form.
        :param tokens: A list of input tokens.
        :return: A list of stemmed tokens.
        """
        stemmer = SnowballStemmer(language='english')
        return [stemmer.stem(token).strip("'") for token in tokens]
    
if __name__=="__main__":
    pass




