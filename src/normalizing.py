from abc import ABC, abstractmethod
from typing import Optional

from src.utils import WordTools

import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers. Subclasses must implement the 'model' property
    and the 'tokenize' method to provide specific tokenization functionality.

    Attributes:
    - None

    Methods:
    - model (property):
        This property should return the specific tokenizer model or configuration used by the subclass.
        Subclasses must implement this property.

    - tokenize(text: str) -> None:
        Tokenizes the input text according to the tokenizer's model or configuration.
        Subclasses must implement this method.

    """
    @property
    @abstractmethod
    def model(self):
        """
        Property that should return the specific tokenizer model or configuration used by the subclass.
        Subclasses must implement this property.
        """
        raise NotImplementedError("Subclasses must implement the 'model' property.")

    @abstractmethod
    def tokenize(self, text:str) -> None:
        """
        Tokenizes the input text according to the tokenizer's model or configuration.
        Subclasses must implement this method.
        
        :param text: The input text to tokenize.
        """
        raise NotImplementedError("Subclasses must implement the 'tokenize' method.")


class SimpleTokenizer(Tokenizer):
    """
    A concrete implementation of the Tokenizer abstract class that provides a simple
    tokenization method based on splitting text by whitespace. It also replaces newline
    escape characters with spaces and gets rid of commas and dots, and removes empty tokens.

    Attributes:
    - None

    Methods:
    - model (property):
        This property returns the specific tokenizer model or configuration used by the class.

    - tokenize(text: str) -> list:
        Tokenizes the input text by splitting it based on whitespace.

    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        """
        Property that returns the specific tokenizer model or configuration used by the class.
        In this case, it is always set to None since SimpleTokenizer doesn't have a complex model.
        """
        return self._model

    def tokenize(self, text:str):
        """
        Tokenizes the input text by splitting it based on whitespace. It also replaces newline
        escape characters with spaces and gets rid of commas and dots, and removes empty tokens.
        
        :param text: The input text to tokenize.
        :return: A list of tokens obtained by making some transformations and splitting the
        input text.
        """
        if isinstance(text, str):
            output = text.replace('\n', ' ')
            output = output.replace('.', '')
            output = output.replace(',', '')
            tokens = output.split()
            tokens = [string for string in tokens if string != '']
            return tokens
        else:
            return []


class SpaCyTokenizer(Tokenizer):
    """
    A concrete implementation of the Tokenizer abstract class that utilizes the SpaCy library for tokenization
    and lemmatization. The size of the SpaCy model can be specified during initialization.

    Attributes:
    - None

    Methods:
    - __init__(size: str = 'sm'):
        Initializes the SpaCyTokenizer with the specified model size.
        :param size: The size of the SpaCy model ('sm' for small (17Mb), 'md' for medium (45Mb), or 'lg' for large (780Mb)).

    - model (property):
        This property returns the specific SpaCy model used by the class.

    - tokenize(text: str) -> list[str]:
        Tokenizes the input text using the SpaCy model.
        :param text: The input text to tokenize.
        :return: A list of tokens obtained from the input text.

    - lemmatize(text: str) -> list[str]:
        Lemmatizes the input text using the SpaCy model.
        :param text: The input text to lemmatize.
        :return: A list of lemmatized tokens obtained from the input text.

    - pipeline() -> list[str]:
        Returns the names of the processing pipeline components used in the SpaCy model.

    """

    def __init__(self, size: str='sm'):
        """
        Initializes the SpaCyTokenizer with the specified model size.
        If the specified model is not installed, it will be downloaded.
        :param size: The size of the SpaCy model ('sm' for small, 'md' for medium, or 'lg' for large).
        """
        if not spacy.util.is_package(f'en_core_web_{size}'):
            spacy.cli.download(f'en_core_web_{size}')

        self._model = spacy.load(f'en_core_web_{size}')

    @property
    def model(self):
        """
        Property that returns the specific SpaCy model used by the class.
        """
        return self._model.__class__

    def tokenize(self, text:str) -> list[str]:
        """
        Tokenizes the input text using the SpaCy model.
        :param text: The input text to tokenize.
        :return: A list of tokens obtained from the input text.
        """
        if isinstance(text, str):
            string = text.replace('\n', ' ')
            doc = self._model(string)
            return [token.text for token in doc if ' ' not in token.text]
        else:
            return []
        
    def lemmatize(self, text:str) -> list[str]:
        """
        Lemmatizes the input text using the SpaCy model.
        :param text: The input text to lemmatize.
        :return: A list of lemmatized tokens obtained from the input text.
        """
        if isinstance(text, str):
            string = text.replace('\n', ' ')
            doc = self._model(string)
            return [token.lemma_ for token in doc if ' ' not in token.lemma_]
        
    def pipeline(self) -> list[str]:
        """
        Returns the names of the processing pipeline components used in the SpaCy model.
        """
        return self._model.pipe_names


class NLTKTokenizer(Tokenizer):
    """
    A concrete implementation of the Tokenizer abstract class that utilizes the NLTK library for tokenization.

    Attributes:
    - None

    Methods:
    - __init__():
        Initializes the NLTKTokenizer and ensures the NLTK Punkt tokenizer model is available.

    - model (property):
        This property returns the specific NLTK tokenizer model used by the class.

    - tokenize(text: str) -> list[str]:
        Tokenizes the input text using the NLTK tokenizer.
        :param text: The input text to tokenize.
        :return: A list of tokens obtained from the input text.

    """

    def __init__(self):
        """
        Initializes the NLTKTokenizer and ensures the NLTK Punkt tokenizer model is available.
        If the model is not found, it will be downloaded.
        """
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        self._model = word_tokenize

    @property
    def model(self):
        """
        Property that returns the specific NLTK tokenizer model used by the class.
        """
        return self._model.__name__

    def tokenize(self, text:str) -> list[str]:
        """
        Tokenizes the input text using the NLTK tokenizer.
        :param text: The input text to tokenize.
        :return: A list of tokens obtained from the input text.
        """
        if isinstance(text, str):
            string = text.replace('\n', ' ')
            return self._model(string)
        else:
            return []

def normalize(text: str, tkn: Tokenizer, lemma: bool = False, stop_words: bool = True,
              punct_signs: bool = False, stemmer: Optional[str] = None, case_folding: bool = True):
    """
    Normalize a given text using specified tokenization and text processing techniques.
    It also creates a dictionary with the options used for creating the tokens

    Parameters:
    - text (str): The input text to be normalized.
    - tkn (Tokenizer): An instance of a tokenizer to be applied to the text.
    - lemma (bool, optional): If True and the tokenizer is a SpaCyTokenizer, apply lemmatization. Default is False.
    - stop_words (bool, optional): If True, remove common stop words from the tokenized text. Default is True.
    - punct_signs (bool, optional): If True, remove punctuation signs from the tokenized text. Default is False.
    - stemmer (str, optional): Specify the stemming algorithm to use ("Porter" or "Snowball"). Default is None.
    - case_folding (bool, optional): If True, perform case folding (convert to lowercase). Default is True.

    Returns:
    - dict: A dictionary with the normalization pipeline parameters
    - list[str]: A list of normalized tokens based on the specified processing techniques.

    """
    args = dict(locals())
    parameters = {}
    parameters['tokenizer'] = args['tkn'].__class__.__name__
    args.pop('text', None)
    args.pop('tkn', None)
    parameters['args'] = args
    
    # Apply the tokenizer to the text
    if lemma & isinstance(tkn, SpaCyTokenizer):
        tokens = tkn.lemmatize(text)
    else:
        tokens = tkn.tokenize(text)

    # Stemming
    if stemmer == "Porter":
        tokens = tokens = WordTools.porter_stemmer(tokens)
    elif stemmer == "Snowball":
        tokens = WordTools.snowball_stemmer(tokens)

    # Remove the stop words
    if stop_words:
        tokens = WordTools.stopword_filter(tokens)

    # Case folding
    if case_folding:
        tokens = WordTools.case_folding(tokens)

    if punct_signs:
        tokens = WordTools.punct_remover(tokens)

    return parameters, tokens

def df_pipeline(df: pd.DataFrame, tkn: Tokenizer, **kwargs) -> set[dict, pd.DataFrame]:
    """
    Process a DataFrame by applying a tokenization pipeline to a specified column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - **kwargs: Additional keyword arguments to be passed to the 'normalize' function.

    Returns:
    pd.DataFrame: A modified DataFrame with an additional 'tokens' column containing the processed tokens.

    Example:
    >>> input_df = pd.DataFrame({'overview': ['Text 1', 'Text 2', 'Text 3']})
    >>> processed_df = df_pipeline(input_df, param1=value1, param2=value2)
    >>> print(processed_df)
            overview    tokens
    0       Text 1      processed_tokens_1
    1       Text 2      processed_tokens_2
    2       Text 3      processed_tokens_3
    """
    df.loc[:, 'tokens'] = df.descriptor.map(lambda x: normalize(x, tkn, **kwargs))
    params = df.tokens[0][0]
    df.loc[:, 'tokens'] = df.tokens.map(lambda x: x[1])

    return params, df

if __name__=="__main__":
    pass