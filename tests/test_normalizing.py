import pytest

import spacy
import nltk

from src.normalizing import SimpleTokenizer, SpaCyTokenizer, NLTKTokenizer

@pytest.fixture()
def text():
    # Our reference test text is the abstract for the paper "Attention is all you need"
    #
    # Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information
    # processing systems (p./pp. 5998--6008) [arXiv:1706.03762v7 [cs.CL] 2 Aug 2023]
    
    t = "The dominant sequence transduction models are based on complex recurrent or\n" \
        "convolutional neural networks that include an encoder and a decoder. The best\n " \
        "performing models also connect the encoder and decoder through an attention\n " \
        "mechanism. We propose a new simple network architecture, the Transformer,\n" \
        "based solely on attention mechanisms, dispensing with recurrence and convolutions\n" \
        "entirely. Experiments on two machine translation tasks show these models to\n" \
        "be superior in quality while being more parallelizable and requiring significantly\n" \
        "less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-" \
        "to-German translation task, improving over the existing best results, including\n" \
        "ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task,\n" \
        "our model establishes a new single-model state-of-the-art BLEU score of 41.8 after\n" \
        "training for 3.5 days on eight GPUs, a small fraction of the training costs of the\n" \
        "best models from the literature. We show that the Transformer generalizes well to\n" \
        "other tasks by applying it successfully to English constituency parsing both with\n" \
        "large and limited training data."
    return t

class TestSimpleTokenizer:
    @pytest.fixture(scope="class")
    def tker(self):
        simpt = SimpleTokenizer()
        return simpt    

    def test_model_type(self, tker):
        # Ensure that the tokenizer model is None
        assert tker.model == None

    def test_non_string_input(self, tker):
        # Ensure that the tokenizer can handle non-string input
        tokens = tker.tokenize(123)  # 123 is not a string
        assert tokens == []

    def test_n_tokens(self, tker, text):
        # Ensure that the tokenizer returns the expected number of tokens
        tokens = tker.tokenize(text)
        
        assert len(tokens) == len(text.split())

class TestSpaCyTokenizer:
    @pytest.fixture(scope="class")
    def tker(self):
        spacyt = SpaCyTokenizer()
        return spacyt    

    def test_model_type(self, tker):
        # Ensure that the tokenizer model is correct
        assert isinstance(tker._model, spacy.lang.en.English)

    def test_non_string_input(self, tker):
        # Ensure that the tokenizer can handle non-string input
        tokens = tker.tokenize(123)  # 123 is not a string
        assert tokens == []

    def test_model_pipe(self, tker):
        # Ensure that the model pipeline is the standard 
        assert tker.pipeline() == ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

    def test_lemmatize(self, tker):
        # Ensure that the lemmatizer works properly
        text = "cats running quickly"
        lemmatized_tokens = tker.lemmatize(text)
        assert lemmatized_tokens == ["cat", "run", "quickly"]

class TestNLTKTokenizer:
    @pytest.fixture(scope="class")
    def tker(self):
        nltkt = NLTKTokenizer()
        return nltkt

    def test_model_type(self, tker):
        assert tker._model == nltk.tokenize.word_tokenize

    def test_non_string_input(self, tker):
        # Ensure that the tokenizer can handle non-string input
        tokens = tker.tokenize(123)  # 123 is not a string
        assert tokens == []