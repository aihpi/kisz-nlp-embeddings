import pytest

from src.utils import WordTools


class TestWordTools:
    @pytest.fixture(scope="class")
    def wtools(self):
        wt = WordTools
        return wt

    def test_stopword_filter(self, wtools):
        input_tokens = ["this", "is", "a", "test", "sentence"]
        filtered_tokens = wtools.stopword_filter(input_tokens)
        assert filtered_tokens == ["test", "sentence"]

    def test_case_folding(self, wtools):
        input_tokens = ["This", "Is", "A", "Test", "Sentence"]
        folded_tokens = wtools.case_folding(input_tokens)
        assert folded_tokens == ["this", "is", "a", "test", "sentence"]

    def test_porter_stemmer(self, wtools):
        word_tools = WordTools()
        input_tokens = ["running", "flies", "happily"]
        stemmed_tokens = wtools.porter_stemmer(input_tokens)
        assert stemmed_tokens == ["run", "fli", "happili"]

    def test_snowball_stemmer(self, wtools):
        input_tokens = ["running", "flies", "happily"]
        stemmed_tokens = wtools.snowball_stemmer(input_tokens)
        assert stemmed_tokens == ["run", "fli", "happili"]

if __name__ == '__main__':
    pass



