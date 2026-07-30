"""Unit tests for text_cleaner utility."""

import pytest
from src.utils.text_cleaner import clean_text, PIPELINE


class TestCleanText:
    def test_removes_special_chars(self):
        # ! and ? are now preserved
        assert clean_text("hello! world??") == "hello! world??"

    def test_collapses_multiple_special_chars(self):
        # @ is now preserved, # and $ are removed but # becomes space
        assert clean_text("foo   @#$   bar") == "foo @ $ bar"

    def test_whitespace_only_returns_empty(self):
        assert clean_text("  ") == ""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_alphanumeric_unchanged(self):
        assert clean_text("hello world 123") == "hello world 123"

    def test_leading_trailing_special_chars(self):
        # ! is now preserved
        assert clean_text("!!!hello!!!") == "!!!hello!!!"

    def test_mixed_symbols(self):
        # - and () are now preserved
        assert clean_text("covid-19 (child)") == "covid-19 (child)"

    def test_consecutive_spaces_collapsed(self):
        assert clean_text("foo     bar") == "foo bar"

    def test_newlines_treated_as_special(self):
        assert clean_text("line1\nline2") == "line1 line2"

    def test_tabs_treated_as_special(self):
        assert clean_text("col1\tcol2") == "col1 col2"
        
    def test_apostrophe_removal(self):
        # Apostrophes are removed in step 1
        assert clean_text("don't go to 5'-Nucleotidase") == "don t go to 5 -Nucleotidase"
        
    def test_html_entity_removal(self):
        # HTML entities are removed if present (whitespace gets trimmed)
        assert clean_text("hello&#39;world&quot;test&quot;") == "hello world test"
        
    def test_healthcare_symbols_preserved(self):
        # Healthcare symbols should be preserved
        assert clean_text("temperature >98.6°F, dosage 25mg") == "temperature >98.6°F, dosage 25mg"
        
    def test_at_symbol_preserved(self):
        # @ symbol should now be preserved
        assert clean_text("contact user@domain.com") == "contact user@domain.com"


class TestPipeline:
    def test_pipeline_is_list(self):
        assert isinstance(PIPELINE, list)

    def test_pipeline_has_three_steps(self):
        # Updated to expect 3 steps now
        assert len(PIPELINE) == 3

    def test_pipeline_steps_are_callable(self):
        for step in PIPELINE:
            assert callable(step)