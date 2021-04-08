

import pytest
import spacy
from allennlp import predictors
import allennlp.models
from allennlp_rc import predictors as rc_predictors

from allennlp.common.testing import AllenNlpTestCase
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

def _load_predictor(archive_file: str, predictor_name: str) -> Predictor:
    """
    Helper to load the desired predictor from the given archive.
    """
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)

def open_information_extraction_stanovsky_2018() -> predictors.OpenIePredictor:
    predictor = _load_predictor(
        "https://allennlp.s3.amazonaws.com/models/openie-model.2020.02.10.tar.gz",
        "open-information-extraction",
    )
    return predictor

def test_openie():
        predictor = open_information_extraction_stanovsky_2018()
        result = predictor.predict_json({"sentence": "I'm against picketing, but I don't know how to show it."})
        assert "verbs" in result
        assert "words" in result
        return result

result = test_openie()

print(result)