"""Python unittests"""
import unittest

import stanfordnlp

from utils import (
    contains_adjective_noun_pair,
    contains_verb_noun_pair,
    get_objects_for_noun,
    get_objects_for_verb,
)


class UtilsTests(unittest.TestCase):

    nlp_pipeline = stanfordnlp.Pipeline()

    def test_contains_adjective_noun_pair(self):
        caption = "a white car is driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_plural(self):
        caption = "two white cars are driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_conjunction(self):
        caption = "a white and blue car is driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_hyphen(self):
        caption = "a white-blue car is driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_hyphen_2(self):
        caption = "a blue-white car is driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_hyphen_on_noun(self):
        caption = "a white compact-car is driving down the street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_nominal_modifier(self):
        caption = "the car that is driving down the street is white"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_wrong_noun(self):
        caption = "a blue car is driving down the white street"
        nouns = {"car"}
        adjectives = {"white"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, False)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_wrong_noun_2(self):
        caption = "person inside display area with a young elephant"
        nouns = {"person"}
        adjectives = {"young"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, False)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_adjective_noun_pair_wrong_noun_3(self):
        caption = (
            "a gray shaggy dog hanging out the driver side window of a blue minivan."
        )
        nouns = {"window"}
        adjectives = {"blue"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, False)
        self.assertEqual(
            expected_pattern,
            contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives),
        )

    def test_contains_verb_noun_pair(self):
        caption = "a man sits on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern, contains_verb_noun_pair(pos_tagged_caption, nouns, verbs)
        )

    def test_contains_verb_noun_pair_2(self):
        caption = "a man that is sitting on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern, contains_verb_noun_pair(pos_tagged_caption, nouns, verbs)
        )

    def test_contains_verb_noun_pair_3(self):
        caption = "a man that sits on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern, contains_verb_noun_pair(pos_tagged_caption, nouns, verbs)
        )

    def test_contains_verb_noun_pair_4(self):
        caption = "a man sitting on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern, contains_verb_noun_pair(pos_tagged_caption, nouns, verbs)
        )

    def test_contains_verb_noun_pair_5(self):
        caption = "a man is sitting on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        expected_pattern = (True, True, True)
        self.assertEqual(
            expected_pattern, contains_verb_noun_pair(pos_tagged_caption, nouns, verbs)
        )

    def test_get_objects(self):
        caption = "a man is sitting on a chair."
        nouns = {"man"}
        verbs = {"sit"}

        pos_tagged_caption = self.nlp_pipeline(caption).sentences[0]

        objects = get_objects_for_noun(
            pos_tagged_caption, nouns
        ) | get_objects_for_verb(pos_tagged_caption, verbs)
        self.assertEqual({"chair"}, objects)


if __name__ == "__main__":
    unittest.main()
