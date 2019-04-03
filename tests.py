import unittest

import stanfordnlp

from utils import contains_adjective_noun_pair


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


if __name__ == "__main__":
    unittest.main()
