from typing import List
from unittest import TestCase

from text_depixelizer.training_pipeline.text_generator import RegexTextGenerator, NumberTextGenerator


class TestRegexTextGenerator(TestCase):
    def test_regex_text_generator_digits(self):
        # Arrange
        pattern: str = r'\d{1,5}'
        text_generator: RegexTextGenerator = RegexTextGenerator(pattern=pattern)

        # Act
        random_texts: List[str] = [text_generator.generate_text() for _ in range(100)]

        # Assert
        for random_text in random_texts:
            self.assertRegex(random_text, pattern)


class TestNumberTextGenerator(TestCase):
    def test_number_text_generator(self):
        # Arrange
        text_length: int = 5
        text_generator: NumberTextGenerator = NumberTextGenerator(text_length=text_length)

        # Act
        random_text: str = text_generator.generate_text()

        # Assert
        self.assertEqual(len(random_text), text_length)
