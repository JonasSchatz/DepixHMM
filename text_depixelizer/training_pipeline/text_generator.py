import random
import string
from abc import ABC, abstractmethod

import rstr


class TextGenerator(ABC):
    @abstractmethod
    def generate_text(self) -> str:
        pass


class RegexTextGenerator(TextGenerator):

    def __init__(self, pattern: str):
        self.pattern = pattern

    def generate_text(self) -> str:
        return rstr.xeger(self.pattern)


class NumberTextGenerator(TextGenerator):

    def __init__(self, text_length: int):
        self.text_length = text_length

    def generate_text(self) -> str:
        digits = string.digits
        return ''.join(random.choice(digits) for i in range(self.text_length))
