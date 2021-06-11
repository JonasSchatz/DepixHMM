import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DemoFontPaths:
    arial: Path = Path(__file__).parent / 'arial.ttf'
    micr: Path = Path(__file__).parent / 'micrenc.ttf'
