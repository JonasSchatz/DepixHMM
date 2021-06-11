from pathlib import Path
import unittest
from typing import List

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.parameters import PictureParameters
from text_depixelizer.training_pipeline.training_pipeline import create_training_data


class GenerateSampleImages(unittest.TestCase):

    def test_generate_sample_images(self):

        # Editable parameters
        font_size: int = 50
        block_sizes: List[int] = [9] # [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        font_path: str = str(DemoFontPaths.arial)
        text: str = 'H  E  L  L  O'

        # Act
        folder_name = f'{Path(font_path).stem}_{font_size}'
        output_path = Path(__file__).parent.parent / 'resources' / 'images' / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        for block_size in block_sizes:
            picture_parameters: PictureParameters = PictureParameters(
                pattern=rf'{text}',
                block_size=block_size,
                randomize_pixelization_origin_x=False,
                font=ImageFont.truetype(font_path, font_size)
            )

            _, _, pixelized_images, _ = create_training_data(n_img=1, picture_parameters=picture_parameters)
            pixelized_images[0].image.save(output_path / f'{text}_blocksize-{block_size}.png')
