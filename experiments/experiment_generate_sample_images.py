from pathlib import Path
import unittest
from typing import List, Tuple

from PIL import ImageFont

from resources.fonts import DemoFontPaths
from text_depixelizer.parameters import PictureParameters
from text_depixelizer.training_pipeline.training_pipeline import create_training_data


class GenerateSampleImages(unittest.TestCase):

    def test_generate_sample_images(self):

        # Editable parameters
        font_size: int = 30
        block_sizes: List[int] = [10] # [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        offset_ys: List[int] = list(range(8))
        font_path: str = str(DemoFontPaths.arial)
        text: str = 'v Abjkly 123Bac'
        font_color: Tuple[int, int, int] = (255, 255, 255)
        background_color: Tuple[int, int, int] = (39, 48, 70)

        # Act
        folder_name = f'{Path(font_path).stem}_{font_size}'
        output_path = Path(__file__).parent.parent / 'resources' / 'images' / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        for block_size in block_sizes:
            for offset_y in offset_ys:
                picture_parameters: PictureParameters = PictureParameters(
                    pattern=rf'{text}',
                    block_size=block_size,
                    randomize_pixelization_origin_x=False,
                    font=ImageFont.truetype(font_path, font_size),
                    offset_y=offset_y,
                    font_color=font_color,
                    background_color=background_color
                )

                _, _, pixelized_images, _ = create_training_data(n_img=1, picture_parameters=picture_parameters)
                pixelized_images[0].image.save(output_path / f'{text}_blocksize-{block_size}_offset_y-{offset_y}.png')
