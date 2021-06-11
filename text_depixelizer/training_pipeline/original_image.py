from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont


@dataclass
class ImageCreationOptions:
    padding: Tuple[int, int]
    font: FreeTypeFont
    background_color: Tuple[int, int, int] = (255, 255, 255)
    font_color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class CharacterBoundingBox:
    char: str
    top: int
    bottom: int
    left: int
    right: int


@dataclass
class OriginalImage:
    text: str
    img: Image
    character_bounding_boxes: List[CharacterBoundingBox]
    image_creation_options: ImageCreationOptions

    @property
    def text_size(self) -> Tuple[int, int]:
        return self.image_creation_options.font.getsize(self.text)

    @property
    def font_metrics(self) -> Tuple[int, int]:
        ascent, descent = self.image_creation_options.font.getmetrics()
        return ascent, descent

def generate_image_from_text(text: str, options: ImageCreationOptions) -> OriginalImage:
    width, height = options.font.getsize(text)
    ascent, descent = options.font.getmetrics()
    image_size: Tuple[int, int] = (2*options.padding[0] + width, 2*options.padding[1] + ascent + descent)
    font = options.font

    image: Image = Image.new('RGB', image_size, options.background_color)
    draw = ImageDraw.Draw(image)
    draw.text(options.padding, text, font=font, fill=options.font_color)
    character_bounding_boxes: List[CharacterBoundingBox] = generate_character_bounding_boxes(text, options)

    return OriginalImage(text=text, img=image, character_bounding_boxes=character_bounding_boxes, image_creation_options=options)


def generate_character_bounding_boxes(text: str, options: ImageCreationOptions) -> List[CharacterBoundingBox]:
    """
    Calculate the bounding boxes for every character.
    Source: https://github.com/python-pillow/Pillow/issues/3921
    """
    character_bounding_boxes: List[CharacterBoundingBox] = []
    for i, char in enumerate(text):
        bottom_1 = options.font.getsize(text[i])[1]
        right, bottom_2 = options.font.getsize(text[:i + 1])
        bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
        width, height = options.font.getmask(char).size
        right += options.padding[0]
        bottom += options.padding[1]
        top = bottom-height
        left = right-width
        bb: CharacterBoundingBox = CharacterBoundingBox(char=char, top=top, bottom=bottom, left=left, right=right)
        character_bounding_boxes.append(bb)
    return character_bounding_boxes


def draw_character_bounding_boxes(original_image: OriginalImage) -> Image:
    """
    Return a copy of an original image with the character bounding boxes drawn onto it for visualization
    """
    image: Image = original_image.img.copy()
    draw = ImageDraw.Draw(image)
    bbs: List[CharacterBoundingBox] = original_image.character_bounding_boxes
    for bb in bbs:
        draw.rectangle((bb.left, bb.top, bb.right, bb.bottom), fill=None, outline=(255, 0, 0))
    return image
