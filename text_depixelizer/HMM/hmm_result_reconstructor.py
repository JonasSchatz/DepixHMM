from typing import List, Tuple

from PIL import ImageFont


def reconstruct_string_from_window_characters(window_characters: List[Tuple[str]], block_size: int, font: ImageFont) -> str:
    """
    Reconstruct the string from the HMM results, e.g.
    [('a', 'b'), ('b', 'c')] -> 'abc'
    """

    reconstructed_result: List[str] = []
    estimated_positions: List[Tuple[int, int]] = []

    for index, characters_in_one_window in enumerate(window_characters):
        block_start_position: int = index * block_size
        possible_overlap_area = [
            char
            for char, pos
            in zip(reconstructed_result, estimated_positions)
            if pos[1] >= (block_start_position - font.getsize(characters_in_one_window[0])[0])]
        overlap: int = get_overlap(possible_overlap_area, characters_in_one_window)

        offset: int = 0
        for i in range(overlap, len(list(characters_in_one_window))):
            character_to_be_added = characters_in_one_window[i]
            estimated_start: int = block_start_position + offset
            estimated_end: int = block_start_position + font.getsize(character_to_be_added)[0] + offset
            estimated_positions.append((estimated_start, estimated_end))
            reconstructed_result.append(characters_in_one_window[i])

            offset = offset + font.getsize(character_to_be_added)[0]

    reconstructed_string: str = ''.join(reconstructed_result)
    return reconstructed_string


def get_overlap(reconstructed_data: List[str], new_characters: Tuple[str, ...]) -> int:
    largest_overlap = 0
    for possible_overlap in range(1, len(new_characters) + 1):
        if reconstructed_data[-possible_overlap:] == list(new_characters)[:possible_overlap]:
            largest_overlap = possible_overlap
    return largest_overlap


def string_similarity(original_string: str, recovered_string: str) -> float:
    """
    Modified edit distance, normalizing the Levenshtein distance between 0 and 1,
    where 1 indicates a perfect match of the recovered string to the original string
    """
    return 1 - levenshteinDistance(original_string, recovered_string)/len(original_string)


def levenshteinDistance(s1: str, s2: str) -> int:
    """
    https://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
