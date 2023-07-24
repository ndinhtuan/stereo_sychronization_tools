def binary_to_gray(num: int) -> int:
    """adapted from wikipedia: https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code"""
    assert num >= 0
    return num ^ (num >> 1)


def gray_to_binary(num: int) -> int:
    """adapted from wikipedia: https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code"""
    assert num >= 0
    mask = num
    while mask:
        mask = mask >> 1
        num = num ^ mask
    return num


def compose_rows(rows) -> int:
    """
    Board orientation: frowny-face in the top left
    :param rows string of five bits ("0" or "1"), input must already be zero-padded to equal length
    :return decoded integer
    """
    composed = ""
    for row in rows:
        composed += row[::-1]
    return gray_to_binary(int(composed, 2))
