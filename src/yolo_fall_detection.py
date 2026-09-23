def calculate_bbox_ratio(bbox):
    """
    Calculate the width-to-height ratio
    of a bounding box.
    """

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if height <= 0:
        return 0.0

    return width / height


def classify_fall(ratio, threshold):
    """
    Classify a bounding box as falling or not falling
    based on the supplied width-to-height threshold.
    """

    return ratio >= threshold