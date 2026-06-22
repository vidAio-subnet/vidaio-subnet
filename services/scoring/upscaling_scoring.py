import math


def calculate_length_score(content_length):
    """
    Convert content length in seconds to a normalized length score.

    Args:
        content_length (float): Video duration in seconds (5-320s)

    Returns:
        float: Normalized length score (0-1)
    """
    return math.log(1 + content_length) / math.log(1 + 320)


def calculate_preliminary_score(quality_score, length_score, quality_weight=0.5, length_weight=0.5):
    """
    Calculate the preliminary score from quality and length scores.

    Args:
        quality_score (float): Normalized quality score (0-1)
        length_score (float): Normalized length score (0-1)
        quality_weight (float): Weight for quality component (default: 0.5)
        length_weight (float): Weight for length component (default: 0.5)

    Returns:
        float: Preliminary combined score (0-1)
    """
    return (quality_score * quality_weight) + (length_score * length_weight)


def calculate_final_score(s_pre):
    """
    Transform preliminary score into final score using exponential function.

    Args:
        s_pre (float): Preliminary score (0-1)

    Returns:
        float: Final exponentially-transformed score
    """
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calculate_quality_score(pieapp_score):
    sigmoid_normalized_score = sigmoid(pieapp_score)

    original_at_zero = (1 - (math.log10(sigmoid(0) + 1) / math.log10(3.5))) ** 2.5
    original_at_two = (1 - (math.log10(sigmoid(2.0) + 1) / math.log10(3.5))) ** 2.5

    original_value = (1 - (math.log10(sigmoid_normalized_score + 1) / math.log10(3.5))) ** 2.5

    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))

    return scaled_value
