import math


def remove_small(segments, min_segment=0.15):
    result = []
    for start, end in segments:
        if end - start >= min_segment:
            result.append([start, end])
    return result


def generate_segment(labels):
    segments = []
    start, end = 0, 0
    for i, y in enumerate(labels):
        if y == 1 and (i == 0 or labels[i - 1] == 0):
            start = 10 * i

        if y == 1 and (i == len(labels) - 1 or labels[i + 1] == 0):
            end = 10 * (i + 1)
            segments.append([start, end])

    segments_raw = [(float(start) / 1000, float(end) / 1000) for start, end in segments]

    segments = remove_small(segments)
    segments_suf_process = [(float(start) / 1000, float(end) / 1000) for start, end in segments]

    return segments_suf_process, segments_raw
