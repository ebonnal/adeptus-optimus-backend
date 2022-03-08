import re

from time import time, sleep

min_exec_duration_seconds = 3

def apply_mask_matrix(matrix, mask_matrix, predicate_on_mask_matrix):
    return [
        [
            e if predicate_on_mask_matrix(e_mask) else None
            for e, e_mask in zip(l, l_mask)
        ]
        for l, l_mask in zip(matrix, mask_matrix)
    ]


def assert_float_eq(a, b, max_ratio=1.0001, verbose=False):
    res = float_eq(a, b, max_ratio, verbose)
    if not res:
        raise AssertionError(f"{a} too different from {b}")


def assert_float_neq(a, b, min_ratio=1.00001, verbose=False):
    res = float_eq(a, b, min_ratio, verbose)
    if res:
        raise AssertionError(f"a={a} too close to b={b}")


def assert_matrix_float_eq(m1, m2, min_ratio=1.00001):
    require(len(m1) == len(m2), f"{len(m1)} != {len(m2)}")
    require(len(m1[0]) == len(m2[0]), f"{len(m1[0])} != {len(m2[0])}")
    for line_1, line_2 in zip(m1, m2):
        for e_1, e_2 in zip(line_1, line_2):
            assert_float_eq(e_1, e_2, min_ratio)


def float_eq(a, b, ratio, verbose):
    assert (int(ratio) != ratio)
    assert (1 < ratio < 1.5)
    if verbose and is_dev_execution():
        print(a, b, max(abs(b / a), abs(a / b)))
    if a == 0 or b == 0:
        return a == b
    return a == b or max(abs(b / a), abs(a / b)) <= ratio


_is_dev_execution = False


def is_dev_execution():
    return _is_dev_execution


def set_is_dev_execution(boolean):
    global _is_dev_execution
    _is_dev_execution = boolean


def delay_from(from_instant, for_seconds):
    if time() - from_instant < for_seconds:
        sleep(for_seconds - (time() - from_instant))


def with_timer(func):
    start = time()
    res = func()
    if is_dev_execution():
        print(f"Took {(time() - start) * 1000} ms")
    return res


def map_7_to_None(v):
    return None if v == 7 else v


class RequirementError(Exception):
    pass


def require(predicate, error_message):
    try:
        error_message = error_message()
    except TypeError:
        pass
    if not (predicate):
        raise RequirementError(error_message)


def compute_necessary_wound_roll(f, e):
    if f >= 2 * e:
        return 2
    elif f > e:
        return 3
    elif f == e:
        return 4
    elif f <= e / 2:
        return 6
    else:
        assert (f < e)
        return 5


def get_avg_of_density(d):
    return sum([float(v) * float(p) for v, p in d.items()])
