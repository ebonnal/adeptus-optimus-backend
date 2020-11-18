import re

from time import time, sleep


def apply_mask_matrix(matrix, mask_matrix, predicate_on_mask_matrix):
    return [
        [
            e if predicate_on_mask_matrix(e_mask) else None
            for e, e_mask in zip(l, l_mask)
        ]
        for l, l_mask in zip(matrix, mask_matrix)
    ]


def float_eq(a, b, n_same_decimals=8, verbose=False):
    assert (type(n_same_decimals) is int and type(verbose) is bool)
    if verbose:
        print(f"a={a}, b={b}")
        print(f'%.{n_same_decimals}E' % a, f'%.{n_same_decimals}E' % b)
    return f'%.{n_same_decimals}E' % a == f'%.{n_same_decimals}E' % b


_is_dev_execution = False


def is_dev_execution():
    return _is_dev_execution


def set_is_dev_execution(boolean):
    global _is_dev_execution
    _is_dev_execution = boolean


def with_minimum_exec_time(seconds_min_exec_time, callable, seconds_step=0.1):
    start = time()
    res = callable()
    while time() - start < seconds_min_exec_time:
        sleep(seconds_step)
    return res


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


class DiceExpr:
    star = None

    def __init__(self, n, dices_type=None):
        self.n = n
        self.dices_type = dices_type
        require(dices_type in {None, 3, 6}, f"Dices used must be either D3 or D6, not D{dices_type}")

        if self.dices_type is None:
            self.avg = n
            self.min = n
            self.max = n
        else:
            self.avg = n * (self.dices_type + 1) / 2
            self.min = n
            self.max = n * self.dices_type

    def __str__(self):
        if self.dices_type is None:
            return str(self.n)
        else:
            return f"{self.n if self.n > 1 else ''}D{self.dices_type}"

    def __eq__(self, other):
        if not isinstance(other, DiceExpr):
            return False
        else:
            return self.n == other.n and self.dices_type == other.dices_type


DiceExpr.star = DiceExpr(-1, None)


def parse_dice_expr(d, complexity_threshold=18, raise_on_failure=False, allow_star=False):
    if isinstance(d, DiceExpr):
        return d
    assert (type(d) is str)
    groups = re.fullmatch(r"([1-9][0-9]*)?D([36])?|([0-9]+)", d)
    res = None
    invalidity_details = ""
    try:
        if d == "*" and allow_star:
            res = DiceExpr.star
        else:
            dices_type = int(groups.group(2))
            # at this point dices type is known
            if groups.group(1) is not None and int(groups.group(1)) == 1:
                res = None  # 1D6 is not canonical, should enter D6
                invalidity_details = f" must be noted 'D{dices_type}'"
            else:
                if groups.group(1) is None:
                    n_dices = 1
                else:
                    n_dices = int(groups.group(1))
                res = DiceExpr(n_dices, dices_type)

    except TypeError:
        try:
            flat = int(groups.group(3))
            res = DiceExpr(flat)
        except TypeError:
            res = None
    finally:
        # not too many cases splits
        if res is not None and res.max > complexity_threshold:
            if res.dices_type is None:
                invalidity_details = f": Value is too high"
            else:
                invalidity_details = f": Maximum value is too high"
            res = None

        if raise_on_failure:
            require(res is not None, f"Invalid input '{d}'{invalidity_details}")
        return res


def parse_roll(roll):
    res = re.fullmatch(r"([23456])\+", roll)
    if res is None:
        return None
    else:
        return int(res.group(1))


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
