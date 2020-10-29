import random
import re

import time


def with_timer(func):
    start = time.time()
    res = func()
    print(f"Took {(time.time() - start) * 1000} ms")
    return res


def map_7_to_None(v):
    return None if v == 7 else v


class RequirementFailError(Exception):
    pass


def require(predicate, error_message):
    if not (predicate):
        raise RequirementFailError(error_message)


class DiceExpr:
    def __init__(self, n, dices_type=None):
        self.n = n
        self.dices_type = dices_type

        if self.dices_type is None:
            self.avg = n
        else:
            self.avg = n * (self.dices_type + 1) / 2

    def __str__(self):
        if self.dices_type is None:
            return str(self.n)
        else:
            return f"{self.n if self.n > 1 else ''}D{self.dices_type}"


def parse_dice_expr(d, complexity_threshold=16, raise_on_failure=False):
    assert (type(d) is str)
    groups = re.fullmatch(r"([1-9][0-9]*)?D([36])?|([0-9]+)", d)
    res = None
    invalidity_details = ""
    try:

        dices_type = int(groups.group(2))
        # at this point dices type is known
        if groups.group(1) is not None and int(groups.group(1)) == 1:
            res = None  # 1D6 is not canonical, should enter D6
            invalidity_details = f"must be noted 'D{dices_type}'"
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
        if res is not None and res.n * (1 if res.dices_type is None else res.dices_type) > complexity_threshold:
            res = None
        if raise_on_failure:
            require(res is not None, f"Invalid dices expression: '{d}' {invalidity_details}")
        return res


def parse_roll(roll):
    res = re.fullmatch(r"([23456])\+", roll)
    if res is None:
        return None
    else:
        return int(res.group(1))


def float_eq(a, b, n_same_decimals=4, verbose=False):
    if verbose: print(f'%.{n_same_decimals}E' % a, f'%.{n_same_decimals}E' % b)
    return f'%.{n_same_decimals}E' % a == f'%.{n_same_decimals}E' % b


def prob_by_roll_result(dice_expr):
    if dice_expr.dices_type is None:
        return {dice_expr.n: 1}
    else:
        roll_results_counts = {}

        def f(n, current_sum):
            if n == 0:
                roll_results_counts[current_sum] = roll_results_counts.get(current_sum, 0) + 1
            else:
                for i in range(1, dice_expr.dices_type + 1):
                    f(n - 1, current_sum + i)

        f(dice_expr.n, 0)
        n_cases = sum(roll_results_counts.values())
        for key in roll_results_counts.keys():
            roll_results_counts[key] /= n_cases
        return roll_results_counts


assert (prob_by_roll_result(parse_dice_expr("D3")) == {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
assert (prob_by_roll_result(parse_dice_expr("7")) == {7: 1})
assert (float_eq(1, sum(prob_by_roll_result(parse_dice_expr("2D6")).values())))
assert (prob_by_roll_result(parse_dice_expr("2D6")) == {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
                                                        7: 6 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36,
                                                        12: 1 / 36})


# In[ ]:


# In[228]:


# Core Classes
class Bonuses:
    def __init__(self, to_hit, to_wound, props=None):
        assert (to_hit in {-1, 0, 1})
        assert (to_wound in {-1, 0, 1})

        self.to_hit = to_hit
        self.to_wound = to_wound
        self.props = props if props is not None else {}

    @classmethod
    def empty(cls):
        return Bonuses(0, 0)


Bonuses.empty().to_hit = 0


class Weapon:
    def __init__(self, hit, a, s, ap, d, bonuses=Bonuses.empty(), points=1):
        # prob by roll result: O(n*dice_type)
        self.hit = parse_dice_expr(hit, complexity_threshold=24, raise_on_failure=True)  # only one time O(n*dice_type)
        self.a = parse_dice_expr(a, complexity_threshold=64, raise_on_failure=True)  # only one time 0(n)
        self.s = parse_dice_expr(s, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.ap = parse_dice_expr(ap, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.d = parse_dice_expr(d, complexity_threshold=6, raise_on_failure=True)  # exponential exponential compl

        self.bonuses = bonuses

        try:
            self.points = int(points)
        except ValueError:
            self.points = None
        except Exception as e:
            raise e
        require(self.points is not None and self.points > 0, f"Invalid points value: '{points}'")


class Target:
    def __init__(self, t, sv, invu=None, fnp=None, w=1):
        assert (invu is None or (type(invu) is int and invu > 0 and invu <= 6))
        self.invu = invu

        assert (fnp is None or (type(fnp) is int and fnp > 0 and fnp <= 6))
        self.fnp = fnp

        assert (type(t) is int and t > 0)
        self.t = t

        assert (type(sv) is int and sv > 0 and sv <= 6)
        self.sv = sv

        assert (type(w) is int and w > 0)
        self.w = w


def compute_successes_ratio(modified_necessary_roll, auto_success_on_6=True):
    necessary_roll = modified_necessary_roll
    if modified_necessary_roll <= 1:
        necessary_roll = 2  # roll of 1 always fails
    if modified_necessary_roll >= 7:
        if auto_success_on_6:
            necessary_roll = 6  # roll of 6 always succeeds
        else:
            return 0
    return (7 - necessary_roll) / 6


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
    l = [float(v) * float(p) for v, p in d.items()]
    return sum(l)


try:
    # python version >= 3.8
    from math import comb
except:
    # fallback requiring scipy
    from scipy.special import comb


def dispatch_density_key(previous_density_key, next_density_prob):
    assert (type(previous_density_key) is int)
    assert (previous_density_key >= 0)
    assert (0 < next_density_prob and next_density_prob <= 1)
    n = previous_density_key
    p = next_density_prob
    return {k: comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(0, n + 1)}
