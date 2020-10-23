import random
import re

import math


# Utils
def map_7_to_None(v):
    return None if v == 7 else v


class RequirementFailError(Exception):
    pass


def require(predicate, error_message):
    if not (predicate):
        raise RequirementFailError(error_message)


require(True, "bla")

thrown_message = ""
try:
    require(False, "bla")
except RequirementFailError as e:
    thrown_message = str(e)
assert (thrown_message == "bla")


def roll_D6():
    return random.randint(1, 6)


assert (len({roll_D6() for _ in range(1000)}) == 6)


def roll_D3():
    return (random.randint(1, 6) + 1) // 2


assert (len({roll_D3() for _ in range(1000)}) == 3)


class DiceExpr:
    def __init__(self, n, dices_type=None):
        self.n = n
        self.dices_type = dices_type

        if self.dices_type is None:
            self.avg = n
        else:
            self.avg = n * (self.dices_type + 1) / 2

    def roll(self):
        if self.dices_type is None:
            return self.n
        else:
            if self.dices_type == 3:
                return sum([roll_D3() for _ in range(self.n)])
            elif self.dices_type == 6:
                return sum([roll_D6() for _ in range(self.n)])
            else:
                raise AttributeError(f"Unsupported dices_type D{self.dices_type}")

    def __str__(self):
        if self.dices_type is None:
            return str(self.n)
        else:
            return f"{self.n if self.n > 1 else ''}D{self.dices_type}"


assert (str(DiceExpr(5, 3)) == "5D3")
assert (str(DiceExpr(1, 6)) == "D6")
assert (str(DiceExpr(10, None)) == "10")

dice_5D3 = DiceExpr(5, 3)
assert (len({dice_5D3.roll() for _ in range(10000)}) == 5 * 3 - 5 + 1)
dice_4D6 = DiceExpr(4, 6)
assert (len({dice_4D6.roll() for _ in range(10000)}) == 4 * 6 - 4 + 1)


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


assert (parse_dice_expr("4D3").avg == 8)
assert (parse_dice_expr("5").avg == 5)
assert (parse_dice_expr("D7") is None)
assert (parse_dice_expr("0D6") is None)
assert (parse_dice_expr("0").avg == 0)
assert (parse_dice_expr("7D6") is None)
assert (parse_dice_expr("D3").avg == 2)
assert (parse_dice_expr("3D3").avg == 6)
assert (parse_dice_expr("D6").avg == 3.5)
assert (parse_dice_expr("1D6") is None)


def parse_roll(roll):
    res = re.fullmatch(r"([23456])\+", roll)
    if res is None:
        return None
    else:
        return int(res.group(1))


assert (parse_roll("1+") is None)
assert (parse_roll("1+") is None)
assert (parse_roll("2+") == 2)
assert (parse_roll("3+") == 3)
assert (parse_roll("6+") == 6)
assert (parse_roll("7+") is None)
assert (parse_roll("3") is None)


def float_eq(a, b, n_same_decimals=4):
    return f'%.{n_same_decimals}E' % a == f'%.{n_same_decimals}E' % b


assert (float_eq(1, 1.01, 1))
assert (float_eq(0.3333, 0.3334, 2))
assert (float_eq(0.03333, 0.03334, 2))
assert (not float_eq(0.3333, 0.334, 2))
assert (not float_eq(0.03333, 0.0334, 2))


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
        self.hit = parse_dice_expr(hit, raise_on_failure=True)
        self.a = parse_dice_expr(a, raise_on_failure=True)
        self.s = parse_dice_expr(s, raise_on_failure=True)
        self.ap = parse_dice_expr(ap, raise_on_failure=True)
        self.d = parse_dice_expr(d, complexity_threshold=6, raise_on_failure=True)

        self.bonuses = bonuses

        try:
            self.points = int(points)
        except ValueError:
            self.points = None
        except Exception as e:
            raise e
        require(self.points is not None and self.points > 0, f"Invalid points value: '{points}'")


Weapon(hit="5", a="2", s="4D3", ap="1", d="D3", bonuses=Bonuses.empty())


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


Target(8, 3, 5, 6)
Target(8, 3)


# In[229]:


# Engine v1

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


def exact_avg_figs_fraction_slained_per_unsaved_wound(d, w):
    return 1 / math.ceil(w / d)


assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5) == 0.5)
assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2) == 1)
assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16) == 1 / 3)


def update_slained_figs_ratios(n_unsaved_wounds_left,
                               current_wound_n_damages_left,
                               n_figs_slained_so_far,
                               remaining_target_wounds,
                               prob_node,
                               start_target_wounds,
                               fnp_fail_ratio,
                               n_figs_slained_weighted_ratios,
                               weapon_d,
                               target_fnp,
                               target_wounds,
                               n_unsaved_wounds_init,
                               prob_min_until_cut,
                               current_wound_init_n_damages):
    assert (remaining_target_wounds >= 0)
    assert (n_unsaved_wounds_left >= 0)
    assert (current_wound_n_damages_left >= 0)
    if prob_node == 0:
        return

    # resolve a model kill
    if remaining_target_wounds == 0:
        remaining_target_wounds = target_wounds
        n_figs_slained_so_far += 1
        # additionnal damages are not propagated to other models
        current_wound_n_damages_left = 0
        update_slained_figs_ratios(n_unsaved_wounds_left,
                                   current_wound_n_damages_left,
                                   n_figs_slained_so_far,
                                   remaining_target_wounds,
                                   prob_node,
                                   start_target_wounds,
                                   fnp_fail_ratio,
                                   n_figs_slained_weighted_ratios,
                                   weapon_d=weapon_d, target_fnp=target_fnp, target_wounds=target_wounds,
                                   n_unsaved_wounds_init=n_unsaved_wounds_init,
                                   prob_min_until_cut=prob_min_until_cut,
                                   current_wound_init_n_damages=current_wound_init_n_damages)
        return

    # leaf: no more damages to fnp no more wounds to consume or p(leaf) < threshold
    if prob_node < prob_min_until_cut or (n_unsaved_wounds_left == 0 and current_wound_n_damages_left == 0):
        # print("leaf: n_figs_slained_so_far =", n_figs_slained_so_far, "remaining_target_wounds=", remaining_target_wounds )
        if current_wound_n_damages_left > 0:
            # wounds not used when branch is cut
            unused_unsaved_wounds_portion = n_unsaved_wounds_left + current_wound_n_damages_left / current_wound_init_n_damages
        else:
            unused_unsaved_wounds_portion = n_unsaved_wounds_left
        if n_unsaved_wounds_init == unused_unsaved_wounds_portion:
            assert (n_figs_slained_so_far == 0)
        else:
            used_unsaved_wounds_portion = n_unsaved_wounds_init - unused_unsaved_wounds_portion
            assert (used_unsaved_wounds_portion > 0)
            n_figs_slained_weighted_ratios.append(
                # prob, n_figs_slained_ratio_per_wound
                (
                        prob_node *
                        (n_figs_slained_so_far +
                         (-1 + start_target_wounds / target_wounds) +  # portion of the first model cleaned
                         (1 - remaining_target_wounds / target_wounds)) /  # portion of the last model injured
                        (used_unsaved_wounds_portion)
                )
            )
        return

    # consume a wound
    if current_wound_n_damages_left == 0:
        n_unsaved_wounds_left -= 1
        # random doms handling
        for d, prob_d in prob_by_roll_result(weapon_d).items():
            current_wound_n_damages_left = d
            update_slained_figs_ratios(n_unsaved_wounds_left,
                                       current_wound_n_damages_left,
                                       n_figs_slained_so_far,
                                       remaining_target_wounds,
                                       prob_node * prob_d,
                                       start_target_wounds,
                                       fnp_fail_ratio,
                                       n_figs_slained_weighted_ratios,
                                       weapon_d=weapon_d, target_fnp=target_fnp, target_wounds=target_wounds,
                                       n_unsaved_wounds_init=n_unsaved_wounds_init,
                                       prob_min_until_cut=prob_min_until_cut,
                                       current_wound_init_n_damages=current_wound_n_damages_left)
        return

    # FNP success
    update_slained_figs_ratios(
        n_unsaved_wounds_left,
        current_wound_n_damages_left - 1,
        n_figs_slained_so_far,
        remaining_target_wounds,
        prob_node * (1 - fnp_fail_ratio),
        start_target_wounds,
        fnp_fail_ratio,
        n_figs_slained_weighted_ratios,
        weapon_d=weapon_d, target_fnp=target_fnp, target_wounds=target_wounds,
        n_unsaved_wounds_init=n_unsaved_wounds_init,
        prob_min_until_cut=prob_min_until_cut,
        current_wound_init_n_damages=current_wound_init_n_damages)

    # FNP fail
    update_slained_figs_ratios(
        n_unsaved_wounds_left,
        current_wound_n_damages_left - 1,
        n_figs_slained_so_far,
        remaining_target_wounds - 1,
        prob_node * fnp_fail_ratio,
        start_target_wounds,
        fnp_fail_ratio,
        n_figs_slained_weighted_ratios,
        weapon_d=weapon_d, target_fnp=target_fnp, target_wounds=target_wounds,
        n_unsaved_wounds_init=n_unsaved_wounds_init,
        prob_min_until_cut=prob_min_until_cut,
        current_wound_init_n_damages=current_wound_init_n_damages)


def compute_slained_figs_ratios_per_unsaved_wound(weapon_d, target_fnp, target_wounds,
                                                  n_unsaved_wounds_init=5,
                                                  prob_min_until_cut=0.0001):
    n_figs_slained_weighted_ratios = []
    fnp_fail_ratio = 1 if target_fnp is None else 1 - compute_successes_ratio(target_fnp)
    for start_target_wounds in range(target_wounds, target_wounds + 1):
        update_slained_figs_ratios(
            n_unsaved_wounds_left=n_unsaved_wounds_init,
            current_wound_n_damages_left=0,
            n_figs_slained_so_far=0,
            remaining_target_wounds=start_target_wounds,
            prob_node=1,
            start_target_wounds=start_target_wounds,
            fnp_fail_ratio=fnp_fail_ratio,
            n_figs_slained_weighted_ratios=n_figs_slained_weighted_ratios,
            weapon_d=weapon_d, target_fnp=target_fnp, target_wounds=target_wounds,
            n_unsaved_wounds_init=n_unsaved_wounds_init,
            prob_min_until_cut=prob_min_until_cut,
            current_wound_init_n_damages=0)
    # print(n_figs_slained_weighted_ratios)
    # print(f"{len(n_figs_slained_weighted_ratios)/1} leafs by single tree, for depth={n_unsaved_wounds_init}")
    # return sum(map(lambda tup: tup[0] * tup[1], n_figs_slained_weighted_ratios))/1
    return sum(n_figs_slained_weighted_ratios) / 1


# FNP
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 6, 1), 5 / 6, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 5, 1), 4 / 6, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 4, 1), 0.5, 0))
# on W=2
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), None, 2), 0.5, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(2), None, 2), 1, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(2, 3), None, 2), 1, 0))
# random doms
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 6), None, 35), 0.1, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 6), 4, 70, n_unsaved_wounds_init=32,
                                                               prob_min_until_cut=0.0001), 0.025, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 6), 5, 70, n_unsaved_wounds_init=32,
                                                               prob_min_until_cut=0.0001), 2 / 60, 0))


# last step numeric averaging: damage roll + fnp
def get_avg_figs_fraction_slained_per_unsaved_wound(weapon, target):
    """
    Random damage value is resolved once per unsaved wound

    :param N: number of consecutive wounds resolved:
          - N=1000 leads to a result precise at +- 1.5%
          - N=10000 leads to a result precise at +- 0.5%
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    if weapon.d.dices_type is None and target.fnp is None:
        return exact_avg_figs_fraction_slained_per_unsaved_wound(d=weapon.d.n, w=target.w)

    return compute_slained_figs_ratios_per_unsaved_wound(weapon.d, target.fnp, target.w)


assert (get_avg_figs_fraction_slained_per_unsaved_wound(
    Weapon("5", "10", "2D6", "1", "1"),
    Target(t=8, sv=6, invu=None, fnp=6, w=1)
) == get_avg_figs_fraction_slained_per_unsaved_wound(
    Weapon("5", "10", "7", "1", "1"),
    Target(t=8, sv=6, invu=None, fnp=6, w=1)
))


def get_avg_of_density(d):
    l = [float(v) * float(p) for v, p in d.items()]
    return sum(l)


assert (get_avg_of_density({0: 0.2, 1: 0.5, 2: 0.3}) == 0.5 + 0.3 * 2)


# V3


def get_hit_ratio(weapon):
    assert (isinstance(weapon, Weapon))
    hit_ratio = 0
    for hit_roll, prob_hit_roll in prob_by_roll_result(weapon.hit).items():
        hit_ratio += prob_hit_roll * compute_successes_ratio(hit_roll - weapon.bonuses.to_wound)
    return hit_ratio


def get_wound_ratio(weapon, target):
    """
    Random strength value is resolved once per weapon
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    wound_ratio = 0
    for s_roll, prob_s_roll in prob_by_roll_result(weapon.s).items():
        wound_ratio += compute_successes_ratio(
            compute_necessary_wound_roll(s_roll, target.t) - weapon.bonuses.to_wound) * prob_s_roll
    return wound_ratio


def get_unsaved_wound_ratio(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    unsaved_wound_ratio = 0
    for ap_roll, prob_ap_roll in prob_by_roll_result(weapon.ap).items():
        save_roll = target.sv + ap_roll
        if target.invu is not None:
            save_roll = min(save_roll, target.invu)
        save_fail_ratio = 1 - compute_successes_ratio(save_roll, auto_success_on_6=False)
        unsaved_wound_ratio += save_fail_ratio * prob_ap_roll
    return unsaved_wound_ratio


def score_weapon_on_target(w, t, avg_n_attacks=None, hit_ratio=None):
    """
    avg_figs_fraction_slained by point
    """
    avg_n_attacks = w.a.avg if avg_n_attacks is None else avg_n_attacks
    hit_ratio = get_hit_ratio(w) if hit_ratio is None else hit_ratio
    return avg_n_attacks * hit_ratio * get_wound_ratio(w, t) * get_unsaved_wound_ratio(w, t) \
           * get_avg_figs_fraction_slained_per_unsaved_wound(w, t) / w.points


# Sv=1 : ignore PA -1
wea = Weapon(hit="4", a="4", s="4", ap="1", d="3", bonuses=Bonuses(0, 0), points=120)
wea2 = Weapon(hit="4", a="4", s="4", ap="0", d="3", bonuses=Bonuses(0, 0), points=120)
tar = Target(t=4, sv=1, invu=5, fnp=6, w=16)
assert (abs(score_weapon_on_target(wea, tar) / score_weapon_on_target(wea2, tar) - 1) <= 0.25)
# S=2D6 triggers upper threshold effect on T=8 and is better than S=7, but not on other Toughnesses
w1, w2 = Weapon("5", "10", "2D6", "1", "1", bonuses=Bonuses.empty()), Weapon("5", "10", "7", "1", "1",
                                                                             bonuses=Bonuses.empty())
t1 = Target(t=8, sv=6, invu=None, fnp=6, w=1)
t2 = Target(t=7, sv=6, invu=None, fnp=6, w=1)
assert (score_weapon_on_target(w1, t1) > 1.1 * score_weapon_on_target(w2, t1))
assert (score_weapon_on_target(w1, t2) < 1.1 * score_weapon_on_target(w2, t2))
w3, w4 = Weapon("5", "7", "2D6", "1", "1", bonuses=Bonuses.empty()), Weapon("5", "2D6", "2D6", "1", "1",
                                                                            bonuses=Bonuses.empty())
assert (float_eq(score_weapon_on_target(w3, t1), score_weapon_on_target(w4, t1)))


def scores_to_comparison_score(score_a, score_b):
    if score_a > score_b:
        return + (1 - score_b / score_a)
    else:
        return - (1 - score_a / score_b)


assert (scores_to_comparison_score(10000, 1) == 0.9999)
assert (scores_to_comparison_score(1, 10000) == -0.9999)
assert (scores_to_comparison_score(1, 1) == 0)


def y_dims_to_str(l):
    return f"""T:{l[0]}, W:{l[1]}, fnp:{"-" if l[2] is None else f"{l[2]}+"}"""


def scores_to_ratio(score_a, score_b):
    if score_a > score_b:
        return round(score_a / score_b, 2)
    else:
        return round(score_b / score_a, 2)


assert (scores_to_ratio(1, 1) == 0.0)
assert (scores_to_ratio(1, 2) == 2.0)
assert (scores_to_ratio(4, 2) == 2.0)


def x_dims_to_str(l):
    return f"""Sv:{"-" if l[0] is None else f"{l[0]}+"}, invu:{"-" if l[1] is None else f"{l[1]}+"}"""


def compute_heatmap(weapon_a, weapon_b):
    res = {}
    ws_ts_fnps = []
    for w, ts in zip(
            [1, 2, 3, 4, 6, 8, 10, 12, 16],
            [
                [2, 3, 4],
                [2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 8],
                [4, 5, 6, 8],
                [5, 6, 8],
                [6, 8],
                [6, 8]
            ]
    ):
        fnps = [7] if w > 6 else [7, 6, 5]
        for fnp in fnps:
            for t in ts:
                ws_ts_fnps.append((t, w, fnp))

    ws_ts_fnps.sort(key=lambda e: e[2] * 10000 - e[0] * 100 - e[1])
    ws_ts_fnps = list(map(lambda l: list(map(map_7_to_None, l)), ws_ts_fnps))

    res["y"] = list(map(y_dims_to_str, ws_ts_fnps))

    svs = []
    for invu in [2, 3, 4, 5, 6, 7]:
        for sv in range(1, min(invu + 1, 6 + 1)):
            svs.append((sv, invu))
    svs.sort(key=lambda e: -e[0] * 10 + -e[1])
    svs = list(map(lambda l: list(map(map_7_to_None, l)), svs))

    res["x"] = list(map(x_dims_to_str, svs))

    # target independant
    avg_attack_a, avg_attack_b = weapon_a.a.avg, weapon_b.a.avg
    hit_ratio_a, hit_ratio_b = get_hit_ratio(weapon_a), get_hit_ratio(weapon_b)

    score_a_score_b_tuples = \
        [
            [
                (
                    score_weapon_on_target(
                        weapon_a,
                        Target(t, sv, invu=invu, fnp=fnp, w=w),
                        avg_attack_a,
                        hit_ratio_a
                    ),
                    score_weapon_on_target(
                        weapon_b,
                        Target(t, sv, invu=invu, fnp=fnp, w=w),
                        avg_attack_b,
                        hit_ratio_b)
                )
                for sv, invu in svs
            ]
            for t, w, fnp in ws_ts_fnps
        ]
    res["z"] = [[scores_to_comparison_score(score_a, score_b) for score_a, score_b in line] for line in
                score_a_score_b_tuples]
    res["ratios"] = [[scores_to_ratio(score_a, score_b) for score_a, score_b in line] for line in
                     score_a_score_b_tuples]
    print(res)

    # TODO: return 2 scores to be in hover log
    return res
