import random
import re

import math

import scipy.special
from engineutils import *
from enginev3 import compute_slained_figs_ratios_per_unsaved_wound



# Engine v2
def dispatch_density_key(previous_density_key, next_density_prob):
    assert (type(previous_density_key) is int)
    assert (previous_density_key >= 0)
    assert (0 < next_density_prob and next_density_prob <= 1)
    n = previous_density_key
    p = next_density_prob
    return {k: scipy.special.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(0, n + 1)}


assert (dispatch_density_key(3, 0.5) == {0: 0.125, 1: 0.375, 2: 0.375, 3: 0.125})



def exact_avg_figs_fraction_slained_per_unsaved_wound(d, w):
    return 1 / math.ceil(w / d)


assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5) == 0.5)
assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2) == 1)
assert (exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16) == 1 / 3)




# FNP
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 6, 1), 5 / 6, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 5, 1), 4 / 6, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), 4, 1), 0.5, 0))
# on W=2
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1), None, 2), 0.5, 0))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(2), None, 2), 1, 0))
print(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(2, 3), None, 2), 1, 0)
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

## LEGACY

# new version:
# 3 A, c 4, f 4 endu 4
# [0, 0, 0, 1] attacks density
def get_attack_density(weapon):
    assert (isinstance(weapon, Weapon))
    return {a: prob for a, prob in prob_by_roll_result(weapon.a).items()}


# [1/8, 3/8, 3/8 ,1/8] hit density
def get_hits_density(weapon, attack_density):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(attack_density, dict))
    hits_density = {}
    for a, prob_a in attack_density.items():
        # {1: 0.3333333333333333, 2: 0.3333333333333333, 3: 0.3333333333333333}
        for hit_roll, prob_hit_roll in prob_by_roll_result(weapon.hit).items():
            # {5: 1}
            hits_ratio = compute_successes_ratio(hit_roll - weapon.bonuses.to_hit)
            # 0.5
            for hits, prob_hits in dispatch_density_key(a, hits_ratio).items():
                hits_density[hits] = hits_density.get(hits, 0) + prob_hits * prob_hit_roll * prob_a
    return hits_density


# [......]  woud density
def get_wounds_density(weapon, target, hits_density):
    """
    Random strength value is resolved once per weapon
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    assert (isinstance(hits_density, dict))
    wounds_density = {}
    for hits, prob_hits in hits_density.items():
        for s_roll, prob_s_roll in prob_by_roll_result(weapon.s).items():
            wounds_ratio = compute_successes_ratio(
                compute_necessary_wound_roll(s_roll, target.t) - weapon.bonuses.to_wound)
            for wounds, prob_wounds in dispatch_density_key(hits, wounds_ratio).items():
                wounds_density[wounds] = wounds_density.get(wounds, 0) + prob_wounds * prob_s_roll * prob_hits
    return wounds_density


# [......] unsaved wounds density
def get_unsaved_wounds_density(weapon, target, wounds_density):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    assert (isinstance(wounds_density, dict))
    unsaved_wounds_density = {}
    for wounds, prob_wounds in wounds_density.items():
        for ap_roll, prob_ap_roll in prob_by_roll_result(weapon.ap).items():
            save_roll = target.sv + ap_roll
            if target.invu is not None:
                save_roll = min(save_roll, target.invu)
            unsaved_wounds_ratio = 1 - compute_successes_ratio(save_roll, auto_success_on_6=False)
            for unsaved_wounds, prob_unsaved_wounds in dispatch_density_key(wounds, unsaved_wounds_ratio).items():
                unsaved_wounds_density[unsaved_wounds] = \
                    unsaved_wounds_density.get(unsaved_wounds, 0) + prob_unsaved_wounds * prob_ap_roll * prob_wounds
    return unsaved_wounds_density


def score_weapon_on_target_legacy(w, t):
    """
    avg_figs_fraction_slained by point
    """
    a_d = {1: 1}
    h_d = get_hits_density(w, a_d)
    w_d = get_wounds_density(w, t, h_d)
    uw_d = get_unsaved_wounds_density(w, t, w_d)
    assert (float_eq(sum(a_d.values()), 1))
    assert (float_eq(sum(h_d.values()), 1))
    assert (float_eq(sum(w_d.values()), 1))
    assert (float_eq(sum(uw_d.values()), 1))
    return get_avg_figs_fraction_slained_per_unsaved_wound(w, t) * \
           uw_d[1] * \
           get_avg_of_density(get_attack_density(w)) / \
           w.points
## END LEGACY

assert (score_weapon_on_target_legacy(Weapon("D6", "D6", "D6", "D6", "D6", bonuses=Bonuses.empty()), Target(t=8, sv=4, invu=6, fnp=5, w=2))
        == score_weapon_on_target(Weapon("D6", "D6", "D6", "D6", "D6", bonuses=Bonuses.empty()), Target(t=8, sv=4, invu=6, fnp=5, w=2)))
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


assert (scores_to_ratio(1, 1) == 1)
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
