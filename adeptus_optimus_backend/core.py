import math

from .utils import *


# Core engine logic
class Options:
    """
    Notes about rules:
    - Rerolls apply before modifiers
    """
    ones = "ones"
    onestwos = "onestwos"
    all = "all"
    none = "none"

    def __init__(self,
                 hit_modifier=0,
                 wound_modifier=0,
                 reroll_hits="none",
                 reroll_wounds="none"):
        assert (hit_modifier in {-1, 0, 1})
        assert (wound_modifier in {-1, 0, 1})
        assert (reroll_hits in {Options.none, Options.ones, Options.onestwos, Options.all})
        assert (reroll_wounds in {Options.none, Options.ones, Options.onestwos, Options.all})

        self.hit_modifier = hit_modifier
        self.wound_modifier = wound_modifier
        self.reroll_hits = reroll_hits
        self.reroll_wounds = reroll_wounds

    @staticmethod
    def empty():
        return Options()

    @staticmethod
    def parse(options):
        if isinstance(options, Options):
            return options
        else:
            return Options(
                hit_modifier=int(options["hit_modifier"]),
                wound_modifier=int(options["wound_modifier"]),
                reroll_hits=options["reroll_hits"],
                reroll_wounds=options["reroll_wounds"]
            )


class Profile:
    def __init__(self, weapons, points):
        assert (isinstance(weapons, list))
        require(len(weapons) > 0, "An attacking profile must have at least one declared weapon")
        assert (isinstance(weapons[0], Weapon))
        self.weapons = weapons
        try:
            self.points = int(points)
        except ValueError:
            self.points = None
        except Exception as e:
            raise e
        require(self.points is not None and self.points > 0, f"Invalid points value: '{points}'")


class Weapon:
    def __init__(self, hit, a, s, ap, d, options=Options.empty()):
        # prob by roll result: O(n*dice_type)
        self.hit = parse_dice_expr(hit, complexity_threshold=24, raise_on_failure=True)  # only one time O(n*dice_type)
        require(self.hit.dices_type is None, "Balistic/Weapon Skill cannot be a dice expression")
        self.a = parse_dice_expr(a, complexity_threshold=128, raise_on_failure=True)  # only one time 0(n)
        require(self.a.avg != 0, "Number of Attacks cannot be 0")
        self.s = parse_dice_expr(s, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        require(self.s.avg != 0, "Strength cannot be 0")
        self.ap = parse_dice_expr(ap, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.d = parse_dice_expr(d, complexity_threshold=6, raise_on_failure=True)  # exponential exponential compl
        require(self.d.avg != 0, "Damage cannot be 0")
        self.options = Options.parse(options)

        self.avg_attack = self.a.avg
        self.hit_ratio = get_hit_ratio(self)


class Target:
    def __init__(self, t, sv=6, invu=None, fnp=None, w=1):
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


def get_hit_ratio(weapon):
    assert (isinstance(weapon, Weapon))
    hit_ratio = 0
    for hit_roll, prob_hit_roll in prob_by_roll_result(weapon.hit).items():
        hit_ratio += prob_hit_roll * compute_successes_ratio(hit_roll - weapon.options.hit_modifier)
    return hit_ratio


wound_ratios_cache = {}


def compute_successes_ratio(modified_necessary_roll, auto_success_on_6=True, reroll=Options.none):
    assert (reroll in {Options.none, Options.ones, Options.onestwos, Options.all})

    necessary_roll = modified_necessary_roll
    if modified_necessary_roll <= 1:
        necessary_roll = 2  # roll of 1 always fails
    if modified_necessary_roll >= 7:
        if auto_success_on_6:
            necessary_roll = 6  # roll of 6 always succeeds
        else:
            return 0
    base_successes_ratio = (7 - necessary_roll) / 6
    if reroll == Options.none:
        return base_successes_ratio
    elif reroll == Options.ones or (reroll == Options.onestwos and necessary_roll == 2):
        return base_successes_ratio + base_successes_ratio / 6
    elif reroll == Options.onestwos:
        # guaranteed that necessary_roll > 2
        return base_successes_ratio + 2*base_successes_ratio / 6
    elif reroll == Options.all:
        return base_successes_ratio + (1-base_successes_ratio)*base_successes_ratio#1 - (1 - base_successes_ratio)**2



def get_wound_ratio(weapon, target):
    """
    Random strength value is resolved once per weapon:
    "Each time this unit is chosen to shoot with, roll once to
    determine the Strength characteristic of this weapon."
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    key = f"{weapon.s}{weapon.options.wound_modifier}{target.t}"
    wound_ratio = wound_ratios_cache.get(key, None)
    if wound_ratio is None:
        wound_ratio = 0
        for s_roll, prob_s_roll in prob_by_roll_result(weapon.s).items():
            wound_ratio += compute_successes_ratio(
                compute_necessary_wound_roll(s_roll, target.t) - weapon.options.wound_modifier) * prob_s_roll
        wound_ratios_cache[key] = wound_ratio

    return wound_ratio


unsaved_wound_ratios_cache = {}


def get_unsaved_wound_ratio(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    key = f"{weapon.ap}{target.sv}{target.invu}"
    unsaved_wound_ratio = unsaved_wound_ratios_cache.get(key, None)
    if unsaved_wound_ratio is None:
        unsaved_wound_ratio = 0
        for ap_roll, prob_ap_roll in prob_by_roll_result(weapon.ap).items():
            save_roll = target.sv + ap_roll
            if target.invu is not None:
                save_roll = min(save_roll, target.invu)
            save_fail_ratio = 1 - compute_successes_ratio(save_roll, auto_success_on_6=False)
            unsaved_wound_ratio += save_fail_ratio * prob_ap_roll
        unsaved_wound_ratios_cache[key] = unsaved_wound_ratio

    return unsaved_wound_ratio


class Cache:
    def __init__(self):
        self.dict = {}
        self.hits = 0
        self.tries = 0

    def __str__(self):
        return f"tries={self.tries}, hits={self.hits}, misses={self.tries - self.hits}"

    def add(self, state, cached_unweighted_downstream):
        key = Cache._keyify(state)
        if self.dict.get(key, (0, 0))[0] < state.n_unsaved_wounds_left:
            self.dict[key] = (state.n_unsaved_wounds_left, cached_unweighted_downstream)

    def get(self, state):
        res = self.dict.get(Cache._keyify(state), (None, None))
        self.tries += 1
        if res[0] is not None:
            self.hits += 1
        return res

    def reset(self):
        del self.dict
        self.dict = {}
        self.hits = 0
        self.tries = 0

    @staticmethod
    def _keyify(state):
        return f"{state.current_wound_n_damages_left},{state.remaining_target_wounds},{state.n_unsaved_wounds_left}"


class State:
    weapon_d = None
    target_wounds = None
    n_unsaved_wounds_init = None
    n_figs_slained_weighted_ratios = None
    fnp_fail_ratio = None
    start_target_wounds = None
    cache = Cache()

    def __init__(self,
                 n_unsaved_wounds_left,  # key field, 0 when resolved
                 current_wound_n_damages_left,  # key field, 0 when resolved
                 n_figs_slained_so_far,  # value field
                 remaining_target_wounds,  # key field
                 ):
        self.n_unsaved_wounds_left = n_unsaved_wounds_left
        self.current_wound_n_damages_left = current_wound_n_damages_left
        self.n_figs_slained_so_far = n_figs_slained_so_far
        self.remaining_target_wounds = remaining_target_wounds

    def copy(self):
        return State(self.n_unsaved_wounds_left,
                     self.current_wound_n_damages_left,
                     self.n_figs_slained_so_far,
                     self.remaining_target_wounds)


def get_slained_figs_ratio(state_):
    assert (isinstance(state_, State))
    assert (state_.remaining_target_wounds >= 0)
    assert (state_.n_unsaved_wounds_left >= 0)
    assert (state_.current_wound_n_damages_left >= 0)
    state = state_.copy()

    # resolve a model kill
    if state.remaining_target_wounds == 0:
        state.remaining_target_wounds = State.target_wounds
        # additionnal damages are not propagated to other models
        state.current_wound_n_damages_left = 0
        downstream = get_slained_figs_ratio(state)
        downstream += 1
        return downstream  # upstream propagation of figs slained count

    last_model_injured_frac = 1 - state.remaining_target_wounds / State.target_wounds

    if state.current_wound_n_damages_left == 0 and state.n_unsaved_wounds_left == 0:
        # leaf: no more damages to fnp no more wounds to consume or p(leaf) < threshold
        # portion of the last model injured
        State.cache.add(state, last_model_injured_frac)
        return last_model_injured_frac
    else:
        # test cache
        # cached_downstream = None
        cached_res_n_unsaved_wounds_left, cached_downstream = State.cache.get(state)
        if cached_downstream is not None and cached_res_n_unsaved_wounds_left >= state.n_unsaved_wounds_left:
            # use cached res if deep enough
            return cached_downstream
        else:
            if state.current_wound_n_damages_left == 0:
                if cached_downstream is None or cached_res_n_unsaved_wounds_left < state.n_unsaved_wounds_left:
                    # consume a wound
                    # random doms handling
                    res = [
                        prob_d *
                        get_slained_figs_ratio(State(n_unsaved_wounds_left=state.n_unsaved_wounds_left - 1,
                                                     current_wound_n_damages_left=d,
                                                     n_figs_slained_so_far=state.n_figs_slained_so_far,
                                                     remaining_target_wounds=state.remaining_target_wounds))
                        for d, prob_d in prob_by_roll_result(State.weapon_d).items()
                    ]
                    downstream = sum(res)
                    State.cache.add(state, downstream)
                    return downstream
            else:
                # FNP fail
                f = get_slained_figs_ratio(State(state.n_unsaved_wounds_left,
                                                 state.current_wound_n_damages_left - 1,
                                                 state.n_figs_slained_so_far,
                                                 state.remaining_target_wounds - 1))

                # FNP success
                if State.fnp_fail_ratio != 1:
                    s = get_slained_figs_ratio(State(state.n_unsaved_wounds_left,
                                                     state.current_wound_n_damages_left - 1,
                                                     state.n_figs_slained_so_far,
                                                     state.remaining_target_wounds))
                    downstream = (1 - State.fnp_fail_ratio) * s + State.fnp_fail_ratio * f
                else:
                    downstream = f
                State.cache.add(state, downstream)
                return downstream


slained_figs_ratio_per_unsaved_wound_cache = {}


def get_slained_figs_ratio_per_unsaved_wound(weapon_d, target_fnp, target_wounds):
    """
    n_unsaved_wounds_init=32: 14 sec, res prec +-0.02 compared to 64
    n_unsaved_wounds_init=10:  5 sec, res prec +-0.1  compared to 64
    """
    key = f"{weapon_d}{target_fnp}{target_wounds}"
    slained_figs_ratio_per_unsaved_wound = slained_figs_ratio_per_unsaved_wound_cache.get(key, None)
    if slained_figs_ratio_per_unsaved_wound is None:
        State.weapon_d = weapon_d
        State.target_wounds = target_wounds
        State.n_unsaved_wounds_init = 16
        State.n_figs_slained_weighted_ratios = []
        State.fnp_fail_ratio = 1 if target_fnp is None else 1 - compute_successes_ratio(target_fnp)
        State.start_target_wounds = target_wounds
        State.cache.reset()

        slained_figs_ratio_per_unsaved_wound = get_slained_figs_ratio(State(
            n_unsaved_wounds_left=State.n_unsaved_wounds_init,
            current_wound_n_damages_left=0,
            n_figs_slained_so_far=0,
            remaining_target_wounds=target_wounds)) / State.n_unsaved_wounds_init
        slained_figs_ratio_per_unsaved_wound_cache[key] = slained_figs_ratio_per_unsaved_wound
    return slained_figs_ratio_per_unsaved_wound


def exact_avg_figs_fraction_slained_per_unsaved_wound(d, w):
    return 1 / math.ceil(w / d)


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

    return get_slained_figs_ratio_per_unsaved_wound(weapon.d, target.fnp, target.w)


def score_weapon_on_target(w, t, avg_n_attacks, hit_ratio):
    """
    avg_figs_fraction_slained by point
    """
    avg_n_attacks = w.a.avg if avg_n_attacks is None else avg_n_attacks
    hit_ratio = get_hit_ratio(w) if hit_ratio is None else hit_ratio
    return avg_n_attacks * hit_ratio * get_wound_ratio(w, t) * get_unsaved_wound_ratio(w, t) \
           * get_avg_figs_fraction_slained_per_unsaved_wound(w, t)


def scores_to_z(score_a, score_b):
    """
    z is in ]-1, 1[
    :return z rounded in [-1, 1]
    """
    if score_a > score_b:
        z = + (1 - score_b / score_a)
    else:
        z = - (1 - score_a / score_b)
    return round(z, 2)  # round(z, 3) 91.8kB


def y_dims_to_str(l):
    return f"""T:{l[0]}, W:{l[1]}, fnp:{"-" if l[2] is None else f"{l[2]}+"}"""


def scores_to_ratio(score_a, score_b):
    if score_a > score_b:
        return round(score_a / score_b, 2)
    else:
        return round(score_b / score_a, 2)


def x_dims_to_str(l):
    return f"""Sv:{"-" if l[0] is None else f"{l[0]}+"}, invu:{"-" if l[1] is None else f"{l[1]}+"}"""


def compute_heatmap(profile_a, profile_b):
    assert (isinstance(profile_a, Profile))
    assert (isinstance(profile_b, Profile))

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
    exact_scores = \
        [
            [
                [
                    [score_weapon_on_target(
                        weapon_a,
                        Target(t, sv, invu=invu, fnp=fnp, w=w),
                        weapon_a.avg_attack,
                        weapon_a.hit_ratio
                    ) for weapon_a in profile_a.weapons],
                    [score_weapon_on_target(
                        weapon_b,
                        Target(t, sv, invu=invu, fnp=fnp, w=w),
                        weapon_b.avg_attack,
                        weapon_b.hit_ratio
                    ) for weapon_b in profile_b.weapons]
                ]
                for sv, invu in svs
            ]
            for t, w, fnp in ws_ts_fnps
        ]

    score_a_score_b_tuples = [
        [(sum(scores_weapons_a) / profile_a.points, sum(scores_weapons_b) / profile_b.points)
         for scores_weapons_a, scores_weapons_b in line]
        for line in exact_scores
    ]

    res["scores"] = [
        [
            [
                list(map(lambda x: round(x, 4), scores_weapons_a)),
                list(map(lambda x: round(x, 4), scores_weapons_b))
            ]
            for scores_weapons_a, scores_weapons_b in line]
        for line in exact_scores
    ]

    res["z"] = [[scores_to_z(score_a, score_b) for score_a, score_b in line] for line in
                score_a_score_b_tuples]

    res["ratios"] = [[scores_to_ratio(score_a, score_b) for score_a, score_b in line] for line in
                     score_a_score_b_tuples]
    # print(res)

    # TODO: return 2 scores to be in hover log
    return res
