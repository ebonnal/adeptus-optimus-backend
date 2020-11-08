import math

from .utils import *


# Core engine logic
class Options:
    """
    Notes & rules:
    - Rerolls apply before modifiers
    - Bad moon reroll & dakka3
      Also goes the other way. So if you roll a 1 then reroll to a 6,
      you score a hit plus you get to make an extra hit roll
      (which can also be rerolled if it's a 1). You just can't
      get another bonus attack from a bonus attack, or reroll
      the same hit roll again.
    - if an attack inflicts mortal wounds in addition to the normal damage,
      resolve the normal damage first.
    - Blast weapons, when applied against 6-10 models, "always makes a minimum of 3 attacks."
      This does not mean each die you roll can't roll lower than 3.
      It means the total result of all the weapon's dice to determine the number of attacks can't be lower than 3.
    - Marksman bolt carbine rule: "When resolving an attack made with this weapon, an unmodified hit roll of 6
      automatically scores a hit and successfully wounds the target (do not make a wound roll)."
    - Rail Riffle rule: "For each wound roll of 6+ made for this weapon, the target unit suffers a mortal wound
      in addition to the normal damage."
    - Smasha gun rule: " Instead of making a wound roll for this weapon, roll 2D6. If the result is equal to or greater
      than the target’s Toughness characteristic, the attack successfully wounds."
    - SAG rule: "Before firing this weapon, roll once to determine the Strength of all its shots. If the result is 11+
      each successful hit inflicts D3 mortal wounds on the target in addition to any normal damage"
    """
    none = None
    ones = "ones"
    onestwos = "onestwos"
    full = "full"

    n_models_1to5 = 1
    n_models_6to10 = 6
    n_models_11_plus = 11

    def __init__(self,
                 hit_modifier=0,
                 wound_modifier=0,
                 reroll_hits=None,
                 reroll_wounds=None,
                 dakka3=None,
                 auto_wounds_on=None,
                 is_blast=False,
                 auto_hit=False,
                 wounds_by_2D6=False,
                 reroll_damages=False):
        assert (hit_modifier in {-1, 0, 1})
        assert (wound_modifier in {-1, 0, 1})
        assert (reroll_hits in {Options.none, Options.ones, Options.onestwos, Options.full})
        assert (reroll_wounds in {Options.none, Options.ones, Options.onestwos, Options.full})
        assert (dakka3 in {Options.none, 5, 6})
        assert (auto_wounds_on in {Options.none, 5, 6})
        assert (type(is_blast) is bool)
        assert (type(auto_hit) is bool)
        assert (type(wounds_by_2D6) is bool)
        assert (type(reroll_damages) is bool)

        self.hit_modifier = hit_modifier
        self.wound_modifier = wound_modifier
        self.reroll_hits = reroll_hits
        self.reroll_wounds = reroll_wounds
        self.dakka3 = dakka3
        self.auto_wounds_on = auto_wounds_on
        self.is_blast = is_blast
        self.auto_hit = auto_hit
        self.wounds_by_2D6 = wounds_by_2D6
        self.reroll_damages = reroll_damages

    @staticmethod
    def empty():
        return Options()

    @staticmethod
    def parse(options):
        if isinstance(options, Options):
            return options
        else:
            assert (len(options) == 10)
            return Options(
                hit_modifier=int(options["hit_modifier"]),
                wound_modifier=int(options["wound_modifier"]),
                reroll_hits=Options.none if options["reroll_hits"] == "none" else options["reroll_hits"],
                reroll_wounds=Options.none if options["reroll_wounds"] == "none" else options["reroll_wounds"],
                dakka3=Options.none if options["dakka3"] == "none" else int(options["dakka3"]),
                auto_wounds_on=Options.none if options["auto_wounds_on"] == "none" else int(options["auto_wounds_on"]),
                is_blast=True if options["is_blast"] == "yes" else False if options["is_blast"] == "no" else None,
                auto_hit=True if options["auto_hit"] == "yes" else False if options["auto_hit"] == "no" else None,
                wounds_by_2D6=
                True if options["wounds_by_2D6"] == "yes" else False if options["wounds_by_2D6"] == "no" else None,
                reroll_damages=
                True if options["reroll_damages"] == "yes" else False if options["reroll_damages"] == "no" else None
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
    at_least_one_blast_weapon = False

    def __init__(self, hit, a, s, ap, d, options=Options.empty()):
        # prob by roll result: O(n*dice_type)
        self.hit = parse_dice_expr(hit, complexity_threshold=24, raise_on_failure=True)  # only one time O(n*dice_type)
        require(self.hit.dices_type is None, "Random Balistic/Weapon Skill is not allowed")
        self.a = parse_dice_expr(a, complexity_threshold=128, raise_on_failure=True)  # only one time 0(n)
        require(self.a.avg != 0, "Number of Attacks cannot be 0")

        self.ap = parse_dice_expr(ap, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.d = parse_dice_expr(d, complexity_threshold=6, raise_on_failure=True)  # exponential exponential compl
        require(self.d.avg != 0, "Damage cannot be 0")
        self.options = Options.parse(options)
        require(not self.options.is_blast or self.a.dices_type is not None,
                f"Cannot activate blast option with a non random attack characteristic: {self.a}")
        self.s = parse_dice_expr(s,
                                 complexity_threshold=12,
                                 raise_on_failure=True,
                                 allow_star=self.options.wounds_by_2D6)  # per each target O(n*dice_type)
        require(self.s.avg != 0, "Strength cannot be 0")
        if self.options.is_blast:
            Weapon.at_least_one_blast_weapon = True
        self.hit_ratio = get_hit_ratio(self)


class Target:
    def __init__(self, t, sv=6, invu=None, fnp=None, w=1, n_models=1):
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

        assert (type(n_models) is int and n_models > 0)
        self.n_models = n_models


# Function runtime caches:
success_ratios_cache = {}
n_attacks_cache = {}
hit_ratios_cache = {}
wound_ratios_cache = {}
unsaved_wound_ratios_cache = {}
slained_figs_ratio_per_unsaved_wound_cache = {}


def get_n_attacks(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    if not weapon.options.is_blast or target.n_models <= 5:
        key = f"{weapon.a}"
    else:
        key = f"{weapon.a}{weapon.options.is_blast}{target.n_models}"

    n_attacks = n_attacks_cache.get(key, None)
    if n_attacks is None:
        if not weapon.options.is_blast or target.n_models <= 5:
            n_attacks = weapon.a.avg
        else:
            if target.n_models <= 10:
                n_attacks = sum(list(map(
                    lambda n_a_and_prob: max(3, n_a_and_prob[0]) * n_a_and_prob[1],
                    get_prob_by_roll_result(weapon.a).items()
                )))
            else:
                n_attacks = weapon.a.n * weapon.a.dices_type

        n_attacks_cache[key] = n_attacks
    return n_attacks


def _visit_hit_tree(reroll_consumed, dakka3_consumed, necessary_roll, reroll, dakka3):
    successes_ratio = 0
    for i in range(1, 7):
        if i >= necessary_roll:
            successes_ratio += 1 / 6
        elif not reroll_consumed and reroll != Options.none:
            if reroll == Options.ones and i == 1:
                successes_ratio += 1 / 6 * _visit_hit_tree(True, dakka3_consumed, necessary_roll, reroll, dakka3)
            elif reroll == Options.onestwos and i <= 2:
                successes_ratio += 1 / 6 * _visit_hit_tree(True, dakka3_consumed, necessary_roll, reroll, dakka3)
            elif reroll == Options.full:
                successes_ratio += 1 / 6 * _visit_hit_tree(True, dakka3_consumed, necessary_roll, reroll, dakka3)
        if not dakka3_consumed and dakka3 != Options.none and i >= dakka3:
            # dakka result in a new dice roll that may be rerolled
            successes_ratio += 1 / 6 * _visit_hit_tree(reroll == Options.none, True, necessary_roll, reroll, dakka3)
    return successes_ratio


def get_success_ratio(modified_necessary_roll, auto_success_on_6=True, reroll=Options.none, dakka3=Options.none):
    assert (reroll in {Options.none, Options.ones, Options.onestwos, Options.full})
    assert (dakka3 in {Options.none, 5, 6})
    key = f"{modified_necessary_roll}{auto_success_on_6}{reroll}{dakka3}"
    success_ratio = success_ratios_cache.get(key, None)
    if success_ratio is None:
        necessary_roll = modified_necessary_roll
        if modified_necessary_roll <= 1:
            necessary_roll = 2  # roll of 1 always fails
        if modified_necessary_roll >= 7:
            if auto_success_on_6:
                necessary_roll = 6  # roll of 6 always succeeds
            else:
                return 0

        success_ratio = _visit_hit_tree(
            reroll_consumed=reroll == Options.none,
            dakka3_consumed=dakka3 == Options.none,
            necessary_roll=necessary_roll,
            reroll=reroll,
            dakka3=dakka3
        )

        success_ratios_cache[key] = success_ratio
    return success_ratio


def get_hit_ratio(weapon):
    assert (isinstance(weapon, Weapon))
    key = f"{weapon.hit}" \
          f"{weapon.options.hit_modifier}" \
          f"{weapon.options.reroll_hits}" \
          f"{weapon.options.dakka3}" \
          f"{weapon.options.auto_hit}"
    hit_ratio = hit_ratios_cache.get(key, None)
    if hit_ratio is None:
        if weapon.options.auto_hit:
            hit_ratio = 1
        else:
            hit_ratio = 0
            for hit_roll, prob_hit_roll in get_prob_by_roll_result(weapon.hit).items():
                hit_ratio += prob_hit_roll * get_success_ratio(hit_roll - weapon.options.hit_modifier,
                                                               reroll=weapon.options.reroll_hits,
                                                               dakka3=weapon.options.dakka3)
        hit_ratios_cache[key] = hit_ratio
    return hit_ratio


def get_wound_ratio(weapon, target):
    """
    Random strength value is resolved once per weapon:
    "Each time this unit is chosen to shoot with, roll once to
    determine the Strength characteristic of this weapon."
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    key = f"{weapon.s}" \
          f"{weapon.options.wound_modifier}" \
          f"{weapon.options.reroll_wounds}" \
          f"{weapon.options.auto_wounds_on}" \
          f"{weapon.options.wounds_by_2D6}" \
          f"{target.t}"

    wound_ratio = wound_ratios_cache.get(key, None)
    if wound_ratio is None:
        wound_ratio = 0
        if weapon.options.wounds_by_2D6:
            for roll, prob_roll in get_prob_by_roll_result(DiceExpr(2, 6)).items():
                if roll >= target.t:
                    wound_ratio += prob_roll
        else:
            for s_roll, prob_s_roll in get_prob_by_roll_result(weapon.s).items():
                success_ratio = get_success_ratio(
                    compute_necessary_wound_roll(s_roll, target.t) - weapon.options.wound_modifier,
                    reroll=weapon.options.reroll_wounds
                )
                if weapon.options.auto_wounds_on == Options.none:
                    wound_ratio += success_ratio * prob_s_roll
                else:
                    wound_ratio += (weapon.options.auto_wounds_on - 1) / 6 * success_ratio + \
                                   (7 - weapon.options.auto_wounds_on) / 6  # auto wounds
        wound_ratios_cache[key] = wound_ratio

    return wound_ratio


def get_unsaved_wound_ratio(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    key = f"{weapon.ap}{target.sv}{target.invu}"
    unsaved_wound_ratio = unsaved_wound_ratios_cache.get(key, None)
    if unsaved_wound_ratio is None:
        unsaved_wound_ratio = 0
        for ap_roll, prob_ap_roll in get_prob_by_roll_result(weapon.ap).items():
            save_roll = target.sv + ap_roll
            if target.invu is not None:
                save_roll = min(save_roll, target.invu)
            save_fail_ratio = 1 - get_success_ratio(save_roll, auto_success_on_6=False)
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
    reroll_damages = None
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
                    if State.reroll_damages:
                        # Damages reroll policy: reroll if
                        # - roll < expected damage roll and
                        # - roll < expected_required_damages_to_destroy_current_target
                        # TODO improve, especially with mortal wounds, and take into account that damages in excess are lost
                        # TODO  ... for expected value calculus
                        expected_required_damages_to_destroy_current_target = state.remaining_target_wounds/State.fnp_fail_ratio
                        prob_by_roll_result = get_prob_by_roll_result(
                            State.weapon_d,
                            reroll_if_less_than=min(
                                expected_required_damages_to_destroy_current_target,
                                State.weapon_d.avg  #
                            )
                        )
                    else:
                        prob_by_roll_result = get_prob_by_roll_result(State.weapon_d)

                    res = [
                        prob_d *
                        get_slained_figs_ratio(State(n_unsaved_wounds_left=state.n_unsaved_wounds_left - 1,
                                                     current_wound_n_damages_left=d,
                                                     n_figs_slained_so_far=state.n_figs_slained_so_far,
                                                     remaining_target_wounds=state.remaining_target_wounds))
                        for d, prob_d in prob_by_roll_result.items()
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


def get_slained_figs_ratio_per_unsaved_wound(weapon_d, target_fnp, target_wounds, reroll_damages=False):
    """
    n_unsaved_wounds_init=32: 14 sec, res prec +-0.02 compared to 64
    n_unsaved_wounds_init=10:  5 sec, res prec +-0.1  compared to 64
    """
    key = f"{weapon_d}{target_fnp}{target_wounds}"
    slained_figs_ratio_per_unsaved_wound = slained_figs_ratio_per_unsaved_wound_cache.get(key, None)
    if slained_figs_ratio_per_unsaved_wound is None:
        State.reroll_damages = reroll_damages
        State.weapon_d = weapon_d
        State.target_wounds = target_wounds
        State.n_unsaved_wounds_init = 16
        State.n_figs_slained_weighted_ratios = []
        State.fnp_fail_ratio = 1 if target_fnp is None else 1 - get_success_ratio(target_fnp)
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
    show_n_models = Weapon.at_least_one_blast_weapon
    if show_n_models:
        n_models_label = f"n°models:{map_n_models_to_label(l[3])}, "
    else:
        n_models_label = ""
    return f"""{n_models_label}fnp:{"-" if l[2] is None else f"{l[2]}+"}, T:{l[0]}, W:{l[1]}, """


def x_dims_to_str(l):
    return f"""Sv:{"-" if l[0] is None else f"{l[0]}+"}, invu:{"-" if l[1] is None else f"{l[1]}+"}"""


def scores_to_ratio(score_a, score_b):
    if score_a > score_b:
        return round(score_a / score_b, 2)
    else:
        return round(score_b / score_a, 2)


def map_n_models_to_label(n_models):
    if n_models == 1:
        return "1to5"
    elif n_models == 6:
        return "6to10"
    elif n_models == 11:
        return "11+"
    else:
        raise RuntimeError


def compute_heatmap(profile_a, profile_b):
    assert (isinstance(profile_a, Profile))
    assert (isinstance(profile_b, Profile))

    res = {}
    ts_ws_fnps_nm = []
    for w, ts in zip(
            [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 25],
            [
                [2, 3, 4],
                [4, 5],
                [4, 5, 6],
                [4, 5, 6],
                [5, 6],
                [5, 6, 7],
                [5, 6, 7],
                [6, 7, 8],
                [7, 8],
                [7, 8],
                [8]
            ]
    ):
        fnps = [7] if w > 5 else [7, 6, 5]

        if Weapon.at_least_one_blast_weapon:
            if w > 2:
                ns_models = [Options.n_models_1to5]
            elif w == 2:
                ns_models = [Options.n_models_1to5, Options.n_models_6to10]
            else:
                ns_models = [Options.n_models_1to5, Options.n_models_6to10, Options.n_models_11_plus]
        else:
            ns_models = [Options.n_models_1to5]

        for fnp in fnps:
            for t in ts:
                for n_models in ns_models:
                    ts_ws_fnps_nm.append((t, w, fnp, n_models))

    # n_models then fnp, then toughness, then wounds
    ts_ws_fnps_nm.sort(key=lambda e: e[2] * 10000 - e[3] * 1000000 - e[0] * 100 - e[1])
    ts_ws_fnps_nm = list(map(lambda l: [l[0], l[1], map_7_to_None(l[2]), l[3]], ts_ws_fnps_nm))

    res["y"] = list(map(y_dims_to_str, ts_ws_fnps_nm))

    svs = []
    for invu in [2, 3, 4, 5, 6, 7]:
        for sv in range(1, min(invu + 1, 6 + 1)):
            svs.append((sv, invu))
    svs.sort(key=lambda e: -e[0] * 10 + -e[1])
    svs = list(map(lambda l: list(map(map_7_to_None, l)), svs))

    res["x"] = list(map(x_dims_to_str, svs))

    targets_matrix = [
        [
            Target(t, sv, invu=invu, fnp=fnp, w=w, n_models=n_models)
            for sv, invu in svs
        ]
        for t, w, fnp, n_models in ts_ws_fnps_nm
    ]

    exact_scores = \
        [
            [
                [
                    [score_weapon_on_target(
                        weapon_a,
                        target,
                        get_n_attacks(weapon_a, target),
                        weapon_a.hit_ratio
                    ) for weapon_a in profile_a.weapons],
                    [score_weapon_on_target(
                        weapon_b,
                        target,
                        get_n_attacks(weapon_b, target),
                        weapon_b.hit_ratio
                    ) for weapon_b in profile_b.weapons]
                ]
                for target in line
            ]
            for line in targets_matrix
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

    if is_dev_execution():
        print(f"caches stats:")
        print(f"\tsize of success_ratios_cache={len(success_ratios_cache)}")
        print(f"\tsize of n_attacks_cache={len(n_attacks_cache)}")
        print(f"\tsize of hit_ratios_cache={len(hit_ratios_cache)}")
        print(f"\tsize of wound_ratios_cache={len(wound_ratios_cache)}")
        print(f"\tsize of unsaved_wound_ratios_cache={len(unsaved_wound_ratios_cache)}")
        print(f"\tsize of slained_figs_ratio_per_unsaved_wound_cache={len(slained_figs_ratio_per_unsaved_wound_cache)}")

    return res
