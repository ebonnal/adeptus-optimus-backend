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
    - mortals and saves/FNP: "mortal wounds ignore saves (regular and invulnerable) completely but don't usually
      ignore Feel No Pain type effects"
    """
    none = None
    ones = "ones"
    onestwos = "onestwos"
    full = "full"

    wound = "wound"
    strength = "strength"

    n_models_1to5 = 1
    n_models_6to10 = 6
    n_models_11_plus = 11

    snipe_roll_type = "roll_type"
    snipe_threshold = "threshold"
    snipe_n_mortals = "n_mortals"

    hit_modifier_key = "hit_modifier"
    wound_modifier_key = "wound_modifier"
    save_modifier_key = "save_modifier"
    reroll_hits_key = "reroll_hits"
    reroll_wounds_key = "reroll_wounds"
    dakka3_key = "dakka3"
    auto_wounds_on_key = "auto_wounds_on"
    is_blast_key = "is_blast"
    auto_hit_key = "auto_hit"
    wounds_by_2D6_key = "wounds_by_2D6"
    reroll_damages_key = "reroll_damages"
    roll_damages_twice_key = "roll_damages_twice"
    snipe_key = "snipe"

    opt_key_to_repr = {
        hit_modifier_key: "Hit roll modifier",
        wound_modifier_key: "Wound roll modifier",
        save_modifier_key: "Save roll modifier",
        reroll_hits_key: "Hits reroll",
        reroll_wounds_key: "Wounds reroll",
        dakka3_key: "Dakka Dakka Dakka on _+",
        auto_wounds_on_key: "An unmodified hit roll of _+ automatically wounds",
        is_blast_key: "Is a blast weapon",
        auto_hit_key: "Automatically hits",
        wounds_by_2D6_key: "Wounds if the result of 2D6 >= target’s Toughness",
        reroll_damages_key: "Damage rolls reroll",
        roll_damages_twice_key: "Make random damage rolls twice and discard the lowest result",
        snipe_key: "For each _ roll of _+ , inflicts _ mortal wound(s)"
    }

    not_activated_value = {
        hit_modifier_key: 0,
        wound_modifier_key: 0,
        save_modifier_key: 0,
        reroll_hits_key: none,
        reroll_wounds_key: none,
        dakka3_key: none,
        auto_wounds_on_key: none,
        is_blast_key: False,
        auto_hit_key: False,
        wounds_by_2D6_key: False,
        reroll_damages_key: False,
        roll_damages_twice_key: False,
        snipe_key: none
    }

    incompatibilities = {
        hit_modifier_key: {},
        wound_modifier_key: {},
        save_modifier_key: {},
        reroll_hits_key: {},
        reroll_wounds_key: {},
        dakka3_key: {},
        auto_wounds_on_key: {},
        is_blast_key: {},
        auto_hit_key: {hit_modifier_key, auto_hit_key, reroll_hits_key, dakka3_key, auto_wounds_on_key},
        wounds_by_2D6_key: {wound_modifier_key, auto_wounds_on_key, reroll_wounds_key},
        reroll_damages_key: {},
        roll_damages_twice_key: {reroll_damages_key},
        snipe_key: {auto_wounds_on_key, wounds_by_2D6_key}
    }

    def __init__(self,
                 hit_modifier=0,
                 wound_modifier=0,
                 save_modifier=0,
                 reroll_hits=None,
                 reroll_wounds=None,
                 dakka3=None,
                 auto_wounds_on=None,
                 is_blast=False,
                 auto_hit=False,
                 wounds_by_2D6=False,
                 reroll_damages=False,
                 roll_damages_twice=False,
                 snipe=None):

        assert (hit_modifier in {-1, 0, 1})
        assert (wound_modifier in {-1, 0, 1})
        assert (save_modifier in {-3, -2, -1, 0, 1, 2, 3})
        assert (reroll_hits in {Options.none, Options.ones, Options.onestwos, Options.full})
        assert (reroll_wounds in {Options.none, Options.ones, Options.onestwos, Options.full})
        assert (dakka3 in {Options.none, 5, 6})
        assert (auto_wounds_on in {Options.none, 5, 6})
        assert (type(is_blast) is bool)
        assert (type(auto_hit) is bool)
        assert (type(wounds_by_2D6) is bool)
        assert (type(reroll_damages) is bool)
        assert (type(roll_damages_twice) is bool)

        self.hit_modifier = hit_modifier
        self.wound_modifier = wound_modifier
        self.save_modifier = save_modifier
        self.reroll_hits = reroll_hits
        self.reroll_wounds = reroll_wounds
        self.dakka3 = dakka3
        self.auto_wounds_on = auto_wounds_on
        self.is_blast = is_blast
        self.auto_hit = auto_hit
        self.wounds_by_2D6 = wounds_by_2D6
        self.reroll_damages = reroll_damages
        self.roll_damages_twice = roll_damages_twice
        self.snipe = snipe  # a part of snipe validation is in Options.parse_snipe and another part in Weapon.__init__
        require(self.snipe == Options.not_activated_value[Options.snipe_key],
                f"Option '{Options.opt_key_to_repr[Options.snipe_key]}' is temporarily unavailable")

        # Compatibility check:
        for opt_key1, incompatible_opt_keys in Options.incompatibilities.items():
            if self.__dict__[opt_key1] != Options.not_activated_value[opt_key1]:
                for opt_key2 in Options.opt_key_to_repr.keys():
                    if opt_key2 != opt_key1 and self.__dict__[opt_key2] != Options.not_activated_value[opt_key2]:
                        require(
                            opt_key2 not in incompatible_opt_keys,
                            f"Options '{Options.opt_key_to_repr[opt_key1]}' and '{Options.opt_key_to_repr[opt_key2]}' are incompatible"
                        )

    @staticmethod
    def empty():
        return Options()

    @staticmethod
    def parse(options):
        if isinstance(options, Options):
            return options
        else:
            assert (len(options) == 13)
            return Options(
                hit_modifier=
                int(options[Options.hit_modifier_key]) if len(options[Options.hit_modifier_key]) else 0,
                wound_modifier=
                int(options[Options.wound_modifier_key]) if len(options[Options.wound_modifier_key]) else 0,
                save_modifier=
                int(options[Options.save_modifier_key]) if len(options[Options.save_modifier_key]) else 0,
                reroll_hits=
                options[Options.reroll_hits_key] if len(options[Options.reroll_hits_key]) else Options.none,
                reroll_wounds=
                options[Options.reroll_wounds_key] if len(options[Options.reroll_wounds_key]) else Options.none,
                dakka3=
                int(options[Options.dakka3_key]) if len(options[Options.dakka3_key]) else Options.none,
                auto_wounds_on=
                int(options[Options.auto_wounds_on_key]) if len(options[Options.auto_wounds_on_key]) else Options.none,
                is_blast=bool(options[Options.is_blast_key]) if len(options[Options.is_blast_key]) else False,
                auto_hit=bool(options[Options.auto_hit_key]) if len(options[Options.auto_hit_key]) else False,
                wounds_by_2D6=
                bool(options[Options.wounds_by_2D6_key]) if len(options[Options.wounds_by_2D6_key]) else False,
                reroll_damages=
                bool(options[Options.reroll_damages_key]) if len(options[Options.reroll_damages_key]) else False,
                roll_damages_twice=
                bool(options[Options.roll_damages_twice_key]) if len(
                    options[Options.roll_damages_twice_key]) else False,
                snipe=
                Options.parse_snipe(options[Options.snipe_key]) if len(options[Options.snipe_key]) else Options.none
            )

    @staticmethod
    def parse_snipe(v):
        roll_type, threshold, n_mortals = v.split(",")
        assert (roll_type in {Options.wound, Options.strength})
        threshold = int(threshold)
        require(threshold > 0, f"Threshold input for option '{Options.opt_key_to_repr[Options.snipe_key]}' must be > 0")
        n_mortals = parse_dice_expr(n_mortals, raise_on_failure=True)
        return {
            Options.snipe_roll_type: roll_type,  # *D3* mortals
            Options.snipe_threshold: threshold,  # on *"wound_roll"*
            Options.snipe_n_mortals: n_mortals  # of *5*+
        }


class Profile:
    allowed_points_expr_chars = set("0123456789/*-+() ")

    def __init__(self, weapons, points_expr):
        assert (isinstance(weapons, list))
        require(len(weapons) > 0, "An attacking profile must have at least one declared weapon")
        assert (isinstance(weapons[0], Weapon))
        self.weapons = weapons
        points_expr_chars = {c for c in points_expr}
        invalid_chars = points_expr_chars - Profile.allowed_points_expr_chars
        require(not len(invalid_chars), f"Invalid characters found in points expression: {invalid_chars}")
        try:
            points_expr_evaluated = eval(points_expr)  # safe eval: contains only arithmetic characters
        except Exception as e:
            raise RequirementError(f"Invalid arithmetic expression for points '{points_expr}': {e}")
        try:
            self.points = int(points_expr_evaluated)
        except ValueError:
            self.points = None
        except Exception as e:
            raise e
        require(self.points is not None and self.points > 0, f"Invalid points value: '{points_expr}'")


class Weapon:
    at_least_one_blast_weapon = False

    def __init__(self, hit="4", a="1", s="4", ap="0", d="1", options=Options.empty()):
        # prob by roll result: O(n*dice_type)
        self.hit = parse_dice_expr(hit, complexity_threshold=float("inf"),
                                   raise_on_failure=True)  # only one time O(n*dice_type)
        require(self.hit.dices_type is None, "Random Ballistic/Weapon Skill is not allowed")
        require(2 <= self.hit.n <= 6, f"Ballistic/Weapon Skill must be between 2 and 6 (included), not '{self.hit}'")
        self.a = parse_dice_expr(a, complexity_threshold=128, raise_on_failure=True)  # only one time 0(n)
        require(self.a.avg != 0, "Number of Attacks cannot be 0")

        self.ap = parse_dice_expr(ap, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.d = parse_dice_expr(d, complexity_threshold=12, raise_on_failure=True)  # exponential exponential compl
        require(self.d.avg != 0, "Damage cannot be 0")
        self.options = Options.parse(options)
        require(not self.options.is_blast or self.a.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.is_blast_key]}' "
                f"option with a non random attack characteristic: {self.a}")
        require(not self.options.reroll_damages or self.d.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.reroll_damages_key]}' "
                f"option with a non random Damage characteristic: {self.d}")
        require(not self.options.roll_damages_twice or self.d.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.roll_damages_twice_key]}' "
                f"option with a non random Damage characteristic: {self.d}")
        self.s = parse_dice_expr(s,
                                 complexity_threshold=12,
                                 raise_on_failure=True,
                                 allow_star=self.options.wounds_by_2D6)  # per each target O(n*dice_type)
        require(
            self.options.snipe is None or self.options.snipe[Options.snipe_roll_type] != Options.strength or
            self.s.dices_type is not None,
            lambda: f"""Cannot activate '{Options.opt_key_to_repr[Options.snipe_key]}': The {self.options.snipe[Options.snipe_roll_type]} roll '{self.s}' is not random."""
        )
        require(
            self.options.snipe is None or self.options.snipe[Options.snipe_threshold] <=
            {Options.strength: self.s.max, Options.wound: 6 + self.options.wound_modifier}[
                self.options.snipe[Options.snipe_roll_type]],
            lambda: f"""Cannot activate '{Options.opt_key_to_repr[Options.snipe_key]}': A {self.options.snipe[Options.snipe_roll_type]} roll of {self.options.snipe[Options.snipe_threshold]}+ is impossible"""
        )

        require(self.s.avg != 0, "Strength cannot be 0")
        if self.options.is_blast:
            Weapon.at_least_one_blast_weapon = True


class Target:
    def __init__(self, t=4, sv=6, invu=None, fnp=None, w=1, n_models=1):
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
class CachesHolder:
    prob_by_roll_result_cache = {}
    success_ratios_cache = {}
    n_attacks_cache = {}
    hit_ratios_cache = {}
    wound_ratios_cache = {}
    unsaved_wound_ratios_cache = {}
    slained_figs_percent_per_unsaved_wound_cache = {}


def get_prob_by_roll_result(dice_expr, reroll_if_less_than=0, roll_twice=False):
    """
    :param reroll_if_less_than: dictates the reroll (reroll all dices) policy, 0 means a reroll never occurs
    """
    assert (reroll_if_less_than >= 0)
    assert (reroll_if_less_than == 0 or not roll_twice)
    key = f"{dice_expr},{reroll_if_less_than},{roll_twice},"
    prob_by_roll_result = CachesHolder.prob_by_roll_result_cache.get(key, None)
    if prob_by_roll_result is None:
        if dice_expr.dices_type is None:
            prob_by_roll_result = {dice_expr.n: 1}
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
            prob_by_roll_result = {k: v / n_cases for k, v in roll_results_counts.items()}
            if reroll_if_less_than > 0:
                prob_by_roll_result_items = prob_by_roll_result.items()
                prob_by_roll_result = {k: (0 if k < reroll_if_less_than else v) for k, v in prob_by_roll_result_items}
                # reach depth 2 nodes (reroll) participations
                for roll, prob_roll in prob_by_roll_result_items:
                    if roll < reroll_if_less_than:
                        for r, prob_r in prob_by_roll_result_items:
                            prob_by_roll_result[r] += prob_roll * prob_r

            elif roll_twice:

                prob_by_roll_result_items = prob_by_roll_result.items()
                prob_by_roll_result = {k: 0 for k, v in prob_by_roll_result_items}
                # reach depth 2 nodes (reroll) participations
                for r1, prob_r1 in prob_by_roll_result_items:
                    for r2, prob_r2 in prob_by_roll_result_items:
                        prob_by_roll_result[max(r1, r2)] += prob_r1 * prob_r2

        CachesHolder.prob_by_roll_result_cache[key] = prob_by_roll_result
        assert (float_eq(sum(prob_by_roll_result.values()), 1))
    return prob_by_roll_result


def get_n_attacks(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    key = f"{weapon.a},"
    if weapon.options.is_blast and target.n_models > 5:
        key += f"{target.n_models},"

    n_attacks = CachesHolder.n_attacks_cache.get(key, None)
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

        CachesHolder.n_attacks_cache[key] = n_attacks
    return n_attacks


def get_success_ratio(modified_necessary_roll, auto_success_on_6=True, reroll=Options.none, dakka3=Options.none):
    assert (reroll in {Options.none, Options.ones, Options.onestwos, Options.full})
    assert (dakka3 in {Options.none, 5, 6})
    key = f"{modified_necessary_roll},{auto_success_on_6},{reroll},{dakka3},"
    success_ratio = CachesHolder.success_ratios_cache.get(key, None)
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

        CachesHolder.success_ratios_cache[key] = success_ratio

    return success_ratio


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


def get_hit_ratio(weapon):
    assert (isinstance(weapon, Weapon))
    key = f"{weapon.hit}," \
          f"{weapon.options.hit_modifier}," \
          f"{weapon.options.reroll_hits}," \
          f"{weapon.options.dakka3}," \
          f"{weapon.options.auto_hit},"
    hit_ratio = CachesHolder.hit_ratios_cache.get(key, None)
    if hit_ratio is None:
        if weapon.options.auto_hit:
            hit_ratio = 1
        else:
            hit_ratio = get_success_ratio(weapon.hit.avg - weapon.options.hit_modifier,
                                          reroll=weapon.options.reroll_hits,
                                          dakka3=weapon.options.dakka3)

        CachesHolder.hit_ratios_cache[key] = hit_ratio

    return hit_ratio


def get_wound_ratio(weapon, target):
    """
    Random strength value is resolved once per weapon:
    "Each time this unit is chosen to shoot with, roll once to
    determine the Strength characteristic of this weapon."
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = f"{weapon.s}," \
          f"{weapon.options.wound_modifier}," \
          f"{weapon.options.reroll_wounds}," \
          f"{weapon.options.wounds_by_2D6}," \
          f"{target.t},"

    if weapon.options.auto_wounds_on:
        key += f"{weapon.options.auto_wounds_on},{weapon.hit},{weapon.options.hit_modifier},"

    wound_ratio = CachesHolder.wound_ratios_cache.get(key, None)
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
                    # modified
                    necessary_roll_to_hit = weapon.hit.avg - weapon.options.hit_modifier
                    # unmodified
                    auto_wounds_necessary_hit_roll = weapon.options.auto_wounds_on
                    auto_wounding_hit_rolls_ratio = min(
                        1,
                        (7 - auto_wounds_necessary_hit_roll) / (7 - necessary_roll_to_hit)
                    )
                    wound_ratio += prob_s_roll * \
                                   ((1 - auto_wounding_hit_rolls_ratio) * success_ratio + auto_wounding_hit_rolls_ratio)

        CachesHolder.wound_ratios_cache[key] = wound_ratio

    return wound_ratio


def get_unsaved_wound_ratio(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = f"{weapon.ap}," \
          f"{target.sv}," \
          f"{target.invu}," \
          f"{weapon.options.save_modifier},"

    unsaved_wound_ratio = CachesHolder.unsaved_wound_ratios_cache.get(key, None)
    if unsaved_wound_ratio is None:
        unsaved_wound_ratio = 0
        for ap_roll, prob_ap_roll in get_prob_by_roll_result(weapon.ap).items():
            save_roll = target.sv + ap_roll
            if target.invu is not None:
                save_roll = min(save_roll, target.invu)
            save_fail_ratio = 1 - get_success_ratio(save_roll - weapon.options.save_modifier, auto_success_on_6=False)
            unsaved_wound_ratio += save_fail_ratio * prob_ap_roll
        CachesHolder.unsaved_wound_ratios_cache[key] = unsaved_wound_ratio

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
    roll_damages_twice = None
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


def get_slained_figs_percent(state_):
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
        downstream = get_slained_figs_percent(state)
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
                        prob_by_roll_result_list = [
                            get_prob_by_roll_result(State.weapon_d,
                                                    reroll_if_less_than=roll,
                                                    roll_twice=State.roll_damages_twice)
                            for roll in range(State.weapon_d.min, State.weapon_d.max)
                        ]
                    else:
                        prob_by_roll_result_list = \
                            [get_prob_by_roll_result(State.weapon_d, roll_twice=State.roll_damages_twice)]

                    downstream = max([
                        sum([
                            prob_d *
                            get_slained_figs_percent(State(n_unsaved_wounds_left=state.n_unsaved_wounds_left - 1,
                                                           current_wound_n_damages_left=d,
                                                           n_figs_slained_so_far=state.n_figs_slained_so_far,
                                                           remaining_target_wounds=state.remaining_target_wounds))
                            for d, prob_d in prob_by_roll_result.items()
                        ])
                        for prob_by_roll_result in prob_by_roll_result_list])

                    State.cache.add(state, downstream)
                    return downstream
            else:
                # FNP fail
                f = get_slained_figs_percent(State(state.n_unsaved_wounds_left,
                                                   state.current_wound_n_damages_left - 1,
                                                   state.n_figs_slained_so_far,
                                                   state.remaining_target_wounds - 1))

                # FNP success
                if State.fnp_fail_ratio != 1:
                    s = get_slained_figs_percent(State(state.n_unsaved_wounds_left,
                                                       state.current_wound_n_damages_left - 1,
                                                       state.n_figs_slained_so_far,
                                                       state.remaining_target_wounds))
                    downstream = (1 - State.fnp_fail_ratio) * s + State.fnp_fail_ratio * f
                else:
                    downstream = f
                State.cache.add(state, downstream)
                return downstream


def get_slained_figs_percent_per_unsaved_wound(weapon, target):
    """
    n_unsaved_wounds_init=32: 14 sec, res prec +-0.02 compared to 64
    n_unsaved_wounds_init=10:  5 sec, res prec +-0.1  compared to 64
    """
    key = f"{weapon.d}," \
          f"{target.fnp}," \
          f"{target.w}," \
          f"{weapon.options.reroll_damages}," \
          f"{weapon.options.roll_damages_twice},"

    slained_figs_percent_per_unsaved_wound = CachesHolder.slained_figs_percent_per_unsaved_wound_cache.get(key, None)

    if slained_figs_percent_per_unsaved_wound is None:
        assert (isinstance(weapon, Weapon))
        assert (isinstance(target, Target))
        if weapon.d.dices_type is None and target.fnp is None:
            slained_figs_percent_per_unsaved_wound = exact_avg_figs_fraction_slained_per_unsaved_wound(d=weapon.d.n,
                                                                                                       w=target.w)
        else:
            State.reroll_damages = weapon.options.reroll_damages
            State.weapon_d = weapon.d
            State.target_wounds = target.w
            State.n_unsaved_wounds_init = 16
            State.n_figs_slained_weighted_ratios = []
            State.fnp_fail_ratio = 1 if target.fnp is None else 1 - get_success_ratio(target.fnp)
            State.start_target_wounds = target.w
            State.roll_damages_twice = weapon.options.roll_damages_twice
            State.cache.reset()

            slained_figs_percent_per_unsaved_wound = get_slained_figs_percent(State(
                n_unsaved_wounds_left=State.n_unsaved_wounds_init,
                current_wound_n_damages_left=0,
                n_figs_slained_so_far=0,
                remaining_target_wounds=target.w)) / State.n_unsaved_wounds_init

        CachesHolder.slained_figs_percent_per_unsaved_wound_cache[key] = slained_figs_percent_per_unsaved_wound

    return slained_figs_percent_per_unsaved_wound


def exact_avg_figs_fraction_slained_per_unsaved_wound(d, w):
    return 1 / math.ceil(w / d)


def score_weapon_on_target(w, t, avg_n_attacks, hit_ratio):
    """
    avg_figs_fraction_slained by point
    """
    avg_n_attacks = w.a.avg if avg_n_attacks is None else avg_n_attacks
    hit_ratio = get_hit_ratio(w) if hit_ratio is None else hit_ratio
    return avg_n_attacks * hit_ratio * get_wound_ratio(w, t) * get_unsaved_wound_ratio(w, t) \
           * get_slained_figs_percent_per_unsaved_wound(w, t)


def scores_to_z(score_a, score_b):
    """
    z is in ]-1, 1[
    :return z rounded in [-1, 1]
    """
    if score_a > score_b:
        z = + (1 - score_b / score_a)
    else:
        z = - (1 - score_a / score_b)
    return round(z, 4)  # round(z, 3) 91.8kB


def construct_y_label(l):
    show_n_models = Weapon.at_least_one_blast_weapon
    if show_n_models:
        n_models_label = f"unit size:{get_n_models_label(l[3])}, "
    else:
        n_models_label = ""
    return f"""{n_models_label}FNP:{"-" if l[2] is None else f"{l[2]}+"}, T:{l[0]}, W:{l[1]}"""


def construct_x_label(l):
    return f"""Sv:{"-" if l[0] is None else f"{l[0]}+"}, Invu:{"-" if l[1] is None else f"{l[1]}+"}"""


def scores_to_ratio(score_a, score_b):
    if score_a > score_b:
        return round(score_a / score_b, 4)
    else:
        return round(score_b / score_a, 4)


def get_n_models_label(n_models):
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
    ts_ws_fnps_nm.sort(key=lambda e: (- e[3], e[2], - e[0], - e[1]))
    ts_ws_fnps_nm = list(map(lambda l: [l[0], l[1], map_7_to_None(l[2]), l[3]], ts_ws_fnps_nm))

    res["y"] = list(map(construct_y_label, ts_ws_fnps_nm))

    svs = []
    for invu in [2, 3, 4, 5, 6, 7]:
        for sv in range(1, min(invu + 1, 6 + 1)):
            svs.append((sv, invu))
    svs.sort(key=lambda e: (-e[0], -e[1]))
    svs = list(map(lambda l: list(map(map_7_to_None, l)), svs))

    res["x"] = list(map(construct_x_label, svs))

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
                        get_hit_ratio(weapon_a)
                    ) for weapon_a in profile_a.weapons],
                    [score_weapon_on_target(
                        weapon_b,
                        target,
                        get_n_attacks(weapon_b, target),
                        get_hit_ratio(weapon_b)
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

    zs = [[scores_to_z(score_a, score_b) for score_a, score_b in line] for line in score_a_score_b_tuples]

    res["z"] = apply_mask_matrix(
        matrix=zs,
        mask_matrix=targets_matrix,
        predicate_on_mask_matrix=lambda target: 14 > target.t + target.sv > 5
    )

    res["ratios"] = [[scores_to_ratio(score_a, score_b) for score_a, score_b in line] for line in
                     score_a_score_b_tuples]

    if is_dev_execution():
        print(f"caches stats:")
        print(f"\tsize of prob_by_roll_result_cache={len(CachesHolder.prob_by_roll_result_cache)}")
        print(f"\tsize of success_ratios_cache={len(CachesHolder.success_ratios_cache)}")
        print(f"\tsize of n_attacks_cache={len(CachesHolder.n_attacks_cache)}")
        print(f"\tsize of hit_ratios_cache={len(CachesHolder.hit_ratios_cache)}")
        print(f"\tsize of wound_ratios_cache={len(CachesHolder.wound_ratios_cache)}")
        print(f"\tsize of unsaved_wound_ratios_cache={len(CachesHolder.unsaved_wound_ratios_cache)}")
        print(f"\tsize of slained_figs_percent_per_unsaved_wound_cache="
              f"{len(CachesHolder.slained_figs_percent_per_unsaved_wound_cache)}")

    return res
