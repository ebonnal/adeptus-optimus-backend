from .models import *
from math import ceil
import numpy as np


# Function runtime caches:
class UnfillableDict(dict):
    def __setitem__(self, key, value):
        pass


class Caches:
    prob_by_roll_result_cache = None
    success_ratios_cache = None
    n_attacks_cache = None
    hit_ratios_cache = None
    wound_ratios_cache = None
    unsaved_wound_ratios_cache = None
    slained_figs_percent_per_unsaved_wound_cache = None
    prob_mortals_after_wound_cache = None

    @staticmethod
    def reset_and_enable():
        Caches.prob_by_roll_result_cache = {}
        Caches.success_ratios_cache = {}
        Caches.n_attacks_cache = {}
        Caches.hit_ratios_cache = {}
        Caches.wound_ratios_cache = {}
        Caches.unsaved_wound_ratios_cache = {}
        Caches.slained_figs_percent_per_unsaved_wound_cache = {}
        Caches.prob_mortals_after_wound_cache = {}

    @staticmethod
    def disable():
        Caches.prob_by_roll_result_cache = UnfillableDict()
        Caches.success_ratios_cache = UnfillableDict()
        Caches.n_attacks_cache = UnfillableDict()
        Caches.hit_ratios_cache = UnfillableDict()
        Caches.wound_ratios_cache = UnfillableDict()
        Caches.unsaved_wound_ratios_cache = UnfillableDict()
        Caches.slained_figs_percent_per_unsaved_wound_cache = UnfillableDict()
        Caches.prob_mortals_after_wound_cache = UnfillableDict()


Caches.reset_and_enable()


def get_prob_by_roll_result(dice_expr, reroll_if_less_than=0, roll_twice=False):
    """
    :param reroll_if_less_than: dictates the reroll (reroll all dices) policy, 0 means a reroll never occurs
    """
    assert (reroll_if_less_than >= 0)
    assert (reroll_if_less_than == 0 or not roll_twice)
    key = f"{dice_expr},{reroll_if_less_than},{roll_twice},"
    prob_by_roll_result = Caches.prob_by_roll_result_cache.get(key, None)
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

        Caches.prob_by_roll_result_cache[key] = prob_by_roll_result
        if is_dev_execution():
            assert_float_eq(sum(prob_by_roll_result.values()), 1)
    return prob_by_roll_result


def get_n_attacks_key(weapon, target):
    key = f"{weapon.a},"
    if weapon.options.is_blast and target.n_models > 5:
        key += f"{target.n_models},"
    return key


def get_n_attacks(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = get_n_attacks_key(weapon, target)

    n_attacks = Caches.n_attacks_cache.get(key, None)
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

        Caches.n_attacks_cache[key] = n_attacks
    return n_attacks


def apply_roll_border_cases(modified_necessary_roll, auto_success_on_6):
    if modified_necessary_roll <= 1:
        modified_necessary_roll = 2  # roll of 1 always fails
    if modified_necessary_roll >= 7:
        if auto_success_on_6:
            modified_necessary_roll = 6  # roll of 6 always succeeds
        else:
            modified_necessary_roll = 7
    return modified_necessary_roll


def get_success_ratio(necessary_roll,
                      modifier,
                      auto_success_on_6=True,
                      reroll=Options.none,
                      dakka3=Options.none,
                      explodes=Options.none,
                      ):
    """
    reroll and dakka3 are applied regarding unmodified rolls.
    """
    assert (reroll in {Options.none, Options.ones, Options.onestwos, Options.full})
    assert (dakka3 in {Options.none, 5, 6})
    assert (type(modifier) is int)
    key = f"{necessary_roll},{modifier},{auto_success_on_6},{reroll},{dakka3},{explodes}"
    success_ratio = Caches.success_ratios_cache.get(key, None)
    if success_ratio is None:
        modified_necessary_roll = apply_roll_border_cases(necessary_roll - modifier, auto_success_on_6)

        success_ratio = _visit_rolls_tree(
            reroll_consumed=reroll == Options.none,
            dakka3_consumed=dakka3 == Options.none,
            modified_necessary_roll=modified_necessary_roll,
            reroll=reroll,
            dakka3=dakka3,
            explodes=explodes
        )

        Caches.success_ratios_cache[key] = success_ratio

    return success_ratio


def _visit_rolls_tree(reroll_consumed, dakka3_consumed, modified_necessary_roll, reroll, dakka3, explodes):
    successes_ratio = 0
    # unmodified dice roll possible events: 1, 2, 3, 4, 5, 6
    for i in range(1, 7):
        # explodes
        if explodes != Options.none and i >= explodes:
            successes_ratio += 1 / 6
        # succeeds or reroll
        if i >= modified_necessary_roll:
            successes_ratio += 1 / 6
        elif not reroll_consumed and reroll != Options.none:
            if reroll == Options.ones and i == 1:
                successes_ratio += 1 / 6 * _visit_rolls_tree(True, dakka3_consumed, modified_necessary_roll, reroll,
                                                             dakka3, explodes)
            elif reroll == Options.onestwos and i <= 2:
                successes_ratio += 1 / 6 * _visit_rolls_tree(True, dakka3_consumed, modified_necessary_roll, reroll,
                                                             dakka3, explodes)
            elif reroll == Options.full:
                successes_ratio += 1 / 6 * _visit_rolls_tree(True, dakka3_consumed, modified_necessary_roll, reroll,
                                                             dakka3, explodes)
        # dakka3
        if not dakka3_consumed and dakka3 != Options.none and i >= dakka3:
            # dakka result in a new dice roll that may be rerolled
            successes_ratio += 1 / 6 * _visit_rolls_tree(reroll == Options.none, True, modified_necessary_roll, reroll,
                                                         dakka3, explodes)
    return successes_ratio


def get_hit_ratio_key(weapon):
    return f"{weapon.hit}," \
           f"{weapon.options.hit_modifier}," \
           f"{weapon.options.reroll_hits}," \
           f"{weapon.options.dakka3}," \
           f"{weapon.options.auto_hit}," \
           f"{weapon.options.hit_explodes}," \
           f"{weapon.options.auto_wounds_on}"


def get_hit_ratio(weapon):
    assert (isinstance(weapon, Weapon))
    key = get_hit_ratio_key(weapon)
    hit_ratio = Caches.hit_ratios_cache.get(key, None)
    if hit_ratio is None:
        if weapon.options.auto_hit:
            hit_ratio = 1
        else:
            necessary_roll = weapon.hit.avg
            modifier = weapon.options.hit_modifier
            if weapon.options.is_activated(Options.auto_wounds_on_key):
                if weapon.hit.avg - modifier >= weapon.options.auto_wounds_on:
                    # always hit on roll >= auto_wounds_on
                    necessary_roll = weapon.options.auto_wounds_on
                    modifier = 0

            hit_ratio = get_success_ratio(necessary_roll,
                                          modifier,
                                          reroll=weapon.options.reroll_hits,
                                          dakka3=weapon.options.dakka3,
                                          explodes=weapon.options.hit_explodes)

        Caches.hit_ratios_cache[key] = hit_ratio

    return hit_ratio


def get_wound_ratio_key(weapon, target):
    key = f"{weapon.s}," \
          f"{weapon.options.wound_modifier}," \
          f"{weapon.options.reroll_wounds}," \
          f"{weapon.options.wounds_by_2D6}," \
          f"{target.t},"

    if weapon.options.auto_wounds_on:
        key += f"{weapon.options.auto_wounds_on},{weapon.hit},{weapon.options.hit_modifier},"

    return key


def get_wound_ratio(weapon, target):
    """
    Random strength value is resolved once per weapon:
    "Each time this unit is chosen to shoot with, roll once to
    determine the Strength characteristic of this weapon."
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = get_wound_ratio_key(weapon, target)

    wound_ratio = Caches.wound_ratios_cache.get(key, None)
    if wound_ratio is None:
        wound_ratio = 0
        if weapon.options.wounds_by_2D6:
            for roll, prob_roll in get_prob_by_roll_result(DiceExpr(2, 6)).items():
                if roll >= target.t:
                    wound_ratio += prob_roll
        else:
            for s_roll, prob_s_roll in get_prob_by_roll_result(weapon.s).items():
                success_ratio = get_success_ratio(
                    compute_necessary_wound_roll(s_roll, target.t),
                    weapon.options.wound_modifier,
                    reroll=weapon.options.reroll_wounds
                )
                if not weapon.options.is_activated(Options.auto_wounds_on_key):
                    wound_ratio += success_ratio * prob_s_roll
                else:
                    # modified
                    necessary_roll_to_hit = weapon.hit.avg - weapon.options.hit_modifier
                    # unmodified
                    auto_wounds_necessary_hit_roll = weapon.options.auto_wounds_on
                    auto_wounding_hit_rolls_ratio = min(
                        1, (7 - auto_wounds_necessary_hit_roll) / (7 - necessary_roll_to_hit)
                    )

                    wound_ratio += prob_s_roll * \
                                   ((1 - auto_wounding_hit_rolls_ratio) * success_ratio + auto_wounding_hit_rolls_ratio)

        Caches.wound_ratios_cache[key] = wound_ratio

    return wound_ratio


def get_unsaved_wound_ratio_key(weapon, target):
    return f"{weapon.ap}," \
           f"{target.sv}," \
           f"{target.invu}," \
           f"{weapon.options.save_modifier},"


def get_unsaved_wound_ratio(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = get_unsaved_wound_ratio_key(weapon, target)

    unsaved_wound_ratio = Caches.unsaved_wound_ratios_cache.get(key, None)
    if unsaved_wound_ratio is None:
        unsaved_wound_ratio = 0
        for ap_roll, prob_ap_roll in get_prob_by_roll_result(weapon.ap).items():
            save_roll = target.sv + ap_roll
            if target.invu is not None:
                save_roll = min(save_roll, target.invu)
            save_fail_ratio = 1 - get_success_ratio(save_roll,
                                                    weapon.options.save_modifier,
                                                    auto_success_on_6=False)
            unsaved_wound_ratio += save_fail_ratio * prob_ap_roll
        Caches.unsaved_wound_ratios_cache[key] = unsaved_wound_ratio

    return unsaved_wound_ratio


def get_prob_snipe_mortals_after_wound_key(weapon, target):
    if not weapon.options.is_activated(Options.snipe_key):
        key = ""
    else:
        key = f"{weapon.options.snipe[Options.snipe_roll_type]}," \
              f"{weapon.options.snipe[Options.snipe_threshold]},"
        if weapon.options.snipe[Options.snipe_roll_type] == Options.wound:
            key += f"{weapon.s},{weapon.options.wound_modifier},{target.t},"
        else:
            key += f"{weapon.s},"
    return key


def get_prob_snipe_mortals_after_wound(weapon, target):
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))

    key = get_prob_snipe_mortals_after_wound_key(weapon, target)

    prob_mortals_after_wound = Caches.prob_mortals_after_wound_cache.get(key, None)
    if prob_mortals_after_wound is None:
        if not weapon.options.is_activated(Options.snipe_key):
            prob_mortals_after_wound = 0
        else:
            assert (not weapon.options.is_activated(Options.wounds_by_2D6_key))
            assert (not weapon.options.is_activated(Options.auto_wounds_on_key))
            if weapon.options.snipe[Options.snipe_roll_type] == Options.wound:
                prob_mortals_after_wound = 0
                for s_roll, prob_s_roll in get_prob_by_roll_result(weapon.s).items():
                    # modified
                    modified_necessary_roll_to_wound = apply_roll_border_cases(
                        compute_necessary_wound_roll(s_roll, target.t) - weapon.options.wound_modifier,
                        auto_success_on_6=True
                    )
                    # modified
                    snipe_modified_necessary_wound_roll = apply_roll_border_cases(
                        weapon.options.snipe[Options.snipe_threshold] - weapon.options.wound_modifier,
                        auto_success_on_6=False  # snipe on 6+ is desactivated by a modifier of -1
                    )
                    prob_mortals_after_wound += prob_s_roll * min(
                        1.0,
                        (7 - snipe_modified_necessary_wound_roll) / (7 - modified_necessary_roll_to_wound)
                    )
            else:
                prob_mortals_after_wound = 0
                for s_roll, prob_s_roll in get_prob_by_roll_result(weapon.s).items():
                    if s_roll >= weapon.options.snipe[Options.snipe_threshold]:
                        prob_mortals_after_wound += prob_s_roll

        Caches.prob_mortals_after_wound_cache[key] = prob_mortals_after_wound

    return prob_mortals_after_wound


class DmgAllocCache:
    def __init__(self):
        self.dict = {}
        self.hits = 0
        self.tries = 0

    def __str__(self):
        return f"tries={self.tries}, hits={self.hits}, misses={self.tries - self.hits}"

    def add(self, state, cached_unweighted_downstream):
        assert (isinstance(cached_unweighted_downstream, np.ndarray))
        assert (len(cached_unweighted_downstream.shape) == 1)
        assert (cached_unweighted_downstream.shape[0] == len(DmgAllocNode.vectorized_dims_comb))
        key = DmgAllocCache._keyify(state)
        if self.dict.get(key, (-1, None))[0] < state.n_wounds_left:  # -1: massive optim fix
            self.dict[key] = (state.n_wounds_left, cached_unweighted_downstream)

    def get(self, state):
        res = self.dict.get(DmgAllocCache._keyify(state), None)
        self.tries += 1
        if res is not None:
            self.hits += 1
            return res
        else:
            return None, None

    def reset(self):
        del self.dict
        self.dict = {}
        self.hits = 0
        self.tries = 0

    @staticmethod
    def _keyify(state):
        return f"{state.current_wound_n_damages_left}," \
               f"{state.remaining_target_wounds}," \
               f"{state.n_wounds_left}," \
               f"{state.current_wound_damages_are_mortal}"


class HeatMapConsts:
    w_vector = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 25]
    ts_dim = [
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

    sv_invu_dim = [(sv, invu) for invu in [2, 3, 4, 5, 6, 7] for sv in range(1, min(invu + 1, 6 + 1))]
    sv_invu_dim.sort(key=lambda e: (-e[0], -e[1]))
    sv_invu_dim = list(map(lambda l: list(map(map_7_to_None, l)), sv_invu_dim))

    all_fnps_dim = [None, 6, 5]

    @staticmethod
    def get_ts_ws_fnps_nm_dim(at_least_one_blast_weapon):
        ts_ws_fnps_nm_dim = []

        for w, ts in zip(HeatMapConsts.w_vector, HeatMapConsts.ts_dim):
            fnps = [7] if w > float('inf') else [7, 6, 5]  # w>inf: 4 times slower than w>5

            if at_least_one_blast_weapon:
                if w > 3:
                    ns_models = [Options.n_models_1to5]
                elif w == 3:
                    ns_models = [Options.n_models_1to5, Options.n_models_6to10]
                else:
                    ns_models = [Options.n_models_1to5, Options.n_models_6to10, Options.n_models_11_plus]
            else:
                ns_models = [Options.n_models_1to5]

            for fnp in fnps:
                for t in ts:
                    for n_models in ns_models:
                        ts_ws_fnps_nm_dim.append((t, w, fnp, n_models))

            # n_models then fnp, then toughness, then wounds
        ts_ws_fnps_nm_dim.sort(key=lambda e: (- e[3], e[2], - e[0], - e[1]))
        ts_ws_fnps_nm_dim = list(map(
            lambda l: [l[0], l[1], map_7_to_None(l[2]), l[3]],
            ts_ws_fnps_nm_dim
        ))
        return ts_ws_fnps_nm_dim


class DmgAllocNode:
    vectorized_dims_comb = [
        (fnp, sv, invu)
        for fnp in HeatMapConsts.all_fnps_dim
        for sv, invu in HeatMapConsts.sv_invu_dim
    ]

    reroll_damages = None
    weapon_d = None
    weapon_options_snipe = None
    target_wounds = None
    start_target_wounds = None
    roll_damages_twice = None
    unsaved_wound_ratios = None
    prob_mortals_after_wound = None

    fnp_fail_ratios = np.fromiter(
        (1 - (get_success_ratio(fnp, modifier=0, auto_success_on_6=False) if fnp is not None else 0)
         for fnp, _, _ in vectorized_dims_comb),
        dtype=np.float64,
        count=len(vectorized_dims_comb))

    cache = DmgAllocCache()

    def __init__(self,
                 n_wounds_left,  # key field, 0 when resolved
                 current_wound_n_damages_left,  # key field, 0 when resolved
                 current_wound_damages_are_mortal,  # key field
                 n_figs_slained_so_far,  # value field
                 remaining_target_wounds,  # key field
                 ):
        self.n_wounds_left = n_wounds_left
        self.current_wound_n_damages_left = current_wound_n_damages_left
        self.current_wound_damages_are_mortal = current_wound_damages_are_mortal
        self.n_figs_slained_so_far = n_figs_slained_so_far
        self.remaining_target_wounds = remaining_target_wounds

    def copy(self):
        return DmgAllocNode(n_wounds_left=self.n_wounds_left,
                            current_wound_n_damages_left=self.current_wound_n_damages_left,
                            current_wound_damages_are_mortal=self.current_wound_damages_are_mortal,
                            n_figs_slained_so_far=self.n_figs_slained_so_far,
                            remaining_target_wounds=self.remaining_target_wounds)


def get_slained_figs_percent(state_):
    assert (isinstance(state_, DmgAllocNode))
    assert (state_.remaining_target_wounds >= 0)
    assert (state_.n_wounds_left >= 0)
    assert (state_.current_wound_n_damages_left >= 0)
    state = state_.copy()

    # resolve a model kill
    if state.remaining_target_wounds == 0:
        state.remaining_target_wounds = DmgAllocNode.target_wounds
        if not state.current_wound_damages_are_mortal:
            # additionnal damages are not propagated to other models
            state.current_wound_n_damages_left = 0
        downstream = get_slained_figs_percent(state)
        downstream = downstream + 1
        return downstream  # upstream propagation of figs slained count

    if state.current_wound_n_damages_left == 0 and state.n_wounds_left == 0 \
            and state.current_wound_damages_are_mortal in {None, True}:
        # leaf: no more damages to fnp no more wounds to consume or p(leaf) < threshold, no more potential mortals
        # portion of the last model injured

        last_model_injured_frac_vect = np.full(
            len(DmgAllocNode.vectorized_dims_comb),
            1 - state.remaining_target_wounds / DmgAllocNode.target_wounds,
            dtype=np.float64
        )
        DmgAllocNode.cache.add(state, last_model_injured_frac_vect)
        return last_model_injured_frac_vect
    else:
        # test cache
        # cached_downstream = None
        cached_res_n_wounds_left, cached_downstream = DmgAllocNode.cache.get(state)
        if cached_downstream is not None and cached_res_n_wounds_left >= state.n_wounds_left:
            # use cached res if deep enough
            return cached_downstream
        else:
            if state.current_wound_n_damages_left == 0:
                if cached_downstream is None or cached_res_n_wounds_left < state.n_wounds_left:
                    if state.current_wound_damages_are_mortal is None or state.current_wound_damages_are_mortal:
                        # consume a wound
                        # no mortals triggered after last wound consumption or just finished to alloc them

                        if DmgAllocNode.reroll_damages:
                            prob_by_roll_result_list = [
                                get_prob_by_roll_result(DmgAllocNode.weapon_d,
                                                        reroll_if_less_than=roll,
                                                        roll_twice=DmgAllocNode.roll_damages_twice)
                                for roll in range(DmgAllocNode.weapon_d.min, DmgAllocNode.weapon_d.max)
                            ]
                        else:
                            prob_by_roll_result_list = \
                                [get_prob_by_roll_result(DmgAllocNode.weapon_d,
                                                         roll_twice=DmgAllocNode.roll_damages_twice)]

                        downstream_saved_wound = get_slained_figs_percent(
                            DmgAllocNode(n_wounds_left=state.n_wounds_left - 1,
                                         current_wound_n_damages_left=state.current_wound_n_damages_left,
                                         # triggers attempt for mortals after consuming this saved wound
                                         current_wound_damages_are_mortal=False,
                                         n_figs_slained_so_far=state.n_figs_slained_so_far,
                                         remaining_target_wounds=state.remaining_target_wounds))

                        downstream_unsaved_wound = np.array(
                            [np.array(
                                [prob_d *
                                 get_slained_figs_percent(
                                     DmgAllocNode(n_wounds_left=state.n_wounds_left - 1,
                                                  current_wound_n_damages_left=d,
                                                  current_wound_damages_are_mortal=False,
                                                  n_figs_slained_so_far=state.n_figs_slained_so_far,
                                                  remaining_target_wounds=state.remaining_target_wounds))
                                 for d, prob_d in prob_by_roll_result.items()]
                            ).sum(axis=0)
                             for prob_by_roll_result in prob_by_roll_result_list]
                        ).max(axis=0)

                        downstream = DmgAllocNode.unsaved_wound_ratios * downstream_unsaved_wound + \
                                     (1 - DmgAllocNode.unsaved_wound_ratios) * downstream_saved_wound
                    else:
                        # potential mortal: it does never consume a wound
                        downstream_no_mortals = get_slained_figs_percent(
                            DmgAllocNode(n_wounds_left=state.n_wounds_left,
                                         current_wound_n_damages_left=state.current_wound_n_damages_left,
                                         # signals to next node that a new wound has to be consumed
                                         current_wound_damages_are_mortal=None,
                                         n_figs_slained_so_far=state.n_figs_slained_so_far,
                                         remaining_target_wounds=state.remaining_target_wounds))
                        if DmgAllocNode.weapon_options_snipe != Options.none:
                            prob_by_roll_result = \
                                get_prob_by_roll_result(DmgAllocNode.weapon_options_snipe[Options.snipe_n_mortals])
                            downstream_with_mortals = np.array([
                                prob_n_mortals *
                                get_slained_figs_percent(
                                    DmgAllocNode(n_wounds_left=state.n_wounds_left,
                                                 current_wound_n_damages_left=n_mortals,
                                                 current_wound_damages_are_mortal=True,
                                                 n_figs_slained_so_far=state.n_figs_slained_so_far,
                                                 remaining_target_wounds=state.remaining_target_wounds))
                                for n_mortals, prob_n_mortals in prob_by_roll_result.items()

                            ], dtype=np.float64).sum(axis=0)

                            downstream = DmgAllocNode.prob_mortals_after_wound * downstream_with_mortals + \
                                         (1 - DmgAllocNode.prob_mortals_after_wound) * downstream_no_mortals
                        else:
                            downstream = downstream_no_mortals

                    DmgAllocNode.cache.add(state, downstream)
                    return downstream
            else:
                # consume damage

                # FNP fail
                f = get_slained_figs_percent(
                    DmgAllocNode(n_wounds_left=state.n_wounds_left,
                                 current_wound_n_damages_left=state.current_wound_n_damages_left - 1,
                                 current_wound_damages_are_mortal=state.current_wound_damages_are_mortal,
                                 n_figs_slained_so_far=state.n_figs_slained_so_far,
                                 remaining_target_wounds=state.remaining_target_wounds - 1))

                # FNP success
                s = get_slained_figs_percent(
                    DmgAllocNode(n_wounds_left=state.n_wounds_left,
                                 current_wound_n_damages_left=state.current_wound_n_damages_left - 1,
                                 current_wound_damages_are_mortal=state.current_wound_damages_are_mortal,
                                 n_figs_slained_so_far=state.n_figs_slained_so_far,
                                 remaining_target_wounds=state.remaining_target_wounds))
                downstream = (1 - DmgAllocNode.fnp_fail_ratios) * s + DmgAllocNode.fnp_fail_ratios * f

                DmgAllocNode.cache.add(state, downstream)
                return downstream


def get_slained_figs_percent_per_unsaved_wound_key(weapon, target):
    key = f"{weapon.d}," \
          f"{target.fnp}," \
          f"{target.w}," \
          f"{weapon.options.reroll_damages}," \
          f"{weapon.options.roll_damages_twice}," \
          f"{get_unsaved_wound_ratio(weapon, target)},"

    if weapon.options.is_activated(Options.snipe_key):
        key += f"{weapon.options.snipe}," \
               f"{get_prob_snipe_mortals_after_wound_key(weapon, target)}"
    return key


def get_slained_figs_percent_per_unsaved_wound(weapon, target, exact_optim=False):
    """
    n_unsaved_wounds_init=32: 14 sec, res prec +-0.02 compared to 64
    n_unsaved_wounds_init=10:  5 sec, res prec +-0.1  compared to 64
    """
    assert (isinstance(weapon, Weapon))
    assert (isinstance(target, Target))
    if exact_optim and weapon.d.dices_type is None and target.fnp is None and not weapon.options.is_activated(
            Options.snipe_key):
        return exact_avg_figs_fraction_slained_per_unsaved_wound(d=weapon.d.n, w=target.w)
    else:
        key = get_slained_figs_percent_per_unsaved_wound_key(weapon, target)
        slained_figs_percent_per_unsaved_wound = Caches.slained_figs_percent_per_unsaved_wound_cache.get(key, None)
        if slained_figs_percent_per_unsaved_wound is None:

            DmgAllocNode.cache.reset()
            DmgAllocNode.n_wounds_init = 16
            DmgAllocNode.reroll_damages = weapon.options.reroll_damages
            DmgAllocNode.weapon_d = weapon.d
            DmgAllocNode.weapon_options_snipe = weapon.options.snipe
            DmgAllocNode.target_wounds = target.w
            DmgAllocNode.start_target_wounds = target.w
            DmgAllocNode.roll_damages_twice = weapon.options.roll_damages_twice
            DmgAllocNode.prob_mortals_after_wound = get_prob_snipe_mortals_after_wound(weapon, target)

            # TODO: only different unsaved_wound_ratios, not all the different (sv, invu) pairs
            DmgAllocNode.unsaved_wound_ratios = np.fromiter((
                # 1 if not weapon.options.is_activated(Options.snipe_key) else
                get_unsaved_wound_ratio(weapon, target.copy(sv=sv, invu=invu))
                for fnp, sv, invu in DmgAllocNode.vectorized_dims_comb),
                dtype=np.float64,
                count=len(DmgAllocNode.vectorized_dims_comb)
            )

            slained_figs_percents_per_unsaved_wound = get_slained_figs_percent(DmgAllocNode(
                n_wounds_left=DmgAllocNode.n_wounds_init,
                current_wound_n_damages_left=0,
                current_wound_damages_are_mortal=None,
                n_figs_slained_so_far=0,
                remaining_target_wounds=target.w)
            ) / (DmgAllocNode.n_wounds_init * DmgAllocNode.unsaved_wound_ratios)
            slained_figs_percent_per_unsaved_wound = slained_figs_percents_per_unsaved_wound[
                DmgAllocNode.vectorized_dims_comb.index((target.fnp, target.sv, target.invu))
            ]
            assert ((target.fnp, target.sv, target.invu) in DmgAllocNode.vectorized_dims_comb)
            for percent, (fnp, sv, invu) in zip(slained_figs_percents_per_unsaved_wound,
                                                DmgAllocNode.vectorized_dims_comb):
                t = target.copy(fnp=fnp, sv=sv, invu=invu)
                k = get_slained_figs_percent_per_unsaved_wound_key(weapon, t)
                if k not in Caches.slained_figs_percent_per_unsaved_wound_cache:
                    Caches.slained_figs_percent_per_unsaved_wound_cache[k] = percent

        return slained_figs_percent_per_unsaved_wound


def exact_avg_figs_fraction_slained_per_unsaved_wound(d, w):
    return 1 / ceil(w / d)


# TODO remove
def score_weapon_on_target(w, t, foo, bar):
    return _score_weapon_on_target(w, t)


def _score_weapon_on_target(w, t):
    """
    avg_figs_fraction_slained by point
    """
    avg_n_attacks = get_n_attacks(w, t)  # almost always in cache
    hit_ratio = get_hit_ratio(w)  # almost always in cache
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


def construct_y_label(l, show_n_models):
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

    at_least_one_blast_weapon = any(map(
        lambda weapon: weapon.options.is_activated(Options.is_blast_key),
        profile_a.weapons + profile_b.weapons
    ))

    ts_ws_fnps_nm = HeatMapConsts.get_ts_ws_fnps_nm_dim(at_least_one_blast_weapon)

    res = {
        "x": list(map(construct_x_label, HeatMapConsts.sv_invu_dim)),
        "y": list(map(lambda l: construct_y_label(l, at_least_one_blast_weapon), ts_ws_fnps_nm))
    }

    targets_matrix = [
        [
            Target(t, sv, invu=invu, fnp=fnp, w=w, n_models=n_models)
            for sv, invu in HeatMapConsts.sv_invu_dim
        ]
        for t, w, fnp, n_models in ts_ws_fnps_nm
    ]

    exact_scores = \
        [
            [
                [
                    [_score_weapon_on_target(
                        weapon_a,
                        target
                    ) for weapon_a in profile_a.weapons],
                    [_score_weapon_on_target(
                        weapon_b,
                        target
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
        print(f"\tsize of prob_by_roll_result_cache={len(Caches.prob_by_roll_result_cache)}")
        print(f"\tsize of success_ratios_cache={len(Caches.success_ratios_cache)}")
        print(f"\tsize of n_attacks_cache={len(Caches.n_attacks_cache)}")
        print(f"\tsize of hit_ratios_cache={len(Caches.hit_ratios_cache)}")
        print(f"\tsize of wound_ratios_cache={len(Caches.wound_ratios_cache)}")
        print(f"\tsize of unsaved_wound_ratios_cache={len(Caches.unsaved_wound_ratios_cache)}")
        print(f"\tsize of slained_figs_percent_per_unsaved_wound_cache="
              f"{len(Caches.slained_figs_percent_per_unsaved_wound_cache)}")
        print(f"\tsize of prob_mortals_after_wound_cache={len(Caches.prob_mortals_after_wound_cache)}")

    return res
