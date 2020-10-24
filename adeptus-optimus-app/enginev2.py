from engine import prob_by_roll_result, compute_successes_ratio, DiceExpr, float_eq

class State:
    def __init__(self,
                 n_unsaved_wounds_left,  # key field, 0 when resolved
                 current_wound_n_damages_left,  # key field, 0 when resolved
                 n_figs_slained_so_far,  # value field
                 remaining_target_wounds,  # key field
                 prob_node,  # involved in n_figs_slained_so_far update
                 ):
        self.n_unsaved_wounds_left = n_unsaved_wounds_left
        self.current_wound_n_damages_left = current_wound_n_damages_left
        self.n_figs_slained_so_far = n_figs_slained_so_far
        self.remaining_target_wounds = remaining_target_wounds
        self.prob_node = prob_node

    def copy(self):
        return State(self.n_unsaved_wounds_left,
                     self.current_wound_n_damages_left,
                     self.n_figs_slained_so_far,
                     self.remaining_target_wounds,
                     self.prob_node)


class Cache:
    def __init__(self):
        self.dict = {}

    def add(self, state, n_figs_slained_frac):
        self.dict[Cache._keyify(state)] = n_figs_slained_frac

    def get(self, state):
        self.dict.get(Cache._keyify(state), None)

    def reset(self):
        del self.dict
        self.dict = {}
    
    @staticmethod
    def _keyify(state):
        return f"{state.n_unsaved_wounds_left},{state.current_wound_n_damages_left},{state.remaining_target_wounds}"


class Node:
    weapon_d = None
    target_wounds = None
    n_unsaved_wounds_init = None
    n_figs_slained_weighted_ratios = None
    fnp_fail_ratio = None
    start_target_wounds = None
    cache = Cache()

    def __init__(self, state, parents_states, children_states):
        self.state = state
        self.parents_states = parents_states
        self.children_states = children_states

    def n_figs_to_ratio(n_figs_slained_so_far):
        pass

    def resolve(cache):
        # cached_res = Node.cache.get(state)
        # if cached_res is not None:
        #     self.state.n_figs_slained_so_far = cached_res
        #     return n_figs_slained_so_far * self.state.bla
        # else:
        #     update_slained_figs_ratios(self.state)
        #     n_unsaved_wounds_init = self.state.n_unsaved_wounds_init
        pass


def update_slained_figs_ratios(state_):
    assert (isinstance(state_, State))
    assert (state_.remaining_target_wounds >= 0)
    assert (state_.n_unsaved_wounds_left >= 0)
    assert (state_.current_wound_n_damages_left >= 0)
    state = state_.copy()
    if state.prob_node == 0:
        return 0

    # resolve a model kill
    if state.remaining_target_wounds == 0:
        state.remaining_target_wounds = Node.target_wounds
        state.n_figs_slained_so_far += 1
        # additionnal damages are not propagated to other models
        state.current_wound_n_damages_left = 0
        return update_slained_figs_ratios(state)

    # leaf: no more damages to fnp no more wounds to consume or p(leaf) < threshold
    if state.n_unsaved_wounds_left == 0 and state.current_wound_n_damages_left == 0:
        used_unsaved_wounds_portion = Node.n_unsaved_wounds_init - state.n_unsaved_wounds_left
        assert (used_unsaved_wounds_portion > 0)
        n_figs_slained_frac = (state.n_figs_slained_so_far +
                               # portion of the first model cleaned
                               (-1 + Node.start_target_wounds / Node.target_wounds) +
                               # portion of the last model injured
                               (1 - state.remaining_target_wounds / Node.target_wounds)) / \
                              used_unsaved_wounds_portion
        Node.cache.add(state, n_figs_slained_frac)
        return state.prob_node * n_figs_slained_frac

    # consume a wound

    if state.current_wound_n_damages_left == 0:
        state.n_unsaved_wounds_left -= 1
        # random doms handling
        return sum([update_slained_figs_ratios(State(state.n_unsaved_wounds_left,
                                                     d,
                                                     state.n_figs_slained_so_far,
                                                     state.remaining_target_wounds,
                                                     state.prob_node * prob_d))
                    for d, prob_d in prob_by_roll_result(Node.weapon_d).items()])

        # FNP success
    s = update_slained_figs_ratios(State(state.n_unsaved_wounds_left,
                                         state.current_wound_n_damages_left - 1,
                                         state.n_figs_slained_so_far,
                                         state.remaining_target_wounds,
                                         state.prob_node * (1 - Node.fnp_fail_ratio)))

    # FNP fail
    f = update_slained_figs_ratios(State(state.n_unsaved_wounds_left,
                                         state.current_wound_n_damages_left - 1,
                                         state.n_figs_slained_so_far,
                                         state.remaining_target_wounds - 1,
                                         state.prob_node * Node.fnp_fail_ratio))
    return s + f


def compute_slained_figs_ratios_per_unsaved_wound(weapon_d, target_fnp, target_wounds,
                                                  n_unsaved_wounds_init=5):
    Node.weapon_d = weapon_d
    Node.target_wounds = target_wounds
    Node.n_unsaved_wounds_init = n_unsaved_wounds_init
    Node.n_figs_slained_weighted_ratios = []
    Node.fnp_fail_ratio = 1 if target_fnp is None else 1 - compute_successes_ratio(target_fnp)
    Node.start_target_wounds = target_wounds
    Node.cache.reset()

    return update_slained_figs_ratios(State(
        n_unsaved_wounds_left=n_unsaved_wounds_init,
        current_wound_n_damages_left=0,
        n_figs_slained_so_far=0,
        remaining_target_wounds=target_wounds,
        prob_node=1))


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
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 6), 4, 70, n_unsaved_wounds_init=2), 0.025, 1))
assert (float_eq(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 6), 5, 70, n_unsaved_wounds_init=2), 2 / 60, 1))

print(Node.cache.dict)