from engineutils import prob_by_roll_result, compute_successes_ratio, DiceExpr

verbose = False
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
        self.hits = 0
        self.tries = 0

    def add(self, state, cached_unweighted_downstream):
        key = Cache._keyify(state)
        if verbose: print("add ", key)
        if self.dict.get(key, (0, 0))[0] < state.n_unsaved_wounds_left:
            self.dict[key] = (state.n_unsaved_wounds_left, cached_unweighted_downstream)

    def get(self, state):
        if verbose: print("tries get ", Cache._keyify(state))
        res = self.dict.get(Cache._keyify(state), (None, None))
        self.tries += 1
        if res[0] != None:
            if verbose: print("hit")
            self.hits += 1
        return res

    def reset(self):
        del self.dict
        self.dict = {}
        self.hits = 0
        self.tries = 0

    @staticmethod
    def _keyify(state):
        return f"dmgs_left:{state.current_wound_n_damages_left},rem_t_ws:{state.remaining_target_wounds}"


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


def compute_slained_figs_frac(state_):
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
        return compute_slained_figs_frac(state)

    if verbose: print("state.n_figs_slained_so_far",  state.n_figs_slained_so_far)
    # test cache
    cached_res_n_unsaved_wounds_left, cached_downstream_n_figs_slained_frac = Node.cache.get(state)
    if cached_downstream_n_figs_slained_frac is not None and cached_res_n_unsaved_wounds_left >= state.n_unsaved_wounds_left:
        # use cached res if deep enough
        return state.prob_node * (state.n_figs_slained_so_far + cached_downstream_n_figs_slained_frac)

    # leaf: no more damages to fnp no more wounds to consume or p(leaf) < threshold
    if state.n_unsaved_wounds_left == 0 and state.current_wound_n_damages_left == 0:
        n_figs_slained_frac = (state.n_figs_slained_so_far +
                               # portion of the first model cleaned
                               (-1 + Node.start_target_wounds / Node.target_wounds) +
                               # portion of the last model injured
                               (1 - state.remaining_target_wounds / Node.target_wounds))
        if verbose: print("leaf n_figs_slained_frac=", n_figs_slained_frac, ", n_figs_slained_so_far=", state.n_figs_slained_so_far)
        Node.cache.add(state, n_figs_slained_frac)
        return state.prob_node * n_figs_slained_frac

    # consume a wound

    if state.current_wound_n_damages_left == 0:
        # random doms handling
        res = [compute_slained_figs_frac(State(n_unsaved_wounds_left=state.n_unsaved_wounds_left - 1,
                                               current_wound_n_damages_left=d,
                                               n_figs_slained_so_far=state.n_figs_slained_so_far,
                                               remaining_target_wounds=state.remaining_target_wounds,
                                               prob_node=state.prob_node * prob_d))
               for d, prob_d in prob_by_roll_result(Node.weapon_d).items()]
        downstream_n_figs_slained_frac = sum(res)
        Node.cache.add(state, downstream_n_figs_slained_frac/state.prob_node)
        return downstream_n_figs_slained_frac

        # FNP success
    s = compute_slained_figs_frac(State(state.n_unsaved_wounds_left,
                                        state.current_wound_n_damages_left - 1,
                                        state.n_figs_slained_so_far,
                                        state.remaining_target_wounds,
                                        state.prob_node * (1 - Node.fnp_fail_ratio)))

    # FNP fail
    f = compute_slained_figs_frac(State(state.n_unsaved_wounds_left,
                                        state.current_wound_n_damages_left - 1,
                                        state.n_figs_slained_so_far,
                                        state.remaining_target_wounds - 1,
                                        state.prob_node * Node.fnp_fail_ratio))

    downstream_n_figs_slained_frac = s + f
    Node.cache.add(state, downstream_n_figs_slained_frac/state.prob_node)
    return downstream_n_figs_slained_frac


def compute_slained_figs_ratios_per_unsaved_wound(weapon_d, target_fnp, target_wounds,
                                      n_unsaved_wounds_init=5):
    if verbose: print("compute_slained_figs_frac_wrapper called")
    Node.weapon_d = weapon_d
    Node.target_wounds = target_wounds
    Node.n_unsaved_wounds_init = n_unsaved_wounds_init
    Node.n_figs_slained_weighted_ratios = []
    Node.fnp_fail_ratio = 1 if target_fnp is None else 1 - compute_successes_ratio(target_fnp)
    Node.start_target_wounds = target_wounds
    Node.cache.reset()

    return compute_slained_figs_frac(State(
        n_unsaved_wounds_left=n_unsaved_wounds_init,
        current_wound_n_damages_left=0,
        n_figs_slained_so_far=0,
        remaining_target_wounds=target_wounds,
        prob_node=1)) / Node.n_unsaved_wounds_init

# n_unsaved_wounds_init=10: (3044, 3070)
print(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(1, 3), target_fnp=None, target_wounds=1, n_unsaved_wounds_init=200), 5 / 6)
print(Node.cache.dict)
print(Node.cache.hits, Node.cache.tries)

print(compute_slained_figs_ratios_per_unsaved_wound(DiceExpr(2, 3), None, 2), 1, 0)
