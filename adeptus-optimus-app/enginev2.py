from engineutils import prob_by_roll_result, compute_successes_ratio, DiceExpr

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
                                                  n_unsaved_wounds_init=None,
                                                  prob_min_until_cut=0):


    n_figs_slained_weighted_ratios = []
    fnp_fail_ratio = 1 if target_fnp is None else 1 - compute_successes_ratio(target_fnp)

    if n_unsaved_wounds_init is None:
        if fnp_fail_ratio == 1:
            if weapon_d.dices_type is None:
                n_unsaved_wounds_init = 5
            else:
                n_unsaved_wounds_init = 5
        else:
            if weapon_d.dices_type is None:
                n_unsaved_wounds_init = 3
            else:
                n_unsaved_wounds_init = 2

    start_target_wounds = target_wounds
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
    # print(f"{len(n_figs_slained_weighted_ratios)} leafs by single tree, for depth={n_unsaved_wounds_init}")
    # return sum(map(lambda tup: tup[0] * tup[1], n_figs_slained_weighted_ratios))/1
    return sum(n_figs_slained_weighted_ratios)
