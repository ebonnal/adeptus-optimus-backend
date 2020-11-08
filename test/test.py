import unittest

from adeptus_optimus_backend import *
from time import time


def float_eq(a, b, n_same_decimals=8, verbose=False):
    if verbose:
        print(f'%.{n_same_decimals}E' % a, f'%.{n_same_decimals}E' % b)
    return f'%.{n_same_decimals}E' % a == f'%.{n_same_decimals}E' % b


class Test(unittest.TestCase):

    def test_doms_alloc(self):
        # FNP
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(1), 6, 1), 5 / 6, 0))
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(1), 5, 1), 4 / 6, 0))
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(1), 4, 1), 0.5, 0))
        # on W=2
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(1), None, 2), 0.5, 0))
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(2), None, 2), 1, 0))
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(2, 3), None, 2), 1, 0))
        # random doms
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(1, 6), None, 35), 0.1, 0))
        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(
            DiceExpr(1, 6), 4, 175), 0.01, 0)
        )

        self.assertTrue(float_eq(get_slained_figs_ratio_per_unsaved_wound(
            DiceExpr(1, 6), 5, 70), 2 / 3 * 3.5 / 70, 0)
        )
        # lost damages
        self.assertTrue(
            float_eq(get_slained_figs_ratio_per_unsaved_wound(DiceExpr(5), target_fnp=None, target_wounds=6), 0.5, 0))

    def test_compute_successes_ratio(self):
        self.assertTrue(float_eq(get_success_ratio(8, True, Options.none), 1 / 6))
        self.assertTrue(float_eq(get_success_ratio(6, True, Options.none), 1 / 6))
        self.assertTrue(float_eq(get_success_ratio(4, True, Options.none), 3 / 6))
        self.assertTrue(float_eq(get_success_ratio(8, False, Options.none), 0))

        self.assertTrue(float_eq(get_success_ratio(8, True, Options.ones), 1 / 6 + 1 / 6 / 6))
        self.assertTrue(float_eq(get_success_ratio(2, True, Options.ones), 5 / 6 + 5 / 6 / 6))
        self.assertTrue(float_eq(get_success_ratio(2, True, Options.onestwos),
                                 get_success_ratio(2, True, Options.ones)))
        self.assertTrue(float_eq(get_success_ratio(8, True, Options.onestwos), 1 / 6 + 2 / 6 / 6))
        self.assertTrue(float_eq(get_success_ratio(4, True, Options.onestwos),
                                 1 - (1 / 6 + 2 * 1 / 2 / 6)))  # only 3 or reroll 1,2,3 fail
        self.assertTrue(float_eq(get_success_ratio(3, True, Options.onestwos),
                                 get_success_ratio(3, True, Options.full)))
        self.assertTrue(float_eq(get_success_ratio(2, True, Options.full), 1 - 1 / 6 * 1 / 6))
        self.assertTrue(float_eq(get_success_ratio(8, True, Options.full), 1 - 5 / 6 * 5 / 6))
        self.assertTrue(float_eq(get_success_ratio(8, True, Options.none, 6),
                                 1 / 6 + 1 / 6 / 6))
        self.assertTrue(float_eq(get_success_ratio(8, True, Options.onestwos, 5),
                                 1 / 6 +  # direct success
                                 2 / 6 * 1 / 6 +  # reroll -> success
                                 2 / 6 * 1 / 6 +  # dakka3 -> success
                                 2 / 6 * 2 / 6 * 1 / 6 +  # dakka3 -> reroll -> success
                                 2 / 6 * 2 / 6 * 1 / 6 +  # reroll -> dakka3 -> success
                                 2 / 6 * 2 / 6 * 2 / 6 * 1 / 6  # reroll -> dakka3 -> reroll -> success
                                 ))

        self.assertTrue(float_eq(get_success_ratio(4, True, Options.onestwos, 5),
                                 3 / 6 +  # direct success
                                 2 / 6 * 3 / 6 +  # dakka3 -> success
                                 2 / 6 * 2 / 6 * 3 / 6 +  # dakka3 -> reroll -> success
                                 2 / 6 * 3 / 6 +  # reroll -> success
                                 2 / 6 * 2 / 6 * 3 / 6 +  # reroll -> dakka3 -> success
                                 2 / 6 * 2 / 6 * 2 / 6 * 3 / 6  # reroll -> dakka3 -> reroll -> success
                                 ))

        self.assertTrue(float_eq(get_success_ratio(4, True, Options.full, 6),
                                 3 / 6 +  # direct success
                                 1 / 6 * 3 / 6 +  # dakka3 -> success
                                 1 / 6 * 3 / 6 * 3 / 6 +  # dakka3 -> reroll -> success
                                 3 / 6 * 3 / 6 +  # reroll -> success
                                 3 / 6 * 1 / 6 * 3 / 6 +  # reroll -> dakka3 -> success
                                 3 / 6 * 1 / 6 * 3 / 6 * 3 / 6  # reroll -> dakka3 -> reroll -> success
                                 ))

    def test_engine_core(self):
        self.assertEqual(Options.parse({"hit_modifier": "0",
                                        "wound_modifier": "0",
                                        "reroll_hits": "none",
                                        "reroll_wounds": "none",
                                        "dakka3": "none",
                                        "auto_wounds_on": "none",
                                        "is_blast": "no"}).wound_modifier, 0)
        self.assertEqual(Options.parse({"hit_modifier": "0",
                                        "wound_modifier": "0",
                                        "reroll_hits": "none",
                                        "reroll_wounds": "none",
                                        "dakka3": "none",
                                        "auto_wounds_on": "none",
                                        "is_blast": "no"}).hit_modifier, 0)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5) == 0.5)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2) == 1)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16) == 1 / 3)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5) ==
                        get_slained_figs_ratio_per_unsaved_wound(DiceExpr(3), None, 5))
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2) ==
                        get_slained_figs_ratio_per_unsaved_wound(DiceExpr(2), None, 2))
        self.assertTrue(float_eq(exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16),
                                 get_slained_figs_ratio_per_unsaved_wound(DiceExpr(6), None, 16), 0))
        self.assertTrue(get_avg_figs_fraction_slained_per_unsaved_wound(
            Weapon("5", "10", "2D6", "1", "1"),
            Target(t=8, sv=6, invu=None, fnp=6, w=1)
        ) == get_avg_figs_fraction_slained_per_unsaved_wound(
            Weapon("5", "10", "7", "1", "1"),
            Target(t=8, sv=6, invu=None, fnp=6, w=1)
        ))
        self.assertTrue(get_avg_of_density({0: 0.2, 1: 0.5, 2: 0.3}) == 0.5 + 0.3 * 2)
        wea = Weapon(hit="4", a="4", s="4", ap="1", d="3", options=Options(hit_modifier=0, wound_modifier=0))
        wea2 = Weapon(hit="4", a="4", s="4", ap="0", d="3", options=Options(hit_modifier=0, wound_modifier=0))
        tar = Target(t=4, sv=1, invu=5, fnp=6, w=16)
        self.assertTrue(abs(
            score_weapon_on_target(wea, tar, None, None) / score_weapon_on_target(wea2, tar, None, None) - 1) <= 0.25)
        # S=2D6 triggers upper threshold effect on T=8 and is better than S=7, but not on other Toughnesses
        w1, w2 = Weapon("5", "10", "2D6", "1", "1", options=Options.empty()), Weapon("5", "10", "7", "1", "1",
                                                                                     options=Options.empty())
        t1 = Target(t=8, sv=6, invu=None, fnp=6, w=1)
        t2 = Target(t=7, sv=6, invu=None, fnp=6, w=1)
        self.assertTrue(score_weapon_on_target(w1, t1, None, None) > 1.1 * score_weapon_on_target(w2, t1, None, None))
        self.assertTrue(score_weapon_on_target(w1, t2, None, None) < 1.1 * score_weapon_on_target(w2, t2, None, None))
        w3, w4 = Weapon("5", "7", "2D6", "1", "1", options=Options.empty()), Weapon("5", "2D6", "2D6", "1", "1",
                                                                                    options=Options.empty())
        self.assertTrue(
            float_eq(score_weapon_on_target(w3, t1, None, None), score_weapon_on_target(w4, t1, None, None)))  # options
        t = Target(t=4, sv=5, invu=None, fnp=6, w=6)
        self.assertTrue(
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)), t,
                None,
                None) ==
            score_weapon_on_target(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=1, wound_modifier=0)), t,
                None,
                None))
        self.assertTrue(
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)), t,
                None,
                None) ==
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="3", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=1)), t,
                None,
                None))
        self.assertTrue(
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=-1, wound_modifier=0)), t,
                None,
                None) ==
            score_weapon_on_target(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)), t,
                None,
                None))
        self.assertTrue(
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=-1)), t,
                None,
                None) ==
            score_weapon_on_target(
                Weapon(hit="5", a="D6", s="3", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)), t,
                None,
                None))
        # blast on 11+
        self.assertTrue(
            get_n_attacks(
                Weapon(hit="4", a="2D6", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=11)
            ) ==
            get_n_attacks(
                Weapon(hit="4", a="4D3", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=11)
            ) == 12
        )
        # blast on 6-10 models
        self.assertTrue(
            get_n_attacks(
                Weapon(hit="4", a="2D3", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=10)
            ) == 3 * 1 / 9 + 3 * 2 / 9 + 4 * 3 / 9 + 5 * 2 / 9 + 6 * 1 / 9
        )
        self.assertTrue(
            get_n_attacks(
                Weapon(hit="4", a="D3", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=6)
            ) == 3
        )
        # hit on 4+ and auto wound at 5+ == hit at 4+ and wound_modifier +1: 2/6+4/6*3/6 == (3+1)/6 == 2/3
        self.assertTrue(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(auto_wounds_on=5)),
                Target(t=4, sv=6)
            ) ==
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(wound_modifier=+1)),
                Target(t=4, sv=6)
            ) == 2 / 3
        )
        # Assert DakkaDakkaDakka and 1s reroll is the same
        self.assertTrue(float_eq(
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(dakka3=6))),
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.ones)))))
        # Assert 1s and 2s is like full for WSBS=4+ and hit modifier +1
        self.assertTrue(float_eq(
            get_hit_ratio(
                Weapon(hit="3", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.full))),
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6",
                       options=Options(reroll_hits=Options.full, hit_modifier=+1)))))
        # rerolls hierarchie
        self.assertLess(
            1.1 * get_hit_ratio(
                Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.none))),
            get_hit_ratio(Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.ones))))
        self.assertLess(
            1.1 * get_hit_ratio(
                Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.ones))),
            get_hit_ratio(
                Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.onestwos))))
        self.assertLess(
            1.1 * get_hit_ratio(
                Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.onestwos))),
            get_hit_ratio(Weapon(hit="6", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.full))))
        # Assert six is always a success to hit or wound
        #   1) Modifiers
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))),
            get_hit_ratio(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=-1, wound_modifier=0))))
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="6", a="D6", s="2", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)),
                Target(4)),
            get_wound_ratio(
                Weapon(hit="6", a="D6", s="2", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=-1)),
                Target(4)))
        #  2) WS|BS > 6
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="10", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))),
            get_hit_ratio(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))))
        # assert 1 is always a failure to hit or wound
        #  1) Modifiers
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))),
            get_hit_ratio(
                Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=+1, wound_modifier=0))))
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="6", a="D6", s="8", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0)),
                Target(4)),
            get_wound_ratio(
                Weapon(hit="6", a="D6", s="8", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=+1)),
                Target(4)))
        #  2) WS|BS < 2
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="0", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))),
            get_hit_ratio(
                Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=0, wound_modifier=0))))

        self.assertTrue(scores_to_z(10000, 1) == 1)
        self.assertTrue(scores_to_z(1, 10000) == -1)
        self.assertTrue(scores_to_z(100, 1) == 0.99)
        self.assertTrue(scores_to_z(1, 100) == -0.99)
        self.assertTrue(scores_to_z(1, 1) == 0)
        self.assertTrue(scores_to_ratio(1, 1) == 1)
        self.assertTrue(scores_to_ratio(1, 2) == 2.0)
        self.assertTrue(scores_to_ratio(4, 2) == 2.0)

    def test_utils(self):
        start = time()
        with_minimum_exec_time(0.3, lambda: 1)
        self.assertGreater(time() - start, 0.3)
        self.assertTrue(str(DiceExpr(5, 3)) == "5D3")
        self.assertTrue(str(DiceExpr(1, 6)) == "D6")
        self.assertTrue(str(DiceExpr(10, None)) == "10")
        self.assertTrue(parse_dice_expr("4D3").avg == 8)
        self.assertTrue(parse_dice_expr("5").avg == 5)
        self.assertTrue(parse_dice_expr("D7") is None)
        self.assertTrue(parse_dice_expr("0D6") is None)
        self.assertTrue(parse_dice_expr("0").avg == 0)
        self.assertTrue(parse_dice_expr("7D6") is None)
        self.assertTrue(parse_dice_expr("D3").avg == 2)
        self.assertTrue(parse_dice_expr("3D3").avg == 6)
        self.assertTrue(parse_dice_expr("D6").avg == 3.5)
        self.assertTrue(parse_dice_expr("1D6") is None)
        self.assertTrue(parse_roll("1+") is None)
        self.assertTrue(parse_roll("1+") is None)
        self.assertTrue(parse_roll("2+") == 2)
        self.assertTrue(parse_roll("3+") == 3)
        self.assertTrue(parse_roll("6+") == 6)
        self.assertTrue(parse_roll("7+") is None)
        self.assertTrue(parse_roll("3") is None)

        # assert(float_eq(0.025, 0.0249, 0))  # TODO: make it pass

        self.assertTrue(float_eq(1, 1.01, 1))
        self.assertTrue(float_eq(0.3333, 0.3334, 2))
        self.assertTrue(float_eq(0.03333, 0.03334, 2))
        self.assertTrue(not float_eq(0.3333, 0.334, 2))
        self.assertTrue(not float_eq(0.03333, 0.0334, 2))

        self.assertTrue(prob_by_roll_result(parse_dice_expr("D3")) == {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertTrue(prob_by_roll_result(parse_dice_expr("7")) == {7: 1})
        self.assertTrue(float_eq(1, sum(prob_by_roll_result(parse_dice_expr("2D6")).values())))
        self.assertTrue(
            prob_by_roll_result(parse_dice_expr("2D6")) == {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
                                                            7: 6 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36,
                                                            12: 1 / 36})
        self.assertEqual(f"{DiceExpr(2, 3)}", "2D3")
