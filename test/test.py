import unittest

from adeptus_optimus_backend import *
from time import time


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

    def test_engine_core(self):
        self.assertEqual(Options.parse({"hit_modifier": "0", "wound_modifier": "0"}).wound_modifier, 0)
        self.assertEqual(Options.parse({"hit_modifier": "0", "wound_modifier": "0"}).hit_modifier, 0)
        self.assertTrue(dispatch_density_key(3, 0.5) == {0: 0.125, 1: 0.375, 2: 0.375, 3: 0.125})
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
        wea = Weapon(hit="4", a="4", s="4", ap="1", d="3", options=Options(0, 0))
        wea2 = Weapon(hit="4", a="4", s="4", ap="0", d="3", options=Options(0, 0))
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
            score_weapon_on_target(Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0)), t, None,
                                   None) ==
            score_weapon_on_target(Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(1, 0)), t, None,
                                   None))
        self.assertTrue(
            score_weapon_on_target(Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0)), t, None,
                                   None) ==
            score_weapon_on_target(Weapon(hit="5", a="D6", s="3", ap="D6", d="D6", options=Options(0, 1)), t, None,
                                   None))
        self.assertTrue(
            score_weapon_on_target(Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(-1, 0)), t, None,
                                   None) ==
            score_weapon_on_target(Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0)), t, None,
                                   None))
        self.assertTrue(
            score_weapon_on_target(Weapon(hit="5", a="D6", s="4", ap="D6", d="D6", options=Options(0, -1)), t, None,
                                   None) ==
            score_weapon_on_target(Weapon(hit="5", a="D6", s="3", ap="D6", d="D6", options=Options(0, 0)), t, None,
                                   None))

        # Assert six is always a success to hit or wound
        #   1) Modifiers
        self.assertEqual(
            get_hit_ratio(Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))),
            get_hit_ratio(Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(-1, 0))))
        self.assertEqual(
            get_wound_ratio(Weapon(hit="6", a="D6", s="2", ap="D6", d="D6", options=Options(0, 0)), Target(4)),
            get_wound_ratio(Weapon(hit="6", a="D6", s="2", ap="D6", d="D6", options=Options(0, -1)), Target(4)))
        #  2) WS|BS > 6
        self.assertEqual(
            get_hit_ratio(Weapon(hit="10", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))),
            get_hit_ratio(Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))))
        # assert 1 is always a failure to hit or wound
        #  1) Modifiers
        self.assertEqual(
            get_hit_ratio(Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))),
            get_hit_ratio(Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(+1, 0))))
        self.assertEqual(
            get_wound_ratio(Weapon(hit="6", a="D6", s="8", ap="D6", d="D6", options=Options(0, 0)), Target(4)),
            get_wound_ratio(Weapon(hit="6", a="D6", s="8", ap="D6", d="D6", options=Options(0, +1)), Target(4)))
        #  2) WS|BS < 2
        self.assertEqual(
            get_hit_ratio(Weapon(hit="0", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))),
            get_hit_ratio(Weapon(hit="2", a="D6", s="4", ap="D6", d="D6", options=Options(0, 0))))

        self.assertTrue(scores_to_z(10000, 1) == 1)
        self.assertTrue(scores_to_z(1, 10000) == -1)
        self.assertTrue(scores_to_z(1000, 1) == 0.999)
        self.assertTrue(scores_to_z(1, 1000) == -0.999)
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
