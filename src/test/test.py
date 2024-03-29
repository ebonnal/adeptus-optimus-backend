import unittest

from adeptus_optimus_backend.engine import *
from adeptus_optimus_backend.models import *
import adeptus_optimus_backend.engine_no_vect as legacy

from time import time

set_is_dev_execution(True)


class Test(unittest.TestCase):
    """
    Spot differences:
        # for sv, invu in HeatMapConsts.sv_invu_dim:
        #     for t, w, fnp, nm in HeatMapConsts.get_ts_ws_fnps_nm_dim(False):
        #         target = Target(t=t, w=w, sv=sv, fnp=fnp, invu=invu)
        #         a, b = (score_weapon_on_target(profile_c.weapons[0], target, None, None),
        #         legacy.score_weapon_on_target(profile_c.weapons[0], target, None, None))
        #         print(
        #             float_eq(a, b, 1.0001, False),
        #             a >= b,
        #             target,
        #             a, b
        #         )
    """
    sag = Weapon(a="D6", hit="5", s="2D6", ap="5", d="D6", options=Options(
        dakka3=6,
        reroll_hits=Options.ones,
        is_blast=True,
        snipe={
            Options.snipe_roll_type: Options.strength,
            Options.snipe_threshold: 11,
            Options.snipe_n_mortals: DiceExpr(1, 3)
        }
    ))

    def test_10_percent_diff_with_exact_modes(self):
        for weapon in [Weapon(d="2"), self.sag]:
            for sv, invu in HeatMapConsts.sv_invu_dim:
                for t, w, fnp, nm in HeatMapConsts.get_ts_ws_fnps_nm_dim(False):
                    target = Target(t=t, w=w, sv=sv, fnp=fnp, invu=invu)
                    a, b = (score_weapon_on_target(weapon, target, None, None),
                            legacy.score_weapon_on_target(weapon, target, None, None))
                    assert_float_eq(a, b, 1.10)
                    # print("a", float_eq(a, b), target, a, b, abs(a - b) / abs(a))

    def test_two_times_same_query(self):
        profile_a = Profile([Test.sag], "1")
        profile_b = Profile([Weapon(d="D3")], "1")
        z_matrix_1 = with_timer(lambda: compute_heatmap(profile_a, profile_b)["z"])
        z_matrix_1b = with_timer(lambda: compute_heatmap(profile_a, profile_b)["z"])
        assert_matrix_float_eq(z_matrix_1, z_matrix_1b)

    def _test_compute_heatmap_with_and_without_caches(self):
        profile_a = Profile([Weapon(hit="2", s="1")], "1")
        profile_b = Profile([Weapon(s="8")], "100")
        # with cache:
        z_matrix_1 = with_timer(lambda: compute_heatmap(profile_a, profile_b)["z"])
        self.assertEqual(None, z_matrix_1[-1][-1])
        self.assertEqual(0.97, z_matrix_1[-1][0])

        Caches.disable()

        z_matrix_2 = with_timer(lambda: compute_heatmap(profile_a, profile_b)["z"])
        self.assertTrue(
            all([all([e_1 == e_2 for e_1, e_2 in zip(line_1, line_2)]) for line_1, line_2 in
                 zip(z_matrix_1, z_matrix_2)])
        )

        assert_matrix_float_eq(z_matrix_1, z_matrix_2)

        Caches.reset_and_enable()

    def test_snipe(self):
        # Snipe on strength roll
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="2D6", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.strength,
                        Options.snipe_threshold: 12,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=+1  # make wounding on 3+ but does not impact sniping with prob=1/36
                )),
                Target(t=4, w=1000, sv=6, fnp=5)
            ), (1 +  # normal damages
                (1 / 36 * 2)  # mortals
                / (5 / 6)  # wounds to unsave wounds
                ) / 1000 *  # W
               2 / 3  # FNP
        )
        # SAG case
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="2D6", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.strength,
                        Options.snipe_threshold: 11,
                        Options.snipe_n_mortals: DiceExpr(1, 3)
                    },
                    wound_modifier=+1  # make wounding on 3+ but does not impact sniping with prob=1/36
                )),
                Target(t=4, w=1000, sv=6, fnp=5)
            ), (1 + ((1 + 2) / 36 * (1 + 3) / 2) / (5 / 6)) / 1000 * 2 / 3
        )

        # compares with no snipe equivalent without regular damages losses
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="2D6", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.strength,
                        Options.snipe_threshold: 11,
                        Options.snipe_n_mortals: DiceExpr(12, None)
                    }
                )),
                Target(w=2, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="2", ap="1"),
                Target(w=2, sv=6)
            )
        )
        # with some wounds saved
        self.assertGreater(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="2D6", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.strength,
                        Options.snipe_threshold: 11,
                        Options.snipe_n_mortals: DiceExpr(12, None)  # 12* 3/36 = 1 mortal per wound in average
                    }
                )),
                Target(w=2, sv=6)
            ),
            1.08 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="2"),
                Target(w=2, sv=6)
            )
        )
        # with 1/4 regular damages lost
        assert_float_eq(
            3 / 4 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="2D6", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.strength,
                        Options.snipe_threshold: 11,
                        Options.snipe_n_mortals: DiceExpr(12, None)
                    }
                )),
                Target(w=3, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="2", ap="1"),
                Target(w=3, sv=6),
            )
        )

        # Snipe on wound roll
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="4", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 5,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=0
                )),
                Target(t=4, w=1000, sv=6)
            ), (1 + ((7 - 5) / (7 - 4) * 2) / (5 / 6)) / 1000
        )
        # modifier impacts snipe_threshold
        self.assertLess(
            1.1 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(1, None)
                    },
                    wound_modifier=+1
                )),
                Target(w=100, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 5,
                        Options.snipe_n_mortals: DiceExpr(1, None)
                    },
                    wound_modifier=0
                )),
                Target(w=100, sv=6)
            )
        )
        # ... if wounds on 2+, then snipe on 5+ and snipe on 6+ with wound modif +1 is the same
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="8", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(1, None)
                    },
                    wound_modifier=+1
                )),
                Target(w=100, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", s="8", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 5,
                        Options.snipe_n_mortals: DiceExpr(1, None)
                    },
                    wound_modifier=0
                )),
                Target(w=100, sv=6)
            )
        )

        # snipe with random n_mortals equivalent to the flat avg on infinite target wounds
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="3", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=+1
                )),
                Target(w=100, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="3", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(1, 3)
                    },
                    wound_modifier=+1
                )),
                Target(w=100, sv=6)
            )
        )
        # ... on few target max wounds, with d=1
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=+1
                )),
                Target(w=2, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="1", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(1, 3)
                    },
                    wound_modifier=+1
                )),
                Target(w=2, sv=6)
            )
        )
        # ... on few target max wounds, with w%d == 0, D3 is make triggers regular damages losses
        self.assertGreater(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="2", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=+1
                )),
                Target(w=4, sv=6)
            ),
            1.05 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="2", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 6,
                        Options.snipe_n_mortals: DiceExpr(1, 3)
                    },
                    wound_modifier=+1
                )),
                Target(w=4, sv=6)
            )
        )
        # with wounds on 5+ (s=3, t=4), on w=4 it is equivalent to have d=4 or d=2 and snipe on 5+/2 mortals
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(s="3", d="2", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 5,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=0
                )),
                Target(w=4, sv=6)
            ),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(s="3", d="4", ap="1"),
                Target(w=4, sv=6)
            )
        )
        # losing half of the regular damages (d=4 for w=2), while having this half passed as mortals is optimum
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(s="3", d="2", ap="1", options=Options(
                    snipe={
                        Options.snipe_roll_type: Options.wound,
                        Options.snipe_threshold: 5,
                        Options.snipe_n_mortals: DiceExpr(2, None)
                    },
                    wound_modifier=0
                )),
                Target(w=2, sv=6)
            ) /
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(s="3", d="4", ap="1"),
                Target(w=2, sv=6)
            ),
            2
        )

        # Without snipe on, changing unsaved_wound_ratio does not impact result
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1, 3)), Target(w=100, fnp=5, sv=2, invu=6)),
            get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1, 3)), Target(w=100, fnp=5, sv=6))
        )

    def test_doms_alloc(self):
        # Damages reroll
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1, 3)), Target(w=100, fnp=5)),
                        1 / 100 * 2 * 2 / 3)
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d=DiceExpr(1, 3), options=Options(reroll_damages=True)),
                Target(w=100, fnp=5)
            ), 0.5 * (1 / 9 + 2 * (1 / 3 + 1 / 9) + 3 * (1 / 3 + 1 / 9)) / 100 * 2 * 2 / 3
        )
        # FNP
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1)), Target(w=1, fnp=6)), 5 / 6)
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1)), Target(w=1, fnp=5)), 4 / 6)
        # on W=2
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1)), Target(w=2, fnp=None)), 0.5)
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(2)), Target(w=2, fnp=None)), 1)
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(2, 3)), Target(w=2, fnp=None)), 1)
        # random doms
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(1, 6)), Target(w=350, fnp=None)),
                        0.01)
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(
            Weapon(d=DiceExpr(1, 6)), Target(w=175, fnp=5)), 0.01 * 2 * 2 / 3)

        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(
            Weapon(d=DiceExpr(1, 6)), Target(w=70, fnp=5)), 2 / 3 * 3.5 / 70)

        # lost damages
        assert_float_eq(get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(5)), Target(w=6, fnp=None)),
                        0.5,
                        1.03)

        # exact cases
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5) == 0.5)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2) == 1)
        self.assertTrue(exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16) == 1 / 3)
        assert_float_eq(exact_avg_figs_fraction_slained_per_unsaved_wound(d=3, w=5),
                        get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(3)), Target(w=5, fnp=None)),
                        1.05)
        assert_float_eq(exact_avg_figs_fraction_slained_per_unsaved_wound(d=2, w=2),
                        get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(2)), Target(w=2, fnp=None)),
                        1.05)
        assert_float_eq(
            exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16),
            get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(6)), Target(w=16, fnp=None)),
            1.05
        )
        assert_float_eq(
            exact_avg_figs_fraction_slained_per_unsaved_wound(d=6, w=16),
            get_slained_figs_percent_per_unsaved_wound(Weapon(d=DiceExpr(6)), Target(w=16, fnp=None), exact_optim=True)
        )
        self.assertTrue(get_slained_figs_percent_per_unsaved_wound(
            Weapon("5", "10", "2D6", "1", "1"),
            Target(t=8, sv=6, invu=None, fnp=6, w=1)
        ) == get_slained_figs_percent_per_unsaved_wound(
            Weapon("5", "10", "7", "1", "1"),
            Target(t=8, sv=6, invu=None, fnp=6, w=1)
        ))
        # roll damages twice and take best is better than reroll  TODO: add more precise testing
        self.assertGreater(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(roll_damages_twice=True)),
                Target(w=5)),
            1.01 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(reroll_damages=True)),
                Target(w=5))
        )
        self.assertGreater(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(roll_damages_twice=True)),
                Target(w=3)),
            1.01 * get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(reroll_damages=True)),
                Target(w=3))
        )
        # same result for reroll and roll twice if d=D3 w=2: reroll cannot lead to less than min roll
        assert_float_eq(
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(roll_damages_twice=True)),
                Target(w=2)),
            get_slained_figs_percent_per_unsaved_wound(
                Weapon(d="D3", options=Options(reroll_damages=True)),
                Target(w=2))
        )

    def test_compute_successes_ratio(self):
        assert_float_eq(get_success_ratio(8, 0, True, Options.none), 1 / 6)
        assert_float_eq(get_success_ratio(6, 0, True, Options.none), 1 / 6)
        assert_float_eq(get_success_ratio(4, 0, True, Options.none), 3 / 6)
        assert_float_eq(get_success_ratio(8, 0, False, Options.none), 0)
        assert_float_eq(get_success_ratio(8, 0, True, Options.ones), 1 / 6 + 1 / 6 / 6)
        assert_float_eq(get_success_ratio(2, 0, True, Options.ones), 5 / 6 + 5 / 6 / 6)
        assert_float_eq(get_success_ratio(2, 0, True, Options.onestwos),
                        get_success_ratio(2, 0, True, Options.ones))
        assert_float_eq(get_success_ratio(8, 0, True, Options.onestwos), 1 / 6 + 2 / 6 / 6)
        assert_float_eq(get_success_ratio(4, 0, True, Options.onestwos),
                        1 - (1 / 6 + 2 * 1 / 2 / 6))  # only 3 or reroll 1,2,3 fail
        assert_float_eq(get_success_ratio(3, 0, True, Options.onestwos),
                        get_success_ratio(3, 0, True, Options.full))
        assert_float_eq(get_success_ratio(2, 0, True, Options.full), 1 - 1 / 6 * 1 / 6)
        assert_float_eq(get_success_ratio(8, 0, True, Options.full), 1 - 5 / 6 * 5 / 6)
        assert_float_eq(get_success_ratio(8, 0, True, Options.none, 6),
                        1 / 6 + 1 / 6 / 6)
        assert_float_eq(get_success_ratio(8, 0, True, Options.onestwos, 5),
                        1 / 6 +  # direct success
                        2 / 6 * 1 / 6 +  # reroll -> success
                        2 / 6 * 1 / 6 +  # dakka3 -> success
                        2 / 6 * 2 / 6 * 1 / 6 +  # dakka3 -> reroll -> success
                        2 / 6 * 2 / 6 * 1 / 6 +  # reroll -> dakka3 -> success
                        2 / 6 * 2 / 6 * 2 / 6 * 1 / 6  # reroll -> dakka3 -> reroll -> success
                        )

        assert_float_eq(get_success_ratio(4, 0, True, Options.onestwos, 5),
                        3 / 6 +  # direct success
                        2 / 6 * 3 / 6 +  # dakka3 -> success
                        2 / 6 * 2 / 6 * 3 / 6 +  # dakka3 -> reroll -> success
                        2 / 6 * 3 / 6 +  # reroll -> success
                        2 / 6 * 2 / 6 * 3 / 6 +  # reroll -> dakka3 -> success
                        2 / 6 * 2 / 6 * 2 / 6 * 3 / 6  # reroll -> dakka3 -> reroll -> success
                        )

        assert_float_eq(get_success_ratio(4, 0, True, Options.full, 6),
                        3 / 6 +  # direct success
                        1 / 6 * 3 / 6 +  # dakka3 -> success
                        1 / 6 * 3 / 6 * 3 / 6 +  # dakka3 -> reroll -> success
                        3 / 6 * 3 / 6 +  # reroll -> success
                        3 / 6 * 1 / 6 * 3 / 6 +  # reroll -> dakka3 -> success
                        3 / 6 * 1 / 6 * 3 / 6 * 3 / 6  # reroll -> dakka3 -> reroll -> success
                        )
        # explodes
        assert_float_eq(
            get_success_ratio(4, 0, True, explodes=6),
            3 / 6 + 1 / 6
        )
        assert_float_eq(
            get_success_ratio(4, 0, True, explodes=5),
            3 / 6 + 1 / 6 + 1 / 6
        )
        # explosion > hit
        assert_float_eq(
            get_success_ratio(6, 0, True, explodes=5),
            1 / 6 + 1 / 6 + 1 / 6
        )
        # ratio can be > 1
        assert_float_eq(
            get_success_ratio(2, 0, True, explodes=5),
            5 / 6 + 1 / 6 + 1 / 6
        )
        # modifiers
        assert_float_eq(
            get_success_ratio(2, -1, True, explodes=5),
            4 / 6 + 1 / 6 + 1 / 6
        )
        assert_float_eq(
            get_success_ratio(2, -1, True, explodes=5),
            get_success_ratio(4, +1, True, explodes=5)
        )
        assert_float_eq(
            get_success_ratio(2, +1, True, explodes=5),
            get_success_ratio(2, 0, True, explodes=5)
        )
        # explodes mixed with rerolls and dakka3
        assert_float_eq(get_success_ratio(4, 0, True, Options.full, 6, 5),
                        3 / 6 +  # direct success
                        2 / 6 +  # direct success explodes
                        1 / 6 * 3 / 6 +  # dakka3 -> success
                        1 / 6 * 2 / 6 +  # dakka3 -> success explodes
                        1 / 6 * 3 / 6 * 3 / 6 +  # dakka3 -> reroll -> success
                        1 / 6 * 3 / 6 * 2 / 6 +  # dakka3 -> reroll -> success explodes
                        3 / 6 * 3 / 6 +  # reroll -> success
                        3 / 6 * 2 / 6 +  # reroll -> success explodes
                        3 / 6 * 1 / 6 * 3 / 6 +  # reroll -> dakka3 -> success
                        3 / 6 * 1 / 6 * 2 / 6 +  # reroll -> dakka3 -> success explodes
                        3 / 6 * 1 / 6 * 3 / 6 * 3 / 6 +  # reroll -> dakka3 -> reroll -> success
                        3 / 6 * 1 / 6 * 3 / 6 * 2 / 6  # reroll -> dakka3 -> reroll -> success explodes
                        )
        # assert get_success_ratio equal to corresponding get_hit_ratio
        assert_float_eq(get_success_ratio(4, 0, True, Options.full, 6, 5),
                        get_hit_ratio(Weapon(hit="4", options=Options(hit_modifier=0,
                                                                      reroll_hits=Options.full,
                                                                      dakka3=6,
                                                                      hit_explodes=5)))
                        )

    def test_compute_necessary_wound_roll(self):
        self.assertEqual(compute_necessary_wound_roll(1, 4), 6)
        self.assertEqual(compute_necessary_wound_roll(2, 4), 6)
        self.assertEqual(compute_necessary_wound_roll(3, 4), 5)
        self.assertEqual(compute_necessary_wound_roll(4, 4), 4)
        self.assertEqual(compute_necessary_wound_roll(5, 4), 3)
        self.assertEqual(compute_necessary_wound_roll(6, 4), 3)
        self.assertEqual(compute_necessary_wound_roll(7, 4), 3)
        self.assertEqual(compute_necessary_wound_roll(8, 4), 2)

    def test_engine_core(self):
        opt1 = Options.parse({"hit_modifier": "",
                              "wound_modifier": "",
                              "save_modifier": "",
                              "reroll_hits": "ones",
                              "reroll_wounds": "",
                              "dakka3": "5",
                              "auto_wounds_on": "",
                              "is_blast": "yes",
                              "auto_hit": "",
                              "wounds_by_2D6": "",
                              "reroll_damages": "yes",
                              "roll_damages_twice": "",
                              "snipe": "",
                              "hit_explodes": "5"})
        self.assertEqual(opt1.dakka3, 5)
        # missing keys
        opt2 = Options.parse({"hit_modifier": "",
                              "reroll_hits": "ones",
                              "reroll_wounds": "",
                              "dakka3": "5",
                              "auto_wounds_on": "",
                              "is_blast": "yes",
                              "reroll_damages": "yes",
                              "snipe": "",
                              "hit_explodes": "5"})
        self.assertEqual(opt2.dakka3, 5)
        self.assertDictEqual(opt1.__dict__, opt2.__dict__)

        self.assertEqual(Options.parse_snipe("wound,3,D3")[Options.snipe_n_mortals], DiceExpr(1, 3))
        self.assertRaises(RequirementError, lambda: Options.parse_snipe("wound,3,2D3")[Options.snipe_n_mortals])

        self.assertRaises(RequirementError, lambda: Options(wounds_by_2D6=True, wound_modifier=-1))

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
        self.assertTrue(score_weapon_on_target(w1, t1, None, None) > 1.25 * score_weapon_on_target(w2, t1, None, None))
        self.assertTrue(1.02 * score_weapon_on_target(w1, t2, None, None) < score_weapon_on_target(w2, t2, None, None))
        w3, w4 = Weapon("5", "7", "2D6", "1", "1", options=Options.empty()), Weapon("5", "2D6", "2D6", "1", "1",
                                                                                    options=Options.empty())
        assert_float_eq(score_weapon_on_target(w3, t1, None, None),
                        score_weapon_on_target(w4, t1, None, None))  # options
        t = Target(t=4, sv=5, invu=None, fnp=6, w=6)
        self.assertEqual(
            score_weapon_on_target(
                Weapon(hit="6", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=1, wound_modifier=0)), t,
                None,
                None),
            score_weapon_on_target(
                Weapon(hit="4", a="D6", s="4", ap="D6", d="D6", options=Options(hit_modifier=-1, wound_modifier=0)), t,
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
        assert_float_eq(
            get_n_attacks(
                Weapon(hit="4", a="3D3", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=10)
            ),
            get_n_attacks(
                Weapon(hit="4", a="3D3", s="4", ap="D6", d="D6", options=Options(is_blast=False)),
                Target(t=4, sv=6, n_models=10)
            )
        )
        self.assertTrue(
            get_n_attacks(
                Weapon(hit="4", a="D3", s="4", ap="D6", d="D6", options=Options(is_blast=True)),
                Target(t=4, sv=6, n_models=6)
            ) == 3
        )

        # Assert DakkaDakkaDakka and 1s reroll is the same
        assert_float_eq(
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(dakka3=6))),
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.ones))))
        assert_float_eq(
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(dakka3=6, hit_modifier=1))),
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6",
                       options=Options(reroll_hits=Options.ones, hit_modifier=1))))
        # Assert 1s and 2s is like full for WSBS=4+ and hit modifier +1
        assert_float_eq(
            get_hit_ratio(
                Weapon(hit="3", a="1", s="4", ap="D6", d="D6", options=Options(reroll_hits=Options.full))),
            get_hit_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6",
                       options=Options(reroll_hits=Options.full, hit_modifier=+1))))
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
        # wounds_by_2D6
        assert_float_eq(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(wounds_by_2D6=True)),
                Target(t=2, sv=6)
            ), 1
        )
        assert_float_eq(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(wounds_by_2D6=True)),
                Target(t=3, sv=6)
            ), 1 - 1 / 36
        )
        assert_float_eq(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(wounds_by_2D6=True)),
                Target(t=13, sv=6)
            ), 0
        )
        assert_float_eq(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="4", ap="D6", d="D6", options=Options(wounds_by_2D6=True)),
                Target(t=7, sv=6)
            ), 15 / 36 + 6 / 36
        )

        # autohit on 6+ makes 1/4 of the hits auto wounding if necessary_hit_roll was 3+ ...
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="3", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=6,
                                                                               reroll_hits=Options.none,
                                                                               hit_modifier=0)),
                Target(t=4, sv=6)
            ), 1 / 4 + 3 / 4 * 1 / 3
        )
        # ... or 4+ and +1 wound_modifier (shows that auto_wounds_on considers unmodified hit roll
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="4", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=6,
                                                                               reroll_hits=Options.none,
                                                                               hit_modifier=+1)),
                Target(t=4, sv=6)
            ), 1 / 4 + 3 / 4 * 1 / 3
        )
        # hits reroll does not impact auto_wounds_on
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="3", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=6,
                                                                               reroll_hits=Options.full,
                                                                               hit_modifier=0)),
                Target(t=4, sv=6)
            ), 1 / 4 + 3 / 4 * 1 / 3
        )
        # cannot be greater than 1
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=5,
                                                                               hit_modifier=+1)),
                Target(t=4, sv=6)
            ), 1
        )
        # test auto_wounds_on part "An unmodified hit roll of _+ **always hits** [...]"
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="6", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=5))
            ), 2 / 6
        )
        self.assertEqual(
            get_hit_ratio(
                Weapon(hit="6", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=5, hit_modifier=-1))
            ), 2 / 6
        )

        # auto hit on 6+ makes 100% of the hits auto wounding if necessary_hit_roll was 5+ or 6+
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=5,
                                                                               reroll_hits=Options.full,
                                                                               hit_modifier=+1)),
                Target(t=4, sv=6)
            ), 1
        )
        self.assertEqual(
            get_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="D6", d="D6", options=Options(auto_wounds_on=6,
                                                                               reroll_hits=Options.full,
                                                                               hit_modifier=0)),
                Target(t=4, sv=6)
            ), 1
        )
        # save roll*
        self.assertEqual(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=0)),
                Target(t=4, sv=6)
            ), 5 / 6
        )
        # save can be ignored with modifier
        self.assertEqual(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=-1)),
                Target(t=4, sv=6)
            ), 1
        )
        # unmodified roll of 1 is always a fail for save roll
        assert_float_eq(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=+1)),
                Target(t=4, sv=3)
            ), 1 / 6
        )
        assert_float_eq(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=+2)),
                Target(t=4, sv=3)
            ), 1 / 6
        )
        # when not reaching invul, -2 save modifier and -2 AP are equivalent
        assert_float_eq(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=-2)),
                Target(t=4, sv=3)
            ),
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="2", d="D6", options=Options(save_modifier=0)),
                Target(t=4, sv=3)
            )
        )
        # when reaching Invu, -2 save modifier and -2 AP are equivalent
        self.assertEqual(
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="0", d="D6", options=Options(save_modifier=-2)),
                Target(t=4, sv=3, invu=4)
            ),
            get_unsaved_wound_ratio(
                Weapon(hit="6", a="1", s="3", ap="2", d="D6", options=Options(save_modifier=0)),
                Target(t=4, sv=3, invu=4)
            )
        )
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
        # auto_hit
        self.assertTrue(
            get_hit_ratio(
                Weapon(hit="4", a="10", s="4", ap="D6", d="D6", options=Options(auto_hit=True))
            ) ==
            2 * get_hit_ratio(
                Weapon(hit="4", a="10", s="4", ap="D6", d="D6", options=Options(auto_hit=False))
            ) == 1
        )

        self.assertTrue(scores_to_z(100000, 1) == 1)
        self.assertTrue(scores_to_z(1, 100000) == -1)
        self.assertTrue(scores_to_z(100, 1) == 0.99)
        self.assertTrue(scores_to_z(1, 100) == -0.99)
        self.assertTrue(scores_to_z(1000, 1) == 0.999)
        self.assertTrue(scores_to_z(1, 1000) == -0.999)
        self.assertTrue(scores_to_z(1, 1) == 0)
        self.assertTrue(scores_to_ratio(1, 1) == 1)
        self.assertTrue(scores_to_ratio(1, 2) == 2.0)
        self.assertTrue(scores_to_ratio(4, 2) == 2.0)

    def test_utils(self):
        start = time()
        delay_from(start, 0.5)
        self.assertGreater(time() - start, 0.3)

        self.assertEqual(DiceExpr(5, 3).n_cases(), 11)

        self.assertTrue(DiceExpr(5, 3) == DiceExpr(5, 3))
        self.assertTrue(DiceExpr(5, 3) != DiceExpr(5, 6))
        self.assertTrue(DiceExpr(15, 3) != DiceExpr(5, 3))
        self.assertTrue(str(DiceExpr(5, 3)) == "5D3")
        self.assertTrue(str(DiceExpr(1, 6)) == "D6")
        self.assertTrue(str(DiceExpr(10, None)) == "10")
        self.assertEqual(f"{DiceExpr(2, 3)}", "2D3")
        self.assertEqual(DiceExpr.parse("4D3").min, 4)
        self.assertEqual(DiceExpr.parse("4D3").avg, 8)
        self.assertEqual(DiceExpr.parse("4D3").max, 12)
        self.assertEqual(DiceExpr.parse("5").avg, 5)
        self.assertIsNone(DiceExpr.parse("D7"))
        self.assertIsNone(DiceExpr.parse("0D6"))
        self.assertEqual(DiceExpr.parse("0").avg, 0)
        self.assertIsNone(DiceExpr.parse("7D6"))
        self.assertEqual(DiceExpr.parse("D3").avg, 2)
        self.assertEqual(DiceExpr.parse("3D3").avg, 6)
        self.assertEqual(DiceExpr.parse("D6").avg, 3.5)
        self.assertIsNone(DiceExpr.parse("1D6"))

        assert_float_eq(0.01, 0.011, max_ratio=1.1)
        assert_float_neq(0.01, 0.012, min_ratio=1.1)

        self.assertTrue(get_prob_by_roll_result(DiceExpr.parse("D3")) == {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertTrue(get_prob_by_roll_result(DiceExpr.parse("7")) == {7: 1})
        assert_float_eq(1, sum(get_prob_by_roll_result(DiceExpr.parse("2D6")).values()))
        self.assertTrue(
            get_prob_by_roll_result(DiceExpr.parse("2D6")) == {2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
                                                               7: 6 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36,
                                                               12: 1 / 36})

        # get_prob_by_roll_result with reroll_if_less_than
        assert_float_eq(
            sum(get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=7).values()),
            1
        )
        assert_float_eq(
            get_prob_by_roll_result(DiceExpr.parse("3D6", raise_on_failure=True), reroll_if_less_than=4)[3],
            1 / (6 ** 3) ** 2
        )
        # when expected value is in events, reroll if "less" or "less or equal to" it is equivalent
        assert_float_eq(
            get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=7)),
            get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=8))
        )
        assert_float_eq(
            get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("D3"), reroll_if_less_than=2)),
            get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("D3"), reroll_if_less_than=3))
        )
        # reroll if roll <= expected value is optimum
        for reroll_if_less_than in range(0, 13):
            self.assertLess(
                get_avg_of_density(
                    get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=reroll_if_less_than)),
                1.0000001 * get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=7))
            )

        # reroll is always better than nothing
        for reroll_if_less_than in range(3, 13):
            self.assertLess(
                get_avg_of_density(get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=0)),
                1.0000001 * get_avg_of_density(
                    get_prob_by_roll_result(DiceExpr.parse("2D6"), reroll_if_less_than=reroll_if_less_than))
            )
