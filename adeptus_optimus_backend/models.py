from .utils import *


class DiceExpr:
    star = None

    def __init__(self, n, dices_type=None):
        self.n = n
        self.dices_type = dices_type
        require(dices_type in {None, 3, 6}, f"Dices used must be either D3 or D6, not D{dices_type}")

        if self.dices_type is None:
            self.avg = n
            self.min = n
            self.max = n
        else:
            self.avg = n * (self.dices_type + 1) / 2
            self.min = n
            self.max = n * self.dices_type

    def n_cases(self):
        return self.max - self.min + 1

    def __repr__(self):
        if self.dices_type is None:
            return str(self.n)
        else:
            return f"{self.n if self.n > 1 else ''}D{self.dices_type}"

    def __eq__(self, other):
        if not isinstance(other, DiceExpr):
            return False
        else:
            return self.n == other.n and self.dices_type == other.dices_type

    @staticmethod
    def parse(d, complexity_threshold=18, raise_on_failure=False, allow_star=False):
        if isinstance(d, DiceExpr):
            return d
        assert (type(d) is str)
        groups = re.fullmatch(r"([1-9][0-9]*)?D([36])?|([0-9]+)", d)
        res = None
        invalidity_details = ""
        try:
            if d == "*" and allow_star:
                res = DiceExpr.star
            else:
                dices_type = int(groups.group(2))
                # at this point dices type is known
                if groups.group(1) is not None and int(groups.group(1)) == 1:
                    res = None  # 1D6 is not canonical, should enter D6
                    invalidity_details = f" must be noted 'D{dices_type}'"
                else:
                    if groups.group(1) is None:
                        n_dices = 1
                    else:
                        n_dices = int(groups.group(1))
                    res = DiceExpr(n_dices, dices_type)

        except TypeError:
            try:
                flat = int(groups.group(3))
                res = DiceExpr(flat)
            except TypeError:
                res = None
        finally:
            # not too many cases splits
            if res is not None and res.max > complexity_threshold:
                if res.dices_type is None:
                    invalidity_details = f": Value is too high"
                else:
                    invalidity_details = f": Maximum value is too high"
                res = None

            if raise_on_failure:
                require(res is not None, f"Invalid input '{d}'{invalidity_details}")
            return res


DiceExpr.star = DiceExpr(-1, None)


# Core engine logic
class Options:
    """
    Notes & rules:

    - Rerolls apply before modifiers
    - Dakka! Dakka! Dakka: "Each time you roll an unmodified hit roll of 6 for an attack..."
      dakka 5+, touche sur 6+
            1---1/6--->0

            2---1/6--->0

            3---1/6--->1

            4---1/6--->1

            5---1/6--->_--1/6-->1
                      |`--4/6-->0
                      `---1/6-->_
            6---1/6--->_--1/6-->2
                      |`--5/6-->1
                      `---1/6-->_

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
    hit_explodes_key = "hit_explodes"

    opt_key_to_repr = {
        hit_modifier_key: "Hit roll modifier",
        wound_modifier_key: "Wound roll modifier",
        save_modifier_key: "Save roll modifier",
        reroll_hits_key: "Hit roll reroll",
        reroll_wounds_key: "Wound roll reroll",
        dakka3_key: "An unmodified hit roll of _+ triggers one additional hit roll",
        auto_wounds_on_key: "An unmodified hit roll of _+ always hits and automatically wounds",
        is_blast_key: "Is a blast weapon",
        auto_hit_key: "Automatically hits",
        wounds_by_2D6_key: "Wounds if the result of 2D6 >= target’s Toughness",
        reroll_damages_key: "Damage roll reroll",
        roll_damages_twice_key: "Roll random damages twice and discard the lowest result",
        snipe_key: "Each _ roll of _+, inflicts _ mortal wound(s)",
        hit_explodes_key: "An unmodified hit roll of _+ scores one additional hit"
    }

    opt_keys = set(opt_key_to_repr.keys())

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
        snipe_key: none,
        hit_explodes_key: none
    }

    def is_activated(self, option_key):
        return self.__getattribute__(option_key) != Options.not_activated_value[option_key]

    incompatibilities = {
        hit_modifier_key: {},
        wound_modifier_key: {},
        save_modifier_key: {},
        reroll_hits_key: {},
        reroll_wounds_key: {},
        dakka3_key: {},
        hit_explodes_key: {},
        auto_wounds_on_key: {},
        is_blast_key: {},
        auto_hit_key: {hit_modifier_key, reroll_hits_key, dakka3_key, hit_explodes_key, auto_wounds_on_key},
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
                 hit_explodes=None,
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
        assert (hit_explodes in {Options.none, 5, 6})
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
        self.hit_explodes = hit_explodes
        self.auto_wounds_on = auto_wounds_on
        self.is_blast = is_blast
        self.auto_hit = auto_hit
        self.wounds_by_2D6 = wounds_by_2D6
        self.reroll_damages = reroll_damages
        self.roll_damages_twice = roll_damages_twice
        self.snipe = snipe  # a part of snipe validation is in Options.parse_snipe and another part in Weapon.__init__

        # Compatibility check:
        for opt_key1, incompatible_opt_keys in Options.incompatibilities.items():
            if self.is_activated(opt_key1):
                for opt_key2 in Options.opt_key_to_repr.keys():
                    if opt_key2 != opt_key1 and self.is_activated(opt_key2):
                        require(
                            opt_key2 not in incompatible_opt_keys,
                            f"Options '{Options.opt_key_to_repr[opt_key1]}' and"
                            f" '{Options.opt_key_to_repr[opt_key2]}' are incompatible"
                        )

    @staticmethod
    def empty():
        return Options()

    @staticmethod
    def parse(options):
        if isinstance(options, Options):
            return options
        else:
            assert (len(options) <= 14)
            assert (all([opt_key in Options.opt_keys for opt_key in options.keys()]))
            # replace missing options by ""
            for option_key in Options.opt_keys:
                options[option_key] = options.get(option_key, "")
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
                hit_explodes=
                int(options[Options.hit_explodes_key]) if len(options[Options.hit_explodes_key]) else Options.none,
                auto_wounds_on=
                int(options[Options.auto_wounds_on_key]) if len(options[Options.auto_wounds_on_key]) else Options.none,
                is_blast=bool(options[Options.is_blast_key]) if len(options[Options.is_blast_key]) else False,
                auto_hit=bool(options[Options.auto_hit_key]) if len(options[Options.auto_hit_key]) else False,
                wounds_by_2D6=
                bool(options[Options.wounds_by_2D6_key]) if len(options[Options.wounds_by_2D6_key]) else False,
                reroll_damages=
                bool(options[Options.reroll_damages_key]) if len(options[Options.reroll_damages_key]) else False,
                roll_damages_twice=
                bool(options[Options.roll_damages_twice_key])
                if len(options[Options.roll_damages_twice_key])
                else False,
                snipe=
                Options.parse_snipe(options[Options.snipe_key]) if len(options[Options.snipe_key]) else Options.none
            )

    @staticmethod
    def parse_snipe(v):
        roll_type, threshold, n_mortals = v.split(",")
        assert (roll_type in {Options.wound, Options.strength})
        threshold = int(threshold)
        require(threshold > 0, f"Threshold input for option '{Options.opt_key_to_repr[Options.snipe_key]}' must be > 0")
        n_mortals = DiceExpr.parse(n_mortals, raise_on_failure=True, complexity_threshold=3)
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
            raise RequirementError(f"Invalid arithmetic expression for points: '{points_expr}'")
        try:
            self.points = int(points_expr_evaluated)
        except ValueError:
            self.points = None
        except Exception as e:
            raise RuntimeError(e)
        require(self.points is not None and self.points > 0, f"Invalid points value: '{points_expr}'")


class Weapon:

    def __init__(self, hit="4", a="1", s="4", ap="0", d="1", options=Options.empty()):
        # prob by roll result: O(n*dice_type)
        self.hit = DiceExpr.parse(hit,
                                  complexity_threshold=float("inf"),
                                  raise_on_failure=True)  # only one time O(n*dice_type)
        require(self.hit.dices_type is None, "Random Ballistic/Weapon Skill is not allowed")
        require(2 <= self.hit.n <= 6, f"Ballistic/Weapon Skill must be between 2 and 6 (included), not '{self.hit}'")
        self.a = DiceExpr.parse(a, complexity_threshold=128, raise_on_failure=True)  # only one time 0(n)
        require(self.a.avg != 0, "Number of Attacks cannot be 0")

        self.ap = DiceExpr.parse(ap, complexity_threshold=12, raise_on_failure=True)  # per each target O(n*dice_type)
        self.d = DiceExpr.parse(d, complexity_threshold=12, raise_on_failure=True)  # exponential exponential compl
        require(self.d.avg != 0, "Damage cannot be 0")
        self.options = Options.parse(options)
        require(not self.options.is_activated(Options.is_blast_key) or self.a.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.is_blast_key]}' "
                f"option with a non random attack characteristic: {self.a}")
        require(not self.options.is_activated(Options.reroll_damages_key) or self.d.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.reroll_damages_key]}' "
                f"option with a non random Damage characteristic: {self.d}")
        require(not self.options.is_activated(Options.roll_damages_twice_key) or self.d.dices_type is not None,
                f"Cannot activate '{Options.opt_key_to_repr[Options.roll_damages_twice_key]}' "
                f"option with a non random Damage characteristic: {self.d}")
        self.s = DiceExpr.parse(s,
                                complexity_threshold=12,
                                raise_on_failure=True,
                                allow_star=self.options.wounds_by_2D6)  # per each target O(n*dice_type)
        require(
            not self.options.is_activated(Options.snipe_key) or self.options.snipe[
                Options.snipe_roll_type] != Options.strength or
            self.s.dices_type is not None,
            lambda: f"""Cannot activate '{Options.opt_key_to_repr[Options.snipe_key]}': The {self.options.snipe[Options.snipe_roll_type]} roll '{self.s}' is not random."""
        )
        require(
            not self.options.is_activated(Options.snipe_key) or self.options.snipe[Options.snipe_threshold] <=
            {Options.strength: self.s.max, Options.wound: 6 + self.options.wound_modifier}[
                self.options.snipe[Options.snipe_roll_type]],
            lambda: f"""Cannot activate '{Options.opt_key_to_repr[Options.snipe_key]}': A {self.options.snipe[Options.snipe_roll_type]} roll of {self.options.snipe[Options.snipe_threshold]}+ is impossible"""
        )
        require(
            not self.options.is_activated(Options.snipe_key) or self.d.n_cases() <= 6,
            lambda: f"""Cannot activate '{Options.opt_key_to_repr[Options.snipe_key]}' with random damage expression '{self.d}'."""
        )
        require(self.s.avg != 0, "Strength cannot be 0")


class Target:
    def __init__(self, t=4, sv=6, invu=None, fnp=None, w=1, n_models=1):
        assert (invu is None or (type(invu) is int and 0 < invu <= 6))
        self.invu = invu

        assert (fnp is None or (type(fnp) is int and 0 < fnp <= 6))
        self.fnp = fnp

        assert (type(t) is int and t > 0)
        self.t = t

        assert (type(sv) is int and 0 < sv <= 6)
        self.sv = sv

        assert (type(w) is int and w > 0)
        self.w = w

        assert (type(n_models) is int and n_models > 0)
        self.n_models = n_models

    def copy(self, **kwargs):
        assert (set(kwargs.keys()).issubset({"t", "sv", "invu", "fnp", "w", "n_models"}))
        return Target(
            t=kwargs.get("t", self.t),
            sv=kwargs.get("sv", self.sv),
            invu=kwargs.get("invu", self.invu),
            fnp=kwargs.get("fnp", self.fnp),
            w=kwargs.get("w", self.w),
            n_models=kwargs.get("n_models", self.n_models)
        )

    def __str__(self):
        return f"Target(T={self.t}, W={self.w}, Sv={self.sv}, Invu={self.invu}, fnp={self.fnp}, #models={self.n_models}"
