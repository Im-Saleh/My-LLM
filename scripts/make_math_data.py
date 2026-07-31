#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a Persian + English mathematical-reasoning corpus for SPT.

Why generate instead of only downloading
----------------------------------------
A 30M parameter model has no capacity to memorise facts, but it has plenty of
capacity to learn *procedures* - if it sees the procedure written out. Datasets
like GSM8K contain ~7.5k problems: enough to teach the style of an answer, far
too few to teach carrying, borrowing, long multiplication or solving for x. So
this script writes millions of worked examples where every intermediate step is
explicit, and GSM8K is mixed in afterwards for realistic phrasing.

Three rules the format follows, all of which matter at this scale:

1. Every answer shows its work. A model trained on "234 + 567 = 801" learns a
   lookup table it cannot generalise; a model trained on the per-digit carries
   learns an algorithm that works on unseen numbers.
2. One template per skill, phrased several ways. The template makes the pattern
   learnable; the paraphrases stop the model from keying on one exact wording.
3. ASCII digits throughout. The tokenizer's Persian normalisation maps Persian
   and Arabic-Indic digits to ASCII anyway, so writing them here would only
   waste vocabulary on tokens that never survive preprocessing.

Usage:
    python3 scripts/make_math_data.py --out data/math.txt --mb 40
    python3 scripts/make_math_data.py --out data/math.txt --mb 40 --gsm8k
"""

import argparse
import json
import math
import os
import random
import sys
import urllib.request

EOT = "<|endoftext|>"
U, A = "<|user|>", "<|assistant|>"

# --------------------------------------------------------------------- phrasing
# Several ways to ask the same thing, so the skill is attached to the operation
# rather than to one sentence.
ASK_ADD_FA = ["{a} + {b} چند است؟", "حاصل جمع {a} و {b} را حساب کن.",
              "{a} به علاوه {b} مساوی چند؟", "جمع کن: {a} + {b}"]
ASK_ADD_EN = ["What is {a} + {b}?", "Compute {a} plus {b}.",
              "Add {a} and {b}.", "Calculate: {a} + {b}"]
ASK_SUB_FA = ["{a} - {b} چند است؟", "از {a} مقدار {b} را کم کن.",
              "تفاضل {a} و {b} را حساب کن.", "کم کن: {a} - {b}"]
ASK_SUB_EN = ["What is {a} - {b}?", "Subtract {b} from {a}.",
              "Compute the difference of {a} and {b}.", "Calculate: {a} - {b}"]
ASK_MUL_FA = ["{a} × {b} چند است؟", "حاصل ضرب {a} در {b} را حساب کن.",
              "{a} ضربدر {b} مساوی چند؟", "ضرب کن: {a} * {b}"]
ASK_MUL_EN = ["What is {a} * {b}?", "Compute {a} times {b}.",
              "Multiply {a} by {b}.", "Calculate: {a} × {b}"]
ASK_DIV_FA = ["{a} تقسیم بر {b} چند است؟", "حاصل تقسیم {a} بر {b} را با باقیمانده بنویس.",
              "{a} را بر {b} تقسیم کن.", "تقسیم کن: {a} / {b}"]
ASK_DIV_EN = ["What is {a} divided by {b}?", "Divide {a} by {b} with remainder.",
              "Compute {a} / {b}.", "Calculate: {a} ÷ {b}"]

ANS_FA, ANS_EN = "پاسخ:", "Answer:"
PLACE_FA = ["یکان", "دهگان", "صدگان", "هزارگان", "ده‌هزارگان", "صدهزارگان"]
PLACE_EN = ["ones", "tens", "hundreds", "thousands", "ten-thousands",
            "hundred-thousands"]

NAMES_FA = ["علی", "سارا", "رضا", "مریم", "حسین", "زهرا", "امیر", "نگار", "کاوه",
            "لیلا", "بهرام", "شیرین", "نیما", "پریسا", "سینا", "الهام"]
NAMES_EN = ["Ali", "Sara", "Omar", "Maryam", "Hassan", "Zahra", "Amir", "Nora",
            "Kaveh", "Layla", "John", "Emma", "Noah", "Olivia", "Liam", "Ava"]
ITEMS_FA = [("سیب", "عدد"), ("کتاب", "جلد"), ("مداد", "عدد"), ("تخم‌مرغ", "عدد"),
            ("گل", "شاخه"), ("توپ", "عدد"), ("صندلی", "عدد"), ("شکلات", "بسته"),
            ("دفتر", "جلد"), ("پرتقال", "عدد")]
ITEMS_EN = [("apple", "apples"), ("book", "books"), ("pencil", "pencils"),
            ("egg", "eggs"), ("flower", "flowers"), ("ball", "balls"),
            ("chair", "chairs"), ("chocolate", "chocolates"),
            ("notebook", "notebooks"), ("orange", "oranges")]


def turn(q, steps, ans, fa):
    """Wrap one problem in the chat control tokens the corpus uses."""
    label = ANS_FA if fa else ANS_EN
    body = "\n".join(steps)
    if body:
        body += "\n"
    return f"{U}{q}{A}\n{body}{label} {ans}{EOT}\n"


# ------------------------------------------------------------------- arithmetic
def gen_add(rng, fa):
    """Column addition with the carries written out."""
    d = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5])
    a = rng.randrange(10 ** (d - 1), 10 ** d)
    b = rng.randrange(10 ** (d - 1), 10 ** d)
    q = rng.choice(ASK_ADD_FA if fa else ASK_ADD_EN).format(a=a, b=b)
    steps, carry = [f"{a} + {b}"], 0
    places = PLACE_FA if fa else PLACE_EN
    sa, sb = str(a)[::-1], str(b)[::-1]
    for i in range(max(len(sa), len(sb))):
        x = int(sa[i]) if i < len(sa) else 0
        y = int(sb[i]) if i < len(sb) else 0
        t = x + y + carry
        name = places[i] if i < len(places) else (f"رقم {i + 1}" if fa else f"digit {i + 1}")
        if t >= 10:
            if fa:
                steps.append(f"{name}: {x} + {y}{f' + {carry}' if carry else ''} = {t}"
                             f" → {t % 10} می‌نویسیم و 1 نگه می‌داریم")
            else:
                steps.append(f"{name}: {x} + {y}{f' + {carry}' if carry else ''} = {t}"
                             f" → write {t % 10}, carry 1")
            carry = 1
        else:
            steps.append(f"{name}: {x} + {y}{f' + {carry}' if carry else ''} = {t}")
            carry = 0
    if carry:
        steps.append("رقم بعدی: 1" if fa else "next digit: 1")
    return turn(q, steps, a + b, fa)


def gen_sub(rng, fa):
    """Column subtraction, borrowing made explicit."""
    d = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5])
    a = rng.randrange(10 ** (d - 1), 10 ** d)
    b = rng.randrange(1, a + 1)
    q = rng.choice(ASK_SUB_FA if fa else ASK_SUB_EN).format(a=a, b=b)
    steps = [f"{a} - {b}"]
    places = PLACE_FA if fa else PLACE_EN
    sa, sb, borrow = str(a)[::-1], str(b)[::-1], 0
    for i in range(len(sa)):
        x = int(sa[i]) - borrow
        y = int(sb[i]) if i < len(sb) else 0
        name = places[i] if i < len(places) else (f"رقم {i + 1}" if fa else f"digit {i + 1}")
        if x < y:
            x += 10
            if fa:
                steps.append(f"{name}: {x} - {y} = {x - y}  (یک واحد از رقم بعدی قرض گرفتیم)")
            else:
                steps.append(f"{name}: {x} - {y} = {x - y}  (borrowed one from the next digit)")
            borrow = 1
        else:
            steps.append(f"{name}: {x} - {y} = {x - y}")
            borrow = 0
    return turn(q, steps, a - b, fa)


def gen_mul(rng, fa):
    """Long multiplication: partial products, then their sum."""
    if rng.random() < 0.45:
        a, b = rng.randrange(2, 100), rng.randrange(2, 13)
    else:
        a, b = rng.randrange(10, 1000), rng.randrange(10, 100)
    q = rng.choice(ASK_MUL_FA if fa else ASK_MUL_EN).format(a=a, b=b)
    steps, parts = [f"{a} × {b}"], []
    for i, ch in enumerate(str(b)[::-1]):
        dv = int(ch)
        p = a * dv * (10 ** i)
        parts.append(p)
        if dv == 0:
            steps.append(f"{a} × 0 = 0")
            continue
        shift = "0" * i
        if fa:
            steps.append(f"{a} × {dv}{f' × 1{shift}' if i else ''} = {p}")
        else:
            steps.append(f"{a} × {dv}{f' × 1{shift}' if i else ''} = {p}")
    if len(parts) > 1:
        steps.append(("جمع حاصل‌های جزئی: " if fa else "sum of partial products: ")
                     + " + ".join(str(p) for p in parts) + f" = {sum(parts)}")
    return turn(q, steps, a * b, fa)


def gen_div(rng, fa):
    """Division as quotient and remainder, with the check multiplication."""
    b = rng.randrange(2, 40)
    qv = rng.randrange(2, 400)
    r = rng.randrange(0, b)
    a = b * qv + r
    q = rng.choice(ASK_DIV_FA if fa else ASK_DIV_EN).format(a=a, b=b)
    if fa:
        steps = [f"{a} ÷ {b}",
                 f"بزرگ‌ترین مضرب {b} که از {a} بیشتر نشود: {b} × {qv} = {b * qv}",
                 f"باقیمانده: {a} - {b * qv} = {r}",
                 f"بررسی: {b} × {qv} + {r} = {a}"]
        ans = f"{qv} و باقیمانده {r}" if r else str(qv)
    else:
        steps = [f"{a} ÷ {b}",
                 f"largest multiple of {b} not above {a}: {b} × {qv} = {b * qv}",
                 f"remainder: {a} - {b * qv} = {r}",
                 f"check: {b} × {qv} + {r} = {a}"]
        ans = f"{qv} remainder {r}" if r else str(qv)
    return turn(q, steps, ans, fa)


def gen_chain(rng, fa):
    """Order of operations on a three-term expression."""
    a, b, c = (rng.randrange(2, 60) for _ in range(3))
    form = rng.choice(["a+b*c", "a*b-c", "(a+b)*c", "a+b-c", "a*b+c"])
    expr = form.replace("a", str(a)).replace("b", str(b)).replace("c", str(c))
    if form == "a+b*c":
        steps = [f"{b} × {c} = {b * c}", f"{a} + {b * c} = {a + b * c}"]
        val = a + b * c
        rule = "اول ضرب، بعد جمع" if fa else "multiplication before addition"
    elif form == "a*b-c":
        steps = [f"{a} × {b} = {a * b}", f"{a * b} - {c} = {a * b - c}"]
        val = a * b - c
        rule = "اول ضرب، بعد تفریق" if fa else "multiplication before subtraction"
    elif form == "(a+b)*c":
        steps = [f"{a} + {b} = {a + b}", f"{a + b} × {c} = {(a + b) * c}"]
        val = (a + b) * c
        rule = "اول پرانتز" if fa else "parentheses first"
    elif form == "a+b-c":
        steps = [f"{a} + {b} = {a + b}", f"{a + b} - {c} = {a + b - c}"]
        val = a + b - c
        rule = "از چپ به راست" if fa else "left to right"
    else:
        steps = [f"{a} × {b} = {a * b}", f"{a * b} + {c} = {a * b + c}"]
        val = a * b + c
        rule = "اول ضرب، بعد جمع" if fa else "multiplication before addition"
    q = (f"حاصل {expr} چند است؟" if fa else f"What is {expr}?")
    return turn(q, [f"{rule}:"] + steps, val, fa)


def gen_compare(rng, fa):
    a, b = rng.randrange(1, 10 ** rng.choice([2, 3, 4])), rng.randrange(1, 10 ** rng.choice([2, 3, 4]))
    q = (f"کدام بزرگ‌تر است، {a} یا {b}؟" if fa else f"Which is larger, {a} or {b}?")
    if a == b:
        steps = ["هر دو تعداد رقم و ارقام یکسانی دارند." if fa else "same digits."]
        ans = (f"{a} و {b} برابرند" if fa else f"{a} and {b} are equal")
    else:
        big, small = max(a, b), min(a, b)
        if len(str(a)) != len(str(b)):
            steps = [(f"{a} دارای {len(str(a))} رقم و {b} دارای {len(str(b))} رقم است."
                      if fa else
                      f"{a} has {len(str(a))} digits, {b} has {len(str(b))}.")]
        else:
            for i, (x, y) in enumerate(zip(str(a), str(b))):
                if x != y:
                    steps = [(f"از چپ، رقم {i + 1}: {x} در برابر {y}" if fa else
                              f"from the left, digit {i + 1}: {x} vs {y}")]
                    break
        ans = (f"{big} بزرگ‌تر است ({big} > {small})" if fa else
               f"{big} is larger ({big} > {small})")
    return turn(q, steps, ans, fa)


# --------------------------------------------------------------------- fractions
def gen_fraction(rng, fa):
    kind = rng.choice(["add", "simplify", "compare", "of"])
    if kind == "add":
        b, d = rng.randrange(2, 13), rng.randrange(2, 13)
        a, c = rng.randrange(1, b), rng.randrange(1, d)
        l = b * d // math.gcd(b, d)
        na, nc = a * (l // b), c * (l // d)
        g = math.gcd(na + nc, l)
        q = (f"{a}/{b} + {c}/{d} چند است؟" if fa else f"What is {a}/{b} + {c}/{d}?")
        steps = [(f"مخرج مشترک: ک.م.م({b}, {d}) = {l}" if fa else
                  f"common denominator: lcm({b}, {d}) = {l}"),
                 f"{a}/{b} = {na}/{l}", f"{c}/{d} = {nc}/{l}",
                 f"{na}/{l} + {nc}/{l} = {na + nc}/{l}"]
        ans = f"{(na + nc) // g}/{l // g}"
        if g > 1:
            steps.append((f"ساده‌سازی با تقسیم بر {g}" if fa else f"divide both by {g}"))
        if l // g == 1:
            ans = str((na + nc) // g)
    elif kind == "simplify":
        g = rng.randrange(2, 13)
        n, d = rng.randrange(1, 20) * g, rng.randrange(2, 20) * g
        gg = math.gcd(n, d)
        q = (f"کسر {n}/{d} را ساده کن." if fa else f"Simplify {n}/{d}.")
        steps = [(f"ب.م.م({n}, {d}) = {gg}" if fa else f"gcd({n}, {d}) = {gg}"),
                 f"{n} ÷ {gg} = {n // gg}", f"{d} ÷ {gg} = {d // gg}"]
        ans = f"{n // gg}/{d // gg}" if d // gg != 1 else str(n // gg)
    elif kind == "compare":
        b, d = rng.randrange(2, 13), rng.randrange(2, 13)
        a, c = rng.randrange(1, b), rng.randrange(1, d)
        left, right = a * d, c * b
        q = (f"کدام بزرگ‌تر است، {a}/{b} یا {c}/{d}؟" if fa else
             f"Which is larger, {a}/{b} or {c}/{d}?")
        steps = [(f"ضرب متقابل: {a} × {d} = {left} و {c} × {b} = {right}" if fa else
                  f"cross multiply: {a} × {d} = {left} and {c} × {b} = {right}")]
        if left == right:
            ans = ("برابرند" if fa else "they are equal")
        else:
            ans = (f"{a}/{b}" if left > right else f"{c}/{d}") + (" بزرگ‌تر است" if fa else " is larger")
    else:
        d = rng.choice([2, 3, 4, 5, 6, 8, 10])
        n = rng.randrange(1, d)
        whole = rng.randrange(1, 60) * d
        q = (f"{n}/{d} از {whole} چند است؟" if fa else f"What is {n}/{d} of {whole}?")
        steps = [f"{whole} ÷ {d} = {whole // d}", f"{whole // d} × {n} = {whole // d * n}"]
        ans = whole // d * n
    return turn(q, steps, ans, fa)


def gen_percent(rng, fa):
    kind = rng.choice(["of", "increase", "discount", "what_percent"])
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    base = rng.randrange(1, 200) * rng.choice([10, 100, 1000])
    val = base * p // 100
    if kind == "of":
        q = (f"{p}% از {base} چند است؟" if fa else f"What is {p}% of {base}?")
        steps = [(f"{p}% = {p}/100" if fa else f"{p}% = {p}/100"),
                 f"{base} × {p} = {base * p}", f"{base * p} ÷ 100 = {val}"]
        ans = val
    elif kind == "increase":
        q = (f"{base} را {p}% افزایش بده." if fa else f"Increase {base} by {p}%.")
        steps = [f"{p}% × {base} = {val}", f"{base} + {val} = {base + val}"]
        ans = base + val
    elif kind == "discount":
        q = (f"قیمت {base} تومان با {p}% تخفیف چند می‌شود؟" if fa else
             f"A price of {base} with a {p}% discount is how much?")
        steps = [f"{p}% × {base} = {val}", f"{base} - {val} = {base - val}"]
        ans = (f"{base - val} تومان" if fa else str(base - val))
    else:
        part = base * rng.choice([1, 2, 3, 4, 5]) // 10
        q = (f"{part} چند درصد از {base} است؟" if fa else
             f"{part} is what percent of {base}?")
        pct = part * 100 / base
        steps = [f"{part} ÷ {base} = {part / base:g}",
                 f"{part / base:g} × 100 = {pct:g}"]
        ans = f"{pct:g}%"
    return turn(q, steps, ans, fa)


# --------------------------------------------------------------------- algebra
def gen_linear(rng, fa):
    kind = rng.choice(["ax+b", "ax-b", "x/a+b", "a(x+b)"])
    a = rng.randrange(2, 13)
    b = rng.randrange(1, 60)
    x = rng.randrange(1, 40)
    if kind == "ax+b":
        c = a * x + b
        q = (f"معادله را حل کن: {a}x + {b} = {c}" if fa else f"Solve for x: {a}x + {b} = {c}")
        steps = [(f"{b} را از دو طرف کم می‌کنیم: {a}x = {c} - {b} = {a * x}" if fa else
                  f"subtract {b} from both sides: {a}x = {c} - {b} = {a * x}"),
                 (f"بر {a} تقسیم می‌کنیم: x = {a * x} ÷ {a} = {x}" if fa else
                  f"divide by {a}: x = {a * x} ÷ {a} = {x}")]
    elif kind == "ax-b":
        c = a * x - b
        q = (f"معادله را حل کن: {a}x - {b} = {c}" if fa else f"Solve for x: {a}x - {b} = {c}")
        steps = [(f"{b} را به دو طرف اضافه می‌کنیم: {a}x = {c} + {b} = {a * x}" if fa else
                  f"add {b} to both sides: {a}x = {c} + {b} = {a * x}"),
                 (f"بر {a} تقسیم می‌کنیم: x = {x}" if fa else f"divide by {a}: x = {x}")]
    elif kind == "x/a+b":
        c = x + b
        q = (f"معادله را حل کن: x/{a} + {b} = {c // 1}" if fa else
             f"Solve for x: x/{a} + {b} = {c}")
        steps = [(f"{b} را کم می‌کنیم: x/{a} = {c} - {b} = {x}" if fa else
                  f"subtract {b}: x/{a} = {c} - {b} = {x}"),
                 (f"در {a} ضرب می‌کنیم: x = {x} × {a} = {x * a}" if fa else
                  f"multiply by {a}: x = {x} × {a} = {x * a}")]
        x = x * a
    else:
        c = a * (x + b)
        q = (f"معادله را حل کن: {a}(x + {b}) = {c}" if fa else
             f"Solve for x: {a}(x + {b}) = {c}")
        steps = [(f"بر {a} تقسیم می‌کنیم: x + {b} = {c} ÷ {a} = {x + b}" if fa else
                  f"divide by {a}: x + {b} = {c} ÷ {a} = {x + b}"),
                 (f"{b} را کم می‌کنیم: x = {x}" if fa else f"subtract {b}: x = {x}")]
    return turn(q, steps, f"x = {x}", fa)


def gen_gcd_lcm(rng, fa):
    a, b = rng.randrange(4, 200), rng.randrange(4, 200)
    if rng.random() < 0.5:
        q = (f"ب.م.م {a} و {b} را حساب کن." if fa else f"Compute gcd({a}, {b}).")
        steps, x, y = [], a, b
        while y:
            steps.append((f"{x} = {y} × {x // y} + {x % y}" if fa else
                          f"{x} = {y} × {x // y} + {x % y}"))
            x, y = y, x % y
        steps.insert(0, ("با الگوریتم اقلیدس:" if fa else "Euclidean algorithm:"))
        ans = x
    else:
        g = math.gcd(a, b)
        q = (f"ک.م.م {a} و {b} را حساب کن." if fa else f"Compute lcm({a}, {b}).")
        steps = [(f"ب.م.م({a}, {b}) = {g}" if fa else f"gcd({a}, {b}) = {g}"),
                 (f"ک.م.م = {a} × {b} ÷ {g} = {a * b} ÷ {g} = {a * b // g}" if fa else
                  f"lcm = {a} × {b} ÷ {g} = {a * b} ÷ {g} = {a * b // g}")]
        ans = a * b // g
    return turn(q, steps, ans, fa)


def gen_factor(rng, fa):
    n = rng.randrange(12, 500)
    m, fs = n, []
    d = 2
    while d * d <= m:
        while m % d == 0:
            fs.append(d)
            m //= d
        d += 1
    if m > 1:
        fs.append(m)
    q = (f"{n} را به عوامل اول تجزیه کن." if fa else f"Factor {n} into primes.")
    steps, cur = [], n
    for f in fs:
        steps.append(f"{cur} ÷ {f} = {cur // f}")
        cur //= f
    ans = " × ".join(str(f) for f in fs)
    if len(fs) == 1:
        ans = f"{n} " + ("اول است" if fa else "is prime")
    return turn(q, steps, ans, fa)


def gen_sequence(rng, fa):
    if rng.random() < 0.6:
        a0, d = rng.randrange(1, 30), rng.randrange(2, 15)
        seq = [a0 + i * d for i in range(5)]
        q = (f"جمله بعدی این دنباله چیست؟ {', '.join(map(str, seq))}" if fa else
             f"What comes next? {', '.join(map(str, seq))}")
        steps = [(f"تفاضل دو جمله متوالی: {seq[1]} - {seq[0]} = {d}" if fa else
                  f"difference between terms: {seq[1]} - {seq[0]} = {d}"),
                 (f"دنباله حسابی با تفاضل {d} است." if fa else
                  f"arithmetic sequence with difference {d}."),
                 f"{seq[-1]} + {d} = {seq[-1] + d}"]
        ans = seq[-1] + d
    else:
        a0, r = rng.randrange(1, 8), rng.randrange(2, 5)
        seq = [a0 * r ** i for i in range(5)]
        q = (f"جمله بعدی این دنباله چیست؟ {', '.join(map(str, seq))}" if fa else
             f"What comes next? {', '.join(map(str, seq))}")
        steps = [(f"نسبت دو جمله متوالی: {seq[1]} ÷ {seq[0]} = {r}" if fa else
                  f"ratio between terms: {seq[1]} ÷ {seq[0]} = {r}"),
                 (f"دنباله هندسی با نسبت {r} است." if fa else
                  f"geometric sequence with ratio {r}."),
                 f"{seq[-1]} × {r} = {seq[-1] * r}"]
        ans = seq[-1] * r
    return turn(q, steps, ans, fa)


# --------------------------------------------------------------------- geometry
def gen_geometry(rng, fa):
    kind = rng.choice(["rect_area", "rect_perim", "triangle", "circle", "cube"])
    if kind == "rect_area":
        w, h = rng.randrange(2, 60), rng.randrange(2, 60)
        q = (f"مساحت مستطیلی با طول {w} و عرض {h} چند است؟" if fa else
             f"Area of a rectangle {w} by {h}?")
        steps = [("مساحت = طول × عرض" if fa else "area = length × width"),
                 f"{w} × {h} = {w * h}"]
        ans = w * h
    elif kind == "rect_perim":
        w, h = rng.randrange(2, 60), rng.randrange(2, 60)
        q = (f"محیط مستطیلی با طول {w} و عرض {h} چند است؟" if fa else
             f"Perimeter of a rectangle {w} by {h}?")
        steps = [("محیط = 2 × (طول + عرض)" if fa else "perimeter = 2 × (length + width)"),
                 f"{w} + {h} = {w + h}", f"2 × {w + h} = {2 * (w + h)}"]
        ans = 2 * (w + h)
    elif kind == "triangle":
        b, h = rng.randrange(2, 60), rng.choice([2, 4, 6, 8, 10, 12, 14, 16])
        q = (f"مساحت مثلثی با قاعده {b} و ارتفاع {h} چند است؟" if fa else
             f"Area of a triangle with base {b} and height {h}?")
        steps = [("مساحت = قاعده × ارتفاع ÷ 2" if fa else "area = base × height ÷ 2"),
                 f"{b} × {h} = {b * h}", f"{b * h} ÷ 2 = {b * h // 2}"]
        ans = b * h // 2
    elif kind == "circle":
        r = rng.randrange(1, 30)
        q = (f"محیط دایره‌ای با شعاع {r} را با π ≈ 3.14 حساب کن." if fa else
             f"Circumference of a circle with radius {r}, using π ≈ 3.14?")
        steps = [("محیط = 2 × π × r" if fa else "circumference = 2 × π × r"),
                 f"2 × 3.14 = 6.28", f"6.28 × {r} = {6.28 * r:g}"]
        ans = f"{6.28 * r:g}"
    else:
        s = rng.randrange(2, 25)
        q = (f"حجم مکعبی با ضلع {s} چند است؟" if fa else
             f"Volume of a cube with side {s}?")
        steps = [("حجم = ضلع × ضلع × ضلع" if fa else "volume = side³"),
                 f"{s} × {s} = {s * s}", f"{s * s} × {s} = {s ** 3}"]
        ans = s ** 3
    return turn(q, steps, ans, fa)


def gen_units(rng, fa):
    table = [("متر", "سانتی‌متر", 100, "m", "cm"), ("کیلومتر", "متر", 1000, "km", "m"),
             ("کیلوگرم", "گرم", 1000, "kg", "g"), ("ساعت", "دقیقه", 60, "hour", "minutes"),
             ("دقیقه", "ثانیه", 60, "minute", "seconds"), ("لیتر", "میلی‌لیتر", 1000, "l", "ml")]
    fa_from, fa_to, f, en_from, en_to = rng.choice(table)
    n = rng.randrange(2, 200)
    if fa:
        q = f"{n} {fa_from} چند {fa_to} است؟"
        steps = [f"1 {fa_from} = {f} {fa_to}", f"{n} × {f} = {n * f}"]
        ans = f"{n * f} {fa_to}"
    else:
        q = f"How many {en_to} are in {n} {en_from}?"
        steps = [f"1 {en_from} = {f} {en_to}", f"{n} × {f} = {n * f}"]
        ans = f"{n * f} {en_to}"
    return turn(q, steps, ans, fa)


# ---------------------------------------------------------------- word problems
def gen_word(rng, fa):
    kind = rng.choice(["buy", "share", "rate", "remain", "total", "age"])
    if fa:
        name, name2 = rng.sample(NAMES_FA, 2)
        item, unit = rng.choice(ITEMS_FA)
    else:
        name, name2 = rng.sample(NAMES_EN, 2)
        item, plural = rng.choice(ITEMS_EN)

    if kind == "buy":
        n, price = rng.randrange(2, 30), rng.randrange(1, 60) * 500
        if fa:
            q = f"{name} {n} {unit} {item} خرید و هر کدام {price} تومان بود. در کل چقدر پرداخت کرد؟"
            steps = [f"تعداد × قیمت هر واحد", f"{n} × {price} = {n * price}"]
            ans = f"{n * price} تومان"
        else:
            q = f"{name} bought {n} {plural} at {price} each. What was the total?"
            steps = ["count × unit price", f"{n} × {price} = {n * price}"]
            ans = n * price
    elif kind == "share":
        people = rng.randrange(2, 12)
        each = rng.randrange(2, 30)
        total = people * each + rng.randrange(0, people)
        r = total % people
        if fa:
            q = (f"{total} {unit} {item} را بین {people} نفر به‌طور مساوی تقسیم می‌کنیم. "
                 f"هر نفر چند تا می‌گیرد و چند تا باقی می‌ماند؟")
            steps = [f"{total} ÷ {people} = {total // people}",
                     f"باقیمانده: {total} - {people} × {total // people} = {r}"]
            ans = f"هر نفر {total // people} تا، و {r} تا باقی می‌ماند"
        else:
            q = (f"{total} {plural} are shared equally among {people} people. "
                 f"How many does each get and how many are left?")
            steps = [f"{total} ÷ {people} = {total // people}",
                     f"remainder: {total} - {people} × {total // people} = {r}"]
            ans = f"{total // people} each, {r} left over"
    elif kind == "rate":
        speed, hours = rng.randrange(20, 120), rng.randrange(2, 12)
        if fa:
            q = f"خودرویی با سرعت {speed} کیلومتر بر ساعت، {hours} ساعت حرکت می‌کند. چند کیلومتر می‌رود؟"
            steps = ["مسافت = سرعت × زمان", f"{speed} × {hours} = {speed * hours}"]
            ans = f"{speed * hours} کیلومتر"
        else:
            q = f"A car travels at {speed} km/h for {hours} hours. How far does it go?"
            steps = ["distance = speed × time", f"{speed} × {hours} = {speed * hours}"]
            ans = f"{speed * hours} km"
    elif kind == "remain":
        start = rng.randrange(20, 300)
        used = rng.randrange(1, start)
        if fa:
            q = f"{name} {start} {unit} {item} داشت و {used} {unit} از آن را مصرف کرد. چند {unit} مانده است؟"
            steps = [f"{start} - {used} = {start - used}"]
            ans = f"{start - used} {unit}"
        else:
            q = f"{name} had {start} {plural} and used {used}. How many are left?"
            steps = [f"{start} - {used} = {start - used}"]
            ans = start - used
    elif kind == "total":
        a, b = rng.randrange(5, 200), rng.randrange(5, 200)
        if fa:
            q = f"{name} {a} {unit} و {name2} {b} {unit} {item} دارد. مجموعاً چند {unit} دارند؟"
            steps = [f"{a} + {b} = {a + b}"]
            ans = f"{a + b} {unit}"
        else:
            q = f"{name} has {a} {plural} and {name2} has {b}. How many in total?"
            steps = [f"{a} + {b} = {a + b}"]
            ans = a + b
    else:
        age, diff = rng.randrange(6, 50), rng.randrange(2, 25)
        if fa:
            q = f"{name} {age} سال دارد و {diff} سال از {name2} بزرگ‌تر است. {name2} چند سال دارد؟"
            steps = [f"{age} - {diff} = {age - diff}"]
            ans = f"{age - diff} سال"
        else:
            q = f"{name} is {age} years old and {diff} years older than {name2}. How old is {name2}?"
            steps = [f"{age} - {diff} = {age - diff}"]
            ans = age - diff
    return turn(q, steps, ans, fa)


GENERATORS = [
    (gen_add, 14), (gen_sub, 12), (gen_mul, 13), (gen_div, 11),
    (gen_chain, 7), (gen_compare, 4), (gen_fraction, 8), (gen_percent, 8),
    (gen_linear, 8), (gen_gcd_lcm, 4), (gen_factor, 3), (gen_sequence, 4),
    (gen_geometry, 6), (gen_units, 4), (gen_word, 14),
]


def fetch_gsm8k(dest, limit=8000):
    """Real word problems, for phrasing the synthetic set cannot cover."""
    url = ("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/"
           "train-00000-of-00001.parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow missing - skipping GSM8K", file=sys.stderr)
        return 0
    tmp = dest + ".gsm8k.parquet"
    try:
        print("  downloading GSM8K ...", file=sys.stderr)
        urllib.request.urlretrieve(url, tmp)
        tbl = pq.read_table(tmp)
        qs = tbl.column("question").to_pylist()
        ans = tbl.column("answer").to_pylist()
    except Exception as exc:  # network, format, anything
        print(f"  GSM8K unavailable ({exc}) - continuing without it", file=sys.stderr)
        return 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    n = 0
    with open(dest, "a", encoding="utf-8") as f:
        for q, a in list(zip(qs, ans))[:limit]:
            # GSM8K marks the final answer with "#### N"; keep the reasoning and
            # relabel the answer so the format matches everything else here.
            a = a.replace("####", "\nAnswer:")
            # The <<...>> calculator annotations are noise for a small model.
            while "<<" in a and ">>" in a:
                i, j = a.index("<<"), a.index(">>")
                a = a[:i] + a[j + 2:]
            f.write(f"{U}{q.strip()}{A}\n{a.strip()}{EOT}\n")
            n += 1
    print(f"  GSM8K: {n} problems", file=sys.stderr)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/math.txt")
    ap.add_argument("--mb", type=float, default=40.0, help="target megabytes")
    ap.add_argument("--fa-ratio", type=float, default=0.62,
                    help="fraction of problems written in Persian")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gsm8k", action="store_true", help="also download GSM8K")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    target = int(args.mb * 1024 * 1024)

    pool = []
    for fn, w in GENERATORS:
        pool.extend([fn] * w)

    written, count = 0, 0
    per_kind = {}
    with open(args.out, "w", encoding="utf-8") as f:
        buf = []
        while written < target:
            fn = rng.choice(pool)
            fa = rng.random() < args.fa_ratio
            try:
                s = fn(rng, fa)
            except Exception:
                continue  # a degenerate random draw; just take another
            buf.append(s)
            written += len(s.encode("utf-8"))
            count += 1
            per_kind[fn.__name__] = per_kind.get(fn.__name__, 0) + 1
            if len(buf) >= 4096:
                f.write("".join(buf))
                buf.clear()
        f.write("".join(buf))

    if args.gsm8k:
        fetch_gsm8k(args.out)

    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {size / 1048576:.1f} MiB, {count} problems "
          f"({args.fa_ratio:.0%} Persian)")
    for k in sorted(per_kind, key=lambda x: -per_kind[x]):
        print(f"  {k:<14} {per_kind[k]:>8}")


if __name__ == "__main__":
    main()
