#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a deterministic, fully synthetic trilingual corpus: Persian, English
and Python.

Why synthetic: the repository stays free of third-party text, the corpus is
reproducible from a seed, and every sample has a *known* answer, which is what
lets the autopilot score the model automatically.  For real training replace it
with the sources listed in docs/MULTILINGUAL.fa.md (`scripts/fetch_data.sh`).

The Persian part deliberately exercises the hard parts of Persian tokenisation:
ZWNJ (نیم‌فاصله) inside words, Persian digits, «guillemets», Arabic punctuation
and compound verbs.

Usage:
    python3 scripts/make_trilingual_data.py                 # -> data/{fa,en,py}.txt
    python3 scripts/make_trilingual_data.py --turns 12000 --out-dir data
"""
import argparse
import os
import random

U, A, E = "<|user|>", "<|assistant|>", "<|endoftext|>"


def turn(q, a):
    return f"{U}{q}{A}{a}{E}\n"


# --------------------------------------------------------------------- Persian
FA_CITIES = [
    ("تهران", "ایران"), ("پاریس", "فرانسه"), ("توکیو", "ژاپن"), ("برلین", "آلمان"),
    ("قاهره", "مصر"), ("اتاوا", "کانادا"), ("مادرید", "اسپانیا"), ("رم", "ایتالیا"),
    ("اسلو", "نروژ"), ("لیسبون", "پرتغال"), ("وین", "اتریش"), ("آتن", "یونان"),
]
FA_COLORS = ["سرخ", "سبز", "آبی", "زرد", "سیاه", "سفید", "نارنجی", "بنفش"]
FA_ANIMALS = {"گربه": 4, "سگ": 4, "اسب": 4, "عقاب": 2, "نهنگ": 0, "عنکبوت": 8, "زنبور": 6, "روباه": 4}
FA_NUM = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده", "یازده", "دوازده"]
FA_TOPICS = {
    "ترنسفورمر": "یک شبکه‌ی عصبی است که به‌جای بازگشتی‌بودن، توکن‌ها را با مکانیزم توجه ترکیب می‌کند",
    "توجه": "میانگینی وزن‌دار روی توکن‌ها است که وزن‌هایش از شباهت پرسش و کلید می‌آید",
    "تنسور": "یک آرایه‌ی چندبعدی از اعداد با یک شکل مشخص است",
    "گرادیان": "بردار مشتق‌های جزئی تابع زیان نسبت به پارامترهاست",
    "بیش‌برازش": "وقتی است که مدل داده‌ی آموزش را حفظ می‌کند و توان تعمیم را از دست می‌دهد",
    "توکنایزر": "برنامه‌ای است که متن را به دنباله‌ای از شناسه‌های عددی تبدیل می‌کند",
    "کوانتیزاسیون": "ذخیره‌ی وزن‌ها با تعداد بیت کمتر برای صرفه‌جویی در حافظه است",
    "نیم‌فاصله": "فاصله‌ی مجازی است که دو بخش یک کلمه را جدا می‌کند بدون آنکه فاصله‌ی کامل بگذارد",
    "چک‌پوینت": "فایلی است که همه‌ی پارامترهای مدل را نگه می‌دارد",
    "یادگیری مداوم": "به‌روزرسانی تدریجی مدل با داده‌ی تازه، بدون فراموش‌کردن دانش قبلی است",
}
FA_PROSE = """مدل دنباله‌ای از توکن‌ها را می‌خواند و توکن بعدی را پیش‌بینی می‌کند.
آموزش، هر وزن را کمی تغییر می‌دهد تا پیش‌بینی بهتر شود.
اگر نرخ یادگیری بزرگ باشد، تابع زیان واگرا می‌شود و وزن‌ها منفجر می‌شوند.
اگر نرخ یادگیری کوچک باشد، تابع زیان خیلی کند کاهش پیدا می‌کند.
مکانیزم توجه اجازه می‌دهد هر موقعیت به همه‌ی موقعیت‌های قبلی نگاه کند.
نرمال‌سازی لایه‌ای فعال‌سازی‌ها را در بازه‌ای پایدار نگه می‌دارد.
اتصال باقی‌مانده، ورودی هر بلوک را به خروجی‌اش اضافه می‌کند.
ذخیره‌سازی مجدد گرادیان، حافظه را به قیمت محاسبه‌ی بیشتر کم می‌کند.
خودآموزی تنها وقتی امن است که هر به‌روزرسانی روی داده‌ی نگه‌داشته‌شده سنجیده شود.
کتاب‌ها را می‌خوانم و درباره‌ی برنامه‌نویسی می‌نویسم.
"""


def make_fa(rng):
    k = rng.randrange(10)
    if k == 0:
        c, kk = rng.choice(FA_CITIES)
        return turn(rng.choice([f"پایتخت {kk} کجاست؟", f"پایتخت {kk} چه شهری است؟",
                                f"نام پایتخت {kk} را بگو"]),
                    f"پایتخت {kk} شهر {c} است.")
    if k == 1:
        x, y = rng.randrange(1, 50), rng.randrange(1, 50)
        op = rng.choice("+-*")
        v = x + y if op == "+" else x - y if op == "-" else x * y
        return turn(f"حاصل {x} {op} {y} چند است؟", f"{x} {op} {y} = {v}")
    if k == 2:
        an = rng.choice(list(FA_ANIMALS))
        return turn(f"{an} چند پا دارد؟", f"{an} {FA_NUM[FA_ANIMALS[an]]} پا دارد.")
    if k == 3:
        t, d = rng.choice(list(FA_TOPICS.items()))
        return turn(rng.choice([f"{t} چیست؟", f"{t} را توضیح بده", f"تعریف {t} را بنویس"]),
                    f"{t} {d}.")
    if k == 4:
        n = rng.randrange(3, 9)
        return turn(f"از ۱ تا {n} بشمار", "، ".join(str(i) for i in range(1, n + 1)) + ".")
    if k == 5:
        col, an = rng.choice(FA_COLORS), rng.choice(list(FA_ANIMALS))
        return turn(f"یک جمله‌ی کوتاه درباره‌ی یک {an} {col} بنویس",
                    f"{an} {col} آرام از میان علف‌های بلند می‌گذشت.")
    if k == 6:
        return turn("چرا از پایتون استفاده می‌کنند؟",
                    "پایتون را به‌خاطر خوانایی بالا، کتابخانه‌های فراوان و سرعت توسعه انتخاب می‌کنند.")
    if k == 7:
        n = rng.randrange(2, 12)
        return turn(f"مربع {n} چند است؟", f"مربع {n} برابر {n * n} است.")
    if k == 8:
        g = rng.choice(["سلام", "درود", "صبح بخیر", "سلام علیکم"])
        return turn(g, "سلام! چطور می‌توانم کمکتان کنم؟")
    n = rng.randrange(2, 8)
    return turn(f"{FA_NUM[n]} رنگ نام ببر", "، ".join(rng.sample(FA_COLORS, n)) + ".")


# --------------------------------------------------------------------- English
EN_CITIES = [("Tehran", "Iran"), ("Paris", "France"), ("Tokyo", "Japan"),
             ("Berlin", "Germany"), ("Cairo", "Egypt"), ("Ottawa", "Canada"),
             ("Madrid", "Spain"), ("Rome", "Italy"), ("Oslo", "Norway"),
             ("Lisbon", "Portugal"), ("Vienna", "Austria"), ("Athens", "Greece")]
EN_COLORS = ["red", "green", "blue", "yellow", "black", "white", "orange", "purple"]
EN_ANIMALS = {"cat": 4, "dog": 4, "horse": 4, "eagle": 2, "whale": 0, "spider": 8, "bee": 6, "fox": 4}
EN_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve"]
EN_TOPICS = {
    "a transformer": "a neural network that mixes tokens with attention instead of recurrence",
    "attention": "a weighted average over tokens where the weights come from query and key similarity",
    "a tensor": "a multi dimensional array of numbers with a shape",
    "a gradient": "the vector of partial derivatives of the loss with respect to the parameters",
    "overfitting": "when a model memorises the training data and stops generalising",
    "a tokenizer": "a program that turns text into a sequence of integer ids",
    "quantisation": "storing weights with fewer bits to save memory",
    "a checkpoint": "a file that stores every parameter of the model",
    "continual learning": "updating a model with fresh data without forgetting what it knew",
}
EN_PROSE = """The model reads a sequence of tokens and predicts the next token.
Training adjusts every weight a little bit so the prediction becomes better.
When the learning rate is too large the loss diverges and the weights explode.
Attention lets each position look back at every earlier position in the sequence.
Layer normalisation keeps the activations in a stable range across depth.
A residual connection adds the input of a block to its output.
Gradient checkpointing trades extra compute for much lower memory use.
Self training is only safe when every update is measured on held out data.
"""


def make_en(rng):
    k = rng.randrange(10)
    if k == 0:
        c, kk = rng.choice(EN_CITIES)
        return turn(rng.choice([f"what is the capital of {kk}?", f"capital of {kk}?",
                                f"name the capital city of {kk}"]),
                    f"The capital of {kk} is {c}.")
    if k == 1:
        x, y = rng.randrange(1, 50), rng.randrange(1, 50)
        op = rng.choice("+-*")
        v = x + y if op == "+" else x - y if op == "-" else x * y
        return turn(f"compute {x} {op} {y}", f"{x} {op} {y} = {v}")
    if k == 2:
        an = rng.choice(list(EN_ANIMALS))
        return turn(f"how many legs does a {an} have?",
                    f"A {an} has {EN_NUM[EN_ANIMALS[an]]} legs.")
    if k == 3:
        t, d = rng.choice(list(EN_TOPICS.items()))
        return turn(rng.choice([f"what is {t}?", f"explain {t}", f"define {t}"]),
                    f"{t.capitalize()} is {d}.")
    if k == 4:
        n = rng.randrange(3, 9)
        return turn(f"count from 1 to {n}", ", ".join(str(i) for i in range(1, n + 1)) + ".")
    if k == 5:
        col, an = rng.choice(EN_COLORS), rng.choice(list(EN_ANIMALS))
        return turn(f"write a short sentence about a {col} {an}",
                    f"The {col} {an} walked quietly through the tall grass.")
    if k == 6:
        return turn("why do people use Python?",
                    "People use Python because it is readable, has a huge ecosystem and is quick to write.")
    if k == 7:
        n = rng.randrange(2, 12)
        return turn(f"what is {n} squared?", f"{n} squared is {n * n}.")
    if k == 8:
        g = rng.choice(["hello", "hi", "good morning", "hey there"])
        return turn(g, "Hello! How can I help you today?")
    n = rng.randrange(2, 8)
    return turn(f"list {EN_NUM[n]} colors", ", ".join(rng.sample(EN_COLORS, n)) + ".")


# ---------------------------------------------------------------------- Python
def py_add(name, op, sym, doc_fa):
    body = f"""def {name}(a, b):
    \"\"\"{doc_fa}\"\"\"
    return a {sym} b
"""
    return body


PY_SNIPPETS = [
    ("sum_list", """def sum_list(values):
    \"\"\"Return the sum of a list of numbers.\"\"\"
    total = 0
    for v in values:
        total += v
    return total
"""),
    ("mean", """def mean(values):
    \"\"\"Return the arithmetic mean, or 0.0 for an empty list.\"\"\"
    if not values:
        return 0.0
    return sum(values) / len(values)
"""),
    ("fib", """def fib(n):
    \"\"\"Return the n-th Fibonacci number iteratively.\"\"\"
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""),
    ("is_prime", """def is_prime(n):
    \"\"\"Return True when n is a prime number.\"\"\"
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
"""),
    ("reverse_words", """def reverse_words(text):
    \"\"\"Reverse the order of the words in a sentence.\"\"\"
    words = text.split()
    words.reverse()
    return " ".join(words)
"""),
    ("count_chars", """def count_chars(text):
    \"\"\"Count how often each character appears.\"\"\"
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
"""),
    ("read_lines", """def read_lines(path):
    \"\"\"Read a text file and return its lines without the newline.\"\"\"
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\\n") for line in f]
"""),
    ("safe_div", """def safe_div(a, b):
    \"\"\"Divide a by b and return None when b is zero.\"\"\"
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""),
    ("Counter", """class Counter:
    \"\"\"A minimal counter with an increment and a total.\"\"\"

    def __init__(self):
        self.values = {}

    def add(self, key, amount=1):
        # dict.get keeps the code short and avoids a KeyError
        self.values[key] = self.values.get(key, 0) + amount

    def total(self):
        return sum(self.values.values())
"""),
    ("squares", """def squares(n):
    \"\"\"Return the squares of the first n natural numbers.\"\"\"
    # a list comprehension is faster than append in a loop
    return [i * i for i in range(1, n + 1)]
"""),
    ("flatten", """def flatten(nested):
    \"\"\"Flatten one level of a nested list.\"\"\"
    out = []
    for item in nested:
        out.extend(item)
    return out
"""),
    ("normalise", """def normalise(values):
    \"\"\"Scale a list of numbers into the range 0..1.\"\"\"
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]
"""),
]

PY_ASK_FA = [
    "یک تابع پایتون بنویس که {}",
    "کد پایتون برای {} بنویس",
    "با پایتون تابعی بنویس که {}",
]
PY_ASK_EN = [
    "write a python function that {}",
    "show me python code that {}",
    "implement a python helper that {}",
]
PY_DESC = {
    "sum_list": ("مجموع یک لیست از اعداد را برگرداند", "returns the sum of a list of numbers"),
    "mean": ("میانگین یک لیست را حساب کند", "computes the mean of a list"),
    "fib": ("عدد فیبوناچی n-ام را حساب کند", "computes the n-th Fibonacci number"),
    "is_prime": ("بررسی کند عدد اول است یا نه", "checks whether a number is prime"),
    "reverse_words": ("ترتیب کلمات یک جمله را برعکس کند", "reverses the words of a sentence"),
    "count_chars": ("تعداد تکرار هر کاراکتر را بشمارد", "counts how often each character appears"),
    "read_lines": ("خطوط یک فایل متنی را بخواند", "reads the lines of a text file"),
    "safe_div": ("تقسیم امن انجام دهد و صفر را مدیریت کند", "divides safely and handles zero"),
    "Counter": ("یک شمارنده‌ی ساده به‌صورت کلاس بسازد", "defines a small counter class"),
    "squares": ("مربع n عدد اول طبیعی را برگرداند", "returns the squares of the first n numbers"),
    "flatten": ("یک لیست تودرتو را یک‌لایه کند", "flattens a nested list"),
    "normalise": ("اعداد را به بازه‌ی صفر تا یک ببرد", "scales numbers into the 0..1 range"),
}


def make_py(rng, with_instruction=True):
    name, code = rng.choice(PY_SNIPPETS)
    if not with_instruction:
        return code + "\n"
    fa_desc, en_desc = PY_DESC[name]
    if rng.random() < 0.5:
        q = rng.choice(PY_ASK_FA).format(fa_desc)
    else:
        q = rng.choice(PY_ASK_EN).format(en_desc)
    return turn(q, "```python\n" + code + "```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--turns", type=int, default=9000, help="dialogue turns per language")
    ap.add_argument("--seed", type=int, default=20240730)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    parts = {"fa": [], "en": [], "py": []}
    for i in range(args.turns):
        parts["fa"].append(make_fa(rng))
        parts["en"].append(make_en(rng))
        # two thirds of the Python data is instruction shaped, one third is raw
        # source: raw code teaches syntax, instructions teach how to answer.
        parts["py"].append(make_py(rng, with_instruction=(i % 3 != 0)))
        if i % 40 == 39:
            parts["fa"].append(FA_PROSE)
            parts["en"].append(EN_PROSE)

    for code, chunks in parts.items():
        path = os.path.join(args.out_dir, f"{code}.txt")
        text = "".join(chunks)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {path}: {len(text.encode('utf-8'))} bytes")

    # A single mixed file for tools that want one corpus.
    mixed = os.path.join(args.out_dir, "mixed.txt")
    with open(mixed, "w", encoding="utf-8") as f:
        for code in ("fa", "en", "py"):
            with open(os.path.join(args.out_dir, f"{code}.txt"), encoding="utf-8") as src:
                f.write(src.read())
    print(f"wrote {mixed}")


if __name__ == "__main__":
    main()
