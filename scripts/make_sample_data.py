#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a small, fully synthetic instruction-style corpus.

The corpus is deterministic (fixed seed) and self-contained, so the repository
carries no third-party text.  It is intentionally template driven: a 5-10M
parameter model can fit it in a few hundred steps, which makes the
self-training pipeline observable end to end on a laptop.

Usage:  python3 scripts/make_sample_data.py [out_path] [--turns N]
"""
import random
import sys

U, A, E = "<|user|>", "<|assistant|>", "<|endoftext|>"

CITIES = [
    ("Tehran", "Iran"), ("Paris", "France"), ("Tokyo", "Japan"),
    ("Berlin", "Germany"), ("Cairo", "Egypt"), ("Ottawa", "Canada"),
    ("Madrid", "Spain"), ("Rome", "Italy"), ("Oslo", "Norway"),
    ("Lisbon", "Portugal"), ("Vienna", "Austria"), ("Athens", "Greece"),
]
COLORS = ["red", "green", "blue", "yellow", "black", "white", "orange", "purple"]
ANIMALS = ["cat", "dog", "horse", "eagle", "whale", "spider", "bee", "fox"]
LEGS = {"cat": 4, "dog": 4, "horse": 4, "eagle": 2, "whale": 0, "spider": 8, "bee": 6, "fox": 4}
LANGS = ["C++", "Python", "Rust", "Go", "Java"]
TOPICS = {
    "a transformer": "a neural network that mixes tokens with attention instead of recurrence",
    "attention": "a weighted average over tokens where the weights come from query and key similarity",
    "a tensor": "a multi dimensional array of numbers with a shape",
    "a gradient": "the vector of partial derivatives of the loss with respect to the parameters",
    "overfitting": "when a model memorises the training data and stops generalising",
    "a tokenizer": "a program that turns text into a sequence of integer ids",
    "quantisation": "storing weights with fewer bits to save memory",
    "a checkpoint": "a file that stores every parameter of the model",
}


def num_word(n):
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve"]
    return words[n] if 0 <= n < len(words) else str(n)


def turn(q, a):
    return f"{U}{q}{A}{a}{E}\n"


def make(rng):
    kind = rng.randrange(10)
    if kind == 0:
        c, k = rng.choice(CITIES)
        return turn(rng.choice([f"what is the capital of {k}?",
                                f"name the capital city of {k}",
                                f"capital of {k}?"]),
                    f"The capital of {k} is {c}.")
    if kind == 1:
        x, y = rng.randrange(1, 50), rng.randrange(1, 50)
        op = rng.choice("+-*")
        v = x + y if op == "+" else x - y if op == "-" else x * y
        return turn(f"compute {x} {op} {y}", f"{x} {op} {y} = {v}.")
    if kind == 2:
        an = rng.choice(ANIMALS)
        return turn(f"how many legs does a {an} have?",
                    f"A {an} has {num_word(LEGS[an])} legs.")
    if kind == 3:
        t, d = rng.choice(list(TOPICS.items()))
        return turn(rng.choice([f"what is {t}?", f"explain {t}", f"define {t}"]),
                    f"{t.capitalize()} is {d}.")
    if kind == 4:
        n = rng.randrange(3, 9)
        seq = ", ".join(str(i) for i in range(1, n + 1))
        return turn(f"count from 1 to {n}", f"{seq}.")
    if kind == 5:
        col = rng.choice(COLORS)
        an = rng.choice(ANIMALS)
        return turn(f"write a short sentence about a {col} {an}",
                    f"The {col} {an} walked quietly through the tall grass.")
    if kind == 6:
        lang = rng.choice(LANGS)
        return turn(f"why do people use {lang}?",
                    f"People use {lang} because it is fast, portable and has a large ecosystem.")
    if kind == 7:
        n = rng.randrange(2, 12)
        return turn(f"what is {n} squared?", f"{n} squared is {n * n}.")
    if kind == 8:
        w = rng.choice(["hello", "hi", "good morning", "hey there"])
        return turn(w, "Hello! How can I help you today?")
    n = rng.randrange(2, 10)
    return turn(f"list {num_word(n)} colors",
                ", ".join(rng.sample(COLORS, min(n, len(COLORS)))) + ".")


PROSE = """The model reads a sequence of tokens and predicts the next token.
Training adjusts every weight a little bit so the prediction becomes better.
A small model with a small context window can still learn a narrow domain well.
When the learning rate is too large the loss diverges and the weights explode.
When the learning rate is too small the loss decreases very slowly.
Attention lets each position look back at every earlier position in the sequence.
Layer normalisation keeps the activations in a stable range across depth.
A residual connection adds the input of a block to its output.
Gradient checkpointing trades extra compute for much lower memory use.
Self training is only safe when every update is measured on held out data.
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "data/sample_corpus.txt"
    turns = 9000
    for i, a in enumerate(sys.argv):
        if a == "--turns" and i + 1 < len(sys.argv):
            turns = int(sys.argv[i + 1])
    rng = random.Random(20240730)
    parts = []
    for i in range(turns):
        parts.append(make(rng))
        if i % 40 == 39:
            parts.append(PROSE)
    text = "".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}: {len(text)} bytes, {turns} dialogue turns")


if __name__ == "__main__":
    main()
