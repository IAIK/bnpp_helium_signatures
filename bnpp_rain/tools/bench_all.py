#!/usr/bin/env python3
import sys
import os

executable = "./bench_free"
parser = "./"
logfile = "bench_output.txt"
iterations = 100

# kappa, rain_rounds, N, Tau
min_parameters = [
    (128, 3, 4, 65),
    (128, 3, 8, 44),
    (128, 3, 16, 33),
    (128, 3, 31, 27),
    (128, 3, 48, 24),
    (128, 3, 57, 23),
    (128, 3, 69, 22),
    (128, 3, 85, 21),
    (128, 3, 107, 20),
    (128, 3, 138, 19),
    (128, 3, 185, 18),
    (128, 3, 256, 17),
    (128, 3, 369, 16),
    (128, 3, 563, 15),
    (128, 3, 916, 14),
    (128, 3, 1615, 13),
    (128, 3, 3159, 12),
    (128, 3, 7064, 11),
    (128, 3, 18890, 10),
    (128, 3, 64579, 9),
    (128, 4, 4, 65),
    (128, 4, 8, 44),
    (128, 4, 16, 33),
    (128, 4, 31, 27),
    (128, 4, 48, 24),
    (128, 4, 57, 23),
    (128, 4, 69, 22),
    (128, 4, 85, 21),
    (128, 4, 107, 20),
    (128, 4, 138, 19),
    (128, 4, 185, 18),
    (128, 4, 256, 17),
    (128, 4, 369, 16),
    (128, 4, 563, 15),
    (128, 4, 916, 14),
    (128, 4, 1615, 13),
    (128, 4, 3159, 12),
    (128, 4, 7064, 11),
    (128, 4, 18890, 10),
    (128, 4, 64579, 9),
]

SCALING_FACTOR = 1000


def parse_bench(filename):
    with open(filename, "r") as f:
        content = f.read()

    testruns = content.split("Instance: ")
    if len(testruns) > 1:
        testruns.pop(0)

    for test in testruns:
        lines = test.splitlines()
        # first line is instance:
        # print(lines[0])
        lines.pop(0)
        # second line is header:
        # print(lines[0])
        lines.pop(0)

        count = 0
        keygen, sign, ver, size = 0, 0, 0, 0

        for line in lines:
            if len(line.strip()) == 0:
                continue
            vals = line.strip().split(",")
            keygen += int(vals[0])
            sign += int(vals[1])
            ver += int(vals[2])
            size += int(vals[3])
            count += 1

        keygen = (keygen / SCALING_FACTOR) / count
        sign = (sign / SCALING_FACTOR) / count
        ver = (ver / SCALING_FACTOR) / count
        size = float(size) / count
        print("{:.2f} & {:.2f} & {:.0f} \\\\".format(
            sign, ver, size))


print("Benchmarking 3-round Rain followed by 4-round Rain.")
for kappa, rounds, N, tau in min_parameters:
    os.system("{executable} -i {iter} {kappa} {rounds} {N} {tau} > {logfile}".format(
        executable=executable, iter=iterations, kappa=kappa/8, N=N, tau=tau, rounds=rounds, logfile=logfile))
    print("{kappa} & {N} & {tau} & ".format(kappa=kappa, N=N, tau=tau), end="")
    parse_bench(logfile)
