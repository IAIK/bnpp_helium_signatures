#!/usr/bin/env python3
import sys
import os

executable = "./bench_free"
parser = "./"
logfile = "bench_output.txt"
iterations = 101


# AES 128
# kappa, sboxes, N, Tau
aes128_parameters = [
    (128, 200, 17, 31),
    (128, 200, 19, 30),
    (128, 200, 31, 26),
    (128, 200, 57, 22),
    (128, 200, 107, 19),
    (128, 200, 139, 18),
    (128, 200, 185, 17),
    (128, 200, 255, 16),
    (128, 200, 371, 15),
    (128, 200, 565, 14),
    (128, 200, 921, 13),
    (128, 200, 1625, 12),
    (128, 200, 3183, 11),
    (128, 200, 7131, 10),
    #    (128, 200, 65535, 8)
]

"""
    # The params below are up-to-date, but the bench program doesn't support them yet
    (192, 256, 8, 68),
    (192, 256, 16, 52),
    (192, 256, 31, 43),
    (192, 256, 64, 36),
    (192, 256, 116, 32),
    (192, 256, 254, 28),
    (192, 256, 418, 26),
    (192, 256, 758, 24),
    (192, 256, 1554, 22),
    (192, 256, 2347, 21),
    (192, 256, 4096, 19),
    (192, 256, 65536, 15),

    (256, 340, 8, 91),
    (256, 340, 16, 69),
    (256, 340, 31, 57),
    (256, 340, 62, 48),
    (256, 340, 122, 42),
    (256, 340, 256, 37),
    (256, 340, 455, 34),
    (256, 340, 921, 31),
    (256, 340, 1626, 29),
    (256, 340, 2242, 28),
    (256, 340, 3184, 27),
    (256, 340, 65536, 21),
"""


#SCALING_FACTOR = 1000 * 3600
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
        lines.pop(0)
        # second line is header:
        lines.pop(0)
        # Drop the first line because it's always an outlier:
        lines.pop(0)
        # Remove last two lines which display the average:
        lines.pop()
        lines.pop()

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


print("Benchmarking parameters")
for kappa, sboxes, N, tau in aes128_parameters:
    command = "{executable} -i {iter} {kappa} {sboxes} {N} {tau} > {logfile}".format(
        executable=executable, iter=iterations, kappa=kappa/8, N=N, tau=tau, sboxes=sboxes, logfile=logfile)
    #print("running: ", command)
    ret = os.system(command)
    if os.WEXITSTATUS(ret) != 0:
        print("Failed to run {executable}".format(executable=executable))
    print("{kappa} & {N} & {tau} & ".format(kappa=kappa, N=N, tau=tau), end="")
    parse_bench(logfile)
