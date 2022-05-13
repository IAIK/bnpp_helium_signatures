#!/usr/bin/env python3
import sys
import os

executable = "./bench_free"
parser = "./"
logfile = "bench_output.txt"
iterations = 101




# LowMC with full S-box layer
# kappa, sboxes, N, Tau
full_sbox_parameters = [
    (128, 172, 8, 45),
    (128, 172, 16, 34),
    (128, 172, 31, 28),
    (128, 172, 27, 29),
    (128, 172, 57, 24),
    (128, 172, 107, 21),
    (128, 172, 139, 20),
    (128, 172, 185, 19),
    (128, 172, 256, 18),
    (128, 172, 371, 17),
    (128, 172, 566, 16),
    (128, 172, 921, 15),
    (128, 172, 1626, 14),
#    (128, 172, 3184, 13),
#    (128, 172, 7132, 12),
#    (128, 172, 65536, 10)
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


# LowMC with partial S-box layer
# kappa, sboxes, N, tau
partial_sbox_parameters = [
    (128, 200, 8, 45),
    (128, 200, 16, 34),
    (128, 200, 31, 28),
    (128, 200, 57, 24),
    (128, 200, 107, 21),
    (128, 200, 256, 18),
    (128, 200, 371, 17),
    (128, 200, 921, 15),
    (128, 200, 1626, 14),
#    (128, 200, 3184, 13),
#    (128, 200, 65536, 10)
]

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


print("Benchmarking full S-box layer parameters, followed by partial S-box layer parameters")
for kappa, sboxes, N, tau in full_sbox_parameters + partial_sbox_parameters:
    command = "{executable} -i {iter} {kappa} {sboxes} {N} {tau} > {logfile}".format(
        executable=executable, iter=iterations, kappa=kappa/8, N=N, tau=tau, sboxes=sboxes, logfile=logfile)
    #print("running: ", command)
    ret = os.system(command)
    if os.WEXITSTATUS(ret) != 0:
        print("Failed to run {executable}".format(executable=executable))
    print("{kappa} & {N} & {tau} & ".format(kappa=kappa, N=N, tau=tau), end="")
    parse_bench(logfile)
