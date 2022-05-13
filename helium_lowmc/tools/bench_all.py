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
    (128, 172, 7, 43),
    (128, 172, 11, 36),
    (128, 172, 17, 31),
    (128, 172, 31, 26),
    (128, 172, 57, 22),
    (128, 172, 107, 19),
    (128, 172, 139, 18),
    (128, 172, 185, 17),
    (128, 172, 255, 16),
    (128, 172, 371, 15),
    (128, 172, 565, 14),
    (128, 172, 921, 13),
    (128, 172, 1625, 12),
    #    (128, 172, 3183, 11),
    #    (128, 172, 65535, 8)
]


# LowMC with partial S-box layer
# kappa, sboxes, N, tau
partial_sbox_parameters = [
    (128, 172, 7, 43),
    (128, 172, 11, 36),
    (128, 200, 17, 31),
    (128, 200, 31, 26),
    (128, 200, 57, 22),
    (128, 200, 107, 19),
    (128, 200, 139, 18),
    (128, 200, 185, 17),
    (128, 200, 256, 16),
    (128, 200, 371, 15),
    (128, 200, 565, 14),
    (128, 200, 921, 13),
    (128, 200, 1625, 12),
    #    (128, 200, 3183, 11),
    #    (128, 200, 65535, 8)
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
