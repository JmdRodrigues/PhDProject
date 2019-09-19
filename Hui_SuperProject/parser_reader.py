import regex as re


def read_string(s):
    #find groups
    g = re.findall(r"\((.*)\) \* (\d+)", s)
    pattern = g[0][0].split(" ")

    divisions = g[0][1]

    return pattern, divisions