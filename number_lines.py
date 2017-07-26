import fileinput
import sys
import re

file_name = sys.argv[1]
i = 0
for line in fileinput.input(file_name, inplace=True):
    print(line.replace(line, re.sub(r'(^.*$)', str(i) + ". " + r'\g<1>', line)), end="")
    i += 1
