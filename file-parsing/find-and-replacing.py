# Generated from Jupyter notebook

import subprocess

raw_input_files = subprocess.check_output(["find",".","-iname","*.i"])

raw_input_files = raw_input_files.decode("utf-8")

input_list = raw_input_files.split("\n")
input_list = input_list[:-1]

def is_ls_in_precond(input_file):
    with open(input_file) as input:
        solve_type_line = None
        ls_line = None
        while True:
            line = input.readline()
            if not line:
                return (False,)
            elif "Preconditioning" in line:
                break
        while True:
            line = input.readline()
            if "../" in line:
                do_replacements = solve_type_line is not None or ls_line is not None
                return (do_replacements, solve_type_line, ls_line)
            elif "line_search" in line:
                ls_line = line
            elif "solve_type" in line:
                solve_type_line = line

from tempfile import NamedTemporaryFile
from shutil import move
import re

def rewrite_file(input_file, replacement_line):
    if replacement_line[1] is not None:
        m = re.search('line_search.*?=[^a-z]*([a-z]*)', replacement_line[1])
        line_search_type = m.groups()[0]
    if replacement_line[0] is not None:
        m = re.search('solve_type.*?=[^A-Za-z]*([A-Za-z]*)', replacement_line[0])
        solve_type = m.groups()[0]
    with open(input_file) as f, NamedTemporaryFile(mode='w',dir=".",delete=False) as out:
        performed_replacements = False
        for line in f:
            if not performed_replacements and ("nl_abs_tol" in line or "nl_rel_tol" in line):
                if replacement_line[0] is not None:
                    out.write("  solve_type = '" + solve_type + "'\n")
                if replacement_line[1] is not None:
                    out.write("  line_search = '" + line_search_type + "'\n")
                out.write(line)
                performed_replacements = True
            elif "line_search" in line:
                if replacement_line[1] is None:
                    out.write(line)
            elif "solve_type" in line:
                if replacement_line[0] is None:
                    out.write(line)
            else:
                out.write(line)
    move(out.name, f.name)

num_matches = 0

for input_file in input_list:
    match = is_ls_in_precond(input_file)
    if match[0]:
        num_matches += 1
        rewrite_file(input_file, match[1:])
