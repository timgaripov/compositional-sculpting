from os.path import dirname

from ml_logger import RUN, instr
from termcolor import colored

assert instr  # single-entry for the instrumentation thunk factory
RUN.project = "gflownet-sculpting"  # Specify the project name
RUN.prefix = "{project}/{project}/{now:%Y/%m-%d}/{file_stem}/{job_name}"
RUN.script_root = dirname(__file__)  # specify that this is the script root.
print(colored('set', 'blue'), colored("RUN.script_root", "yellow"), colored('to', 'blue'), RUN.script_root)
