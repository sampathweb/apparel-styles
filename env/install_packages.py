#! /usr/bin/env python

from __future__ import print_function
import os
from subprocess import call

ENV_ROOT = os.path.abspath(os.path.dirname(__file__))
APP_ROOT = os.path.dirname(ENV_ROOT)
VENV_NAME = os.environ.get("VENV_NAME", "venv")
CONDA_REQUIREMENTS_PATH = os.path.join(APP_ROOT, "requirements_conda.txt")

# update any python packages
print("Installing and/or updating requirements...")
call("conda install --file %s --yes" % CONDA_REQUIREMENTS_PATH,
     shell=True)

print("Virtual Environment is setup and activated.  "
      "In future, run 'source activate env/venv (in Mac/Linix) and activate env/venv (in Windows)' to Activate it")
