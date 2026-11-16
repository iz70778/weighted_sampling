import json
import os
import sys

import numpy as np

from args import args
from scripts.init import global_model, client_loaders, test_loader, validate_loader

from train.server import fed_avg, fed_diff_os


if __name__ == '__main__':
    fed_diff_os(global_model, client_loaders, test_loader)