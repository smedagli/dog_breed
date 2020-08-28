"""
This module contains common tools
"""
from toolz import partial
import tqdm

progr = partial(tqdm.tqdm, ascii=True)
