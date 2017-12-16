import logging
import ntpath
import numpy as np

def create_logger(file_name, file_level = logging.INFO, console_level = logging.DEBUG):

    np.set_printoptions(threshold='nan')
    np.set_printoptions(linewidth=300)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if console_level != None:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch_format = logging.Formatter('%(asctime)s - %(funcName)s - %(message)s')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)

    if file_level != None:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh_format = logging.Formatter('%(asctime)s - %(funcName)s - %(message)s')
        fh.setFormatter(fh_format)
        logger.addHandler(fh)

    return logger

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)