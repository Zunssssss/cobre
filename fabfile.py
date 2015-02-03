#!/usr/bin/env python

"""
This file is a series of tasks to preprocess COBRE dataset

Installation
------------
It runs on Python > 3.3 and uses invoke (or Fabric when a Python3 version is released) to execute
the tasks from the command line.

- requirements
pip install invoke
pip install git@github.com:Neurita/boyle.git

- optional requirement (for caching results):
pip install joblib
"""

from    __future__ import (absolute_import, division, print_function, unicode_literals)

import  os
import  shutil
import  logging
import  os.path                 as     op
from    glob                    import glob
import  subprocess
from    subprocess              import Popen, PIPE

from    boyle.files.search      import recursive_find_match
from    boyle.files.names       import get_extension, remove_ext
from    boyle.utils.strings     import count_hits
from    boyle.mhd               import copy_mhd_and_raw
from    boyle.commands          import which

try:
    from fabric.api import task, local
except:
    from invoke     import task
    from invoke     import run as local


# my own system call
from functools import partial
call = partial(subprocess.call, shell=True)


# setup log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# read configurations
APPNAME   = 'cobre'
CFG       = rcfile(APPNAME)
RAW_DIR   = op.expanduser(CFG['raw_dir'])
DATA_DIR  = op.expanduser(CFG['data_dir'])
CACHE_DIR = op.expanduser(CFG['cache_dir'])


# read files_of_interest section
FOI_CFG = rcfile(APPNAME, 'files_of_interest')


@task
def show_configuration(section=None):
    cfg = rcfile(APPNAME, section)
    for i in cfg:
        print("{} : {}".format(i, cfg[i]))

    if section is not None:
        return

    sections = get_sections(APPNAME)
    for s in sections:
        if APPNAME not in s:
            print('')
            print('[{}]'.format(s))
            cfg = rcfile(APPNAME, s)
            for i in cfg:
                print("{} : {}".format(i, cfg[i]))


@task
def show_sections():
    sections = get_sections(APPNAME)
    [print(s) for s in sections]


def call_cmd_and_logit(cmd, logfile='logfile.log'):
    """Call cmd line with shell=True and saves the output and error output in logfile"""
    p           = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc          = p.returncode

    if not logfile:
        return rc

    try:
        with open(logfile, 'a') as alog:
            alog.write(output.decode("utf-8"))
            alog.write(   err.decode("utf-8"))
    except:
        log.exception('Error writing logfile {}.'.format(logfile))
    finally:
        return rc


@task
def compress_niftis(work_dir=DATA_DIR):
    """Compress nifti files within work_dir using fslchfiletype command."""
    if not which('fslchfiletype'):
        print('Cannot find fslchfiletype to compress NifTi files. Passing.')
        return -1

    niftis = recursive_find_match(work_dir, '.*nii$')
    niftis.sort()

    for nii in niftis:
        log.debug('Compressing {}'.format(nii))
        local('fslchfiletype NIFTI_GZ {}'.format(nii))


@task
def rename_files_of_interest(work_dir=DATA_DIR):
    """Look through the work_dir looking to the patterns matches indicated in the
    files_of_interest section of the config file.
    For each match it creates a copy of the file in the same folder renamed to
    the names of the section configuration option.
    This will keep the file extensions and adding '+' characters if there are
    more than one match.
    """
    def copy_file(src, dst):
        dirname = op.dirname   (src)
        ext     = get_extension(src)
        dst     = op.basename  (dst)
        dst     = op.join(dirname, remove_ext(dst))

        # add many '+' to the files that have repeated names
        #while op.exists(dst + ext):
        #    dst += '+'
        #dst += ext

        # add a counter value to the files that have repeated names
        if op.exists(dst + ext):
            fc = 2
            while op.exists(dst + str(fc) + ext):
                fc += 1
        dst += str(fc) + ext

        # copy the src file to the given dst value
        try:
            if ext == '.mhd':
                return copy_mhd_and_raw(src, dst)
            else:
                shutil.copyfile(src, dst)
                return dst
        except:
            log.exception('Error copying file {} to {}.'.format(src, dst))
            raise

    def has_mhd_with_raws(files):
        """Return True if the number of .raw files is the same as the number of .mhd files"""
        return count_hits(files, '.*\.raw$') == count_hits(files, '.*\.mhd$')

    for foi in FOI_CFG:
        regex = FOI_CFG[foi]
        files = recursive_find_match(work_dir, regex)
        files.sort()

        if not files:
            print('Could not find {} files that match {} within {}.'.format(foi, regex, work_dir))
            continue

        use_copy_mhd_and_raw = has_mhd_with_raws(files)
        print('Copying {} to {}.'.format(regex, foi))

        for fn in files:
            ext = get_extension(fn)
            if use_copy_mhd_and_raw:
                if ext == '.raw':
                    continue

            try:
                new_fn  = op.join(op.dirname(fn), foi) + ext
                new_dst = copy_file(fn, new_fn)
            except:
                log.exception()
                exit(-1)

            if not op.exists(new_dst):
                msg = 'Error copying file'


@task
def remove_files_of_interest(work_dir=DATA_DIR):
    """Look through the work_dir looking to the patterns matches indicated in the
    files_of_interest section of the config file and remove them.
    """
    for foi in FOI_CFG:
        files = recursive_find_match(work_dir, foi)
        files.sort()

        if not files:
            print('Could not find {0} files that match "{0}" within {1}.'.format(foi, work_dir))
            continue

        for fn in files:
            log.debug('Removing {}.'.format(fn))
            os.remove(fn)


@task
def remove_files(pattern, work_dir=DATA_DIR,):
    """Look through the work_dir looking to the patterns matches the pattern argument value
    and remove them.
    """
    import sys
    try:
        from distutils.util import strtobool
        raw_input = input
    except:
        from distutils import strtobool

    def prompt(query):
        sys.stdout.write('%s [y/n]: ' % query)
        val = raw_input()
        try:
            ret = strtobool(val)
        except ValueError:
            sys.stdout.write('Please answer with a y/n\n')
            return prompt(query)
        return ret

    files = recursive_find_match(work_dir, pattern)
    files.sort()

    if not files:
        print('Could not find files that match "{0}" within {1}.'.format(pattern, work_dir))
        return

    print('\n'.join(files))
    if prompt('Found these files. Want to remove?'):
        for fn in files:
            log.debug('Removing {}.'.format(fn))
            os.remove(fn)


@task
def show_files(name, work_dir=DATA_DIR):
    cfg = rcfile(APPNAME, 'files_of_interest')
    if name not in cfg:
        print("Option {} not found in files_of_interest section.".format(name))
        return -1

    regex = cfg[name]
    files = recursive_find_match(work_dir, regex)
    files.sort()

    if not files:
        print('No files that match "{}" found in {}.'.format(regex, work_dir))
    else:
        print('Files that match "{}" in {}:'.format(regex, work_dir))
        [print(f) for f in files]


@task
def clean():
    """Remove a few temporal files and logs in this folder."""
    call('rm *.log')
    call('rm *.pyc')
    shutil.rmtree('__pycache__')
