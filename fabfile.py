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
import  re
import  sys
import  shutil
import  logging
import  os.path                 as     op
from    glob                    import glob
import  subprocess
from    subprocess              import Popen, PIPE

from    boyle.files.search      import recursive_find_match
from    boyle.files.names       import get_extension, remove_ext
from    boyle.utils.rcfile      import rcfile
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
APPNAME       = 'cobre'
CFG           = rcfile(APPNAME)
RAW_DIR       = op.expanduser(CFG['raw_dir'])
PREPROC_DIR   = op.expanduser(CFG['preproc_dir'])
CACHE_DIR     = op.expanduser(CFG['cache_dir'])

DATA_DIR      = PREPROC_DIR

# read files_of_interest section
FOI_CFG = rcfile(APPNAME, 'files_of_interest')


def verbose_switch(verbose=False):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


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


def call_and_logit(cmd, logfile='logfile.log'):
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
def compress_niftis(work_dir=DATA_DIR, verbose=False):
    """Compress nifti files within work_dir using fslchfiletype command."""
    if not which('fslchfiletype'):
        print('Cannot find fslchfiletype to compress NifTi files. Passing.')
        return -1

    verbose_switch(verbose)

    niftis = recursive_find_match(work_dir, '.*nii$')
    niftis.sort()

    for nii in niftis:
        log.debug('Compressing {}'.format(nii))
        local('fslchfiletype NIFTI_GZ {}'.format(nii))


@task
def rename_files_of_interest(work_dir=DATA_DIR, verbose=False):
    """Look through the work_dir looking to the patterns matches indicated in the
    files_of_interest section of the config file.
    For each match it creates a copy of the file in the same folder renamed to
    the names of the section configuration option.
    This will keep the file extensions and adding '+' characters if there are
    more than one match.
    """
    verbose_switch(verbose)

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
        log.debig('Copying {} to {}.'.format(regex, foi))

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
def remove_files_of_interest(work_dir=DATA_DIR, verbose=False):
    """Look through the work_dir looking to the patterns matches indicated in the
    files_of_interest section of the config file and remove them.
    """
    verbose_switch(verbose)

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
def remove_files(pattern, work_dir=DATA_DIR, verbose=False):
    """Look through the work_dir looking to the patterns matches the pattern argument value
    and remove them.
    """
    verbose_switch(verbose)

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


def find_files(regex, work_dir):
    """Returns a list of the files regex value within the files_of_interest section.

    Parameters
    ----------
    regex: str
        Name of the variable in files_of_interest section.

    work_dir: str
        Path of the root folder from where to start the search.s
    """
    files = recursive_find_match(work_dir, regex)
    files.sort()
    return files


def get_file_of_interest_regex(name):
    """Return the regex of the name variable in the files_of_interest section of the app rc file."""
    cfg = rcfile(APPNAME, 'files_of_interest')
    if name not in cfg:
        print("Option {} not found in files_of_interest section.".format(name))
        return -1
    return cfg[name]


@task
def show_files(name, work_dir=DATA_DIR):
    """Lists the files inside work_dir that match the regex value of the variable 'name' within the
    files_of_interest section.

    Parameters
    ----------
    name: str
        Name of the variable in files_of_interest section.

    work_dir: str
        Path of the root folder from where to start the search.s
    """
    regex = get_file_of_interest_regex(name)
    if not regex:
        return -1

    files = find_files(regex, work_dir)
    if not files:
        print('No files that match "{}" found in {}.'.format(regex, work_dir))
    else:
        print('# Files that match "{}" in {}:'.format(regex, work_dir))
        [print(f) for f in files]


@task
def show_my_files(rcpath, app_name=APPNAME):
    """Shows the files within the rcpath, i.e., a string with one '/', in the
    format <variable of folder path>/<variable of files_of_interest regex>.

    Parameters
    ----------
    rcpath: str
        A path with one '/', in the format <variable of folder path>/<variable of files_of_interest regex>.
        For example: 'data_dir/anat' will look for the folder path in the data_dir variable and the regex in the
        anat variable inside the files_of_interest section.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)
    """
    if '/' not in rcpath:
        print("Expected an rcpath in the format <variable of folder path>/<variable of files_of_interest regex>.")
        return -1

    dir_name, foi_name = rcpath.split('/')

    app_cfg = rcfile(app_name)
    if dir_name not in app_cfg:
        print("Option {} not found in {} section.".format(dir_name, app_name))
        return -1

    foi_cfg = rcfile(app_name, 'files_of_interest')
    if foi_name not in foi_cfg:
        print("Option {} not found in files_of_interest section of {}.".format(foi_name, app_name))
        return -1

    work_dir = op.expanduser(app_cfg[dir_name])

    return show_files(foi_name, work_dir)


@task
def clean():
    """Remove a few temporal files and logs in this folder."""
    call('rm *.log')
    call('rm *.pyc')
    shutil.rmtree('__pycache__')

# ----------------------------------------------------------------------------------------------------------------------
# COBRE PROJECT SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
SUBJ_ID_REGEX = CFG['subj_id_regex']
FSURF_DIR     = op.expanduser(CFG['fsurf_dir'])


@task
def recon_all(input_dir=RAW_DIR, out_dir=FSURF_DIR, use_cluster=True, verbose=False):
    """Execute recon_all on all subjects from input_dir/raw_anat"""
    verbose_switch(verbose)

    os.environ['SUBJECTS_DIR'] = out_dir

    regex      = get_file_of_interest_regex('raw_anat')
    subj_anats = find_files(regex, input_dir)
    subj_reg   = re.compile(SUBJ_ID_REGEX)

    recon_all  = which('recon-all')

    for subj_anat_path in subj_anats:

        subj_id = subj_reg.search(subj_anat_path).group()

        #recon-all -all -i raw/0040000/session_1/anat_1/mprage.nii.gz -s 0040000
        cmd = '{} -all -i {} -s {} -sd {}'.format(recon_all, subj_anat_path, subj_id, out_dir)
        log.debug('Calling {}'.format(cmd))

        if use_cluster:
            cmd = 'fsl_sub ' + cmd
            call(cmd)
        else:
            call_and_logit(cmd, 'freesurfer_{}.log'.format(subj_id), wait=True)


@task
def run_cpac(verbose=False):
    """Execute cpac_run.py using the configuration from the rcfile"""

    try:
        conf_dir      = op.realpath(op.join(op.dirname(__file__), CFG['cpac_conf']))
        subjects_list = op.realpath(op.join(conf_dir, CFG['cpac_subjects_list']))
        pipeline_file = op.realpath(op.join(conf_dir, CFG['cpac_pipeline_file']))
    except KeyError as ke:
        log.exception(ke)

    verbose_switch(verbose)

    cpac_path = which('cpac_run.py')

    cmd = '{} {} {}'.format(cpac_path, pipeline_file, subjects_list)
    log.debug('Calling: {}'.format(cmd))
    log.info ('Logging to cpac.log')

    call_and_logit(cmd, 'cpac.log')
