#!/usr/bin/env python

"""
This file is a series of tasks to preprocess COBRE dataset

Installation
------------
It runs on Python > 3.3 or Python2.7 and uses invoke (or Fabric when a Python3 version is released) to execute
the tasks from the command line.

- requirements
pip install invoke
pip install git@github.com:Neurita/boyle.git

- optional requirement (for caching results):
pip install joblib

"""

from   __future__ import (absolute_import, division, print_function, unicode_literals)

import os
import re
import shutil
import logging
import os.path                 as     op
import numpy                   as     np
import subprocess
from   subprocess              import Popen, PIPE

from   boyle.utils.text_files  import read
from   boyle.files.search      import recursive_find_match
from   boyle.files.names       import get_extension, remove_ext
from   boyle.utils.rcfile      import rcfile, get_sections, get_sys_path
from   boyle.utils.strings     import count_hits
from   boyle.mhd               import copy_mhd_and_raw
from   boyle.commands          import which

try:
    from invoke     import task
    from invoke     import run as local
except:
    from fabric.api import task, local

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
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.getLogger().setLevel(log_level)


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


def get_subject_labels(app_name=APPNAME, subj_labels_file_varname='subj_labels'):
    file_path = op.realpath(op.expanduser(CFG.get(subj_labels_file_varname, None)))
    if file_path is None:
        raise KeyError('Could not find variable {} in {} rcfile.'.format(subj_labels_file_varname, app_name))

    return np.loadtxt(file_path, dtype=int, delimiter='\n')


def get_subject_ids(app_name=APPNAME, subj_id_list_varname='subj_id_list_file'):
    file_path = op.realpath(op.expanduser(CFG.get(subj_id_list_varname, None)))
    if file_path is None:
        raise KeyError('Could not find variable {} in {} rcfile.'.format(subj_id_list_varname, app_name))

    log.debug('Reading list of subject ids from file {}.'.format(file_path))
    return read(file_path).split()


@task
def get_filtered_subjects_ids_and_labels(app_name=APPNAME, subj_id_list_varname='subj_id_list_file',
                                         subj_id_regex_varname='subj_id_regex'):
    """Will use the value of subj_id_regex variable to filter out the subject ids that do not match on the
    subj_id_list_file of the rcfile. Will also return filtered labels.

    The recommendation is to add a '#' character in front of the IDs that you want excluded from the experiment.

    Returns
    -------
    filt_ids: list of str
        The subject ids that match the subject_id regex variable from the rcfile.
    """
    subj_ids = get_subject_ids(app_name, subj_id_list_varname)

    subj_id_regex = CFG[subj_id_regex_varname]
    subj_reg      = re.compile(subj_id_regex)
    labels        = get_subject_labels()

    log.debug('Filtering list of files using subjects ids from subject ids file.')
    filt_ids  = []
    filt_labs = []
    for idx, sid in enumerate(subj_ids):
        if subj_reg.match(sid) is not None:
            filt_ids.append(sid)
            filt_labs.append(labels[idx])

    return filt_ids, filt_labs


def get_subject_ids_and_labels(filter_by_subject_ids=False):
    if filter_by_subject_ids:
        subj_ids, labels = get_filtered_subjects_ids_and_labels()
    else:
        subj_ids = get_subject_ids()
        labels   = get_subject_labels()

    return subj_ids, labels


def filter_list_by_subject_ids(files, subject_ids):
    if files is None or not files:
        return files

    if subject_ids is None or not subject_ids:
        return files

    log.debug('Filtering list of files using subjects ids.')
    filt_files = []
    for fn in files:
        if any(re.search(sid, fn) for sid in subject_ids):
            filt_files.append(fn)

    return filt_files


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
        log.error('Cannot find fslchfiletype to compress NifTi files. Passing.')
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
            log.error('Could not find {} files that match {} within {}.'.format(foi, regex, work_dir))
            continue

        use_copy_mhd_and_raw = has_mhd_with_raws(files)
        log.debig('Copying {} to {}.'.format(regex, foi))

        for fn in files:
            ext = get_extension(fn)
            if use_copy_mhd_and_raw:
                if ext == '.raw':
                    continue

            new_fn  = op.join(op.dirname(fn), foi) + ext
            try:
                new_dst = copy_file(fn, new_fn)
            except:
                msg = 'Error copying file {} to {}.'.format(fn, new_fn)
                log.exception(msg)
                raise IOError(msg)

            if not op.exists(new_dst):
                msg = 'Error copying file {} to {}. After trying to copy, the file does not exist.'.format(fn, new_dst)
                log.error(msg)


@task
def remove_files_of_interest(work_dir=DATA_DIR, verbose=True):
    """Look through the work_dir looking to the patterns matches indicated in the
    files_of_interest section of the config file and remove them.
    """
    verbose_switch(verbose)

    for foi in FOI_CFG:
        regex = get_file_of_interest_regex(foi)
        log.info('Removing within {} that match {}.'.format(len(work_dir), regex))
        remove_files(regex, work_dir, verbose)


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

    files = find_files(work_dir, pattern)

    if not files:
        log.info('Could not find files that match r"{0}" within {1}.'.format(pattern, work_dir))
        return

    log.info('\n'.join(files))
    if prompt('Found these files. Want to remove?'):
        for fn in files:
            log.debug('Removing {}.'.format(fn))
            os.remove(fn)


def find_files(work_dir, regex):
    """Returns a list of the files regex value within the files_of_interest section.

    Parameters
    ----------
    work_dir: str
        Path of the root folder from where to start the search.s

    regex: str
        Name of the variable in files_of_interest section.
    """
    if not op.exists(work_dir):
        msg = 'Could not find {} folder.'.format(work_dir)
        log.error(msg)
        return []

    files = recursive_find_match(work_dir, regex)
    files.sort()
    return files


def get_file_of_interest_regex(name):
    """Return the regex of the name variable in the files_of_interest section of the app rc file."""
    cfg = rcfile(APPNAME, 'files_of_interest')
    if name not in cfg:
        msg = "Option {} not found in files_of_interest section.".format(name)
        log.error(msg)
        raise KeyError(msg)
    return cfg[name]


def print_list(alist):
    [print(i) for i in alist]


@task
def show_regex_match(regex, work_dir=DATA_DIR, filter_by_subject_ids=False):
    """Lists the files inside work_dir that match the name of the given regex.

    Parameters
    ----------
    regex: str
        Regular expession

    work_dir: str
        Path of the root folder from where to start the search.
        Or, if the given name is not an existing path, name of the rcfile variable that contains the folder path.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    if not op.exists(work_dir):
        work_dir = op.expanduser(CFG[work_dir])

    files = find_files(work_dir, regex)

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)
    if filter_by_subject_ids:
        files = filter_list_by_subject_ids(files, subj_ids)

    if not files:
        log.info('No files that match "{}" found in {}.'.format(regex, work_dir))
    else:
        log.info('# Files that match "{}" in {}:'.format(regex, work_dir))
        print_list(files)


@task
def show_files(name, work_dir=DATA_DIR, filter_by_subject_ids=False):
    """Show a list of the files inside work_dir that match the regex value of the variable 'name' within the
    files_of_interest section.

    Parameters
    ----------
    name: str
        Name of the variable in files_of_interest section.

    work_dir: str
        Path of the root folder from where to start the search.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    try:
        regex = get_file_of_interest_regex(name)
    except:
        raise

    log.debug('Looking for files that match {} within {}.'.format(regex, work_dir))
    files = find_files(work_dir, regex)

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)
    if filter_by_subject_ids:
        files = filter_list_by_subject_ids(files, subj_ids)

    if not files:
        log.error('No files that match "{}" found in {}.'.format(regex, work_dir))
    else:
        log.debug('# Files that match "{}" in {}:'.format(regex, work_dir))
        print_list(files)
        return files


@task
def show_my_files(rcpath, app_name=APPNAME, filter_by_subject_ids=False):
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

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    if '/' not in rcpath:
        log.error("Expected an rcpath in the format <variable of folder path>/<variable of files_of_interest regex>.")
        return -1

    dir_name, foi_name = rcpath.split('/')

    app_cfg = rcfile(app_name)
    if dir_name not in app_cfg:
        log.error("Option {} not found in {} section.".format(dir_name, app_name))
        return -1

    work_dir = op.expanduser(app_cfg[dir_name])

    return show_files(foi_name, work_dir, filter_by_subject_ids=filter_by_subject_ids)


@task
def clean():
    """Remove a few temporal files and logs in this folder."""
    call('rm *.log')
    call('rm *.pyc')
    shutil.rmtree('__pycache__')


# ----------------------------------------------------------------------------------------------------------------------
# COBRE PROJECT SPECIFIC FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
OLD_COBRE_DIR = op.expanduser(CFG.get('old_cobre_dir', ''))
OLD_COBRE_CFG = rcfile(APPNAME, 'old_cobre')

SUBJ_ID_REGEX = CFG['subj_id_regex']
FSURF_DIR     = op.expanduser(CFG['fsurf_dir'])
PREPROC_DIR   = OLD_COBRE_DIR


@task
def recon_all(input_dir=RAW_DIR, out_dir=FSURF_DIR, use_cluster=True, verbose=False, filter_by_subject_ids=False):
    """Execute recon_all on all subjects from input_dir/raw_anat

    Parameters
    ----------
    input_dir: str
        Path to where the subjects are

    out_dir: str
        Path to output folder where freesurfer will leave results.

    use_cluster: bool
        If True will use fsl_sub to submit the jobs to your set up cluster queue. This is True by default.
        Use the flag -c to set it to False.

    verbose: bool
        If True will show debug logs.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    verbose_switch(verbose)

    os.environ['SUBJECTS_DIR'] = out_dir

    regex      = get_file_of_interest_regex('raw_anat')
    subj_anats = find_files(input_dir, regex)
    subj_reg   = re.compile(SUBJ_ID_REGEX)

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)
    if filter_by_subject_ids:
        subj_anats = filter_list_by_subject_ids(subj_anats, subj_ids)

    recon_all  = which('recon-all')

    for subj_anat_path in subj_anats:

        subj_id = subj_reg.search(subj_anat_path).group()

        #recon-all -all -i raw/0040000/session_1/anat_1/mprage.nii.gz -s 0040000
        cmd = '{} -all -i {} -s {} -sd {}'.format(recon_all, subj_anat_path, subj_id, out_dir)

        if use_cluster:
            cmd = 'fsl_sub ' + cmd

        log.debug('Calling {}'.format(cmd))
        call_and_logit(cmd, 'freesurfer_{}.log'.format(subj_id))


@task
def run_cpac(verbose=False):
    """Execute cpac_run.py using the configuration from the rcfile"""

    try:
        conf_dir      = op.realpath(op.join(op.dirname(__file__), CFG['cpac_conf']))
        subjects_list = op.realpath(op.join(conf_dir, CFG['cpac_subjects_list']))
        pipeline_file = op.realpath(op.join(conf_dir, CFG['cpac_pipeline_file']))
    except KeyError as ke:
        log.exception(ke)
        raise

    verbose_switch(verbose)

    cpac_path = which('cpac_run.py')

    cmd = '{} {} {}'.format(cpac_path, pipeline_file, subjects_list)
    log.debug('Calling: {}'.format(cmd))
    log.info ('Logging to cpac.log')

    # print('import CPAC')
    # print('CPAC.pipeline.cpac_runner.run("{}", "{}")'.format(pipeline_file, subjects_list))
    call_and_logit(cmd, 'cpac.log')


def get_pipeline_files(root_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                       file_name_varname='reho', filter_by_subject_ids=False, app_name=APPNAME):
    """Return a list of the file_name_varname files in the corresponding pipeline

    Parameters
    ----------
    root_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    file_name_varname: str
        RCfile variable name for the file you are looking for.

    verbose: bool
        If verbose will show DEBUG log info.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    Returns
    -------
    fois
        list of matched filepaths
    """

    try:
        settings          = rcfile(app_name, section_name)
        pipe_name         = settings[pipe_varname]
        files_of_interest = rcfile(app_name, 'files_of_interest')
        varname           = files_of_interest[file_name_varname]
        root_dir          = get_sys_path(root_dir, section_name, app_name)
    except IOError:
        raise
    except:
        msg = 'Error looking for variable names in {} rc file.'.format(APPNAME)
        log.exception (msg)
        raise KeyError(msg)

    log.debug('Looking for {} files from pipe {} within {} folder'.format(varname, pipe_name, root_dir))
    files = find_files(root_dir, varname)

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)
    if filter_by_subject_ids:
        files = filter_list_by_subject_ids(files, subj_ids)

    log.debug('Found {} files that match the file name. Now filtering for pipe name.'.format(len(files)))

    return [fpath for fpath in files if re.match(pipe_name, fpath)]


@task
def show_pipeline_files(root_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                        file_name_varname='reho', verbose=False, filter_by_subject_ids=True):
    """Print a list of the file_name_varname files in the corresponding pipeline.

    Parameters
    ----------
    root_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    file_name_varname: str
        RCfile variable name for the file you are looking for.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    verbose_switch(verbose)

    pipe_files = get_pipeline_files(root_dir, section_name, pipe_varname, file_name_varname,
                                    filter_by_subject_ids=filter_by_subject_ids)

    if not pipe_files:
        log.info('Could not find {} files from pipe {} within {} folder'.format(file_name_varname, pipe_varname, root_dir))
    else:
        print_list(pipe_files)


@task
def pack_pipeline_files(root_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                        file_name_varname='reho', mask_file_varname='brain_mask_dil_3mm', smooth_fwhm=0,
                        output_file='cobre_reho_pack.mat', verbose=False, filter_by_subject_ids=True):
    """Mask and compress the data into a file.

    Will save into the file: data, mask_indices, vol_shape

        data: Numpy array with shape N x prod(vol.shape)
              containing the N files as flat vectors.

        mask_indices: matrix with indices of the voxels in the mask

        vol_shape: Tuple with shape of the volumes, for reshaping.

    Parameters
    ----------
    root_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    file_name_varname: str
        RCfile variable name for the file you are looking for.

    mask_file_varname: str
        RCfile variable name for the mask file that you want to use to mask the data.

    smooth_fwhm: int
        FWHM size in mm of a Gaussian smoothing kernel to smooth the images before storage.

    output_file: str
        Path to the output file. The extension of the file will be taken into account for the file format.
        Choices of extensions: '.pyshelf' or '.shelf' (Python shelve)
                               '.mat' (Matlab archive),
                               '.hdf5' or '.h5' (HDF5 file)

    verbose: bool
        If verbose will show DEBUG log info.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    verbose_switch(verbose)

    mask_file = None
    if mask_file_varname:
        mask_file = op.join(op.expanduser(CFG['std_dir']),  CFG[mask_file_varname])

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)

    pipe_files       = get_pipeline_files(root_dir, section_name, pipe_varname, file_name_varname,
                                          filter_by_subject_ids=filter_by_subject_ids)
    pipe_files.sort()

    if not pipe_files:
        log.info('Could not find {} files from pipe {} '
                 'within {} folder'.format(file_name_varname, pipe_varname, root_dir))
        exit(-1)

    log.debug('Parsing {} subjects into a Nifti file set.'.format(len(pipe_files)))
    try:
        _pack_files_to(pipe_files, mask_file=mask_file, labels=labels, subj_ids=subj_ids, smooth_fwhm=smooth_fwhm,
                       output_file=output_file, verbose=verbose)
    except:
        log.exception('Error creating the set of subjects from {} files '
                      'from pipe {} within {} folder'.format(file_name_varname, pipe_varname, root_dir))
        raise


def _pack_files_to(images, output_file, mask_file=None, labels=None, subj_ids=None, smooth_fwhm=0, verbose=False):
    """Get NeuroImage files mask them, put all the data in a matrix and save them into
    output_file together with mask shape and affine information and labels.

    Will save into the file: data, mask_indices, vol_shape, labels

        data: Numpy array with shape N x prod(vol.shape)
              containing the N files as flat vectors.

        mask_indices: matrix with indices of the voxels in the mask

        vol_shape: Tuple with shape of the volumes, for reshaping.

    Parameters
    ----------
    images: list of str or img-like object.
        See boyle.nifti.NeuroImage constructor docstring.

    mask: str or img-like object.
        See boyle.nifti.NeuroImage constructor docstring.

    labels: list or tuple of str or int or float.
        This list shoule have the same length as images.
        If None, will use the info in the rcfile config files.

    subj_ids: list or tuple of str
        This list shoule have the same length as images.
        If None, will use the info in the rcfile config files.

    smooth_fwhm: int
        FWHM size in mm of a Gaussian smoothing kernel to smooth the images before storage.

    output_file: str
        Path to the output file. The extension of the file will be taken into account for the file format.
        Choices of extensions: '.pyshelf' or '.shelf' (Python shelve)
                               '.mat' (Matlab archive),
                               '.hdf5' or '.h5' (HDF5 file)

    verbose: bool
        If verbose will show DEBUG log info.
    """
    from boyle.nifti.sets import NeuroImageSet

    verbose_switch(verbose)

    try:
        subj_set = NeuroImageSet(images, mask=mask_file, labels=labels, all_compatible=True)
        subj_set.others['subj_ids'] = np.array(subj_ids)
    except:
        raise
    else:
        log.debug('Saving masked data into file {}.'.format(output_file))
        subj_set.to_file(output_file, smooth_fwhm=smooth_fwhm)


@task
def pack_files(name, output_file, work_dir=DATA_DIR, mask_file=None, labels=None, subj_ids=None, smooth_fwhm=0,
               verbose=False, filter_by_subject_ids=False):
    """Pack a list of the files inside work_dir that match the regex value of the variable 'name' within the
    files_of_interest section.

    Parameters
    ----------
    name: str
        Name of the variable in files_of_interest section.

    work_dir: str
        Path of the root folder from where to start the search.s

    mask: str
        RCfile variable name for the mask file that you want to use to mask the data.

    labels: list or tuple of str or int or float.
        This list shoule have the same length as images.
        If None, will use the info in the rcfile config files.

    subj_ids: list or tuple of str
        This list shoule have the same length as images.
        If None, will use the info in the rcfile config files.

    smooth_fwhm: int
        FWHM size in mm of a Gaussian smoothing kernel to smooth the images before storage.

    output_file: str
        Path to the output file. The extension of the file will be taken into account for the file format.
        Choices of extensions: '.pyshelf' or '.shelf' (Python shelve)
                               '.mat' (Matlab archive),
                               '.hdf5' or '.h5' (HDF5 file)

    verbose: bool
        If verbose will show DEBUG log info.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    verbose_switch(verbose)

    if mask_file is not None:
        mask_file = op.join(op.expanduser(CFG['std_dir']),  CFG[mask_file])
    if not op.exists(mask_file):
        raise IOError('The mask file {} has not been found.'.format(mask_file))

    try:
        images = show_files(name, work_dir=work_dir, filter_by_subject_ids=filter_by_subject_ids)
    except:
        raise

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)

    if images:
        _pack_files_to(images, output_file, mask_file=mask_file, labels=labels, subj_ids=subj_ids,
                       smooth_fwhm=smooth_fwhm, verbose=verbose)


@task
def pack_my_files(rcpath, output_file, app_name=APPNAME, mask_file=None, labels=None, smooth_fwhm=0,
                  verbose=False, filter_by_subject_ids=False):
    """Pack a list of the files inside within the rcpath, i.e., a string with one '/', in the
    format <variable of folder path>/<variable of files_of_interest regex>.

    Parameters
    ----------
    rcpath: str
        A path with one '/', in the format <variable of folder path>/<variable of files_of_interest regex>.
        For example: 'data_dir/anat' will look for the folder path in the data_dir variable and the regex in the
        anat variable inside the files_of_interest section.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    mask_file: str
        RCfile variable name for the mask file that you want to use to mask the data.

    labels: list or tuple of str or int or float.
        This list shoule have the same length as images.

    smooth_fwhm: int
        FWHM size in mm of a Gaussian smoothing kernel to smooth the images before storage.

    output_file: str
        Path to the output file. The extension of the file will be taken into account for the file format.
        Choices of extensions: '.pyshelf' or '.shelf' (Python shelve)
                               '.mat' (Matlab archive),
                               '.hdf5' or '.h5' (HDF5 file)

    verbose: bool
        If verbose will show DEBUG log info.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.
    """
    verbose_switch(verbose)

    if mask_file is not None:
        mask_file = op.join(op.expanduser(CFG['std_dir']),  CFG[mask_file])
    if not op.exists(mask_file):
        raise IOError('The mask file {} has not been found.'.format(mask_file))

    try:
        images = show_my_files(rcpath, app_name=app_name, filter_by_subject_ids=filter_by_subject_ids)
    except:
        raise

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)

    if images:
        _pack_files_to(images, output_file, mask_file=mask_file, labels=labels, subj_ids=subj_ids,
                       smooth_fwhm=smooth_fwhm, verbose=verbose)
