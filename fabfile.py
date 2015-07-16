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
import os.path                  as     op
import numpy                    as     np
from   functools                import partial
from   subprocess               import Popen, PIPE
from   collections              import OrderedDict

from   boyle.mhd.write          import copy_mhd_and_raw
from   boyle.commands           import which
from   boyle.utils.strings      import count_hits, where_is
from   boyle.utils.text_files   import read
from   boyle.utils.rcfile       import (rcfile, get_sections, get_sys_path, find_in_sections,
                                        get_rcfile_section, get_rcfile_variable_value)

from   boyle.files.search       import recursive_find_match, check_file_exists
from   boyle.files.names        import get_extension, remove_ext
from   boyle.nifti.cpac_helpers import xfm_atlas_to_functional
from   boyle.nifti.roi          import partition_timeseries
from   boyle.storage            import save_variables_to_hdf5

from   invoke                   import task
from   invoke                   import run as local


# setup log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# read configurations
APPNAME = 'cobre'

try:
    CFG           = rcfile(APPNAME)
    RAW_DIR       = op.expanduser(CFG['raw_dir'    ])
    PREPROC_DIR   = op.expanduser(CFG['preproc_dir'])
    CACHE_DIR     = op.expanduser(CFG['cache_dir'  ])
    EXPORTS_DIR   = op.expanduser(CFG['exports_dir'])
    ATLAS_DIR     = op.expanduser(CFG['atlas_dir'  ])
    STD_DIR       = op.expanduser(CFG['std_dir'    ])
    DATA_DIR      = PREPROC_DIR

    # read files_of_interest section
    FOI_CFG = rcfile(APPNAME, 'files_of_interest')
except:
    log.exception('Error reading config variable from settings in {} rcfiles.'.format(APPNAME))
    raise


def verbose_switch(verbose=False):
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.getLogger().setLevel(log_level)


@task
def clean_cache(cache_dir=CACHE_DIR):
    """Remove joblib cache folder"""
    cache_dir = op.expanduser(cache_dir)
    log.info('Removing cache folder {}'.format(cache_dir))
    shutil.rmtree(cache_dir)


@task(autoprint=True)
def get_rc_sections(app_name=APPNAME):
    """Return the available rcfiles sections"""
    sections = get_sections(app_name)
    return sections


@task
def show_configuration(app_name=APPNAME, section=None):
    """ Show the rcfile configuration variables for the given application.

    Parameters
    ----------
    app_name: str
        Name of the application to look for rcfiles.

    section: str
        Rcfile section name
    """
    cfg = rcfile(app_name, section)
    for i in cfg:
        print("{} : {}".format(i, cfg[i]))

    if section is not None:
        return

    sections = get_sections(app_name)
    for s in sections:
        if app_name not in s:
            print('')
            print('[{}]'.format(s))
            cfg = rcfile(app_name, s)
            for i in cfg:
                print("{} : {}".format(i, cfg[i]))


@task(autoprint=True)
def get_subject_labels(app_name=APPNAME, subj_labels_file_varname='subj_labels'):
    """ Return the class labels of all subjects in a list

    Parameters
    ----------
    app_name: str
        Name of the application to look for rcfiles.

    subj_labels_file_varname: str
        Name of the rcfile variable that holds the path to the subject labels file.

    Returns
    -------
    labels
        list of int
    """
    file_path = op.realpath(op.expanduser(CFG.get(subj_labels_file_varname, None)))
    if file_path is None:
        raise KeyError('Could not find variable {} in {} rcfile.'.format(subj_labels_file_varname, app_name))

    return np.loadtxt(file_path, dtype=int, delimiter='\n')


def read_subject_ids_file(app_name=APPNAME, subj_id_list_varname='subj_id_list_file'):
    """ Return the content of the subject_id file in a list.
    Parameters
    ----------
    app_name: str
        Name of the application to look for rcfiles.

    subj_id_list_varname: str
        Name of the rcfile variable that holds the path to the subject ids file.

    Returns
    -------
    subject_ids: list of str
    """
    file_path = op.realpath(op.expanduser(CFG.get(subj_id_list_varname, None)))
    if file_path is None:
        raise KeyError('Could not find variable {} in {} rcfile.'.format(subj_id_list_varname, app_name))

    log.debug('Reading list of subject ids from file {}.'.format(file_path))

    return read(file_path).split('\n')


@task(autoprint=True)
def get_subject_ids(app_name=APPNAME, subj_id_list_varname='subj_id_list_file', remove_commented=False):
    """ Return the class labels of all subjects in a list

    Parameters
    ----------
    app_name: str
        Name of the application to look for rcfiles.

    subj_id_list_varname: str
        Name of the rcfile variable that holds the path to the subject ids file.

    remove_commented: bool
        If True will remove the ids that are commented with a '#'. Will return them all, otherwise.

    Returns
    -------
    subject_ids: list of str
    """
    subj_ids = read_subject_ids_file(app_name, subj_id_list_varname)
    if remove_commented:
        subj_ids = [id for id in subj_ids if not id.startswith('#')]
    else:
        subj_ids = [id.replace('#', '').strip() for id in subj_ids]

    return subj_ids


@task(autoprint=True)
def get_filtered_subjects_ids_and_labels(app_name=APPNAME, subj_id_list_varname='subj_id_list_file',
                                         subj_id_regex_varname='subj_id_regex'):
    """Will use the value of subj_id_regex variable to filter out the subject ids that do not match on the
    subj_id_list_file of the rcfile. Will also return filtered labels.

    The recommendation is to add a '#' character in front of the IDs that you want excluded from the experiment.

    Parameters
    ----------
    app_name: str
        Name of the application to look for rcfiles.

    subj_id_list_varname: str
        Name of the rcfile variable that holds the path to the subject ids file.

    subj_id_regex_varname: str
        Regular expression

    Returns
    -------
    filt_ids: list of str
        The subject ids that match the subject_id regex variable from the rcfile.
    """
    subj_ids           = read_subject_ids_file(app_name, subj_id_list_varname)

    labels             = get_subject_labels()
    matches_subj_regex = partial(re.match, CFG[subj_id_regex_varname])

    log.debug('Filtering list of files using subjects ids from subject ids file.')
    filt_ids  = []
    filt_labs = []
    for idx, sid in enumerate(subj_ids):
        if matches_subj_regex(sid) is not None:
            filt_ids.append (sid)
            filt_labs.append(labels[idx])

    return filt_ids, filt_labs


@task(autoprint=True)
def get_subject_ids_and_labels(filter_by_subject_ids=False):
    if filter_by_subject_ids:
        subj_ids, labels = get_filtered_subjects_ids_and_labels()
    else:
        subj_ids = get_subject_ids()
        labels   = get_subject_labels()

    return subj_ids, labels


def filter_list_by_subject_ids(files, subject_ids):
    """ Look for matches of each subject_id in the files list, if a match is not found, the file is removed.
    The filtered list is then returned.

    Parameters
    ----------
    files: list of str
        List of file paths that contain a subject id

    subject_ids: list of str
        List of subject ids that you want included in files

    Returns
    -------
    filtered_list: list of str
        List of file paths that contain any of the subject ids in subject_ids
    """
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
    """Look through the work_dir looking to the patterns matches indicated in the files_of_interest section of the
    config file.
    For each match it creates a copy of the file in the same folder renamed to the names of the section configuration
    option.
    This will keep the file extensions and adding '+' characters if there are more than one match.
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
    """Look through the work_dir looking to the patterns matches indicated in the files_of_interest section of
    the config file and remove them.
    """
    verbose_switch(verbose)

    for foi in FOI_CFG:
        regex = get_file_of_interest_regex(foi)
        log.info('Removing within {} that match {}.'.format(len(work_dir), regex))
        remove_files(regex, work_dir, verbose)


@task
def remove_files(pattern, work_dir=DATA_DIR, verbose=False):
    """Look through the work_dir looking to the patterns matches the pattern argument value and remove them.
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


@task(autoprint=True)
def find_files(work_dir, regex):
    """ Returns a list of the files that match the regex value within work_dir.

    Parameters
    ----------
    work_dir: str
        Path of the root folder from where to start the search.s

    regex: str
        Name of the variable in files_of_interest section.
    """
    try:
        check_file_exists(work_dir)
    except:
        return []

    files = recursive_find_match(work_dir, regex)
    files.sort()
    return files


@task(autoprint=True)
def get_file_of_interest_regex(name, app_name=APPNAME):
    """Return the regex of the name variable in the files_of_interest section of the app rc file."""
    return get_rcfile_variable_value(name, 'files_of_interest', app_name)


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
    try:
        check_file_exists(work_dir)
    except:
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
    local('rm *.log')
    local('rm *.pyc')
    shutil.rmtree('__pycache__')


@task(autoprint=True)
def get_standard_file(file_name_varname,  app_name=APPNAME):
    """ Return the path to an atlas or a standard template file.
    Looking for 'standard' and 'atlas' section in rcfiles.

    Parameters
    ----------
    file_name_varname: str

    app_name: str

    Returns
    -------
    std_path: str
        Path to the atlas or the standard template.
    """
    section_name, var_value = find_in_sections(file_name_varname, app_name)

    if section_name == 'atlases':
        std_path = op.join(ATLAS_DIR, var_value)

    elif section_name == 'standard':
        std_path = op.join(STD_DIR, var_value)

    else:
        raise KeyError('The variable {} could only be found in section {}. '
                       'I do not know what to do with this.'.format(file_name_varname, section_name))

    return std_path

#
# @task
# def create_cpac_subj_list(anat_file_var='raw_anat', rest_files_vars=['raw_rest'],
#                           output='CPAC_subject_list_file.yaml',
#                           filter_by_subject_ids=False, verbose=False):
#     """Create a C-PAC subject list file including the path to the files represented by the variables in
#     conf_variables.
#
#     Parameters
#     ----------
#     anat_file_var: str
#         Variable name in the application rcfiles which hold the name of the subject anatomical file.
#
#     rest_files_vars: list of str
#         List of variable names in the application rcfiles which hold the name of the subject fMRI files.
#
#     output: str
#         Path of the output file
#
#     filter_by_subject_ids: bool
#         If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
#         and let only matches to the subject ID list values.
#
#     verbose: bool
#         If True will show debug logs.
#     """
#     import yaml
#
#



@task
def run_cpac(cpac_pipeline_file_varname='cpac_pipeline_file', verbose=False):
    """Execute cpac_run.py using the configuration from the rcfile"""

    try:
        conf_dir      = op.realpath(op.join(op.dirname(__file__), CFG['cpac_conf']               ))
        subjects_list = op.realpath(op.join(conf_dir,             CFG['cpac_subjects_list']      ))
        pipeline_file = op.realpath(op.join(conf_dir,             CFG[cpac_pipeline_file_varname]))
    except KeyError as ke:
        log.exception(ke)
        raise

    verbose_switch(verbose)

    cpac_cmd  = 'cpac_run.py'
    cpac_path = which(cpac_cmd)
    if cpac_path is None:
        log.error('Could not find {} command.'.format(cpac_cmd))
        return -1

    if op.exists('cpac.log'):
        log.debug('Remove cpac.log file.')
        os.remove('cpac.log')

    cmd = '"{}" "{}" "{}"'.format(cpac_path, pipeline_file, subjects_list)
    log.debug('Calling: {}'.format(cmd))
    log.info ('Logging to cpac.log')

    # print('import CPAC')
    # print('CPAC.pipeline.cpac_runner.run("{}", "{}")'.format(pipeline_file, subjects_list))
    call_and_logit(cmd, 'cpac.log')

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


def show_pipeline_files(root_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                       file_name_varname='reho', filter_by_subject_ids=False, app_name=APPNAME):
    """ Return a list of the file_name_varname files in the corresponding pipeline.

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
    print_list(get_pipeline_files(root_dir=root_dir, section_name=section_name, pipe_varname=pipe_varname,
                                  file_name_varname=file_name_varname, filter_by_subject_ids=filter_by_subject_ids,
                                  app_name=app_name))


@task(autoprint=True)
def get_pipeline_folder(root_dir=PREPROC_DIR, pipe_section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                        app_name=APPNAME):

    pipe_dirpath = get_rcfile_variable_value(pipe_varname, section_name=pipe_section_name, app_name=app_name)
    root_dir     = get_sys_path             (root_dir,     section_name=pipe_section_name, app_name=app_name)

    return op.join(root_dir, pipe_dirpath)


@task(autoprint=True)
def get_pipeline_files(root_dir=PREPROC_DIR, pipe_section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                       file_name_varname='reho', filter_by_subject_ids=False, app_name=APPNAME):
    """See show_pipeline_files."""

    pipe_dir = get_pipeline_folder(root_dir=root_dir, pipe_varname=pipe_varname,
                                   pipe_section_name=pipe_section_name, app_name=app_name)

    section_name, var_value = find_in_sections(file_name_varname, app_name)

    if section_name == 'files_of_interest':
        varname = var_value

        log.debug('Looking for {} files from pipe {} within {} folder'.format(varname, pipe_varname, pipe_dir))
        files = find_files(pipe_dir, varname)

    elif section_name == 'relative_paths':
        varname = get_rcfile_variable_value('funcfiltx', section_name='files_of_interest', app_name=app_name)
        relpath = var_value

        log.debug('Looking for {} files from pipe {} within {} folder'.format(varname, pipe_varname, pipe_dir))
        files = [op.join(pipe_dir, subj_f, relpath) for subj_f in os.listdir(pipe_dir)]

    else:
        raise KeyError('The variable {} could only be found in section {}. '
                       'I do not know what to do with this.'.format(file_name_varname, section_name))

    if filter_by_subject_ids:
        subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)
        files            = filter_list_by_subject_ids(files, subj_ids)

    log.debug('Found {} files that match the file name in pipeline folder {}.'.format(len(files), pipe_dir))

    return files



@task
def show_pipeline_files(root_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_wglob',
                        file_name_varname='reho', verbose=False, filter_by_subject_ids=False):
    """Return a list of the file_name_varname files in the corresponding pipeline.

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
                        output_file='cobre_reho_pack.mat', verbose=False, filter_by_subject_ids=False):
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
        mask_file = op.join(op.expanduser(CFG['std_dir']),  FOI_CFG[mask_file_varname])

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

    check_file_exists(mask_file)

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
        mask_file = get_standard_file(mask_file)

    check_file_exists(mask_file)

    try:
        images = show_my_files(rcpath, app_name=app_name, filter_by_subject_ids=filter_by_subject_ids)
    except:
        raise

    subj_ids, labels = get_subject_ids_and_labels(filter_by_subject_ids)

    if images:
        _pack_files_to(images, output_file, mask_file=mask_file, labels=labels, subj_ids=subj_ids,
                       smooth_fwhm=smooth_fwhm, verbose=verbose)


def get_cobre_export_data(root_dir=EXPORTS_DIR, section_name='old_cobre', type='timeseries', regex='',
                          app_name=APPNAME):
    """
    Parameters
    ----------
    root_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    type: str
        Type of the data within the exported file archive. Choices:
        'timeseries'      - for smoothed or not raw fMRI timeseries data
        'scalar_activity' - for local activity measures from fMRI timeseries data, e.g., reho, alff, etc.

    regex: str
        Regular expression to match with the archive file name.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Results
    -------
    list
        List of export files found
    """
    type_choices = {'timeseries', 'scalar_activity'}

    try:
        settings              = get_rcfile_section(app_name, section_name)
        feats_dir_name        = settings['features_dir']

        ts_feats_dir_name     = settings['timeseries_feats_dir']
        scalar_feats_dir_name = settings['scalar_wtemp_noglob_feats_dir']
    except IOError:
        raise
    except:
        msg = 'Error looking for variable names in {} rc file in section {}.'.format(app_name, section_name)
        log.exception (msg)
        raise KeyError(msg)

    if type == 'timeseries':
        work_dir = op.join(root_dir, feats_dir_name, ts_feats_dir_name)
    elif type == 'scalar_activity':
        work_dir = op.join(root_dir, feats_dir_name, scalar_feats_dir_name)
    else:
        msg = 'Expected type variable value of {} but got {}.'.format(type_choices, type)
        log.error(msg)
        raise ValueError(msg)

    files = find_files(work_dir, regex)
    if len(files):
        log.debug('Found the following export data files: {}.'.format(files))
    else:
        log.debug('Did not found any export data files within {} with te regex {}.'.format(work_dir, regex))

    return files


def get_cobre_export_timeseries(root_dir=EXPORTS_DIR, section_name='old_cobre', fwhm='4mm'):
    """
    See get_cobre_export_data.

    Parameters
    ----------
    fwhm: str
        Part of the file name with information of FWHM smoothing kernel size, e.g.: '0mm' or '4mm'

    Returns
    -------
    List of files found
    """
    regex = '.*' + fwhm + '.*'
    return get_cobre_export_data(root_dir, section_name=section_name, type='timeseries', regex=regex)


def get_cobre_export_scalar_data(root_dir=EXPORTS_DIR, section_name='old_cobre', type='reho', pipeline='wtemp_noglob'):
    """
    See get_cobre_export_data.

    Parameters
    ----------
    type: str
        Type of scalar fMRI-based activity measure, e.g., 'reho', 'alff', 'falff', 'vmhc'

    pipeline: str
        Pipeline configuration for

    Returns
    -------
    List of files found
    """
    regex = '.*' + type + '.*' + pipeline + '.*'
    return get_cobre_export_data(root_dir, section_name=section_name, type='scalar_activity', regex=regex)


def has_the_correct_subject_order(alist, filter_by_subject_ids=False):
    """Using the subject id list from get_subject_ids_and_labels will match alist for the same length and order.

    Parameters
    ----------
    alist: list of str or list of str
        If list of string will re.search each string item using the corresponding subject id.
        If list of lists of string will look within each sub-list for an exact match of the corresponding subject id.

    Returns
    -------
    has_the_correct_order: bool
        Will return False with any error, length mismatch or element without subject id match.
        True otherwise.
    """
    ids, _ = get_subject_ids_and_labels(filter_by_subject_ids=filter_by_subject_ids)

    if len(ids) < 1:
        msg = 'The list of subjects ids is empty. Expected something else.'
        log.error(msg)
        return False

    if len(alist) < 1:
        msg = 'The given list to be checked is empty. Expected something else.'
        log.error(msg)
        return False

    if len(ids) != len(alist):
        msg = 'The length of the given list and the list of subject ids are different. Expected the same length, ' \
              'got {} and {}. The fist element of the given list is {}.'.format(len(alist), len(ids), alist[0])
        log.error(msg)
        return False

    for items in zip(ids, alist):
        if isinstance(items[1], str):
            if re.search(items[0], items[1]) is None:
                return False
        elif isinstance(items[1], list):
            if where_is(items[1], items[0], lookup_func=re.search) < 0:
                return False
        else:
            log.error('The given list element type is {}. Expected str or list of str.'
                      'The first element of the given list is {}'.format(type(items[1]), alist[0]))
            return False

    return True


@task(autoprint=True)
def get_subject_folders(work_dir=PREPROC_DIR, section_name='old_cobre', pipe_varname='pipe_wtemp_noglob',
                        app_name=APPNAME, verbose=False, filter_by_subject_ids=False, check_order=True):
    """Return the first folder within the pipeline folder that is found with the name subj_id.

    Parameters
    ----------
    subj_id: str
        ID number of the subject.

    work_dir: str (optional)
        Root folder path

    section_name: str (optional)
        Name of the section in the rcfiles to look for the pipe_varname argument value.

    pipe_varname: str (optional)
        Name of the variable in the rcfiles which hold the path to the desired pipeline to look for the subject folder.

    verbose: bool (optional)
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    subj_dir: str
        Path to the subject folder.
    """
    verbose_switch(verbose)

    # get the pipeline folder path
    pipe_dirpath = get_pipeline_folder(root_dir=work_dir, pipe_section_name=section_name, pipe_varname=pipe_varname,
                                       app_name=app_name)

    folders      = [op.join(pipe_dirpath, subj_f) for subj_f in os.listdir(pipe_dirpath)]
    subj_ids, _  = get_subject_ids_and_labels(filter_by_subject_ids)

    subj_folders = []
    for idx, sid in enumerate(subj_ids):
        fidx = where_is(folders, sid, lookup_func=re.search)
        if fidx >= 0:
            subj_folders.append(folders[fidx])

    # check that func and ids have the same length and match
    if check_order:
        if not has_the_correct_subject_order(subj_folders, filter_by_subject_ids=filter_by_subject_ids):
            raise IOError('The list of subject folders found in {} and the list of subject '
                          'ids do not match.'.format(pipe_dirpath))

    return subj_folders


@task(autoprint=True)
def get_subject_folder(subj_id, work_dir=PREPROC_DIR, section_name='old_cobre',
                       pipe_varname='pipe_wtemp_noglob', app_name=APPNAME, verbose=False):
    """Return the first folder within the pipeline folder that is found with the name subj_id.

    Parameters
    ----------
    subj_id: str
        ID number of the subject.

    work_dir: str (optional)
        Root folder path

    section_name: str (optional)
        Name of the section in the rcfiles to look for the pipe_varname argument value.

    pipe_varname: str (optional)
        Name of the variable in the rcfiles which hold the path to the desired pipeline to look for the subject folder.

    verbose: bool (optional)
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    subj_dir: str
        Path to the subject folder.
    """
    verbose_switch(verbose)

    subj_folders = get_subject_folders(work_dir=work_dir, section_name=section_name, pipe_varname=pipe_varname,
                                       app_name=app_name, verbose=verbose, filter_by_subject_ids=False)

    return get_subject_folder_from_list(subj_id, file_list=subj_folders, verbose=verbose)



@task(autoprint=True)
def get_subject_file(file_varname, subj_dir, check_exists=True, app_name=APPNAME):
    """ Return the filepath for the rcfile file_varname for the given subject folder.

    Parameters
    ----------
    file_varname: str

    subj_dir: str

    check_exists: bool

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    filepath: str
    """
    section_name, var_value = find_in_sections(file_varname, app_name)

    if section_name == 'files_of_interest':
        filepath = find_files(subj_dir, var_value)
        if isinstance(filepath, list):
            if len(filepath) == 1:
                filepath = filepath[0]
            else:
                raise IOError('Found more than one file {} within {}.'.format(var_value, subj_dir))

    elif section_name == 'relative_paths':
        filepath = op.join(subj_dir, var_value)

    else:
        raise KeyError('The variable {} could only be found in section {}. '
                       'I do not know what to do with this.'.format(file_varname, section_name))

    if check_exists:
        if not op.exists(filepath):
            raise IOError('File {} not found.'.format(filepath))

    return filepath


@task(autoprint=True)
def get_subject_folder_from_list(subj_id, file_list=None, verbose=False):
    """Return the first folder within the pipeline folder that is found with the name subj_id.

    Parameters
    ----------
    subj_id: str
        ID number of the subject.

    file_list: list of str (optional)
        List of file paths which will be looked through to find the subject folder. All other variables will be ignored.

    verbose: bool (optional)
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    subj_dir: str
        Path to the subject folder.
    """
    verbose_switch(verbose)

    # check that func and ids have the same length and match
    if not has_the_correct_subject_order([f.split(op.sep) for f in file_list]):
        msg = 'The list of functional files found and the list of subject ids do not match.'
        raise RuntimeError(msg)

    ids, _     = get_subject_ids_and_labels()
    idx        = where_is(ids, subj_id)
    functional = file_list[idx]

    # find the subject root dir
    subjid_idx = where_is(functional.split(op.sep), subj_id, lookup_func=re.search)
    subj_dir   = os.sep.join(functional.split(op.sep)[0:subjid_idx + 1])

    return subj_dir


@task
def slicesdir(underlying, outline=None, work_dir=PREPROC_DIR, section_name='old_cobre',
              pipe_varname='pipe_wtemp_noglob', verbose=False, filter_by_subject_ids=False, axials=False):
    """ Call slicesdir using the relative file paths in the files_of_interest section.

    Parameters
    ----------
    underlying: str
        A files_of_interest relative file path variable, that will be used to look for the volume files that will
        be used as background in the slices images.

    outline: str
        If is a path to a file, this will be used as red-outline image on top of all images in underlying.
        If a files_of_interest relative file path variable, will match this list with the underlying subjects list
        and use each of them as red-outline image on top of the corresponding underlying image.

    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    axials: bool
        If True will output every second axial slice rather than just 9 ortho slices.

    verbose: bool
        If verbose will show DEBUG log info.
    """
    verbose_switch(verbose)

    # slicesdir
    slicesdir = op.join(os.environ['FSLDIR'], 'bin', 'slicesdir')

    # get the list of functional files for the given pipeline
    funcs = get_pipeline_files(root_dir=work_dir, section_name=section_name,
                               pipe_varname=pipe_varname, filter_by_subject_ids=filter_by_subject_ids,
                               file_name_varname='funcfiltx')

    # check that func and ids have the same length and match
    if not has_the_correct_subject_order([f.split(op.sep) for f in funcs]):
        msg = 'The list of functional files found and the list of subject ids do not match.'
        raise RuntimeError(msg)

    ids, _ = get_subject_ids_and_labels(filter_by_subject_ids=filter_by_subject_ids)

    outline_filepath = ''
    outline_is_one   = False
    if outline is not None:
        if op.exists(outline):
            outline_filepath = outline
            outline_is_one   = True

    # get relative filepaths
    underlying_filepath = get_file_of_interest_regex(underlying)
    log.debug('Using as background image: {}'.format(underlying_filepath))

    if outline is not None and not outline_filepath:
        outline_filepath = get_file_of_interest_regex(outline)
        log.debug('Using as red outline image: {}'.format(outline_filepath))

    underlyings = []
    outlines    = []
    for idx, subj_id in enumerate(ids):
        subj_dir   = get_subject_folder_from_list(subj_id, file_list=funcs, verbose=verbose)
        underlying = op.join(subj_dir, underlying_filepath)

        if not op.exists(underlying):
            raise IOError('Could not find file {}.'.format(underlying))
        underlyings.append(underlying)

        if not outline_is_one and outline_filepath:
            subj_outline = op.join(subj_dir, outline_filepath)
            if not op.exists(subj_outline):
                raise IOError('Could not find file {}.'.format(subj_outline))

            outlines.append(subj_outline)

    args = ' '
    if axials:
        args += '-S '

    if outlines:
        args += '-o '
        args += ' '.join(['{} {}'.format(i, j) for i, j in zip(underlyings, outlines)])
    elif outline_is_one:
        args += '-p {} '.format(outline_filepath)
        args += ' '.join(underlyings)
    else:
        args += ' '.join(underlyings)

    cmd = slicesdir + args
    log.debug('Running: {}'.format(cmd))
    local(cmd)


@task(autoprint=True)
def register_atlas_to_functionals(work_dir=PREPROC_DIR, atlas='aal_3mm', anat_out_var='aal_3mm_anat',
                                  func_out_var='aal_3mm_func', section_name='old_cobre',
                                  pipe_varname='pipe_wtemp_noglob', verbose=False, filter_by_subject_ids=False,
                                  parallel=False, app_name=APPNAME):
    """Apply the existent transformation from MNI standard to functional MRI to an atlas image in MNI space.

    Parameters
    ----------
    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    atlas: str
        Files of intereste variable name or file path to a 3D atlas volume.

    anat_out_var: str
        Variable name that holds the file name of the resulting registered atlas in a specific subject functional
        space.

    func_out_var: str
        Variable name that holds the file name of the resulting registered atlas in a specific subject functional
        space.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    verbose: bool
        If verbose will show DEBUG log info.

    parallel: bool
        If True will launch the commands using ${FSLDIR}/fsl_sub to use the cluster infrastructure you have setup
        with FSL (SGE or HTCondor).

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)
    """
    verbose_switch(verbose)

    try:
        atlas_filepath = get_standard_file(atlas)
    except:
        atlas_filepath = atlas
    if not op.exists(atlas_filepath):
        raise IOError('Could not find atlas file {}.'.format(atlas_filepath))

    #read relative filepaths
    subj_folders = get_subject_folders(work_dir=work_dir, section_name=section_name, pipe_varname=pipe_varname,
                                       app_name=app_name, verbose=verbose, filter_by_subject_ids=filter_by_subject_ids,
                                       check_order=True)

    for subj_path in subj_folders:
        find_subject_file_and_check = partial(get_subject_file, subj_dir=subj_path, check_exists=True)
        anat_brain      = find_subject_file_and_check('anat_brain'      )
        avg_func        = find_subject_file_and_check('mean_func'       )
        atlas2anat_lin  = find_subject_file_and_check('anat_to_mni_mat' )
        atlas2anat_nlin = find_subject_file_and_check('anat_to_mni_nl'  )
        anat2func_lin   = find_subject_file_and_check('anat_to_func_mat')

        atlas_in_anat   = get_subject_file(anat_out_var, subj_dir=subj_path, check_exists=False)
        atlas_in_func   = get_subject_file(func_out_var, subj_dir=subj_path, check_exists=False)

        log.debug('Registering atlas to functional: {}.\n'.format(' ,'.join([anat_brain, avg_func, atlas2anat_lin,
                                                                             atlas2anat_nlin, anat2func_lin,
                                                                             atlas_in_anat,
                                                                             atlas_in_func])))

        xfm_atlas_to_functional(atlas_filepath, anat_brain, avg_func, atlas2anat_lin, atlas2anat_nlin, False,
                                anat2func_lin, atlas_in_anat, atlas_in_func, interp='nn', verbose=verbose,
                                rewrite=False, parallel=parallel)


@task(autoprint=True)
def get_atlaspartition_hdf5path(subj_id, pipe_varname='pipe_wtemp_noglob', atlas='aal_3mm_func'):
    """ Return the hdf5 path for the atlas partition for the subject timeseries in the pipeline.

    Parameters
    ----------
    pipe_varname:
        Pipeline variable name.

    atlas:
        Atlas variable name

    subj_id: str
        Subject ID

    Returns
    -------
    hdf5path: str
    """
    return '/{}_{}_timeseries/{}'.format(pipe_varname, atlas, subj_id)


@task(autoprint=True)
def get_atlaspartition_hdf5_filepath(atlas='aal_3mm_func', app_name=APPNAME):
    """ Return the path of the HDF5 file which contains the atlas partition timeseries.

    atlas: str
        Atlas variable name

    app_name: str

    Returns
    -------
    hdf5_filepath: str
    """
    if atlas == 'atlas_3mm_func':
        return op.join(EXPORTS_DIR, get_rcfile_variable_value('out_aal_timeseries', app_name=app_name))
    else:
        raise ValueError('Expected the name of a valid atlas variable name as `atlas_3mm_func`, '
                         'but got {}.'.format(atlas))


@task(autoprint=True)
def get_connectivity_hdf5_filepath(atlas='aal_3mm_func', app_name=APPNAME):
    """ Return the path of the HDF5 file which contains the connectivity matrices.

    atlas: str
        Atlas variable name

    app_name: str

    Returns
    -------
    hdf5_filepath: str
    """
    if atlas == 'atlas_3mm_func':
        return op.join(EXPORTS_DIR, get_rcfile_variable_value('out_aal_connectivities', app_name=app_name))
    else:
        raise ValueError('Expected the name of a valid atlas variable name as `atlas_3mm_func`, '
                         'but got {}.'.format(atlas))


@task
def save_atlas_timeseries_packs(work_dir=PREPROC_DIR, atlas='aal_3mm_func', section_name='old_cobre',
                                pipe_varname='pipe_wtemp_noglob', app_name=APPNAME, verbose=False,
                                filter_by_subject_ids=False):
    """ Save the atlas partitioned timeseries into an HDF5 file.

    Parameters
    ----------
    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    atlas: str
        Files of intereste variable name or file path to a 3D atlas volume.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    verbose: bool
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)
    """
    verbose_switch(verbose)

    subj_timeseries = atlas_partition_timeseries(work_dir=work_dir, atlas=atlas, section_name=section_name,
                                                 pipe_varname=pipe_varname, app_name=app_name, verbose=verbose,
                                                 filter_by_subject_ids=filter_by_subject_ids)

    timeseries_filepath = get_atlaspartition_hdf5_filepath(atlas, app_name=app_name)

    for subj_id in subj_timeseries:
        # save_ts_pack into HDF file.
        h5path = get_atlaspartition_hdf5path(subj_id, pipe_varname=pipe_varname, atlas=atlas)

        log.debug('Saving {} {} partitioned functional timeseries in '
                  '{} group {}.'.format(subj_id, atlas, timeseries_filepath, h5path))

        save_variables_to_hdf5(timeseries_filepath, {'{}_timeseries'.format(atlas): subj_timeseries[subj_id]}, mode='a',
                               h5path=h5path)


def atlas_partition_timeseries(work_dir=PREPROC_DIR, atlas='aal_3mm_func', section_name='old_cobre',
                               pipe_varname='pipe_wtemp_noglob', app_name=APPNAME, verbose=False,
                               filter_by_subject_ids=False):
    """ Return a dictionary with each subject's timeseries partitioned by the atlas file.

    Parameters
    ----------
    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    atlas: str
        Files of intereste variable name or file path to a 3D atlas volume.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    verbose: bool
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    subj_timeseries: dict

    """
    verbose_switch(verbose)

    #read relative filepaths
    subj_folders = get_subject_folders(work_dir=work_dir, section_name=section_name, pipe_varname=pipe_varname,
                                       app_name=app_name, verbose=verbose, filter_by_subject_ids=filter_by_subject_ids,
                                       check_order=True)

    ids, _ = get_subject_ids_and_labels(filter_by_subject_ids=filter_by_subject_ids)

    subj_timeseries = OrderedDict()
    for idx, subj_path in enumerate(subj_folders):
        subj_id = ids[idx]

        find_subject_file_and_check = partial(get_subject_file, subj_dir=subj_path, check_exists=True)
        funcbrainmask   = find_subject_file_and_check('funcbrainmask'     )
        functional      = find_subject_file_and_check('func_freq_filtered')
        atlas_in_func   = find_subject_file_and_check(atlas               )

        log.debug('Partitioning subject {} timeseries in {} using atlas {}.'.format(subj_id, functional, atlas_in_func))

        subj_atlas_ts = partition_timeseries(functional, atlas_in_func, funcbrainmask, zeroe=True, roi_values=None,
                                             outdict=True)

        subj_timeseries[subj_id] = subj_atlas_ts

    return subj_timeseries


#@task
#def save_connectivity_matrices(work_dir=PREPROC_DIR, atlas='aal_3mm_func', section_name='old_cobre',
#                               pipe_varname='pipe_wtemp_noglob', app_name=APPNAME, verbose=False,
#                               filter_by_subject_ids=False):
    """ Save the connectivity matrices of with each subject's timeseries partitioned by the atlas file into an HDF5
    file.
    The file will be saved in exports

    Parameters
    ----------
    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    atlas: str
        Files of intereste variable name or file path to a 3D atlas volume.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    verbose: bool
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    """
# connectitity_filepath = get_connectivity_hdf5_filepath(atlas, app_name=APPNAME)


#def create_connectivity_matrices(work_dir=PREPROC_DIR, atlas='aal_3mm_func', section_name='old_cobre',
#                                 pipe_varname='pipe_wtemp_noglob', app_name=APPNAME, verbose=False,
#                                 filter_by_subject_ids=False):
    """ Return a dictionary with each subject's timeseries partitioned by the atlas file.

    Parameters
    ----------
#    work_dir: str
        A real file path or a RCfile variable name which indicates where to start looking for files.
        Note: be sure that if you want it a variable name, don't have a folder with the same name near this script.

    atlas: str
        Files of intereste variable name or file path to a 3D atlas volume.

    section_name: str
        RCfile section name to get the pipe_varname and also look for root_dir if needed.

    pipe_varname: str
        RCfile variable name for the pipeline pattern to match and filter the full paths of the found files.

    verbose: bool
        If verbose will show DEBUG log messages.

    filter_by_subject_ids: bool
        If True will read the file defined by subj_id_list_file variable in the rcfile and filter the resulting list
        and let only matches to the subject ID list values.

    verbose: bool
        If verbose will show DEBUG log info.

    app_name: str
        Name of the app to look for the correspondent rcfile. Default: APPNAME (global variable)

    Returns
    -------
    subj_timeseries: dict

    """
# h5path = get_atlaspartition_hdf5path(subj_id, pipe_varname=pipe_varname, atlas=atlas)
# timeseries_filepath   = get_atlaspartition_hdf5_filepath(atlas, app_name=app_name)
# connectitity_filepath = get_connectivity_hdf5_filepath(atlas, app_name=APPNAME)

#load_variables_from_hdf5

#ts = h5py.File('/Users/alexandre/Dropbox (Neurita)/projects/cobre/cobre_partitioned_timeseries.hdf5')
