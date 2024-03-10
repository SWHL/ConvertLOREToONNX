# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
import tempfile
import threading
import time
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Union

from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger
from .constants import (FILE_HASH, MINIMUM_DOWNLOAD_INTERVAL_SECONDS,
                        MODELSCOPE_DOWNLOAD_PARALLELS,
                        MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB)
from .file_download import (get_file_download_url, http_get_file,
                            parallel_download)
from .utils.caching import ModelFileSystemCache
from .utils.utils import (file_integrity_validation, get_cache_dir,
                          model_id_to_group_owner_name)

logger = get_logger()

recent_downloaded = threading.local()


def snapshot_download(model_id: str,
                      revision: Optional[str] = DEFAULT_MODEL_REVISION,
                      cache_dir: Union[str, Path, None] = None,
                      user_agent: Optional[Union[Dict, str]] = None,
                      local_files_only: Optional[bool] = False,
                      cookies: Optional[CookieJar] = None,
                      ignore_file_pattern: List = None) -> str:
    """Download all files of a repo.
    Downloads a whole snapshot of a repo's files at the specified revision. This
    is useful when you want all files from a repo, because you don't know which
    ones you will need a priori. All files are nested inside a folder in order
    to keep their actual filename relative to that folder.

    An alternative would be to just clone a repo but this would require that the
    user always has git and git-lfs installed, and properly configured.

    Args:
        model_id (str): A user or an organization name and a repo name separated by a `/`.
        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
            commit hash. NOTE: currently only branch and tag name is supported
        cache_dir (str, Path, optional): Path to the folder where cached files are stored.
        user_agent (str, dict, optional): The user-agent info in the form of a dictionary or a string.
        local_files_only (bool, optional): If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        cookies (CookieJar, optional): The cookie of the request, default None.
        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.
    Raises:
        ValueError: the value details.

    Returns:
        str: Local folder path (string) of repo snapshot

    Note:
        Raises the following errors:
        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
        if `use_auth_token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if
        ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
        if some parameter value is invalid
    """

    if cache_dir is None:
        cache_dir = get_cache_dir()
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    temporary_cache_dir = os.path.join(cache_dir, 'temp')
    os.makedirs(temporary_cache_dir, exist_ok=True)

    group_or_owner, name = model_id_to_group_owner_name(model_id)
    name = name.replace('.', '___')

    cache = ModelFileSystemCache(cache_dir, group_or_owner, name)

    is_recent_downloaded = False
    current_time = time.time()
    recent_download_models = getattr(recent_downloaded, 'models', None)
    if recent_download_models is None:
        recent_downloaded.models = {}
    else:
        if model_id in recent_download_models:
            recent_download_time = recent_download_models[model_id]
            if current_time - recent_download_time < MINIMUM_DOWNLOAD_INTERVAL_SECONDS:
                is_recent_downloaded = True
                recent_download_models[model_id] = current_time
    if local_files_only:
        if len(cache.cached_files) == 0:
            raise ValueError(
                'Cannot find the requested files in the cached path and outgoing'
                ' traffic has been disabled. To enable model look-ups and downloads'
                " online, set 'local_files_only' to False.")
        logger.warning('We can not confirm the cached file is for revision: %s'
                       % revision)
        return cache.get_root_location(
        )  # we can not confirm the cached file is for snapshot 'revision'
    elif is_recent_downloaded:
        logger.warning('Download interval is too small, use local cache')
        return cache.get_root_location()
    else:
        # make headers
        headers = {
            'user-agent':
            ModelScopeConfig.get_user_agent(user_agent=user_agent, )
        }
        _api = HubApi()
        if cookies is None:
            cookies = ModelScopeConfig.get_cookies()
        revision_detail = _api.get_valid_revision_detail(
            model_id, revision=revision, cookies=cookies)
        revision = revision_detail['Revision']

        snapshot_header = headers if 'CI_TEST' in os.environ else {
            **headers,
            **{
                'Snapshot': 'True'
            }
        }
        if cache.cached_model_revision is not None:
            snapshot_header[
                'cached_model_revision'] = cache.cached_model_revision

        model_files = _api.get_model_files(
            model_id=model_id,
            revision=revision,
            recursive=True,
            use_cookies=False if cookies is None else cookies,
            headers=snapshot_header,
        )

        if ignore_file_pattern is None:
            ignore_file_pattern = []
        if isinstance(ignore_file_pattern, str):
            ignore_file_pattern = [ignore_file_pattern]

        with tempfile.TemporaryDirectory(
                dir=temporary_cache_dir) as temp_cache_dir:
            for model_file in model_files:
                if model_file['Type'] == 'tree' or \
                        any([re.search(pattern, model_file['Name']) is not None for pattern in ignore_file_pattern]):
                    continue
                # check model_file is exist in cache, if existed, skip download, otherwise download
                if cache.exists(model_file):
                    file_name = os.path.basename(model_file['Name'])
                    logger.debug(
                        f'File {file_name} already in cache, skip downloading!'
                    )
                    continue

                # get download url
                url = get_file_download_url(
                    model_id=model_id,
                    file_path=model_file['Path'],
                    revision=revision)

                if MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB * 1000 * 1000 < model_file[
                        'Size'] and MODELSCOPE_DOWNLOAD_PARALLELS > 1:
                    parallel_download(
                        url,
                        temp_cache_dir,
                        model_file['Name'],
                        headers=headers,
                        cookies=None
                        if cookies is None else cookies.get_dict(),
                        file_size=model_file['Size'])
                else:
                    http_get_file(
                        url,
                        temp_cache_dir,
                        model_file['Name'],
                        headers=headers,
                        cookies=cookies)

                # check file integrity
                temp_file = os.path.join(temp_cache_dir, model_file['Name'])
                if FILE_HASH in model_file:
                    file_integrity_validation(temp_file, model_file[FILE_HASH])
                # put file into to cache
                cache.put_file(model_file, temp_file)

        cache.save_model_version(revision_info=revision_detail)
        recent_downloaded.models[model_id] = time.time()
        return os.path.join(cache.get_root_location())
