#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset module
"""
import logging
import os
import requests
from urllib.parse import urlparse, unquote
from tqdm import tqdm


UNGA_URL = "https://digitallibrary.un.org/record/4060887/files/2025_9_19_ga_voting.csv?ln=en"
UNSC_URL = "https://digitallibrary.un.org/record/4055387/files/2025_11_25_sc_voting.csv?ln=en"

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

def download_dataset(url: str, dest_path: str, overwrite: bool = False, logger: logging.Logger = logging.getLogger(__name__)) -> bool:
    """Downloads the dataset from the given URL to the destination path.

    Parameters
    ----------
    url : str
        The URL to download the dataset from.
    dest_path : str
        The path to save the downloaded dataset.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    if os.path.exists(dest_path) and not overwrite:
        logger.info(f"Dataset already exists at {dest_path}. Skipping download.")
        return True
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))
        logger.debug(f"Making destination directory {os.path.dirname(dest_path)}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length') or 0)
        chunk_size = 8192
        buf = bytearray()
        with tqdm(total=total if total else None, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    buf.extend(chunk)
                    pbar.update(len(chunk))
        # store downloaded bytes so the later write(response.content) still works
        response._content = bytes(buf)
        with open(dest_path, 'wb') as file:
            file.write(response.content)
        logger.info(f"Dataset {url} downloaded to {dest_path}.")
        return True
    except requests.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def dataset_name(url: str, default: str = 'dataset.csv') -> str:
    """Extracts the dataset name from the given URL.

    Parameters
    ----------
    url : str
        The URL to extract the dataset name from.

    Returns
    -------
    str
        The extracted dataset name.
    """
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path) or ''
    if not filename:
        filename = unquote(parsed.query.split('=')[-1]) if parsed.query else default
    return filename

def download_unga(dest_path: str = DATASET_DIR, logger: logging.Logger = logging.getLogger(__name__)) -> bool:
    """Downloads the UNGA dataset.

    Parameters
    ----------
    dest_path : str
        The path to save the downloaded dataset.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    filename = dataset_name(UNGA_URL, default='ga_voting.csv')
    status = download_dataset(UNGA_URL, os.path.join(dest_path, filename), logger=logger)
    if status:
        logger.info(f"UNGA dataset downloaded to {os.path.join(dest_path, filename)}")
    else:
        logger.error("Failed to download UNGA dataset")
    return status

def download_unsc(dest_path: str = DATASET_DIR, logger: logging.Logger = logging.getLogger(__name__)) -> bool:
    """Downloads the UNSC dataset.

    Parameters
    ----------
    dest_path : str
        The path to save the downloaded dataset.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    filename = dataset_name(UNSC_URL, default='sc_voting.csv')
    status = download_dataset(UNSC_URL, os.path.join(dest_path, filename), logger=logger)
    if status:
        logger.info(f"UNSC dataset downloaded to {os.path.join(dest_path, filename)}")
    else:
        logger.error("Failed to download UNSC dataset")
    return status