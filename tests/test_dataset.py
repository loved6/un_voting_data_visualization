#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for dataset module
"""
import os
import logging
import pytest
from _pytest.logging import LogCaptureFixture

from src import dataset

def test_download_dataset_success(tmp_path: str, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture) -> None:
    test_logger = logging.getLogger("test_download_dataset_success")
    with caplog.at_level(logging.DEBUG, logger="test_download_dataset_success"):
        url = "https://example.com/test_dataset.csv"
        dest_path = os.path.join(tmp_path,'dataset', "test_dataset.csv")

        # Mock requests.get to avoid actual network call
        class MockResponse:
            def __init__(self, content):
                self.content = content
                self.headers = {'content-length': str(len(content))}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192):
                """Mock iter_content for streaming downloads"""
                for i in range(0, len(self.content), chunk_size):
                    yield self.content[i:i + chunk_size]

        def mock_get(url, **kwargs):
            return MockResponse(b"col1,col2\nval1,val2")

        monkeypatch.setattr(dataset.requests, 'get', mock_get)

        success = dataset.download_dataset(url, dest_path, False, test_logger)

        assert success
        assert f"Making destination directory {os.path.dirname(dest_path)}" in caplog.text
        assert f"Dataset {url} downloaded to {dest_path}." in caplog.text
        assert os.path.exists(dest_path)
        with open(dest_path, 'rb') as f:
            content = f.read()
        assert content == b"col1,col2\nval1,val2"

def test_download_dataset_file_exists(tmp_path: str, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture) -> None:
    test_logger = logging.getLogger("test_download_dataset_file_exists")
    with caplog.at_level(logging.DEBUG, logger="test_download_dataset_file_exists"):
        url = "https://example.com/test_dataset.csv"
        dest_path = os.path.join(tmp_path, 'dataset', "test_dataset.csv")
        contents = b"col1,col2\nval1,val2"

        # Mock requests.get to avoid actual network call
        class MockResponse:
            def __init__(self, content):
                self.content = content
                self.headers = {'content-length': str(len(content))}

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=8192):
                """Mock iter_content for streaming downloads"""
                for i in range(0, len(self.content), chunk_size):
                    yield self.content[i:i + chunk_size]

        def mock_get(url, **kwargs):
            return MockResponse(contents)

        monkeypatch.setattr(dataset.requests, 'get', mock_get)

        # create the file beforehand
        os.makedirs(os.path.dirname(dest_path))
        with open(dest_path, 'wb') as f:
            f.write(contents)

        success = dataset.download_dataset(url, dest_path, False, logger=test_logger)

        assert success
        assert f"Dataset already exists at {dest_path}. Skipping download." in caplog.text

def test_download_dataset_error(tmp_path: str, monkeypatch: pytest.MonkeyPatch, caplog: LogCaptureFixture) -> None:
    test_logger = logging.getLogger("test_download_dataset_error")
    with caplog.at_level(logging.DEBUG, logger="test_download_dataset_error"):
        url = "https://example.com/test_dataset.csv"
        dest_path = os.path.join(tmp_path, 'dataset', "test_dataset.csv")
        contents = b"col1,col2\nval1,val2"

        # Mock requests.get to avoid actual network call
        class MockResponse:
            def __init__(self, content):
                self.content = content
                self.headers = {'content-length': str(len(content))}

            def raise_for_status(self):
                raise dataset.requests.exceptions.RequestException("Simulated download error")

            def iter_content(self, chunk_size=8192):
                """Mock iter_content for streaming downloads"""
                for i in range(0, len(self.content), chunk_size):
                    yield self.content[i:i + chunk_size]

        def mock_get(url, **kwargs):
            return MockResponse(contents)

        monkeypatch.setattr(dataset.requests, 'get', mock_get)

        success = dataset.download_dataset(url, dest_path, False, logger=test_logger)

        assert not(success)
        assert "Error downloading dataset: Simulated download error" in caplog.text

def test_dataset_name_nondefault() -> None:
    url = "https://example.com/test_dataset.csv"
    default = 'def_dataset.csv'
    filename = dataset.dataset_name(url, default)
    assert filename == "test_dataset.csv"

def test_dataset_name_default() -> None:
    url = "https://example.com/"
    default = 'def_dataset.csv'
    filename = dataset.dataset_name(url, default)
    assert filename == "def_dataset.csv"
