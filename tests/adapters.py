from __future__ import annotations

import os
from typing import Any

import cs336_data.filtering_helper



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return cs336_data.filtering_helper.extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return cs336_data.filtering_helper.identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return cs336_data.filtering_helper.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return cs336_data.filtering_helper.mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return cs336_data.filtering_helper.mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return cs336_data.filtering_helper.classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return cs336_data.filtering_helper.classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return cs336_data.filtering_helper.classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return cs336_data.filtering_helper.gopher_quality_filter(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return cs336_data.filtering_helper.exact_line_deduplication(text)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return cs336_data.filtering_helper.minhash_deduplication(text)
