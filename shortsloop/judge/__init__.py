"""VLM judge adapters. The judge model only ever reports scores + reasons;
floors and PASS/FAIL are applied by the checker (hard rule 1 holds inside L2 too)."""

from .base import JudgeAdapter, RetryableJudgeError, make_adapter

__all__ = ["JudgeAdapter", "RetryableJudgeError", "make_adapter"]
