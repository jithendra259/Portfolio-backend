"""
Prompts and Knowledge package for Jithendra's Portfolio AI.
"""

from .knowledge import (
    BIOGRAPHY,
    EDUCATION,
    PUBLICATIONS,
    PROJECTS,
    TECHNICAL_SKILLS,
    WORK_EXPERIENCE,
    NAVIGATION_TARGETS,
)
from .pronunciations import PRONUNCIATION_REPLACEMENTS
from .system_prompt import SYSTEM_INSTRUCTIONS, build_system_instructions

__all__ = [
    "BIOGRAPHY",
    "EDUCATION",
    "PUBLICATIONS",
    "PROJECTS",
    "TECHNICAL_SKILLS",
    "WORK_EXPERIENCE",
    "NAVIGATION_TARGETS",
    "PRONUNCIATION_REPLACEMENTS",
    "SYSTEM_INSTRUCTIONS",
    "build_system_instructions",
]
