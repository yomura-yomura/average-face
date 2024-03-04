import logging

from .average_face import compute_average_face

__all__ = ["compute_average_face", "logger"]

logger = logging.getLogger(__name__)
