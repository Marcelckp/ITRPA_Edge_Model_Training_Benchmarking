"""slmbench: benchmarking harness and decision-support advisor for SLM adaptation studies."""

import logging

__version__ = "0.1.0"

MOCK_WATERMARK = "SIMULATED - NOT RESEARCH DATA"

logging.getLogger(__name__).addHandler(logging.NullHandler())
