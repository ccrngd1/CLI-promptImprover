"""
Command-line interface for the Bedrock Prompt Optimizer.

This package provides a comprehensive CLI with argument parsing, command structure,
progress indicators, and formatted output for agent results and orchestration decisions.
"""

from .main import PromptOptimizerCLI, main
from .config import ConfigManager

__version__ = "1.0.0"
__all__ = ["PromptOptimizerCLI", "main", "ConfigManager"]