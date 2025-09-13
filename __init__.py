"""
Bedrock Prompt Optimizer

A comprehensive multi-agent system for iteratively improving prompts for Amazon Bedrock
using LLM-powered collaboration, intelligent orchestration, and embedded best practices.
"""

__version__ = "1.0.0"
__author__ = "ccrngd1"
__email__ = ""
__description__ = "Multi-agent system for optimizing prompts for Amazon Bedrock"

from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback

__all__ = [
    "PromptIteration",
    "ExecutionResult", 
    "EvaluationResult",
    "UserFeedback"
]