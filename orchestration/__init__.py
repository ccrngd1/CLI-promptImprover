"""
Orchestration module for LLM-based workflow coordination.

This module provides intelligent orchestration capabilities for coordinating
multiple agents, resolving conflicts, and making strategic decisions about
the prompt optimization process using LLM reasoning.
"""

from .engine import LLMOrchestrationEngine, OrchestrationResult, ConvergenceAnalysis

__all__ = [
    'LLMOrchestrationEngine',
    'OrchestrationResult', 
    'ConvergenceAnalysis'
]