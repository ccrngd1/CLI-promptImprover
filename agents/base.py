"""
Base Agent class for the multi-agent prompt improvement system.

This module defines the abstract base class that all specialized agents inherit from,
providing a common interface for prompt analysis, refinement, and validation operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from models import PromptIteration, UserFeedback


@dataclass
class AgentResult:
    """Represents the result of an agent operation."""
    
    agent_name: str
    success: bool
    analysis: Dict[str, Any]
    suggestions: List[str]
    confidence_score: float  # 0.0 to 1.0
    error_message: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate the agent result data."""
        if not self.agent_name or not isinstance(self.agent_name, str):
            return False
        if not isinstance(self.success, bool):
            return False
        if not isinstance(self.analysis, dict):
            return False
        if not isinstance(self.suggestions, list):
            return False
        if not isinstance(self.confidence_score, (int, float)) or not (0.0 <= self.confidence_score <= 1.0):
            return False
        return True


class Agent(ABC):
    """
    Abstract base class for all prompt improvement agents.
    
    Each agent specializes in a specific aspect of prompt analysis and improvement:
    - AnalyzerAgent: Analyzes prompt structure and clarity
    - RefinerAgent: Generates improved prompt versions
    - ValidatorAgent: Validates syntax and logical consistency
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent with a name and optional configuration.
        
        Args:
            name: The name of the agent
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30.0)  # Default 30 second timeout
    
    @abstractmethod
    def process(self, 
                prompt: str, 
                context: Optional[Dict[str, Any]] = None,
                history: Optional[List[PromptIteration]] = None,
                feedback: Optional[UserFeedback] = None) -> AgentResult:
        """
        Process a prompt and return analysis/suggestions.
        
        Args:
            prompt: The prompt text to process
            context: Optional context about the prompt's intended use
            history: Optional list of previous prompt iterations
            feedback: Optional user feedback from previous iterations
            
        Returns:
            AgentResult containing the agent's analysis and suggestions
        """
        pass
    
    def validate_input(self, prompt: str) -> bool:
        """
        Validate input prompt before processing.
        
        Args:
            prompt: The prompt text to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not prompt or not isinstance(prompt, str):
            return False
        if len(prompt.strip()) == 0:
            return False
        return True
    
    def get_name(self) -> str:
        """Get the agent's name."""
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration."""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update the agent's configuration."""
        self.config.update(config)