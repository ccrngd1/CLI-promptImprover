"""
Core data models for the Bedrock Prompt Optimizer.

This module defines the primary data structures used throughout the application
for representing prompt iterations, execution results, evaluations, and user feedback.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import uuid


@dataclass
class ExecutionResult:
    """Represents the result of executing a prompt against a Bedrock model."""
    
    model_name: str
    response_text: str
    execution_time: float
    token_usage: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the execution result data."""
        if not self.model_name or not isinstance(self.model_name, str):
            return False
        if not isinstance(self.execution_time, (int, float)) or self.execution_time < 0:
            return False
        if not isinstance(self.success, bool):
            return False
        if not isinstance(self.token_usage, dict):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class EvaluationResult:
    """Represents the evaluation scores and feedback for a prompt response."""
    
    overall_score: float
    relevance_score: float
    clarity_score: float
    completeness_score: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    qualitative_feedback: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate the evaluation result data."""
        scores = [self.overall_score, self.relevance_score, self.clarity_score, self.completeness_score]
        for score in scores:
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                return False
        if not isinstance(self.custom_metrics, dict):
            return False
        for metric_score in self.custom_metrics.values():
            if not isinstance(metric_score, (int, float)) or not (0.0 <= metric_score <= 1.0):
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class UserFeedback:
    """Represents user feedback on a prompt iteration."""
    
    satisfaction_rating: int
    specific_issues: List[str] = field(default_factory=list)
    desired_improvements: str = ""
    continue_optimization: bool = True
    
    def validate(self) -> bool:
        """Validate the user feedback data."""
        if not isinstance(self.satisfaction_rating, int) or not (1 <= self.satisfaction_rating <= 5):
            return False
        if not isinstance(self.specific_issues, list):
            return False
        if not isinstance(self.continue_optimization, bool):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class PromptIteration:
    """Represents a single iteration in the prompt optimization process."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    version: int = 1
    prompt_text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    agent_analysis: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[ExecutionResult] = None
    evaluation_scores: Optional[EvaluationResult] = None
    user_feedback: Optional[UserFeedback] = None
    
    def validate(self) -> bool:
        """Validate the prompt iteration data."""
        if not self.id or not isinstance(self.id, str):
            return False
        if not self.session_id or not isinstance(self.session_id, str):
            return False
        if not isinstance(self.version, int) or self.version < 1:
            return False
        if not self.prompt_text or not isinstance(self.prompt_text, str):
            return False
        if not isinstance(self.timestamp, datetime):
            return False
        
        # Validate nested objects if present
        if self.execution_result and not self.execution_result.validate():
            return False
        if self.evaluation_scores and not self.evaluation_scores.validate():
            return False
        if self.user_feedback and not self.user_feedback.validate():
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO string for JSON serialization
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptIteration':
        """Create instance from dictionary."""
        # Convert timestamp back from ISO string
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert nested objects
        if 'execution_result' in data and data['execution_result']:
            data['execution_result'] = ExecutionResult.from_dict(data['execution_result'])
        if 'evaluation_scores' in data and data['evaluation_scores']:
            data['evaluation_scores'] = EvaluationResult.from_dict(data['evaluation_scores'])
        if 'user_feedback' in data and data['user_feedback']:
            data['user_feedback'] = UserFeedback.from_dict(data['user_feedback'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PromptIteration':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)