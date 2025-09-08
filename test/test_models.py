"""
Test script for core data models.
"""

from datetime import datetime
from models import PromptIteration, ExecutionResult, EvaluationResult, UserFeedback


def test_execution_result():
    """Test ExecutionResult creation and validation."""
    result = ExecutionResult(
        model_name="claude-3-sonnet",
        response_text="This is a test response",
        execution_time=1.5,
        token_usage={"input": 100, "output": 50},
        success=True
    )
    
    assert result.validate(), "ExecutionResult validation failed"
    
    # Test serialization
    data = result.to_dict()
    restored = ExecutionResult.from_dict(data)
    assert restored.model_name == result.model_name
    assert restored.success == result.success
    print("✓ ExecutionResult tests passed")


def test_evaluation_result():
    """Test EvaluationResult creation and validation."""
    result = EvaluationResult(
        overall_score=0.85,
        relevance_score=0.9,
        clarity_score=0.8,
        completeness_score=0.85,
        custom_metrics={"creativity": 0.7},
        qualitative_feedback="Good response with room for improvement",
        improvement_suggestions=["Add more specific examples", "Improve clarity"]
    )
    
    assert result.validate(), "EvaluationResult validation failed"
    
    # Test serialization
    data = result.to_dict()
    restored = EvaluationResult.from_dict(data)
    assert restored.overall_score == result.overall_score
    assert len(restored.improvement_suggestions) == 2
    print("✓ EvaluationResult tests passed")


def test_user_feedback():
    """Test UserFeedback creation and validation."""
    feedback = UserFeedback(
        satisfaction_rating=4,
        specific_issues=["Too verbose", "Missing examples"],
        desired_improvements="Make it more concise",
        continue_optimization=True
    )
    
    assert feedback.validate(), "UserFeedback validation failed"
    
    # Test serialization
    data = feedback.to_dict()
    restored = UserFeedback.from_dict(data)
    assert restored.satisfaction_rating == feedback.satisfaction_rating
    assert len(restored.specific_issues) == 2
    print("✓ UserFeedback tests passed")


def test_prompt_iteration():
    """Test PromptIteration creation and validation."""
    # Create nested objects
    execution_result = ExecutionResult(
        model_name="claude-3-sonnet",
        response_text="Test response",
        execution_time=1.0,
        token_usage={"input": 50, "output": 25},
        success=True
    )
    
    evaluation_result = EvaluationResult(
        overall_score=0.8,
        relevance_score=0.85,
        clarity_score=0.75,
        completeness_score=0.8
    )
    
    user_feedback = UserFeedback(
        satisfaction_rating=4,
        specific_issues=["Could be clearer"],
        continue_optimization=True
    )
    
    iteration = PromptIteration(
        session_id="test-session-123",
        version=1,
        prompt_text="Test prompt for optimization",
        agent_analysis={"analyzer": "Good structure", "refiner": "Needs improvement"},
        execution_result=execution_result,
        evaluation_scores=evaluation_result,
        user_feedback=user_feedback
    )
    
    assert iteration.validate(), "PromptIteration validation failed"
    
    # Test JSON serialization
    json_str = iteration.to_json()
    restored = PromptIteration.from_json(json_str)
    
    assert restored.session_id == iteration.session_id
    assert restored.version == iteration.version
    assert restored.prompt_text == iteration.prompt_text
    assert restored.execution_result.model_name == execution_result.model_name
    assert restored.evaluation_scores.overall_score == evaluation_result.overall_score
    assert restored.user_feedback.satisfaction_rating == user_feedback.satisfaction_rating
    
    print("✓ PromptIteration tests passed")


if __name__ == "__main__":
    test_execution_result()
    test_evaluation_result()
    test_user_feedback()
    test_prompt_iteration()
    print("\n✅ All data model tests passed!")