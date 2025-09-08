# Design Document

## Overview

The Bedrock Prompt Optimizer is a Python application that implements a multi-agent system for iterative prompt improvement. The system follows a collaborative architecture where specialized agents work together to analyze, refine, execute, and evaluate prompts against Amazon Bedrock models. The application uses a pipeline-based approach with clear separation of concerns between prompt improvement, execution, evaluation, and user interaction components.

## Architecture

The system follows a modular architecture with the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Session Manager│    │ History Manager │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │   Optimization Engine   │
                    └─────────────┬───────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼────────┐    ┌───────────▼──────────┐    ┌────────▼────────┐
│ Agent Ensemble │    │  Bedrock Executor    │    │   Evaluator     │
│                │    │                      │    │                 │
│ - Analyzer     │    │ - Model Interface    │    │ - Quality Scorer│
│ - Refiner      │    │ - Response Handler   │    │ - Comparator    │
│ - Validator    │    │ - Error Handler      │    │ - Feedback Proc │
└────────────────┘    └──────────────────────┘    └─────────────────┘
```

## Components and Interfaces

### 1. CLI Interface (`cli.py`)
- **Purpose**: Provides command-line interaction for users
- **Key Methods**:
  - `start_optimization_session(initial_prompt, context)`
  - `view_history(session_id)`
  - `configure_settings()`
- **Dependencies**: Session Manager, History Manager

### 2. Session Manager (`session.py`)
- **Purpose**: Orchestrates the optimization workflow and manages session state
- **Key Methods**:
  - `create_session(prompt, context)`
  - `run_optimization_cycle()`
  - `collect_user_feedback()`
  - `finalize_session()`
- **Dependencies**: Optimization Engine, History Manager

### 3. Agent Ensemble (`agents/`)
- **Purpose**: LLM-powered multi-agent system for intelligent collaborative prompt improvement
- **Components**:
  - **LLM Orchestrator**: Coordinates agent interactions using LLM-based decision making and conflict resolution
  - **Analyzer Agent**: Uses LLM with specialized system prompts to analyze prompt structure, clarity, and adherence to best practices
  - **Refiner Agent**: Leverages LLM with prompt engineering expertise to generate improved versions based on analysis and feedback
  - **Validator Agent**: Employs LLM with validation-focused system prompts to check refined prompts for syntax, logical consistency, and best practice compliance
  - **Evaluator Agent**: Uses LLM with evaluation expertise to assess prompt quality and provide reasoned judgments
- **Key Methods**:
  - `orchestrate_improvement_cycle(prompt, context, history)`
  - `analyze_prompt_with_llm(prompt, context, best_practices)`
  - `refine_prompt_with_llm(prompt, analysis, feedback, techniques)`
  - `validate_prompt_with_llm(prompt, requirements, standards)`
  - `evaluate_with_llm_judgment(prompt, response, criteria)`
- **System Prompts**: Each agent uses specialized system prompts containing domain-specific best practices and reasoning frameworks

### 4. Bedrock Executor (`bedrock/executor.py`)
- **Purpose**: Handles all interactions with Amazon Bedrock API
- **Key Methods**:
  - `execute_prompt(prompt, model_config)`
  - `get_available_models()`
  - `handle_rate_limits()`
- **Configuration**: Model selection, temperature, max tokens, etc.
- **Error Handling**: API errors, authentication, rate limiting

### 5. LLM-Enhanced Evaluator (`evaluation/evaluator.py`)
- **Purpose**: LLM-powered automated evaluation of prompt performance with intelligent reasoning
- **Key Methods**:
  - `evaluate_response_with_llm(prompt, response, criteria)`
  - `compare_versions_with_reasoning(current, previous)`
  - `generate_llm_evaluation_report()`
  - `apply_evaluation_best_practices(prompt, response)`
- **LLM Capabilities**: Uses specialized evaluation system prompts with reasoning frameworks for quality assessment
- **Metrics**: Relevance, clarity, completeness, consistency, best practices adherence, task-specific metrics with LLM-generated explanations

### 6. History Manager (`storage/history.py`)
- **Purpose**: Manages persistence of sessions, iterations, and results
- **Key Methods**:
  - `save_iteration(session_id, iteration_data)`
  - `load_session_history(session_id)`
  - `export_final_prompt(session_id, format)`
- **Storage**: JSON-based local storage with optional database backend

### 7. LLM Orchestration Engine (`engine.py`)
- **Purpose**: Coordinates the optimization workflow using LLM-based intelligent orchestration
- **Key Methods**:
  - `run_llm_orchestrated_iteration(prompt, context, feedback)`
  - `determine_convergence_with_reasoning(history)`
  - `generate_llm_improvement_suggestions()`
  - `resolve_agent_conflicts_with_llm(conflicting_recommendations)`
  - `synthesize_agent_outputs_with_llm(agent_results)`
- **LLM Orchestration**: Uses meta-reasoning to coordinate agents, resolve conflicts, and make strategic decisions about the optimization process

### 8. Best Practices Repository (`best_practices/`)
- **Purpose**: Centralized repository of prompt engineering best practices and system prompts
- **Components**:
  - **System Prompt Templates**: Specialized prompts for each agent type with embedded expertise
  - **Best Practices Database**: Curated collection of prompt engineering techniques and patterns
  - **Reasoning Frameworks**: Structured approaches for LLM-based analysis and decision making
- **Key Methods**:
  - `get_agent_system_prompt(agent_type, domain)`
  - `update_best_practices(new_techniques)`
  - `apply_reasoning_framework(task_type, context)`

## Data Models

### PromptIteration
```python
@dataclass
class PromptIteration:
    id: str
    session_id: str
    version: int
    prompt_text: str
    timestamp: datetime
    agent_analysis: Dict[str, Any]
    execution_result: ExecutionResult
    evaluation_scores: EvaluationResult
    user_feedback: Optional[UserFeedback]
```

### ExecutionResult
```python
@dataclass
class ExecutionResult:
    model_name: str
    response_text: str
    execution_time: float
    token_usage: Dict[str, int]
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

### EvaluationResult
```python
@dataclass
class EvaluationResult:
    overall_score: float
    relevance_score: float
    clarity_score: float
    completeness_score: float
    custom_metrics: Dict[str, float]
    qualitative_feedback: str
    improvement_suggestions: List[str]
```

### UserFeedback
```python
@dataclass
class UserFeedback:
    satisfaction_rating: int  # 1-5 scale
    specific_issues: List[str]
    desired_improvements: str
    continue_optimization: bool
```

### AgentRecommendation
```python
@dataclass
class AgentRecommendation:
    agent_type: str
    recommendation_text: str
    confidence_score: float
    reasoning: str
    best_practices_applied: List[str]
    suggested_changes: Dict[str, str]
```

### LLMOrchestrationResult
```python
@dataclass
class LLMOrchestrationResult:
    synthesized_recommendation: str
    agent_consensus_level: float
    conflict_resolutions: List[str]
    orchestrator_reasoning: str
    final_prompt_suggestion: str
    applied_best_practices: List[str]
```

### BestPracticeRule
```python
@dataclass
class BestPracticeRule:
    rule_id: str
    category: str  # e.g., "clarity", "structure", "examples"
    description: str
    system_prompt_fragment: str
    applicability_conditions: List[str]
    priority: int
```

## Error Handling

### Bedrock API Errors
- **Rate Limiting**: Implement exponential backoff with jitter
- **Authentication**: Clear error messages for credential issues
- **Model Availability**: Fallback to alternative models when primary is unavailable
- **Quota Exceeded**: Graceful degradation with user notification

### Agent Collaboration Errors
- **Agent Failure**: Continue with remaining agents, log failures
- **Consensus Issues**: Use voting mechanisms for conflicting recommendations
- **Timeout Handling**: Set reasonable timeouts for agent operations

### Data Persistence Errors
- **Storage Failures**: Implement retry logic and backup mechanisms
- **Corruption Recovery**: Validate data integrity on load
- **Migration Support**: Handle schema changes gracefully

## Testing Strategy

### Unit Testing
- **Agent Testing**: Mock external dependencies, test individual agent logic
- **Executor Testing**: Mock Bedrock API responses, test error scenarios
- **Evaluator Testing**: Test scoring algorithms with known inputs/outputs
- **Storage Testing**: Test persistence operations with temporary storage

### Integration Testing
- **End-to-End Workflows**: Test complete optimization cycles
- **Bedrock Integration**: Test against real Bedrock API in staging environment
- **Agent Collaboration**: Test multi-agent interactions and consensus building
- **CLI Interface**: Test command-line interactions and user flows

### Performance Testing
- **Concurrent Sessions**: Test multiple optimization sessions
- **Large Prompt Handling**: Test with various prompt sizes
- **History Scaling**: Test with large numbers of iterations
- **Memory Usage**: Monitor memory consumption during long sessions

### Configuration Management
- **Environment Variables**: Support for AWS credentials and region configuration
- **Config Files**: YAML/JSON configuration for model preferences and evaluation criteria
- **Runtime Configuration**: Allow dynamic configuration changes during sessions
- **Validation**: Validate configuration on startup with clear error messages

The design emphasizes modularity, testability, and extensibility while maintaining clear separation of concerns between the different aspects of prompt optimization. The multi-agent approach allows for specialized expertise in different areas of prompt improvement, while the pipeline architecture ensures a consistent and repeatable optimization process.