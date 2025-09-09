# Design Document

## Overview

This feature adds a configuration option to enable "LLM-only mode" that bypasses all non-LLM heuristic code and relies exclusively on LLM-based agents for prompt optimization. The system currently uses a hybrid approach with both heuristic agents (AnalyzerAgent, RefinerAgent, ValidatorAgent) and LLM-enhanced agents (LLMAnalyzerAgent, LLMRefinerAgent, LLMValidatorAgent). When LLM-only mode is enabled, the orchestration engine will skip the heuristic agents and use only the LLM-enhanced versions.

## Architecture

### Configuration Layer
- Add `llm_only_mode` boolean parameter to the main configuration system
- Default value: `false` (maintains existing hybrid behavior)
- Configuration location: `config.json` under a new `optimization` section
- Runtime configuration changes supported without restart

### Agent Selection Logic
- Modify `LLMOrchestrationEngine` to check the `llm_only_mode` configuration
- When enabled, skip instantiation and execution of heuristic agents
- Route all agent requests to LLM-enhanced versions only
- Maintain consistent `AgentResult` interfaces regardless of mode

### Orchestration Engine Updates
- Update `_coordinate_agent_analysis()` method to filter agent selection based on mode
- Modify agent ensemble initialization to exclude heuristic agents when in LLM-only mode
- Add logging to indicate which agents are being bypassed
- Ensure fallback behavior remains intact

## Components and Interfaces

### Configuration Schema Extension
```json
{
  "optimization": {
    "llm_only_mode": false,
    "fallback_to_heuristic": true
  }
}
```

### Agent Factory Pattern
```python
class AgentFactory:
    def create_agents(self, config: Dict[str, Any]) -> Dict[str, Agent]:
        if config.get('optimization', {}).get('llm_only_mode', False):
            return self._create_llm_only_agents(config)
        else:
            return self._create_hybrid_agents(config)
```

### Orchestration Engine Interface
```python
class LLMOrchestrationEngine:
    def __init__(self, config: Dict[str, Any]):
        self.llm_only_mode = config.get('optimization', {}).get('llm_only_mode', False)
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        if self.llm_only_mode:
            return {
                'analyzer': LLMAnalyzerAgent(self.config.get('agents', {}).get('analyzer', {})),
                'refiner': LLMRefinerAgent(self.config.get('agents', {}).get('refiner', {})),
                'validator': LLMValidatorAgent(self.config.get('agents', {}).get('validator', {}))
            }
        else:
            # Existing hybrid approach
            return self._create_hybrid_agents()
```

## Data Models

### Configuration Model
```python
@dataclass
class OptimizationConfig:
    llm_only_mode: bool = False
    fallback_to_heuristic: bool = True
    log_bypassed_agents: bool = True
```

### Agent Execution Context
```python
@dataclass
class AgentExecutionContext:
    mode: str  # 'hybrid' or 'llm_only'
    available_agents: List[str]
    bypassed_agents: List[str]
    execution_strategy: str
```

## Error Handling

### LLM Failure Scenarios
- When LLM agents fail and `fallback_to_heuristic` is enabled, temporarily use heuristic agents
- Log all fallback events for monitoring and debugging
- Maintain error context to prevent infinite fallback loops

### Configuration Validation
- Validate `llm_only_mode` parameter type and value
- Provide clear error messages for invalid configurations
- Default to hybrid mode on configuration errors with warning logs

### Agent Availability Checks
- Verify LLM agents are properly configured before enabling LLM-only mode
- Check for required model configurations and API access
- Graceful degradation if LLM services are unavailable

## Testing Strategy

### Unit Tests
- Test configuration parsing and validation
- Test agent factory creation logic for both modes
- Test orchestration engine initialization with different configurations
- Mock LLM responses to test agent behavior in isolation

### Integration Tests
- Test end-to-end prompt optimization in LLM-only mode
- Test fallback behavior when LLM agents fail
- Test configuration changes during runtime
- Compare output quality between hybrid and LLM-only modes

### Performance Tests
- Measure latency differences between modes
- Test token usage and cost implications
- Benchmark throughput with LLM-only processing
- Monitor memory usage patterns

### Regression Tests
- Ensure existing hybrid mode functionality remains unchanged
- Test backward compatibility with existing configurations
- Verify no breaking changes to public APIs
- Test migration scenarios from hybrid to LLM-only mode

## Implementation Considerations

### Backward Compatibility
- Existing configurations without `llm_only_mode` default to hybrid mode
- All existing APIs and interfaces remain unchanged
- No breaking changes to agent result formats or orchestration workflows

### Performance Impact
- LLM-only mode may have higher latency due to API calls
- Increased token usage and associated costs
- Potential for better quality results with sophisticated LLM reasoning
- Consider caching strategies for repeated similar prompts

### Monitoring and Observability
- Add metrics for mode usage and performance
- Log agent selection decisions and bypass events
- Track success rates and quality scores by mode
- Monitor LLM API usage and costs

### Security Considerations
- Ensure LLM API credentials are properly secured
- Validate all inputs before sending to LLM services
- Implement rate limiting and quota management
- Consider data privacy implications of sending prompts to external LLM services