# LLM Fallback Mechanism

This document describes the fallback mechanism implemented for LLM failures in the Bedrock Prompt Optimizer.

## Overview

The fallback mechanism provides resilience when LLM services are unavailable or fail during prompt optimization. When enabled, the system automatically falls back to heuristic agents when LLM agents encounter errors.

## Configuration

The fallback mechanism is controlled by two configuration parameters in the `optimization` section:

```json
{
  "optimization": {
    "llm_only_mode": false,
    "fallback_to_heuristic": true
  }
}
```

### Configuration Parameters

- **`llm_only_mode`** (boolean, default: `false`)
  - When `true`, only LLM-enhanced agents are used for prompt optimization
  - When `false`, both heuristic and LLM agents are used (hybrid mode)

- **`fallback_to_heuristic`** (boolean, default: `true`)
  - When `true`, enables fallback to heuristic agents when LLM agents fail
  - When `false`, disables fallback - failures will result in errors
  - Only relevant when `llm_only_mode` is `true`

## Fallback Scenarios

The fallback mechanism activates in the following scenarios:

### 1. LLM Service Unavailability
- Network connectivity issues
- API service downtime
- Authentication failures
- Rate limiting exceeded

### 2. LLM Response Failures
- Empty or malformed responses
- Timeout errors
- Model-specific errors
- Token limit exceeded

### 3. Agent Creation Failures
- LLM agent initialization errors
- Configuration issues
- Missing model specifications

## Fallback Behavior

### LLM-Only Mode with Fallback Enabled
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true
  }
}
```

**Behavior:**
1. System attempts to use LLM agents first
2. On failure, automatically creates and uses heuristic agents
3. Results include metadata indicating fallback was used
4. Confidence scores are reduced to reflect fallback usage

### LLM-Only Mode without Fallback
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": false
  }
}
```

**Behavior:**
1. System uses only LLM agents
2. On failure, returns error without attempting fallback
3. Optimization process fails if LLM services are unavailable

### Hybrid Mode
```json
{
  "optimization": {
    "llm_only_mode": false,
    "fallback_to_heuristic": true
  }
}
```

**Behavior:**
1. System uses both heuristic and LLM agents
2. Fallback setting is less relevant since heuristic agents are already available
3. Individual LLM agent failures don't affect overall process

## Implementation Details

### Agent Factory Fallback
The `AgentFactory` class handles fallback agent creation:

- **Individual Fallback**: Creates fallback agents for specific failed LLM agents
- **Emergency Fallback**: Creates minimal heuristic agents when all LLM agents fail
- **Service Unavailable Handling**: Manages complete LLM service outages

### Orchestration Engine Integration
The `LLMOrchestrationEngine` includes error recovery strategies:

- **Failure Tracking**: Monitors consecutive LLM failures
- **Automatic Recovery**: Switches to fallback agents after threshold exceeded
- **Error Context**: Provides detailed error information for debugging

### Agent-Level Fallback
Individual LLM agents support fallback processing:

- **`process_with_fallback()`**: Method that attempts LLM processing with fallback
- **Fallback Detection**: Automatically detects when fallback should be used
- **Result Annotation**: Marks results when fallback agents were used

## Usage Examples

### Basic LLM-Only with Fallback
```python
from agents.factory import AgentFactory

config = {
    'optimization': {
        'llm_only_mode': True,
        'fallback_to_heuristic': True
    }
}

factory = AgentFactory(config)
agents = factory.create_agents()  # Will fallback on LLM failure
```

### Orchestration with Fallback
```python
from orchestration.engine import LLMOrchestrationEngine

# Engine automatically handles fallback based on configuration
result = engine.run_llm_orchestrated_iteration(
    prompt="Your prompt here",
    context={'session_id': 'example'}
)

# Check if fallback was used
if result.agent_results:
    for agent_name, agent_result in result.agent_results.items():
        if agent_result.analysis and agent_result.analysis.get('fallback_used'):
            print(f"Agent {agent_name} used fallback")
```

### Manual Fallback Processing
```python
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent

analyzer = LLMAnalyzerAgent(config)
result = analyzer.process_with_fallback(prompt, context)

# Check if fallback was used
if result.analysis and result.analysis.get('fallback_used'):
    print(f"Fallback used due to: {result.analysis.get('llm_error')}")
```

## Monitoring and Logging

The fallback mechanism includes comprehensive logging:

### Log Messages
- **Fallback Activation**: When fallback agents are created
- **LLM Failures**: Details about LLM service failures
- **Recovery Success**: When fallback processing succeeds
- **Configuration Changes**: Mode switches and updates

### Metrics Tracking
- **Failure Counts**: Number of consecutive LLM failures
- **Fallback Usage**: Frequency of fallback activation
- **Agent Selection**: Which agents are used in each mode
- **Performance Impact**: Processing time differences

## Best Practices

### Recommended Configuration
For production environments:
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true
  }
}
```

This provides the best of both worlds:
- Uses advanced LLM capabilities when available
- Maintains service availability during LLM outages
- Provides graceful degradation

### Development and Testing
For development environments:
```json
{
  "optimization": {
    "llm_only_mode": false,
    "fallback_to_heuristic": true
  }
}
```

This hybrid mode provides:
- Consistent behavior regardless of LLM availability
- Ability to compare LLM vs heuristic results
- Reduced dependency on external services

### High-Reliability Scenarios
For scenarios requiring maximum reliability:
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true
  }
}
```

Combined with:
- Multiple LLM model configurations
- Retry mechanisms
- Health checks and monitoring

## Troubleshooting

### Common Issues

1. **Fallback Not Activating**
   - Check `fallback_to_heuristic` is set to `true`
   - Verify `llm_only_mode` is enabled
   - Review error logs for specific failure reasons

2. **Unexpected Fallback Usage**
   - Check LLM service health and connectivity
   - Verify model configurations are correct
   - Review authentication and rate limiting

3. **Poor Fallback Performance**
   - Heuristic agents have different capabilities than LLM agents
   - Consider hybrid mode for consistent performance
   - Tune heuristic agent configurations

### Debugging

Enable verbose logging to debug fallback behavior:
```json
{
  "cli": {
    "verbose_orchestration": true
  }
}
```

Check agent factory validation:
```python
factory = AgentFactory(config)
validation = factory.validate_configuration()
print(validation['warnings'])  # Check for configuration issues
```

## Performance Considerations

### Latency Impact
- **LLM Agents**: Higher latency due to API calls
- **Fallback Agents**: Lower latency, local processing
- **Fallback Switching**: Minimal overhead for detection and switching

### Cost Implications
- **LLM Usage**: Token-based costs for API calls
- **Fallback Usage**: No additional API costs
- **Hybrid Mode**: Balanced cost and performance

### Quality Trade-offs
- **LLM Agents**: Higher quality analysis and suggestions
- **Heuristic Agents**: Consistent but potentially less sophisticated results
- **Fallback Results**: Clearly marked to indicate quality expectations

## Security Considerations

### Data Privacy
- **LLM Agents**: Prompts sent to external services
- **Fallback Agents**: All processing remains local
- **Sensitive Data**: Consider fallback-only mode for sensitive prompts

### Service Dependencies
- **LLM Agents**: Dependent on external API availability
- **Fallback Agents**: No external dependencies
- **Resilience**: Fallback provides independence from external services

## Future Enhancements

Planned improvements to the fallback mechanism:

1. **Smart Fallback Selection**: Choose fallback strategy based on error type
2. **Partial Fallback**: Use fallback for specific agents while maintaining LLM for others
3. **Quality Prediction**: Estimate result quality when using fallback
4. **Adaptive Thresholds**: Dynamically adjust fallback triggers based on service reliability
5. **Fallback Caching**: Cache fallback results to improve performance