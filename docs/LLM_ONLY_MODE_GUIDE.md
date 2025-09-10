# LLM-Only Mode Guide

## Overview

LLM-Only Mode is a configuration option that bypasses all heuristic agents and relies exclusively on LLM-based optimization. This mode is designed for scenarios where sophisticated reasoning and context understanding are more valuable than the speed and cost efficiency of heuristic approaches.

## When to Use LLM-Only Mode

### Recommended Scenarios

1. **Complex Reasoning Tasks**
   - Multi-step problem solving
   - Abstract concept explanation
   - Creative content generation
   - Strategic planning and analysis

2. **Domain-Specific Content**
   - Specialized technical documentation
   - Medical or legal content requiring expertise
   - Financial analysis and compliance
   - Academic research and analysis

3. **Quality-Critical Applications**
   - Customer-facing content
   - Marketing and sales materials
   - Executive communications
   - Public-facing documentation

4. **Heuristic Limitations**
   - When rule-based approaches produce suboptimal results
   - Complex prompts that require nuanced understanding
   - Content requiring cultural or contextual sensitivity

### Not Recommended For

1. **Simple, Repetitive Tasks**
   - Basic formatting improvements
   - Simple grammar corrections
   - Standardized template generation

2. **High-Volume, Low-Complexity Processing**
   - Batch processing of simple prompts
   - Real-time applications requiring sub-second response
   - Cost-sensitive applications with tight budgets

3. **Offline or Air-Gapped Environments**
   - Systems without internet connectivity
   - Environments with strict data privacy requirements

## Performance Implications

### Latency Characteristics

| Mode | Typical Response Time | Factors Affecting Performance |
|------|----------------------|------------------------------|
| Hybrid | 2-5 seconds | Local processing + selective LLM calls |
| LLM-Only | 10-30 seconds | Multiple sequential LLM API calls |

### Throughput Considerations

```python
# Performance comparison example
hybrid_mode = {
    'requests_per_minute': 60-120,
    'concurrent_sessions': 20-50,
    'bottleneck': 'LLM API rate limits'
}

llm_only_mode = {
    'requests_per_minute': 20-40,
    'concurrent_sessions': 5-15,
    'bottleneck': 'LLM API rate limits and latency'
}
```

### Optimization Strategies

#### 1. Model Selection
```bash
# Fast, cost-effective model for simple tasks
bedrock-optimizer optimize "Simple prompt" \
  --llm-only \
  --model anthropic.claude-3-haiku-20240307-v1:0

# High-capability model for complex reasoning
bedrock-optimizer optimize "Complex analysis prompt" \
  --llm-only \
  --model anthropic.claude-3-sonnet-20240229-v1:0
```

#### 2. Iteration Management
```bash
# Reduce iterations for cost control
bedrock-optimizer optimize "Your prompt" \
  --llm-only \
  --max-iterations 3

# Standard iterations for quality optimization
bedrock-optimizer optimize "Your prompt" \
  --llm-only \
  --max-iterations 5-8
```

#### 3. Caching Configuration
```json
{
  "optimization": {
    "llm_only_mode": true,
    "cache_llm_responses": true
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 7200
  }
}
```

## Cost Analysis

### Token Usage Patterns

| Agent Type | Avg Input Tokens | Avg Output Tokens | Total per Iteration |
|------------|------------------|-------------------|-------------------|
| Analyzer | 800-1200 | 300-500 | 1100-1700 |
| Refiner | 1000-1500 | 400-800 | 1400-2300 |
| Validator | 600-1000 | 200-400 | 800-1400 |
| **Total per Iteration** | **2400-3700** | **900-1700** | **3300-5400** |

### Cost Estimation

#### Claude 3 Sonnet Pricing (as of 2024)
- Input tokens: $0.003 per 1K tokens
- Output tokens: $0.015 per 1K tokens

```python
# Cost calculation example
def estimate_llm_only_cost(iterations=5):
    avg_input_tokens_per_iteration = 3000
    avg_output_tokens_per_iteration = 1300
    
    total_input_tokens = avg_input_tokens_per_iteration * iterations
    total_output_tokens = avg_output_tokens_per_iteration * iterations
    
    input_cost = (total_input_tokens / 1000) * 0.003
    output_cost = (total_output_tokens / 1000) * 0.015
    
    return input_cost + output_cost

# Example costs
print(f"5 iterations: ${estimate_llm_only_cost(5):.4f}")  # ~$0.1425
print(f"3 iterations: ${estimate_llm_only_cost(3):.4f}")  # ~$0.0855
```

#### Cost Comparison by Model

| Model | Input Cost (per 1K) | Output Cost (per 1K) | 5-Iteration Session |
|-------|---------------------|----------------------|-------------------|
| Claude 3 Haiku | $0.00025 | $0.00125 | ~$0.0094 |
| Claude 3 Sonnet | $0.003 | $0.015 | ~$0.1425 |
| Claude 3 Opus | $0.015 | $0.075 | ~$0.7125 |

### Cost Management Strategies

#### 1. Model Tiering
```bash
# Use Haiku for initial analysis
bedrock-optimizer optimize "Simple prompt" \
  --llm-only \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --max-iterations 3

# Use Sonnet for complex reasoning
bedrock-optimizer optimize "Complex prompt" \
  --llm-only \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --max-iterations 5
```

#### 2. Budget Controls
```json
{
  "optimization": {
    "llm_only_mode": true,
    "cost_monitoring": true,
    "daily_budget_limit": 50.00,
    "session_cost_limit": 1.00
  }
}
```

#### 3. Caching Strategy
```python
# Implement intelligent caching
cache_config = {
    "cache_similar_prompts": True,
    "similarity_threshold": 0.85,
    "cache_ttl": 86400,  # 24 hours
    "max_cache_size": 1000
}
```

## Configuration Best Practices

### Development Environment
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "cost_monitoring": true,
    "max_concurrent_llm_calls": 3,
    "llm_timeout": 300
  },
  "orchestration": {
    "max_iterations": 3
  }
}
```

### Production Environment
```json
{
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "cost_monitoring": true,
    "rate_limiting": true,
    "max_concurrent_llm_calls": 10,
    "llm_timeout": 600,
    "cache_llm_responses": true
  },
  "orchestration": {
    "max_iterations": 5
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 7200
  }
}
```

## Monitoring and Observability

### Key Metrics to Track

1. **Performance Metrics**
   - Average response time per session
   - Token usage per iteration
   - Cache hit rate
   - API error rate

2. **Cost Metrics**
   - Daily/monthly token costs
   - Cost per optimization session
   - Cost per successful optimization
   - Budget utilization rate

3. **Quality Metrics**
   - Optimization success rate
   - User satisfaction scores
   - Iteration convergence rate
   - Fallback activation rate

### Monitoring Implementation

```python
# Example monitoring setup
class LLMOnlyModeMonitor:
    def __init__(self):
        self.metrics = {
            'total_sessions': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'fallback_rate': 0.0
        }
    
    def track_session(self, session_data):
        self.metrics['total_sessions'] += 1
        self.metrics['total_cost'] += session_data['cost']
        
        # Update averages
        self._update_averages(session_data)
    
    def get_daily_report(self):
        return {
            'date': datetime.now().date(),
            'sessions_processed': self.metrics['total_sessions'],
            'total_cost': self.metrics['total_cost'],
            'avg_cost_per_session': self.metrics['total_cost'] / max(1, self.metrics['total_sessions']),
            'performance_summary': self._get_performance_summary()
        }
```

### Alerting Configuration

```yaml
# monitoring/alerts.yaml
alerts:
  cost_alerts:
    daily_budget_threshold: 80%  # Alert at 80% of daily budget
    session_cost_threshold: 2.00  # Alert for sessions over $2
    
  performance_alerts:
    response_time_threshold: 60  # Alert if response time > 60s
    error_rate_threshold: 5%     # Alert if error rate > 5%
    
  quality_alerts:
    fallback_rate_threshold: 20%  # Alert if fallback rate > 20%
    low_satisfaction_threshold: 3.0  # Alert if avg rating < 3.0
```

## Troubleshooting

### Common Issues

#### 1. High Costs
**Symptoms:** Unexpected high API bills
**Solutions:**
- Enable cost monitoring and budget limits
- Use more cost-effective models (Haiku vs Sonnet)
- Reduce max iterations
- Implement aggressive caching

#### 2. Slow Performance
**Symptoms:** Long response times, timeouts
**Solutions:**
- Increase timeout values
- Reduce concurrent LLM calls
- Use faster models for simple tasks
- Implement request queuing

#### 3. Quality Issues
**Symptoms:** Poor optimization results
**Solutions:**
- Enable fallback to heuristic agents
- Use higher-capability models
- Increase iteration limits
- Provide more detailed context

#### 4. API Rate Limiting
**Symptoms:** Frequent API errors, throttling
**Solutions:**
- Implement exponential backoff
- Reduce concurrent requests
- Enable rate limiting configuration
- Consider API quota upgrades

### Debug Commands

```bash
# Check LLM-only mode status
bedrock-optimizer config --get optimization.llm_only_mode

# Test LLM connectivity
bedrock-optimizer models --test "Hello" --model anthropic.claude-3-haiku-20240307-v1:0

# View detailed session logs
bedrock-optimizer history --session-id abc123 --debug --show-costs

# Monitor real-time performance
bedrock-optimizer monitor --mode llm-only --live
```

## Migration Guide

### From Hybrid to LLM-Only Mode

1. **Assessment Phase**
   ```bash
   # Test with a subset of prompts
   bedrock-optimizer optimize "test prompt" --llm-only --dry-run
   ```

2. **Gradual Migration**
   ```bash
   # Enable for specific domains first
   bedrock-optimizer config --set optimization.llm_only_domains=["healthcare","finance"]
   ```

3. **Full Migration**
   ```bash
   # Enable globally with fallback
   bedrock-optimizer config --set optimization.llm_only_mode=true
   bedrock-optimizer config --set optimization.fallback_to_heuristic=true
   ```

4. **Monitoring and Optimization**
   ```bash
   # Monitor performance and costs
   bedrock-optimizer history --cost-report --since migration-date
   ```

This comprehensive guide provides all the information needed to effectively use LLM-Only Mode, including performance considerations, cost management, and best practices for different environments.