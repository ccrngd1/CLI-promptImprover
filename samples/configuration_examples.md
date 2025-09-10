# Configuration Examples for Hybrid and LLM-Only Modes

## Basic Hybrid Mode Configuration (Default)

```json
{
  "bedrock": {
    "region": "us-east-1",
    "default_model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "timeout": 30,
    "max_retries": 3
  },
  "orchestration": {
    "max_iterations": 10,
    "score_improvement_threshold": 0.02
  },
  "optimization": {
    "llm_only_mode": false,
    "fallback_to_heuristic": true
  },
  "agents": {
    "analyzer": {"enabled": true},
    "refiner": {"enabled": true},
    "validator": {"enabled": true}
  }
}
```

## LLM-Only Mode Configuration

### Development Environment
```json
{
  "bedrock": {
    "region": "us-east-1",
    "default_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "timeout": 60,
    "max_retries": 5
  },
  "orchestration": {
    "max_iterations": 3,
    "score_improvement_threshold": 0.03
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "cost_monitoring": true,
    "max_concurrent_llm_calls": 3,
    "llm_timeout": 300,
    "cache_llm_responses": true
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 3600
  }
}
```

### Production Environment
```json
{
  "bedrock": {
    "region": "us-east-1",
    "default_model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "timeout": 90,
    "max_retries": 5
  },
  "orchestration": {
    "max_iterations": 5,
    "score_improvement_threshold": 0.01,
    "convergence_confidence_threshold": 0.9
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "cost_monitoring": true,
    "rate_limiting": true,
    "max_concurrent_llm_calls": 10,
    "llm_timeout": 600,
    "cache_llm_responses": true,
    "daily_budget_limit": 100.00,
    "session_cost_limit": 5.00
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 7200,
    "connection_pool_size": 15
  },
  "monitoring": {
    "metrics_enabled": true,
    "cost_tracking": true,
    "performance_monitoring": true
  }
}
```

### High-Volume Production
```json
{
  "bedrock": {
    "region": "us-east-1",
    "default_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "timeout": 120,
    "max_retries": 3
  },
  "orchestration": {
    "max_iterations": 3,
    "score_improvement_threshold": 0.05
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": false,
    "cost_monitoring": true,
    "rate_limiting": true,
    "max_concurrent_llm_calls": 20,
    "llm_timeout": 300,
    "cache_llm_responses": true,
    "aggressive_caching": true
  },
  "performance": {
    "cache_enabled": true,
    "cache_ttl": 14400,
    "max_cache_size": 10000,
    "connection_pool_size": 25
  }
}
```

## Domain-Specific Configurations

### Healthcare Domain
```json
{
  "bedrock": {
    "default_model": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  "orchestration": {
    "max_iterations": 6,
    "score_improvement_threshold": 0.01
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "domain_specific_validation": true
  },
  "evaluation": {
    "default_criteria": [
      "medical_accuracy",
      "patient_safety",
      "regulatory_compliance",
      "clarity"
    ]
  },
  "agents": {
    "validator": {
      "model": "anthropic.claude-3-sonnet-20240229-v1:0",
      "temperature": 0.1,
      "specialized_validation": "healthcare"
    }
  }
}
```

### Financial Services
```json
{
  "bedrock": {
    "default_model": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  "orchestration": {
    "max_iterations": 7,
    "score_improvement_threshold": 0.01
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "compliance_checking": true
  },
  "evaluation": {
    "default_criteria": [
      "regulatory_compliance",
      "risk_disclosure",
      "accuracy",
      "clarity"
    ]
  },
  "security": {
    "encrypt_sensitive_data": true,
    "audit_logging": true
  }
}
```

### Creative Content
```json
{
  "bedrock": {
    "default_model": "anthropic.claude-3-sonnet-20240229-v1:0"
  },
  "orchestration": {
    "max_iterations": 8,
    "score_improvement_threshold": 0.02
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": false,
    "creativity_boost": true
  },
  "agents": {
    "refiner": {
      "temperature": 0.7,
      "creative_mode": true
    }
  },
  "evaluation": {
    "default_criteria": [
      "creativity",
      "originality",
      "engagement",
      "coherence"
    ]
  }
}
```

## Cost-Optimized Configurations

### Budget-Conscious Setup
```json
{
  "bedrock": {
    "default_model": "anthropic.claude-3-haiku-20240307-v1:0"
  },
  "orchestration": {
    "max_iterations": 2,
    "score_improvement_threshold": 0.1
  },
  "optimization": {
    "llm_only_mode": true,
    "cost_monitoring": true,
    "daily_budget_limit": 10.00,
    "session_cost_limit": 0.50,
    "cache_llm_responses": true,
    "aggressive_caching": true
  },
  "performance": {
    "cache_ttl": 86400,
    "similarity_threshold": 0.8
  }
}
```

### Premium Quality Setup
```json
{
  "bedrock": {
    "default_model": "anthropic.claude-3-opus-20240229-v1:0"
  },
  "orchestration": {
    "max_iterations": 10,
    "score_improvement_threshold": 0.005,
    "convergence_confidence_threshold": 0.95
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": true,
    "quality_over_cost": true,
    "max_concurrent_llm_calls": 5
  },
  "evaluation": {
    "comprehensive_scoring": true,
    "multi_perspective_analysis": true
  }
}
```

## Environment-Specific Examples

### Docker Configuration
```yaml
# docker-compose.yml environment variables
environment:
  - OPTIMIZER_LLM_ONLY_MODE=true
  - OPTIMIZER_FALLBACK_ENABLED=true
  - OPTIMIZER_COST_MONITORING=true
  - OPTIMIZER_MAX_CONCURRENT_LLM_CALLS=5
  - OPTIMIZER_CACHE_ENABLED=true
  - OPTIMIZER_CACHE_TTL=7200
  - AWS_REGION=us-east-1
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bedrock-optimizer-config
data:
  config.json: |
    {
      "optimization": {
        "llm_only_mode": true,
        "fallback_to_heuristic": true,
        "cost_monitoring": true,
        "rate_limiting": true,
        "max_concurrent_llm_calls": 15,
        "cache_llm_responses": true
      },
      "performance": {
        "cache_enabled": true,
        "cache_ttl": 7200
      }
    }
```

### AWS Lambda Configuration
```json
{
  "bedrock": {
    "region": "us-east-1",
    "default_model": "anthropic.claude-3-haiku-20240307-v1:0",
    "timeout": 60
  },
  "orchestration": {
    "max_iterations": 2,
    "timeout": 240
  },
  "optimization": {
    "llm_only_mode": true,
    "fallback_to_heuristic": false,
    "max_concurrent_llm_calls": 1,
    "llm_timeout": 180,
    "cache_llm_responses": true
  },
  "performance": {
    "memory_optimized": true,
    "cold_start_optimization": true
  }
}
```

## CLI Configuration Commands

### Enable LLM-Only Mode
```bash
# Basic enablement
bedrock-optimizer config --set optimization.llm_only_mode=true

# With fallback and monitoring
bedrock-optimizer config --set optimization.llm_only_mode=true
bedrock-optimizer config --set optimization.fallback_to_heuristic=true
bedrock-optimizer config --set optimization.cost_monitoring=true

# Performance optimization
bedrock-optimizer config --set optimization.cache_llm_responses=true
bedrock-optimizer config --set performance.cache_enabled=true
bedrock-optimizer config --set performance.cache_ttl=7200
```

### Cost Management
```bash
# Set budget limits
bedrock-optimizer config --set optimization.daily_budget_limit=50.00
bedrock-optimizer config --set optimization.session_cost_limit=2.00

# Enable rate limiting
bedrock-optimizer config --set optimization.rate_limiting=true
bedrock-optimizer config --set optimization.max_concurrent_llm_calls=5
```

### Model Selection
```bash
# Cost-effective model
bedrock-optimizer config --set bedrock.default_model=anthropic.claude-3-haiku-20240307-v1:0

# High-quality model
bedrock-optimizer config --set bedrock.default_model=anthropic.claude-3-sonnet-20240229-v1:0

# Premium model
bedrock-optimizer config --set bedrock.default_model=anthropic.claude-3-opus-20240229-v1:0
```

## Migration Scenarios

### Gradual Migration from Hybrid to LLM-Only
```bash
# Phase 1: Test with specific prompts
bedrock-optimizer optimize "test prompt" --llm-only --dry-run

# Phase 2: Enable for specific domains
bedrock-optimizer config --set optimization.llm_only_domains='["healthcare","finance"]'

# Phase 3: Enable globally with fallback
bedrock-optimizer config --set optimization.llm_only_mode=true
bedrock-optimizer config --set optimization.fallback_to_heuristic=true

# Phase 4: Monitor and optimize
bedrock-optimizer history --cost-report --mode-comparison
```

### A/B Testing Configuration
```json
{
  "optimization": {
    "ab_testing_enabled": true,
    "test_variants": {
      "control": {
        "llm_only_mode": false
      },
      "treatment": {
        "llm_only_mode": true,
        "fallback_to_heuristic": true
      }
    },
    "traffic_split": 0.5
  }
}
```

This comprehensive set of configuration examples provides templates for various use cases, environments, and requirements when implementing LLM-Only Mode.