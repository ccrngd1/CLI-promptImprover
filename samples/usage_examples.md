# Usage Examples

## Basic Optimization (Hybrid Mode)
```bash
bedrock-optimizer optimize "Explain machine learning" --context "Educational content for beginners"
```

## LLM-Only Mode Examples

### Enable LLM-Only Mode for Single Session
```bash
bedrock-optimizer optimize "Write a comprehensive analysis of quantum computing applications" \
  --llm-only \
  --context "Technical whitepaper for enterprise audience" \
  --max-iterations 5
```

### Configure LLM-Only Mode Globally
```bash
# Enable LLM-only mode
bedrock-optimizer config --set optimization.llm_only_mode=true

# Optimize with global LLM-only setting
bedrock-optimizer optimize "Create a detailed project proposal for AI implementation"
```

### Cost-Conscious LLM-Only Optimization
```bash
bedrock-optimizer optimize "Explain blockchain technology to investors" \
  --llm-only \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --max-iterations 3 \
  --cost-aware
```

### Complex Reasoning with LLM-Only Mode
```bash
bedrock-optimizer optimize "Develop a multi-step problem-solving framework for software architecture decisions" \
  --llm-only \
  --context "Enterprise software development" \
  --criteria "logical_flow,completeness,practical_applicability" \
  --interactive
```

## Hybrid vs LLM-Only Comparison

### Hybrid Mode (Default)
```bash
# Uses both heuristic and LLM agents
bedrock-optimizer optimize "Write user documentation for API endpoints" \
  --context "Developer documentation" \
  --max-iterations 8
```

### LLM-Only Mode
```bash
# Uses only LLM agents for all optimization steps
bedrock-optimizer optimize "Write user documentation for API endpoints" \
  --llm-only \
  --context "Developer documentation" \
  --max-iterations 5
```

## Interactive Mode
```bash
bedrock-optimizer optimize "Write a product description" --interactive --max-iterations 5
```

## Continue Session
```bash
bedrock-optimizer continue abc123 --rating 4 --feedback "Make it more concise"
```

## View History with Cost Analysis
```bash
# Basic history view
bedrock-optimizer history --session-id abc123 --export results.json

# View with cost information (LLM-only mode sessions)
bedrock-optimizer history --session-id abc123 --show-costs --detailed
```

## Configuration Management

### Basic Configuration
```bash
bedrock-optimizer config --show
bedrock-optimizer config --set bedrock.region=us-west-2
```

### LLM-Only Mode Configuration
```bash
# Enable LLM-only mode with fallback
bedrock-optimizer config --set optimization.llm_only_mode=true
bedrock-optimizer config --set optimization.fallback_to_heuristic=true

# Configure cost monitoring
bedrock-optimizer config --set optimization.cost_monitoring=true
bedrock-optimizer config --set optimization.rate_limiting=true

# Set performance parameters
bedrock-optimizer config --set optimization.max_concurrent_llm_calls=5
bedrock-optimizer config --set optimization.llm_timeout=600
```

## Model Testing
```bash
bedrock-optimizer models --test "Hello, how are you?"
```

## Advanced LLM-Only Scenarios

### Domain-Specific Optimization
```bash
# Healthcare domain with LLM-only mode
bedrock-optimizer optimize "Create patient education material about diabetes management" \
  --llm-only \
  --domain healthcare \
  --context "Patient education, 8th grade reading level" \
  --criteria "medical_accuracy,clarity,patient_safety"

# Financial services with compliance focus
bedrock-optimizer optimize "Draft investment risk disclosure statement" \
  --llm-only \
  --domain finance \
  --context "Regulatory compliance, SEC requirements" \
  --criteria "regulatory_compliance,clarity,completeness"
```

### Batch Processing with LLM-Only Mode
```bash
# Process multiple prompts with LLM-only mode
cat prompts.txt | while read prompt; do
  bedrock-optimizer optimize "$prompt" \
    --llm-only \
    --no-interactive \
    --max-iterations 3 \
    --model anthropic.claude-3-haiku-20240307-v1:0 \
    --save-session
done
```

### Performance Monitoring
```bash
# Monitor LLM-only mode performance
bedrock-optimizer history --list --since 2024-01-01 --mode llm-only
bedrock-optimizer history --cost-report --export monthly-costs.csv
```
