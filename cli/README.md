# Bedrock Prompt Optimizer CLI

A powerful command-line interface for optimizing prompts using Amazon Bedrock with intelligent multi-agent collaboration and LLM orchestration.

## Features

- 🤖 **Multi-Agent Optimization**: Collaborative prompt improvement using specialized AI agents
- 🎭 **LLM Orchestration**: Intelligent coordination and conflict resolution between agents
- 📊 **Real-time Progress**: Visual progress indicators and formatted output
- 🔧 **Flexible Configuration**: Comprehensive configuration management for AWS, models, and orchestration
- 📈 **Performance Tracking**: Detailed evaluation metrics and convergence analysis
- 💾 **Session Management**: Persistent history and session state management
- 🎨 **Rich Output**: Beautiful, colored terminal output with structured information display

## Installation

### Prerequisites

- Python 3.8 or higher
- AWS credentials configured
- Access to Amazon Bedrock

### Quick Setup

1. **Run the setup script:**
   ```bash
   python cli/setup.py
   ```

2. **Add to PATH (optional):**
   ```bash
   export PATH=$PATH:$(pwd)/bin
   ```

3. **Verify installation:**
   ```bash
   bedrock-optimizer config --show
   ```

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements-cli.txt
   ```

2. **Configure AWS credentials:**
   ```bash
   aws configure
   # OR set environment variables:
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   export AWS_DEFAULT_REGION=us-east-1
   ```

3. **Initialize configuration:**
   ```bash
   python -m cli.main config --init
   ```

## Usage

### Basic Commands

#### Start Optimization Session
```bash
# Basic optimization
bedrock-optimizer optimize "Explain quantum computing"

# With context and domain
bedrock-optimizer optimize "Write a product description" \
  --context "E-commerce product page" \
  --domain "retail" \
  --audience "general consumers"

# Interactive mode with feedback collection
bedrock-optimizer optimize "Create a tutorial" \
  --interactive \
  --max-iterations 5
```

#### Continue Existing Session
```bash
# Continue with feedback
bedrock-optimizer continue abc123 \
  --rating 4 \
  --feedback "Make it more concise and add examples"

# Continue without feedback
bedrock-optimizer continue abc123
```

#### View History and Status
```bash
# List all sessions
bedrock-optimizer history --list

# View specific session history
bedrock-optimizer history --session-id abc123

# Export session data
bedrock-optimizer history --session-id abc123 \
  --export results.json \
  --format json

# Show session status
bedrock-optimizer status --session-id abc123

# Show all active sessions
bedrock-optimizer status --all
```

#### Configuration Management
```bash
# Show current configuration
bedrock-optimizer config --show

# Set configuration values
bedrock-optimizer config --set bedrock.region=us-west-2
bedrock-optimizer config --set orchestration.max_iterations=15

# Get specific configuration value
bedrock-optimizer config --get bedrock.default_model

# Initialize default configuration
bedrock-optimizer config --init
```

#### Model Management
```bash
# List available models
bedrock-optimizer models

# Test a model
bedrock-optimizer models --test "Hello, how are you today?"
```

### Advanced Usage

#### Custom Model Configuration
```bash
bedrock-optimizer optimize "Technical documentation" \
  --model anthropic.claude-3-opus-20240229-v1:0 \
  --context "API documentation for developers"
```

#### Batch Processing with Configuration
```bash
# Set up for batch processing
bedrock-optimizer config --set cli.default_interactive=false
bedrock-optimizer config --set orchestration.auto_finalize_on_convergence=true

# Process multiple prompts
for prompt in "Prompt 1" "Prompt 2" "Prompt 3"; do
  bedrock-optimizer optimize "$prompt" --max-iterations 3
done
```

## Configuration

### Configuration File Structure

The CLI uses YAML configuration files located at:
- `~/.bedrock-optimizer/config.yaml` (user-level)
- `./config.yaml` (project-level)

### Key Configuration Sections

#### AWS Bedrock Settings
```yaml
bedrock:
  region: us-east-1
  profile: null  # Use default AWS profile
  default_model: anthropic.claude-3-sonnet-20240229-v1:0
  timeout: 30
  max_retries: 3
```

#### Orchestration Settings
```yaml
orchestration:
  orchestrator_model: anthropic.claude-3-sonnet-20240229-v1:0
  orchestrator_temperature: 0.3
  min_iterations: 3
  max_iterations: 10
  score_improvement_threshold: 0.02
  convergence_confidence_threshold: 0.8
```

#### Agent Configuration
```yaml
agents:
  analyzer:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.2
    max_tokens: 1500
  refiner:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.4
    max_tokens: 2000
  validator:
    enabled: true
    model: anthropic.claude-3-sonnet-20240229-v1:0
    temperature: 0.1
    max_tokens: 1000
```

### Environment Variables

Override configuration with environment variables:

```bash
export AWS_REGION=us-west-2
export BEDROCK_DEFAULT_MODEL=anthropic.claude-3-opus-20240229-v1:0
export OPTIMIZER_MAX_ITERATIONS=15
export OPTIMIZER_STORAGE_PATH=./my_history
```

## Output Formats

### Orchestration Visualization

The CLI provides rich visualization of the orchestration process:

```
🎭 Orchestration Results
├── 🤖 Agent Results
│   ├── Analyzer
│   │   ├── ✅ Success: True
│   │   ├── 🎯 Confidence: 0.85
│   │   └── 💡 Suggestions
│   │       ├── • Improve clarity in technical terms
│   │       ├── • Add concrete examples
│   │       └── • Structure with clear headings
│   ├── Refiner
│   │   ├── ✅ Success: True
│   │   ├── 🎯 Confidence: 0.92
│   │   └── 💡 Suggestions
│   │       └── • Enhanced version with examples
│   └── Validator
│       ├── ✅ Success: True
│       └── 🎯 Confidence: 0.88
├── 🎯 Orchestration Decisions
│   ├── • Prioritized clarity improvements
│   ├── • Integrated concrete examples
│   └── • Applied structured formatting
├── ⚖️ Conflict Resolutions
│   └── • Resolved length vs. detail trade-off
└── 📊 Performance
    ├── ⏱️ Processing Time: 12.34s
    └── 🎯 Confidence: 0.89
```

### Session Summary

```
┌─────────────────────────────────────────┐
│              Session Summary            │
├─────────────────┬───────────────────────┤
│ Session ID      │ abc123...             │
│ Status          │ Active                │
│ Iterations      │ 3                     │
│ Created         │ 2024-01-15 10:30:00   │
│ Last Updated    │ 2024-01-15 10:45:00   │
│ Convergence     │ High user satisfaction │
└─────────────────┴───────────────────────┘
```

## Troubleshooting

### Common Issues

#### AWS Credentials Not Found
```bash
# Check AWS configuration
aws sts get-caller-identity

# Configure credentials
aws configure

# Or use environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

#### Bedrock Access Denied
- Ensure your AWS account has access to Bedrock
- Check that you're using a supported region
- Verify IAM permissions for Bedrock operations

#### Model Not Available
```bash
# List available models
bedrock-optimizer models

# Check region-specific model availability
bedrock-optimizer config --set bedrock.region=us-west-2
```

#### Configuration Issues
```bash
# Validate configuration
bedrock-optimizer config --show

# Reset to defaults
rm ~/.bedrock-optimizer/config.yaml
bedrock-optimizer config --init
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
bedrock-optimizer optimize "test prompt" --verbose
```

### Log Files

Logs are stored in:
- `~/.bedrock-optimizer/logs/` (user-level)
- `./logs/` (project-level)

## Examples

### Educational Content Optimization
```bash
bedrock-optimizer optimize \
  "Explain photosynthesis to a 10-year-old" \
  --context "Elementary science education" \
  --domain "biology" \
  --audience "children" \
  --interactive
```

### Technical Documentation
```bash
bedrock-optimizer optimize \
  "Document the REST API endpoints" \
  --context "Developer documentation" \
  --domain "software" \
  --audience "developers" \
  --max-iterations 8
```

### Marketing Copy
```bash
bedrock-optimizer optimize \
  "Write compelling product copy" \
  --context "E-commerce product page" \
  --domain "marketing" \
  --audience "consumers" \
  --model anthropic.claude-3-opus-20240229-v1:0
```

## Integration

### CI/CD Pipeline Integration
```yaml
# .github/workflows/prompt-optimization.yml
name: Optimize Prompts
on: [push]
jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install CLI
        run: |
          pip install -r requirements-cli.txt
          python cli/setup.py
      - name: Optimize Prompts
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          bedrock-optimizer optimize "$(cat prompts/main.txt)" \
            --export optimized_prompts.json
```

### Programmatic Usage
```python
from cli.main import PromptOptimizerCLI

cli = PromptOptimizerCLI()
cli.run(['optimize', 'Your prompt here', '--max-iterations', '5'])
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the configuration documentation
- Open an issue on GitHub with detailed error information