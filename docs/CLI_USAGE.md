# Bedrock Prompt Optimizer CLI Usage Guide

This guide provides comprehensive documentation for using the Bedrock Prompt Optimizer command-line interface.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Commands Reference](#commands-reference)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- AWS credentials configured (AWS CLI, environment variables, or IAM roles)
- Access to Amazon Bedrock service

### Install from Source

```bash
# Clone the repository
git clone https://github.com/example/bedrock-prompt-optimizer.git
cd bedrock-prompt-optimizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run setup script
python cli/setup.py
```

### Install from PyPI (when available)

```bash
pip install bedrock-prompt-optimizer
```

### Verify Installation

```bash
bedrock-optimizer --version
bedrock-optimizer config --show
```

## Quick Start

### 1. Configure AWS Credentials

Ensure your AWS credentials are configured:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=us-east-1

# Option 3: IAM roles (when running on EC2)
# No additional configuration needed
```

### 2. Basic Optimization

```bash
# Start an interactive optimization session
bedrock-optimizer optimize "Explain machine learning to a beginner" --interactive

# Non-interactive optimization with specific parameters
bedrock-optimizer optimize "Write a product description for a smartphone" \
  --context "E-commerce product page" \
  --max-iterations 5 \
  --model anthropic.claude-3-sonnet-20240229-v1:0
```

### 3. View Results

```bash
# List recent sessions
bedrock-optimizer history --list

# View specific session details
bedrock-optimizer history --session-id abc123 --detailed

# Export results
bedrock-optimizer history --session-id abc123 --export results.json
```

## Configuration

### Configuration File

The CLI uses a YAML configuration file located at:
- `~/.bedrock-optimizer/config.yaml` (user-level)
- `./config.yaml` (project-level)

### Create Default Configuration

```bash
# Create default configuration
bedrock-optimizer config --create-default

# Create sample configuration with comments
bedrock-optimizer config --create-sample
```

### View Current Configuration

```bash
# Show current configuration
bedrock-optimizer config --show

# Show configuration summary
bedrock-optimizer config --summary

# Validate configuration
bedrock-optimizer config --validate
```

### Modify Configuration

```bash
# Set individual values
bedrock-optimizer config --set bedrock.region=us-west-2
bedrock-optimizer config --set orchestration.max_iterations=15
bedrock-optimizer config --set agents.analyzer.temperature=0.3

# Import configuration from file
bedrock-optimizer config --import config.yaml

# Export current configuration
bedrock-optimizer config --export my-config.yaml

# Reset to defaults
bedrock-optimizer config --reset
```

### Environment Variables

Override configuration with environment variables:

```bash
export AWS_REGION=us-west-2
export BEDROCK_DEFAULT_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
export OPTIMIZER_MAX_ITERATIONS=10
export OPTIMIZER_STORAGE_PATH=./my_prompt_history
```

## Commands Reference

### `optimize` - Start Optimization Session

Optimize a prompt using the multi-agent system.

```bash
bedrock-optimizer optimize [OPTIONS] PROMPT
```

**Options:**
- `--context TEXT`: Additional context for the prompt
- `--interactive / --no-interactive`: Enable interactive mode (default: True)
- `--max-iterations INTEGER`: Maximum optimization iterations (default: 10)
- `--model TEXT`: Bedrock model to use
- `--temperature FLOAT`: Model temperature (0.0-1.0)
- `--criteria TEXT`: Evaluation criteria (comma-separated)
- `--domain TEXT`: Domain-specific context (e.g., healthcare, finance)
- `--complexity [low|medium|high]`: Task complexity level
- `--output-format TEXT`: Desired output format
- `--save-session / --no-save-session`: Save session to history (default: True)
- `--session-name TEXT`: Custom name for the session
- `--verbose / --quiet`: Verbose output mode

**Examples:**
```bash
# Basic optimization
bedrock-optimizer optimize "Explain quantum computing"

# With context and domain
bedrock-optimizer optimize "Write a patient care protocol" \
  --context "Emergency department procedures" \
  --domain healthcare \
  --complexity high

# Specific model and parameters
bedrock-optimizer optimize "Generate marketing copy" \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --temperature 0.7 \
  --max-iterations 8 \
  --criteria "creativity,clarity,persuasiveness"
```

### `continue` - Continue Existing Session

Continue an existing optimization session with feedback.

```bash
bedrock-optimizer continue [OPTIONS] SESSION_ID
```

**Options:**
- `--feedback TEXT`: Feedback on the current prompt
- `--rating INTEGER`: Rating from 1-5
- `--max-iterations INTEGER`: Additional iterations to run
- `--interactive / --no-interactive`: Enable interactive mode

**Examples:**
```bash
# Continue with feedback
bedrock-optimizer continue abc123 \
  --feedback "Make it more concise and add examples" \
  --rating 3

# Continue interactively
bedrock-optimizer continue abc123 --interactive
```

### `history` - Manage Session History

View and manage optimization session history.

```bash
bedrock-optimizer history [OPTIONS]
```

**Options:**
- `--list`: List all sessions
- `--session-id TEXT`: Show specific session
- `--detailed`: Show detailed information
- `--export PATH`: Export session data
- `--format [json|yaml|csv]`: Export format
- `--limit INTEGER`: Limit number of sessions shown
- `--since DATE`: Show sessions since date (YYYY-MM-DD)
- `--search TEXT`: Search sessions by prompt content
- `--delete SESSION_ID`: Delete a session
- `--cleanup`: Clean up old sessions

**Examples:**
```bash
# List recent sessions
bedrock-optimizer history --list --limit 10

# View specific session
bedrock-optimizer history --session-id abc123 --detailed

# Export session data
bedrock-optimizer history --session-id abc123 --export results.json

# Search sessions
bedrock-optimizer history --search "machine learning" --since 2024-01-01

# Cleanup old sessions
bedrock-optimizer history --cleanup
```

### `config` - Configuration Management

Manage application configuration.

```bash
bedrock-optimizer config [OPTIONS]
```

**Options:**
- `--show`: Display current configuration
- `--summary`: Show configuration summary
- `--validate`: Validate configuration
- `--create-default`: Create default configuration file
- `--create-sample`: Create sample configuration with comments
- `--set KEY=VALUE`: Set configuration value
- `--get KEY`: Get configuration value
- `--import PATH`: Import configuration from file
- `--export PATH`: Export configuration to file
- `--reset`: Reset to default configuration
- `--backup [NAME]`: Create configuration backup
- `--restore PATH`: Restore from backup

**Examples:**
```bash
# View configuration
bedrock-optimizer config --show
bedrock-optimizer config --summary

# Modify configuration
bedrock-optimizer config --set bedrock.region=us-west-2
bedrock-optimizer config --set orchestration.max_iterations=15

# Backup and restore
bedrock-optimizer config --backup my-backup
bedrock-optimizer config --restore my-backup.yaml
```

### `models` - Model Management

Manage and test Bedrock models.

```bash
bedrock-optimizer models [OPTIONS]
```

**Options:**
- `--list`: List available models
- `--test TEXT`: Test model with prompt
- `--model TEXT`: Specific model to test
- `--compare`: Compare multiple models
- `--benchmark`: Run model benchmarks

**Examples:**
```bash
# List available models
bedrock-optimizer models --list

# Test a model
bedrock-optimizer models --test "Hello, how are you?" \
  --model anthropic.claude-3-sonnet-20240229-v1:0

# Compare models
bedrock-optimizer models --compare --test "Explain AI"
```

### `agents` - Agent Management

Manage and configure agents.

```bash
bedrock-optimizer agents [OPTIONS]
```

**Options:**
- `--list`: List available agents
- `--status`: Show agent status
- `--test AGENT_TYPE`: Test specific agent
- `--configure AGENT_TYPE`: Configure agent settings
- `--enable AGENT_TYPE`: Enable agent
- `--disable AGENT_TYPE`: Disable agent

**Examples:**
```bash
# List agents
bedrock-optimizer agents --list

# Test analyzer agent
bedrock-optimizer agents --test analyzer

# Configure agent
bedrock-optimizer agents --configure refiner
```

### `best-practices` - Best Practices Management

Manage prompt engineering best practices.

```bash
bedrock-optimizer best-practices [OPTIONS]
```

**Options:**
- `--list`: List all best practices
- `--category TEXT`: Filter by category
- `--add`: Add new best practice
- `--update ID`: Update existing practice
- `--delete ID`: Delete practice
- `--export PATH`: Export practices
- `--import PATH`: Import practices
- `--validate`: Validate practices repository

**Examples:**
```bash
# List best practices
bedrock-optimizer best-practices --list

# Filter by category
bedrock-optimizer best-practices --list --category clarity

# Export practices
bedrock-optimizer best-practices --export my-practices.json
```

## Usage Examples

### Example 1: Educational Content Optimization

```bash
# Optimize an educational prompt
bedrock-optimizer optimize \
  "Explain photosynthesis to a 10-year-old" \
  --domain education \
  --complexity medium \
  --context "Elementary science lesson" \
  --criteria "clarity,age_appropriateness,engagement" \
  --interactive
```

### Example 2: Business Communication

```bash
# Optimize business email template
bedrock-optimizer optimize \
  "Write an email declining a meeting request" \
  --domain business \
  --context "Professional communication" \
  --output-format "Email template with subject line" \
  --max-iterations 5
```

### Example 3: Technical Documentation

```bash
# Optimize technical documentation
bedrock-optimizer optimize \
  "Document the API endpoint for user authentication" \
  --domain technical \
  --complexity high \
  --context "REST API documentation" \
  --criteria "completeness,clarity,technical_accuracy"
```

### Example 4: Creative Writing

```bash
# Optimize creative writing prompt
bedrock-optimizer optimize \
  "Write a short story about time travel" \
  --domain creative \
  --context "Science fiction short story" \
  --criteria "creativity,narrative_structure,engagement" \
  --temperature 0.8
```

### Example 5: Batch Processing

```bash
# Process multiple prompts from file
cat prompts.txt | while read prompt; do
  bedrock-optimizer optimize "$prompt" \
    --no-interactive \
    --max-iterations 3 \
    --save-session
done
```

## Advanced Features

### LLM-Only Mode

The CLI supports LLM-only mode for exclusive LLM-based optimization:

```bash
# Enable LLM-only mode for a single optimization
bedrock-optimizer optimize "Your prompt" --llm-only

# Configure LLM-only mode globally
bedrock-optimizer config --set optimization.llm_only_mode=true

# Optimize with LLM-only mode and cost management
bedrock-optimizer optimize "Complex reasoning prompt" \
  --llm-only \
  --max-iterations 3 \
  --model anthropic.claude-3-haiku-20240307-v1:0
```

#### LLM-Only Mode Options

- `--llm-only`: Enable LLM-only mode for this session
- `--no-fallback`: Disable fallback to heuristic agents
- `--cost-aware`: Enable cost monitoring and warnings

#### Performance and Cost Considerations

**Monitor Usage:**
```bash
# View token usage statistics
bedrock-optimizer history --session-id abc123 --show-costs

# Export cost analysis
bedrock-optimizer history --cost-report --since 2024-01-01 --export costs.csv
```

**Cost Management:**
```bash
# Use cost-effective model for LLM-only mode
bedrock-optimizer optimize "Your prompt" \
  --llm-only \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --max-iterations 3

# Enable caching to reduce API calls
bedrock-optimizer config --set performance.cache_enabled=true
```

### Interactive Mode

Interactive mode provides step-by-step guidance:

```bash
bedrock-optimizer optimize "Your prompt here" --interactive
```

Features:
- Real-time feedback collection
- Agent decision visibility
- Manual iteration control
- Custom evaluation criteria

### Session Management

Advanced session management features:

```bash
# Resume interrupted session
bedrock-optimizer continue abc123

# Clone session with modifications
bedrock-optimizer clone abc123 --modify-context "New context"

# Compare sessions
bedrock-optimizer compare abc123 def456

# Merge session insights
bedrock-optimizer merge abc123 def456 --output ghi789
```

### Custom Evaluation Criteria

Define custom evaluation criteria:

```bash
bedrock-optimizer optimize "Your prompt" \
  --criteria "custom:brand_alignment,custom:regulatory_compliance" \
  --custom-criteria-config criteria.yaml
```

### Orchestration Visibility

View detailed orchestration decisions:

```bash
bedrock-optimizer optimize "Your prompt" \
  --verbose \
  --show-orchestration \
  --show-agent-reasoning
```

### Configuration Profiles

Use different configuration profiles:

```bash
# Use development profile
bedrock-optimizer --profile dev optimize "Test prompt"

# Use production profile
bedrock-optimizer --profile prod optimize "Production prompt"
```

## Troubleshooting

### Common Issues

#### 1. AWS Credentials Not Found

```bash
# Check AWS configuration
aws sts get-caller-identity

# Set credentials explicitly
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

#### 2. Bedrock Access Denied

```bash
# Check Bedrock permissions
aws bedrock list-foundation-models --region us-east-1

# Request Bedrock access in AWS Console
# Navigate to Bedrock service and request model access
```

#### 3. Configuration Validation Errors

```bash
# Validate configuration
bedrock-optimizer config --validate

# Reset to defaults if corrupted
bedrock-optimizer config --reset
```

#### 4. Session Not Found

```bash
# List available sessions
bedrock-optimizer history --list

# Check session storage path
bedrock-optimizer config --get storage.path
```

#### 5. Model Not Available

```bash
# List available models
bedrock-optimizer models --list

# Use default model
bedrock-optimizer config --set bedrock.default_model=anthropic.claude-3-sonnet-20240229-v1:0
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Enable debug logging
export OPTIMIZER_LOG_LEVEL=DEBUG
bedrock-optimizer optimize "Your prompt" --verbose

# Save debug logs
bedrock-optimizer optimize "Your prompt" --debug-log debug.log
```

### Performance Issues

#### Slow Optimization

```bash
# Reduce iterations
bedrock-optimizer optimize "Your prompt" --max-iterations 3

# Use faster model
bedrock-optimizer optimize "Your prompt" --model anthropic.claude-3-haiku-20240307-v1:0

# Disable verbose output
bedrock-optimizer optimize "Your prompt" --quiet
```

#### Memory Issues

```bash
# Limit history size
bedrock-optimizer config --set storage.max_history_size=100

# Clean up old sessions
bedrock-optimizer history --cleanup

# Use streaming mode
bedrock-optimizer optimize "Your prompt" --stream
```

### Getting Help

```bash
# General help
bedrock-optimizer --help

# Command-specific help
bedrock-optimizer optimize --help
bedrock-optimizer config --help

# Version information
bedrock-optimizer --version

# System information
bedrock-optimizer --system-info
```

### Support Resources

- **Documentation**: [https://bedrock-prompt-optimizer.readthedocs.io/](https://bedrock-prompt-optimizer.readthedocs.io/)
- **GitHub Issues**: [https://github.com/example/bedrock-prompt-optimizer/issues](https://github.com/example/bedrock-prompt-optimizer/issues)
- **Community Forum**: [https://community.example.com/bedrock-optimizer](https://community.example.com/bedrock-optimizer)
- **Email Support**: support@example.com

## Best Practices

### 1. Configuration Management

- Use version control for configuration files
- Create environment-specific configurations
- Regularly backup configurations
- Validate configurations after changes

### 2. Session Management

- Use descriptive session names
- Export important sessions
- Regular cleanup of old sessions
- Monitor storage usage

### 3. Prompt Optimization

- Start with clear, specific prompts
- Provide sufficient context
- Use appropriate complexity levels
- Iterate based on feedback

### 4. Performance Optimization

- Choose appropriate models for tasks
- Limit iterations for simple prompts
- Use batch processing for multiple prompts
- Monitor API usage and costs

### 5. Security

- Secure AWS credentials
- Use IAM roles when possible
- Avoid sensitive data in prompts
- Regular security updates