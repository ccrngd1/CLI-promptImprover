# LLM Logging Configuration Guide

This document describes the configuration options and security features for LLM agent logging in the Bedrock Prompt Optimizer.

## Overview

The LLM logging system provides comprehensive logging capabilities for LLM agent interactions, including:

- Configurable log levels and output options
- Response truncation and sensitive data filtering
- Separate log file configuration for LLM interactions
- Security options for prompt and response logging

## Configuration Structure

Add the following `logging` section to your configuration file:

```json
{
  "logging": {
    "level": "INFO",
    "log_dir": "./logs",
    "enable_structured_logging": true,
    "enable_performance_logging": true,
    "llm_logging": {
      "level": "DEBUG",
      "log_raw_responses": true,
      "log_prompts": false,
      "max_response_log_length": 5000,
      "max_prompt_log_length": 2000,
      "max_reasoning_log_length": 2500,
      "separate_log_files": true,
      "enable_security_filtering": true,
      "sensitive_data_patterns": [
        "password",
        "api_key",
        "secret",
        "token",
        "credential"
      ],
      "truncation_indicator": "... [truncated for security/length]",
      "log_file_prefix": "llm_",
      "log_file_max_size": "20MB",
      "log_file_backup_count": 10
    }
  }
}
```

## Configuration Options

### General Logging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | "INFO" | Main logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `log_dir` | string | null | Directory for log files (null for console only) |
| `enable_structured_logging` | boolean | true | Whether to use structured JSON logging |
| `enable_performance_logging` | boolean | true | Whether to enable performance logging |

### LLM-Specific Logging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `level` | string | "INFO" | Separate logging level for LLM interactions |
| `log_raw_responses` | boolean | true | Whether to include full LLM responses in logs |
| `log_prompts` | boolean | false | Whether to log outgoing prompts (security consideration) |
| `max_response_log_length` | integer | 5000 | Maximum length for logged responses (truncation) |
| `max_prompt_log_length` | integer | 2000 | Maximum length for logged prompts (truncation) |
| `max_reasoning_log_length` | integer | 2500 | Maximum length for logged reasoning text (truncation) |
| `separate_log_files` | boolean | true | Whether to create separate log files for each agent type |
| `enable_security_filtering` | boolean | true | Whether to filter sensitive data from logs |
| `sensitive_data_patterns` | array | ["password", "api_key", "secret", "token", "credential"] | Patterns to filter from logs |
| `truncation_indicator` | string | "... [truncated for security/length]" | Text to append when content is truncated |
| `log_file_prefix` | string | "llm_" | Prefix for LLM log files |
| `log_file_max_size` | string | "20MB" | Maximum size for log files before rotation |
| `log_file_backup_count` | integer | 10 | Number of backup log files to keep |

## Log Levels

### DEBUG Level
- Logs all LLM interactions including raw prompts (if enabled)
- Includes detailed parsing and extraction information
- Shows confidence calculation details
- Logs agent reasoning and analysis

### INFO Level
- Logs processed outputs and key metrics
- Shows successful LLM responses and parsing results
- Includes confidence scores and extraction success indicators
- Logs agent-specific metadata

### WARNING Level
- Logs LLM errors and fallback usage
- Shows parsing failures and extraction issues
- Includes performance warnings

### ERROR Level
- Only logs critical LLM service failures
- Shows system-level errors that affect functionality

## Security Features

### Sensitive Data Filtering

When `enable_security_filtering` is true, the system automatically filters sensitive data patterns from logs:

- **Pattern Matching**: Searches for configured patterns in text content
- **Key-Value Filtering**: Filters dictionary keys that match sensitive patterns
- **Replacement**: Replaces sensitive values with `[FILTERED]`

Example patterns:
- `password: secret123` → `password: [FILTERED]`
- `api_key=abc123` → `api_key=[FILTERED]`

### Content Truncation

Long content is automatically truncated to prevent log files from becoming too large:

- **Response Truncation**: Limits LLM response length in logs
- **Prompt Truncation**: Limits prompt length when logging is enabled
- **Reasoning Truncation**: Limits reasoning text length
- **Truncation Indicator**: Appends configurable text to show truncation occurred

### Prompt Logging Control

The `log_prompts` option provides fine-grained control over prompt logging:

- **Disabled (default)**: Prompts are not logged for security
- **Enabled**: Prompts are logged with security filtering and truncation applied

## Log File Organization

### Separate Log Files

When `separate_log_files` is enabled, the system creates:

- `llm_interactions.log`: Main LLM interaction log
- `llm_analyzer.log`: LLM analyzer agent specific logs
- `llm_refiner.log`: LLM refiner agent specific logs
- `llm_validator.log`: LLM validator agent specific logs

### Log Rotation

Log files are automatically rotated based on:

- **Size Limit**: Files are rotated when they exceed `log_file_max_size`
- **Backup Count**: Old files are kept according to `log_file_backup_count`
- **Naming**: Rotated files are numbered (e.g., `llm_interactions.log.1`)

## Usage Examples

### Development Configuration

For development and debugging:

```json
{
  "logging": {
    "llm_logging": {
      "level": "DEBUG",
      "log_raw_responses": true,
      "log_prompts": true,
      "enable_security_filtering": true,
      "max_response_log_length": 10000
    }
  }
}
```

### Production Configuration

For production environments:

```json
{
  "logging": {
    "llm_logging": {
      "level": "INFO",
      "log_raw_responses": false,
      "log_prompts": false,
      "enable_security_filtering": true,
      "max_response_log_length": 1000
    }
  }
}
```

### High-Security Configuration

For environments with strict security requirements:

```json
{
  "logging": {
    "llm_logging": {
      "level": "WARNING",
      "log_raw_responses": false,
      "log_prompts": false,
      "enable_security_filtering": true,
      "sensitive_data_patterns": [
        "password", "api_key", "secret", "token", "credential",
        "auth", "key", "pass", "pwd", "access", "private"
      ]
    }
  }
}
```

## Log Format

LLM interaction logs use structured JSON format:

```json
{
  "timestamp": "2025-01-09T10:30:00Z",
  "level": "INFO",
  "logger": "llm_agents.analyzer",
  "message": "LLM response received by LLMAnalyzerAgent",
  "session_id": "session_123",
  "iteration": 2,
  "agent_name": "LLMAnalyzerAgent",
  "interaction_type": "llm_response",
  "model_used": "claude-3-sonnet",
  "tokens_used": 1250,
  "processing_time": 2.3,
  "response_length": 1800,
  "confidence_score": 0.85,
  "extraction_success": true,
  "raw_response": "...",
  "parsed_components": {...},
  "reasoning": "..."
}
```

## Monitoring and Analysis

### Log Analysis

Use the structured logs for:

- **Performance Monitoring**: Track processing times and token usage
- **Quality Assessment**: Monitor confidence scores and extraction success rates
- **Error Analysis**: Identify patterns in LLM failures and parsing issues
- **Usage Patterns**: Analyze agent selection and reasoning patterns

### Metrics Extraction

Key metrics available in logs:

- **Response Times**: `processing_time` field
- **Token Usage**: `tokens_used` field
- **Success Rates**: `extraction_success` and `parsing_success` fields
- **Confidence Scores**: `confidence_score` field
- **Error Rates**: Count of ERROR and WARNING level logs

## Troubleshooting

### Common Issues

1. **Large Log Files**: Reduce `max_response_log_length` or disable `log_raw_responses`
2. **Missing Logs**: Check `log_dir` permissions and `level` configuration
3. **Sensitive Data Exposure**: Enable `enable_security_filtering` and configure `sensitive_data_patterns`
4. **Performance Impact**: Use higher log levels (WARNING/ERROR) in production

### Configuration Validation

The system validates configuration on startup and logs warnings for:

- Invalid log levels
- Inaccessible log directories
- Invalid file size formats
- Missing required configuration sections

## Best Practices

1. **Security First**: Always enable security filtering in production
2. **Appropriate Log Levels**: Use DEBUG for development, INFO for staging, WARNING+ for production
3. **Log Rotation**: Configure appropriate file sizes and backup counts
4. **Monitoring**: Set up log monitoring and alerting for ERROR level logs
5. **Regular Cleanup**: Implement log cleanup policies for long-running systems