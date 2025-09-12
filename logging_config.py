"""
Logging configuration for the Bedrock Prompt Optimizer.

Provides centralized logging setup with different levels for different components,
structured logging for orchestration and agent interactions, and performance monitoring.
"""

import logging
import logging.handlers
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'iteration'):
            log_entry['iteration'] = record.iteration
        if hasattr(record, 'agent_name'):
            log_entry['agent_name'] = record.agent_name
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
        if hasattr(record, 'orchestration_decision'):
            log_entry['orchestration_decision'] = record.orchestration_decision
        
        return json.dumps(log_entry)


class LLMInteractionFormatter(logging.Formatter):
    """Specialized formatter for LLM interaction logs with enhanced metadata and security filtering."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM interaction formatter.
        
        Args:
            llm_config: LLM logging configuration dictionary
        """
        super().__init__()
        self.llm_config = llm_config or {}
        self.enable_security_filtering = self.llm_config.get('enable_security_filtering', True)
        self.sensitive_patterns = self.llm_config.get('sensitive_data_patterns', [])
        self.truncation_indicator = self.llm_config.get('truncation_indicator', '... [truncated]')
        self.max_response_length = self.llm_config.get('max_response_log_length', 5000)
        self.max_prompt_length = self.llm_config.get('max_prompt_log_length', 2000)
        self.max_reasoning_length = self.llm_config.get('max_reasoning_log_length', 2500)
        self.log_prompts = self.llm_config.get('log_prompts', False)
        self.log_raw_responses = self.llm_config.get('log_raw_responses', True)
    
    def format(self, record):
        """Format LLM interaction log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Standard fields
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'iteration'):
            log_entry['iteration'] = record.iteration
        if hasattr(record, 'agent_name'):
            log_entry['agent_name'] = record.agent_name
        
        # LLM-specific fields with security filtering and truncation
        if hasattr(record, 'interaction_type'):
            log_entry['interaction_type'] = record.interaction_type
        if hasattr(record, 'model_used'):
            log_entry['model_used'] = record.model_used
        if hasattr(record, 'tokens_used'):
            log_entry['tokens_used'] = record.tokens_used
        if hasattr(record, 'processing_time'):
            log_entry['processing_time'] = record.processing_time
        if hasattr(record, 'response_length'):
            log_entry['response_length'] = record.response_length
        if hasattr(record, 'confidence_score'):
            log_entry['confidence_score'] = record.confidence_score
        if hasattr(record, 'extraction_success'):
            log_entry['extraction_success'] = record.extraction_success
        
        # Handle raw response with security and length controls
        if hasattr(record, 'raw_response') and self.log_raw_responses:
            raw_response = record.raw_response
            if raw_response:
                # Apply security filtering
                if self.enable_security_filtering:
                    raw_response = self._filter_sensitive_content(raw_response)
                # Apply length truncation
                if len(raw_response) > self.max_response_length:
                    raw_response = raw_response[:self.max_response_length] + self.truncation_indicator
                    log_entry['response_truncated'] = True
                else:
                    log_entry['response_truncated'] = False
                log_entry['raw_response'] = raw_response
        
        # Handle parsed components with security filtering
        if hasattr(record, 'parsed_components'):
            parsed_components = record.parsed_components
            if self.enable_security_filtering and parsed_components:
                parsed_components = self._filter_sensitive_data_dict(parsed_components)
            log_entry['parsed_components'] = parsed_components
        
        # Handle reasoning with length control
        if hasattr(record, 'reasoning'):
            reasoning = record.reasoning
            if reasoning and len(reasoning) > self.max_reasoning_length:
                reasoning = reasoning[:self.max_reasoning_length] + self.truncation_indicator
                log_entry['reasoning_truncated'] = True
            else:
                log_entry['reasoning_truncated'] = False
            log_entry['reasoning'] = reasoning
        
        # Handle prompt text with security and configuration controls
        if hasattr(record, 'prompt_text') and self.log_prompts:
            prompt_text = record.prompt_text
            if prompt_text:
                # Apply security filtering
                if self.enable_security_filtering:
                    prompt_text = self._filter_sensitive_content(prompt_text)
                # Apply length truncation
                if len(prompt_text) > self.max_prompt_length:
                    prompt_text = prompt_text[:self.max_prompt_length] + self.truncation_indicator
                    log_entry['prompt_truncated'] = True
                else:
                    log_entry['prompt_truncated'] = False
                log_entry['prompt_text'] = prompt_text
        
        # Handle context data with security filtering
        if hasattr(record, 'context_data'):
            context_data = record.context_data
            if self.enable_security_filtering and context_data:
                context_data = self._filter_sensitive_data_dict(context_data)
            log_entry['context_data'] = context_data
        
        # Agent-specific metadata
        if hasattr(record, 'analysis_type'):
            log_entry['analysis_type'] = record.analysis_type
        if hasattr(record, 'insights_extracted'):
            log_entry['insights_extracted'] = record.insights_extracted
        if hasattr(record, 'issues_identified'):
            log_entry['issues_identified'] = record.issues_identified
        if hasattr(record, 'refinement_techniques'):
            log_entry['refinement_techniques'] = record.refinement_techniques
        if hasattr(record, 'improvements_made'):
            log_entry['improvements_made'] = record.improvements_made
        if hasattr(record, 'quality_score'):
            log_entry['quality_score'] = record.quality_score
        if hasattr(record, 'validation_criteria'):
            log_entry['validation_criteria'] = record.validation_criteria
        if hasattr(record, 'passes_validation'):
            log_entry['passes_validation'] = record.passes_validation
        if hasattr(record, 'critical_issues_count'):
            log_entry['critical_issues_count'] = record.critical_issues_count
        
        return json.dumps(log_entry)
    
    def _filter_sensitive_content(self, content: str) -> str:
        """
        Filter sensitive content from text using pattern matching.
        
        Args:
            content: Text content to filter
            
        Returns:
            Filtered content with sensitive data replaced
        """
        if not content or not self.sensitive_patterns:
            return content
        
        filtered_content = content
        for pattern in self.sensitive_patterns:
            # Simple case-insensitive pattern matching
            import re
            # Replace patterns like "password: value" or "api_key=value"
            pattern_regex = rf'({re.escape(pattern)}[\s]*[:=][\s]*)[^\s\n,}}\]]+' 
            filtered_content = re.sub(pattern_regex, r'\1[FILTERED]', filtered_content, flags=re.IGNORECASE)
        
        return filtered_content
    
    def _filter_sensitive_data_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter sensitive data from dictionary structures.
        
        Args:
            data: Dictionary to filter
            
        Returns:
            Filtered dictionary with sensitive values replaced
        """
        if not data or not isinstance(data, dict):
            return data
        
        filtered_data = {}
        for key, value in data.items():
            key_lower = key.lower()
            # Check if key matches sensitive patterns
            is_sensitive = any(pattern.lower() in key_lower for pattern in self.sensitive_patterns)
            
            if is_sensitive:
                filtered_data[key] = '[FILTERED]'
            elif isinstance(value, dict):
                filtered_data[key] = self._filter_sensitive_data_dict(value)
            elif isinstance(value, str) and self.enable_security_filtering:
                filtered_data[key] = self._filter_sensitive_content(value)
            else:
                filtered_data[key] = value
        
        return filtered_data


class ModeUsageTracker:
    """Tracker for mode usage and agent selection metrics."""
    
    def __init__(self, logger_name: str = 'mode_metrics'):
        self.logger = logging.getLogger(logger_name)
        self.mode_usage_stats = {
            'llm_only': {'count': 0, 'total_time': 0.0, 'agent_bypasses': 0},
            'hybrid': {'count': 0, 'total_time': 0.0, 'agent_bypasses': 0}
        }
        self.agent_selection_stats = {}
        self.mode_switches = []
    
    def track_mode_usage(self, mode: str, execution_time: float, 
                        bypassed_agents_count: int = 0) -> None:
        """Track usage statistics for a specific mode."""
        if mode in self.mode_usage_stats:
            self.mode_usage_stats[mode]['count'] += 1
            self.mode_usage_stats[mode]['total_time'] += execution_time
            self.mode_usage_stats[mode]['agent_bypasses'] += bypassed_agents_count
            
            self.logger.debug(
                f"Mode usage tracked: {mode}",
                extra={
                    'mode': mode,
                    'execution_time': execution_time,
                    'bypassed_agents_count': bypassed_agents_count,
                    'total_usage_count': self.mode_usage_stats[mode]['count']
                }
            )
    
    def track_agent_selection(self, session_id: str, iteration: int,
                            available_agents: list, selected_agents: list,
                            mode: str) -> None:
        """Track agent selection patterns."""
        selection_key = f"{mode}_{len(available_agents)}_{len(selected_agents)}"
        
        if selection_key not in self.agent_selection_stats:
            self.agent_selection_stats[selection_key] = {
                'count': 0,
                'mode': mode,
                'available_count': len(available_agents),
                'selected_count': len(selected_agents),
                'selection_ratio': len(selected_agents) / len(available_agents) if available_agents else 0
            }
        
        self.agent_selection_stats[selection_key]['count'] += 1
        
        self.logger.debug(
            f"Agent selection tracked: {selection_key}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'selection_pattern': selection_key,
                'available_agents': available_agents,
                'selected_agents': selected_agents,
                'mode': mode
            }
        )
    
    def track_mode_switch(self, old_mode: str, new_mode: str, 
                         trigger: str = 'configuration_change') -> None:
        """Track mode switches."""
        switch_event = {
            'timestamp': datetime.now().isoformat(),
            'old_mode': old_mode,
            'new_mode': new_mode,
            'trigger': trigger
        }
        
        self.mode_switches.append(switch_event)
        
        self.logger.info(
            f"Mode switch tracked: {old_mode} -> {new_mode}",
            extra={
                'old_mode': old_mode,
                'new_mode': new_mode,
                'trigger': trigger,
                'total_switches': len(self.mode_switches)
            }
        )
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of mode usage statistics."""
        summary = {
            'mode_usage': self.mode_usage_stats.copy(),
            'agent_selection_patterns': self.agent_selection_stats.copy(),
            'mode_switches': len(self.mode_switches),
            'recent_switches': self.mode_switches[-5:] if self.mode_switches else []
        }
        
        # Calculate averages
        for mode, stats in summary['mode_usage'].items():
            if stats['count'] > 0:
                stats['avg_execution_time'] = stats['total_time'] / stats['count']
                stats['avg_bypasses_per_execution'] = stats['agent_bypasses'] / stats['count']
            else:
                stats['avg_execution_time'] = 0.0
                stats['avg_bypasses_per_execution'] = 0.0
        
        return summary
    
    def log_usage_summary(self) -> None:
        """Log current usage summary."""
        summary = self.get_usage_summary()
        
        self.logger.info(
            "Mode usage summary",
            extra={
                'usage_summary': summary,
                'total_executions': sum(stats['count'] for stats in summary['mode_usage'].values()),
                'total_switches': summary['mode_switches']
            }
        )


class PerformanceLogger:
    """Logger for performance monitoring and metrics."""
    
    def __init__(self, logger_name: str = 'performance'):
        self.logger = logging.getLogger(logger_name)
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self.start_times[operation_id] = time.time()
        self.logger.debug(f"Started timing operation: {operation_id}")
    
    def end_timer(self, operation_id: str, **extra_fields) -> float:
        """End timing an operation and log the duration."""
        if operation_id not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation_id}")
            return 0.0
        
        duration = time.time() - self.start_times[operation_id]
        del self.start_times[operation_id]
        
        self.logger.info(
            f"Operation completed: {operation_id}",
            extra={
                'processing_time': duration,
                'operation_id': operation_id,
                **extra_fields
            }
        )
        
        return duration
    
    def log_metric(self, metric_name: str, value: float, **extra_fields) -> None:
        """Log a performance metric."""
        self.logger.info(
            f"Metric: {metric_name} = {value}",
            extra={
                'metric_name': metric_name,
                'metric_value': value,
                **extra_fields
            }
        )


class LLMLogger:
    """Specialized logger for LLM agent interactions and outputs."""
    
    def __init__(self, logger_name: str = 'llm_agents'):
        self.logger = logging.getLogger(logger_name)
    
    def log_llm_call(self, agent_name: str, prompt: str, context: Dict[str, Any],
                     session_id: str, iteration: int, model_name: str = None) -> None:
        """Log outgoing LLM call with prompt and context."""
        self.logger.debug(
            f"LLM call initiated by {agent_name}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'llm_call',
                'model_used': model_name,
                'prompt_text': prompt,
                'context_data': context,
                'prompt_length': len(prompt) if prompt else 0
            }
        )
    
    def log_llm_response(self, agent_name: str, response: Dict[str, Any],
                        session_id: str, iteration: int, processing_time: float = None,
                        tokens_used: int = None) -> None:
        """Log incoming LLM response with metadata."""
        response_text = response.get('content', '') if isinstance(response, dict) else str(response)
        
        self.logger.info(
            f"LLM response received by {agent_name}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'llm_response',
                'raw_response': response_text,
                'response_length': len(response_text),
                'processing_time': processing_time,
                'tokens_used': tokens_used,
                'model_used': response.get('model') if isinstance(response, dict) else None
            }
        )
    
    def log_parsed_response(self, agent_name: str, parsed_data: Dict[str, Any],
                           session_id: str, iteration: int, parsing_success: bool = True) -> None:
        """Log parsed response data and parsing results."""
        self.logger.info(
            f"Response parsed by {agent_name}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'response_parsing',
                'parsed_components': parsed_data,
                'extraction_success': parsing_success,
                'components_count': len(parsed_data) if parsed_data else 0
            }
        )
    
    def log_component_extraction(self, agent_name: str, component_type: str,
                               extracted_data: Any, success: bool,
                               session_id: str, iteration: int) -> None:
        """Log component extraction results."""
        self.logger.info(
            f"Component extraction: {component_type} by {agent_name}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'component_extraction',
                'component_type': component_type,
                'extracted_data': extracted_data,
                'extraction_success': success
            }
        )
    
    def log_confidence_calculation(self, agent_name: str, confidence_score: float,
                                 reasoning: str, session_id: str, iteration: int,
                                 factors: Dict[str, Any] = None) -> None:
        """Log confidence calculation details."""
        self.logger.info(
            f"Confidence calculated by {agent_name}: {confidence_score}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'confidence_calculation',
                'confidence_score': confidence_score,
                'reasoning': reasoning,
                'confidence_factors': factors
            }
        )
    
    def log_agent_reasoning(self, agent_name: str, reasoning_type: str,
                          reasoning_text: str, session_id: str, iteration: int,
                          metadata: Dict[str, Any] = None) -> None:
        """Log agent-specific reasoning and analysis."""
        extra_fields = {
            'session_id': session_id,
            'iteration': iteration,
            'agent_name': agent_name,
            'interaction_type': 'agent_reasoning',
            'reasoning_type': reasoning_type,
            'reasoning': reasoning_text
        }
        
        # Add agent-specific metadata
        if metadata:
            extra_fields.update(metadata)
        
        self.logger.info(
            f"Agent reasoning: {reasoning_type} by {agent_name}",
            extra=extra_fields
        )
    
    def log_llm_error(self, agent_name: str, error: Exception, context: Dict[str, Any],
                     session_id: str, iteration: int, fallback_used: bool = False) -> None:
        """Log LLM interaction errors and fallback usage."""
        self.logger.error(
            f"LLM error in {agent_name}: {str(error)}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_name': agent_name,
                'interaction_type': 'llm_error',
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context_data': context,
                'fallback_used': fallback_used
            },
            exc_info=True
        )


class OrchestrationLogger:
    """Specialized logger for orchestration events and decisions."""
    
    def __init__(self, logger_name: str = 'orchestration'):
        self.logger = logging.getLogger(logger_name)
    
    def log_agent_coordination(self, session_id: str, iteration: int, 
                             execution_order: list, strategy_type: str, 
                             reasoning: str) -> None:
        """Log agent coordination decisions."""
        self.logger.info(
            "Agent coordination strategy determined",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'execution_order': execution_order,
                'strategy_type': strategy_type,
                'orchestration_decision': 'agent_coordination',
                'reasoning': reasoning
            }
        )
    
    def log_conflict_resolution(self, session_id: str, iteration: int,
                              conflicts: list, resolution_method: str,
                              final_decision: str) -> None:
        """Log conflict resolution decisions."""
        self.logger.info(
            "Agent conflict resolved",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'conflicts': conflicts,
                'resolution_method': resolution_method,
                'final_decision': final_decision,
                'orchestration_decision': 'conflict_resolution'
            }
        )
    
    def log_convergence_analysis(self, session_id: str, iteration: int,
                               has_converged: bool, convergence_score: float,
                               reasoning: str, confidence: float) -> None:
        """Log convergence analysis results."""
        self.logger.info(
            f"Convergence analysis: {'converged' if has_converged else 'continuing'}",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'has_converged': has_converged,
                'convergence_score': convergence_score,
                'confidence': confidence,
                'reasoning': reasoning,
                'orchestration_decision': 'convergence_analysis'
            }
        )
    
    def log_synthesis_decision(self, session_id: str, iteration: int,
                             agent_results: dict, synthesis_reasoning: str,
                             final_prompt: str, confidence: float) -> None:
        """Log synthesis decisions."""
        self.logger.info(
            "Agent outputs synthesized",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'agent_count': len(agent_results),
                'synthesis_reasoning': synthesis_reasoning,
                'confidence': confidence,
                'orchestration_decision': 'synthesis'
            }
        )
    
    def log_agent_bypass(self, session_id: str, iteration: int,
                        bypassed_agents: list, mode: str, reason: str) -> None:
        """Log when agents are bypassed due to mode configuration."""
        self.logger.info(
            f"Agents bypassed in {mode} mode",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'bypassed_agents': bypassed_agents,
                'bypass_count': len(bypassed_agents),
                'mode': mode,
                'reason': reason,
                'orchestration_decision': 'agent_bypass'
            }
        )
    
    def log_mode_switch(self, old_mode: str, new_mode: str, 
                       session_id: Optional[str] = None,
                       trigger: str = 'configuration_change') -> None:
        """Log when the orchestration mode changes."""
        extra_fields = {
            'old_mode': old_mode,
            'new_mode': new_mode,
            'trigger': trigger,
            'orchestration_decision': 'mode_switch'
        }
        
        if session_id:
            extra_fields['session_id'] = session_id
            
        self.logger.info(
            f"Mode switched from {old_mode} to {new_mode}",
            extra=extra_fields
        )
    
    def log_agent_selection_metrics(self, session_id: str, iteration: int,
                                  available_agents: list, selected_agents: list,
                                  execution_time: float, mode: str) -> None:
        """Log metrics about agent selection and execution."""
        self.logger.info(
            "Agent selection metrics",
            extra={
                'session_id': session_id,
                'iteration': iteration,
                'available_agents': available_agents,
                'selected_agents': selected_agents,
                'available_count': len(available_agents),
                'selected_count': len(selected_agents),
                'execution_time': execution_time,
                'mode': mode,
                'orchestration_decision': 'agent_selection_metrics'
            }
        )


def _parse_file_size(size_str: str) -> int:
    """
    Parse file size string to bytes.
    
    Args:
        size_str: Size string like '20MB', '5GB', etc.
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    # Extract number and unit
    import re
    match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$', size_str)
    if not match:
        # Default to 20MB if parsing fails
        return 20 * 1024 * 1024
    
    number, unit = match.groups()
    number = float(number)
    
    # Convert to bytes
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
        'TB': 1024 * 1024 * 1024 * 1024,
        '': 1  # Default to bytes if no unit
    }
    
    multiplier = multipliers.get(unit, 1024 * 1024)  # Default to MB
    return int(number * multiplier)


def setup_logging(log_level: str = 'INFO', 
                 log_dir: Optional[str] = None,
                 enable_structured_logging: bool = True,
                 enable_performance_logging: bool = True,
                 llm_logging_config: Optional[Dict[str, Any]] = None) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for console only)
        enable_structured_logging: Whether to use structured JSON logging
        enable_performance_logging: Whether to enable performance logging
        llm_logging_config: Dictionary containing LLM-specific logging configuration
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Set up LLM logging configuration with defaults
    if llm_logging_config is None:
        llm_logging_config = {}
    
    llm_config = {
        'level': llm_logging_config.get('level', log_level),
        'log_raw_responses': llm_logging_config.get('log_raw_responses', True),
        'log_prompts': llm_logging_config.get('log_prompts', False),
        'max_response_log_length': llm_logging_config.get('max_response_log_length', 5000),
        'max_prompt_log_length': llm_logging_config.get('max_prompt_log_length', 2000),
        'max_reasoning_log_length': llm_logging_config.get('max_reasoning_log_length', 2500),
        'separate_log_files': llm_logging_config.get('separate_log_files', True),
        'enable_security_filtering': llm_logging_config.get('enable_security_filtering', True),
        'sensitive_data_patterns': llm_logging_config.get('sensitive_data_patterns', [
            'password', 'api_key', 'secret', 'token', 'credential'
        ]),
        'truncation_indicator': llm_logging_config.get('truncation_indicator', '... [truncated for security/length]'),
        'log_file_prefix': llm_logging_config.get('log_file_prefix', 'llm_'),
        'log_file_max_size': llm_logging_config.get('log_file_max_size', '20MB'),
        'log_file_backup_count': llm_logging_config.get('log_file_backup_count', 10)
    }
    
    # Set LLM logging level
    llm_numeric_level = getattr(logging, llm_config['level'].upper(), numeric_level)
    
    # Create log directory if specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Apply the logging level to all existing loggers immediately
    for logger_name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(logger_name)
        existing_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if enable_structured_logging:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers if log directory specified
    loggers = {}
    
    if log_dir:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / 'bedrock_optimizer.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(numeric_level)
        app_handler.setFormatter(console_formatter)
        
        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / 'errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(console_formatter)
        
        # Add file handlers to root logger
        root_logger.addHandler(app_handler)
        root_logger.addHandler(error_handler)
        
        # Orchestration log
        orchestration_logger = logging.getLogger('orchestration')
        orchestration_handler = logging.handlers.RotatingFileHandler(
            log_path / 'orchestration.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        orchestration_handler.setFormatter(console_formatter)
        orchestration_logger.addHandler(orchestration_handler)
        loggers['orchestration'] = orchestration_logger
        
        # Performance log
        if enable_performance_logging:
            performance_logger = logging.getLogger('performance')
            performance_handler = logging.handlers.RotatingFileHandler(
                log_path / 'performance.log',
                maxBytes=5*1024*1024,
                backupCount=3
            )
            performance_handler.setFormatter(console_formatter)
            performance_logger.addHandler(performance_handler)
            loggers['performance'] = performance_logger
        
        # LLM interaction logs with enhanced configuration
        llm_logger = logging.getLogger('llm_agents')
        llm_logger.setLevel(llm_numeric_level)
        
        # Parse log file size
        max_bytes = _parse_file_size(llm_config['log_file_max_size'])
        backup_count = llm_config['log_file_backup_count']
        
        # Main LLM interactions log
        llm_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{llm_config['log_file_prefix']}interactions.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        llm_handler.setLevel(llm_numeric_level)
        
        # Use specialized LLM formatter with configuration
        llm_formatter = LLMInteractionFormatter(llm_config)
        llm_handler.setFormatter(llm_formatter)
        llm_logger.addHandler(llm_handler)
        
        # Add console handler for LLM logs if at appropriate level
        if llm_numeric_level <= logging.INFO:
            llm_console_handler = logging.StreamHandler(sys.stdout)
            llm_console_handler.setLevel(llm_numeric_level)
            llm_console_handler.setFormatter(llm_formatter)
            llm_logger.addHandler(llm_console_handler)
        
        loggers['llm_agents'] = llm_logger
        
        # Individual LLM agent loggers with separate files if configured
        if llm_config['separate_log_files']:
            for agent_type in ['analyzer', 'refiner', 'validator']:
                agent_logger = logging.getLogger(f'llm_agents.{agent_type}')
                agent_logger.setLevel(llm_numeric_level)
                
                agent_handler = logging.handlers.RotatingFileHandler(
                    log_path / f"{llm_config['log_file_prefix']}{agent_type}.log",
                    maxBytes=max_bytes // 2,  # Smaller files for individual agents
                    backupCount=backup_count // 2
                )
                agent_handler.setLevel(llm_numeric_level)
                agent_handler.setFormatter(llm_formatter)
                agent_logger.addHandler(agent_handler)
                
                loggers[f'llm_{agent_type}'] = agent_logger
    
    # Create specialized loggers
    loggers.update({
        'main': logging.getLogger('bedrock_optimizer'),
        'session': logging.getLogger('session'),
        'agents': logging.getLogger('agents'),
        'bedrock': logging.getLogger('bedrock'),
        'evaluation': logging.getLogger('evaluation'),
        'storage': logging.getLogger('storage'),
        'cli': logging.getLogger('cli')
    })
    
    # Ensure LLM loggers exist even without file logging
    if 'llm_agents' not in loggers:
        llm_logger = logging.getLogger('llm_agents')
        llm_logger.setLevel(llm_numeric_level)
        loggers['llm_agents'] = llm_logger
        
        # Individual LLM agent loggers
        for agent_type in ['analyzer', 'refiner', 'validator']:
            agent_logger = logging.getLogger(f'llm_agents.{agent_type}')
            agent_logger.setLevel(llm_numeric_level)
            loggers[f'llm_{agent_type}'] = agent_logger
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exception: Exception, 
                 context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an exception with context information.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        context: Additional context information
    """
    extra_fields = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception)
    }
    
    if context:
        extra_fields.update(context)
    
    logger.error(
        f"Exception occurred: {type(exception).__name__}: {str(exception)}",
        extra=extra_fields,
        exc_info=True
    )


def configure_llm_logging(log_level: str = 'INFO',
                         log_raw_responses: bool = True,
                         log_prompts: bool = False,
                         max_response_length: int = 5000) -> Dict[str, Any]:
    """
    Configure LLM-specific logging settings.
    
    Args:
        log_level: Logging level for LLM interactions
        log_raw_responses: Whether to include full LLM responses
        log_prompts: Whether to log outgoing prompts
        max_response_length: Maximum length for logged responses
    
    Returns:
        Dictionary of LLM logging configuration
    """
    config = {
        'log_level': log_level,
        'log_raw_responses': log_raw_responses,
        'log_prompts': log_prompts,
        'max_response_length': max_response_length,
        'numeric_level': getattr(logging, log_level.upper(), logging.INFO)
    }
    
    # Apply configuration to existing LLM loggers
    llm_logger = logging.getLogger('llm_agents')
    llm_logger.setLevel(config['numeric_level'])
    
    for agent_type in ['analyzer', 'refiner', 'validator']:
        agent_logger = logging.getLogger(f'llm_agents.{agent_type}')
        agent_logger.setLevel(config['numeric_level'])
    
    return config


def get_llm_logger(agent_name: str = None) -> logging.Logger:
    """
    Get an LLM-specific logger instance.
    
    Args:
        agent_name: Specific agent name (analyzer, refiner, validator) or None for general
    
    Returns:
        Logger instance for LLM interactions
    """
    if agent_name and agent_name in ['analyzer', 'refiner', 'validator']:
        return logging.getLogger(f'llm_agents.{agent_name}')
    return logging.getLogger('llm_agents')


def truncate_response_for_logging(response: str, max_length: int = 5000) -> str:
    """
    Truncate LLM response for logging if it exceeds maximum length.
    
    Args:
        response: Raw LLM response text
        max_length: Maximum allowed length
    
    Returns:
        Truncated response with indicator if truncated
    """
    if not response or len(response) <= max_length:
        return response
    
    truncated = response[:max_length]
    return f"{truncated}... [TRUNCATED: original length {len(response)} chars]"


def filter_sensitive_data(data: Dict[str, Any], 
                         sensitive_keys: list = None) -> Dict[str, Any]:
    """
    Filter sensitive data from logging context.
    
    Args:
        data: Data dictionary to filter
        sensitive_keys: List of keys to filter (defaults to common sensitive keys)
    
    Returns:
        Filtered data dictionary
    """
    if not data:
        return data
    
    if sensitive_keys is None:
        sensitive_keys = ['password', 'token', 'secret', 'credential', 'api_key', 'auth_key']
    
    filtered_data = data.copy()
    
    for key in list(filtered_data.keys()):
        key_lower = key.lower()
        # Check for exact matches or specific patterns to avoid over-filtering
        if (key_lower in sensitive_keys or 
            key_lower.endswith('_password') or 
            key_lower.endswith('_token') or 
            key_lower.endswith('_secret') or
            key_lower.endswith('_key') and ('api' in key_lower or 'auth' in key_lower)):
            filtered_data[key] = '[FILTERED]'
    
    return filtered_data


# Global instances for easy access
performance_logger = PerformanceLogger()
orchestration_logger = OrchestrationLogger()
mode_usage_tracker = ModeUsageTracker()
llm_logger = LLMLogger()