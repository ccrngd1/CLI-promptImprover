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


def setup_logging(log_level: str = 'INFO', 
                 log_dir: Optional[str] = None,
                 enable_structured_logging: bool = True,
                 enable_performance_logging: bool = True) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for console only)
        enable_structured_logging: Whether to use structured JSON logging
        enable_performance_logging: Whether to enable performance logging
    
    Returns:
        Dictionary of configured loggers
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory if specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
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


# Global instances for easy access
performance_logger = PerformanceLogger()
orchestration_logger = OrchestrationLogger()
mode_usage_tracker = ModeUsageTracker()