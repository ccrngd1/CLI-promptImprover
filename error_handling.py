"""
Comprehensive error handling utilities for the Bedrock Prompt Optimizer.

Provides custom exceptions, retry mechanisms, timeout handling, and error recovery
strategies for API failures, orchestration edge cases, and system failures.
"""

import time
import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging
from logging_config import get_logger, log_exception


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    API_ERROR = "api_error"
    ORCHESTRATION_ERROR = "orchestration_error"
    AGENT_ERROR = "agent_error"
    VALIDATION_ERROR = "validation_error"
    STORAGE_ERROR = "storage_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    CONFIGURATION_ERROR = "configuration_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class ErrorContext:
    """Context information for errors."""
    session_id: Optional[str] = None
    iteration: Optional[int] = None
    agent_name: Optional[str] = None
    operation: Optional[str] = None
    timestamp: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None


class BedrockOptimizerException(Exception):
    """Base exception for Bedrock Optimizer errors."""
    
    def __init__(self, message: str, 
                 category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.recoverable = recoverable
        self.timestamp = time.time()


class APIError(BedrockOptimizerException):
    """Exception for API-related errors."""
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None,
                 error_code: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.API_ERROR, **kwargs)
        self.api_name = api_name
        self.status_code = status_code
        self.error_code = error_code


class OrchestrationError(BedrockOptimizerException):
    """Exception for orchestration-related errors."""
    
    def __init__(self, message: str, orchestration_stage: str, 
                 agent_results: Optional[Dict] = None, **kwargs):
        super().__init__(message, ErrorCategory.ORCHESTRATION_ERROR, **kwargs)
        self.orchestration_stage = orchestration_stage
        self.agent_results = agent_results or {}


class AgentError(BedrockOptimizerException):
    """Exception for agent-related errors."""
    
    def __init__(self, message: str, agent_name: str, 
                 agent_operation: str, **kwargs):
        super().__init__(message, ErrorCategory.AGENT_ERROR, **kwargs)
        self.agent_name = agent_name
        self.agent_operation = agent_operation


class TimeoutError(BedrockOptimizerException):
    """Exception for timeout-related errors."""
    
    def __init__(self, message: str, operation: str, timeout_duration: float, **kwargs):
        super().__init__(message, ErrorCategory.TIMEOUT_ERROR, **kwargs)
        self.operation = operation
        self.timeout_duration = timeout_duration


class RateLimitError(BedrockOptimizerException):
    """Exception for rate limiting errors."""
    
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, ErrorCategory.RATE_LIMIT_ERROR, **kwargs)
        self.retry_after = retry_after


class ValidationError(BedrockOptimizerException):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 validation_rule: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION_ERROR, **kwargs)
        self.field_name = field_name
        self.validation_rule = validation_rule


class StorageError(BedrockOptimizerException):
    """Exception for storage-related errors."""
    
    def __init__(self, message: str, storage_operation: str, **kwargs):
        super().__init__(message, ErrorCategory.STORAGE_ERROR, **kwargs)
        self.storage_operation = storage_operation


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def with_retry(retry_config: RetryConfig = None,
               retry_on: Union[Type[Exception], tuple] = Exception,
               logger_name: str = 'error_handling'):
    """
    Decorator for adding retry logic to functions.
    
    Args:
        retry_config: Retry configuration
        retry_on: Exception types to retry on
        logger_name: Logger name for retry logging
    """
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after {retry_config.max_attempts} attempts",
                            extra={'function': func.__name__, 'attempts': retry_config.max_attempts}
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{retry_config.max_attempts}), "
                        f"retrying in {delay:.2f}s: {str(e)}",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_attempts': retry_config.max_attempts,
                            'delay': delay,
                            'exception': str(e)
                        }
                    )
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float, operation_name: str = None):
    """
    Decorator for adding timeout handling to functions.
    
    Args:
        timeout_seconds: Timeout duration in seconds
        operation_name: Name of the operation for error messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                op_name = operation_name or func.__name__
                raise TimeoutError(
                    f"Operation '{op_name}' timed out after {timeout_seconds} seconds",
                    operation=op_name,
                    timeout_duration=timeout_seconds
                )
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Clean up timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: BedrockOptimizerException) -> bool:
        """Check if this strategy can recover from the given error."""
        return error.recoverable
    
    def recover(self, error: BedrockOptimizerException, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class FallbackStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that provides fallback values or operations."""
    
    def __init__(self, fallback_value: Any = None, fallback_function: Callable = None):
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function
    
    def recover(self, error: BedrockOptimizerException, context: Dict[str, Any]) -> Any:
        """Return fallback value or execute fallback function."""
        if self.fallback_function:
            return self.fallback_function(error, context)
        return self.fallback_value


class RetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()
    
    def recover(self, error: BedrockOptimizerException, context: Dict[str, Any]) -> Any:
        """Retry the operation based on retry configuration."""
        # This would be implemented based on the specific operation context
        raise NotImplementedError("Retry strategy requires operation-specific implementation")


class ErrorHandler:
    """Central error handler with recovery strategies."""
    
    def __init__(self, logger_name: str = 'error_handling'):
        self.logger = get_logger(logger_name)
        self.recovery_strategies: Dict[ErrorCategory, List[ErrorRecoveryStrategy]] = {}
        self.error_counts: Dict[str, int] = {}
    
    def register_recovery_strategy(self, category: ErrorCategory, 
                                 strategy: ErrorRecoveryStrategy) -> None:
        """Register a recovery strategy for a specific error category."""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The error to handle
            context: Additional context for recovery
        
        Returns:
            Recovery result if successful, otherwise re-raises the error
        """
        context = context or {}
        
        # Convert to BedrockOptimizerException if needed
        if not isinstance(error, BedrockOptimizerException):
            error = BedrockOptimizerException(
                str(error),
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.MEDIUM
            )
        
        # Log the error
        log_exception(self.logger, error, context)
        
        # Track error counts
        error_key = f"{error.category.value}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Try recovery strategies
        strategies = self.recovery_strategies.get(error.category, [])
        
        for strategy in strategies:
            if strategy.can_recover(error):
                try:
                    self.logger.info(
                        f"Attempting recovery with {type(strategy).__name__}",
                        extra={'error_category': error.category.value, 'strategy': type(strategy).__name__}
                    )
                    
                    result = strategy.recover(error, context)
                    
                    self.logger.info(
                        f"Successfully recovered from {type(error).__name__}",
                        extra={'error_category': error.category.value, 'recovery_strategy': type(strategy).__name__}
                    )
                    
                    return result
                    
                except Exception as recovery_error:
                    self.logger.warning(
                        f"Recovery strategy {type(strategy).__name__} failed: {str(recovery_error)}",
                        extra={'original_error': str(error), 'recovery_error': str(recovery_error)}
                    )
                    continue
        
        # No recovery possible, re-raise the error
        self.logger.error(
            f"No recovery strategy succeeded for {type(error).__name__}",
            extra={'error_category': error.category.value, 'strategies_tried': len(strategies)}
        )
        
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'error_categories': list(set(key.split(':')[0] for key in self.error_counts.keys()))
        }


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors with appropriate recovery."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert common API errors to our custom exceptions
            if 'ThrottlingException' in str(e) or 'Rate' in str(e):
                api_error = RateLimitError(
                    f"Rate limit exceeded in {func.__name__}: {str(e)}",
                    severity=ErrorSeverity.MEDIUM
                )
            elif 'ValidationException' in str(e):
                api_error = ValidationError(
                    f"Validation error in {func.__name__}: {str(e)}",
                    severity=ErrorSeverity.LOW
                )
            elif 'AuthenticationException' in str(e) or 'Unauthorized' in str(e):
                api_error = APIError(
                    f"Authentication error in {func.__name__}: {str(e)}",
                    api_name="bedrock",
                    severity=ErrorSeverity.HIGH,
                    recoverable=False
                )
            else:
                api_error = APIError(
                    f"API error in {func.__name__}: {str(e)}",
                    api_name="bedrock",
                    severity=ErrorSeverity.MEDIUM
                )
            
            return global_error_handler.handle_error(
                api_error, 
                {'function': func.__name__, 'args': args, 'kwargs': kwargs}
            )
    
    return wrapper


def handle_orchestration_errors(func: Callable) -> Callable:
    """Decorator for handling orchestration errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not isinstance(e, BedrockOptimizerException):
                orchestration_error = OrchestrationError(
                    f"Orchestration error in {func.__name__}: {str(e)}",
                    orchestration_stage=func.__name__,
                    severity=ErrorSeverity.MEDIUM
                )
            else:
                orchestration_error = e
            
            return global_error_handler.handle_error(
                orchestration_error,
                {'function': func.__name__, 'args': args, 'kwargs': kwargs}
            )
    
    return wrapper


# Register default recovery strategies
global_error_handler.register_recovery_strategy(
    ErrorCategory.RATE_LIMIT_ERROR,
    FallbackStrategy(fallback_value=None)
)

global_error_handler.register_recovery_strategy(
    ErrorCategory.TIMEOUT_ERROR,
    FallbackStrategy(fallback_value=None)
)

global_error_handler.register_recovery_strategy(
    ErrorCategory.AGENT_ERROR,
    FallbackStrategy(fallback_value={'success': False, 'error': 'Agent processing failed'})
)