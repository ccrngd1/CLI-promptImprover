"""
LLMAgentLogger class for specialized LLM interaction logging.

This module provides comprehensive logging capabilities for LLM agent interactions,
including structured logging of LLM calls, responses, parsing results, and 
agent-specific metadata with error handling and fallback mechanisms.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from logging_config import get_logger, log_exception


class LLMAgentLogger:
    """
    Specialized logger for LLM agent interactions with structured logging
    and comprehensive error handling.
    
    This class provides methods for logging all aspects of LLM agent processing
    including calls, responses, parsing, component extraction, and reasoning.
    """
    
    def __init__(self, agent_name: str, logger_name: str = 'llm_agents', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM agent logger.
        
        Args:
            agent_name: Name of the agent using this logger
            logger_name: Base logger name for LLM agents
            config: Configuration dictionary for logging behavior
        """
        self.agent_name = agent_name
        self.logger_name = logger_name
        
        # Load configuration with defaults
        self.config = config or {}
        self.log_prompts = self.config.get('log_prompts', False)
        self.log_raw_responses = self.config.get('log_raw_responses', True)
        self.max_prompt_length = self.config.get('max_prompt_log_length', 2000)
        self.max_response_length = self.config.get('max_response_log_length', 5000)
        self.max_reasoning_length = self.config.get('max_reasoning_log_length', 2500)
        self.enable_security_filtering = self.config.get('enable_security_filtering', True)
        self.sensitive_patterns = self.config.get('sensitive_data_patterns', [
            'password', 'api_key', 'secret', 'token', 'credential'
        ])
        self.truncation_indicator = self.config.get('truncation_indicator', '... [truncated for security/length]')
        
        # Create agent-specific logger
        self.logger = get_logger(f"{logger_name}.{agent_name.lower()}")
        
        # Fallback logger for when primary logging fails
        self.fallback_logger = get_logger('llm_agents.fallback')
        
        # Track logging statistics
        self.logging_stats = {
            'total_logs': 0,
            'failed_logs': 0,
            'llm_calls_logged': 0,
            'responses_logged': 0,
            'parsing_logs': 0,
            'extraction_logs': 0
        }
    
    def log_llm_call(self, 
                     prompt: str, 
                     context: Optional[Dict[str, Any]] = None,
                     session_id: Optional[str] = None,
                     iteration: Optional[int] = None,
                     model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an outgoing LLM call with prompt and context.
        
        Args:
            prompt: The prompt being sent to the LLM
            context: Optional context information
            session_id: Session identifier for traceability
            iteration: Iteration number in the optimization process
            model_config: LLM model configuration details
        """
        try:
            log_entry = {
                'interaction_type': 'llm_call',
                'agent_name': self.agent_name,
                'prompt_length': len(prompt),
                'has_context': context is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if session_id:
                log_entry['session_id'] = session_id
            if iteration is not None:
                log_entry['iteration'] = iteration
            if model_config:
                log_entry['model_config'] = model_config
            if context:
                log_entry['context_keys'] = list(context.keys())
                log_entry['context_size'] = len(str(context))
            
            # Log prompt based on configuration
            if self.log_prompts:
                filtered_prompt = prompt
                # Apply security filtering if enabled
                if self.enable_security_filtering:
                    filtered_prompt = self._filter_sensitive_content(filtered_prompt)
                
                # Apply length truncation
                if len(filtered_prompt) > self.max_prompt_length:
                    log_entry['prompt'] = filtered_prompt[:self.max_prompt_length] + self.truncation_indicator
                    log_entry['prompt_truncated'] = True
                else:
                    log_entry['prompt'] = filtered_prompt
                    log_entry['prompt_truncated'] = False
            else:
                log_entry['prompt_logged'] = False
            
            self.logger.info(
                f"LLM call initiated by {self.agent_name}",
                extra=log_entry
            )
            
            self.logging_stats['llm_calls_logged'] += 1
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_llm_call', e, {
                'prompt_length': len(prompt) if prompt else 0,
                'session_id': session_id,
                'iteration': iteration
            })
    
    def log_llm_response(self, 
                        response: Dict[str, Any],
                        session_id: Optional[str] = None,
                        iteration: Optional[int] = None,
                        processing_time: Optional[float] = None) -> None:
        """
        Log an LLM response with metadata and performance information.
        
        Args:
            response: Dictionary containing LLM response and metadata
            session_id: Session identifier for traceability
            iteration: Iteration number in the optimization process
            processing_time: Time taken for the LLM call
        """
        try:
            log_entry = {
                'interaction_type': 'llm_response',
                'agent_name': self.agent_name,
                'success': response.get('success', False),
                'model_used': response.get('model', 'unknown'),
                'tokens_used': response.get('tokens_used', 0),
                'temperature': response.get('temperature', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if session_id:
                log_entry['session_id'] = session_id
            if iteration is not None:
                log_entry['iteration'] = iteration
            if processing_time is not None:
                log_entry['processing_time'] = processing_time
            
            # Handle response content based on configuration
            response_text = response.get('response', '')
            if response_text and self.log_raw_responses:
                log_entry['response_length'] = len(response_text)
                
                # Apply security filtering if enabled
                filtered_response = response_text
                if self.enable_security_filtering:
                    filtered_response = self._filter_sensitive_content(filtered_response)
                
                # Apply length truncation
                if len(filtered_response) > self.max_response_length:
                    log_entry['response'] = filtered_response[:self.max_response_length] + self.truncation_indicator
                    log_entry['response_truncated'] = True
                else:
                    log_entry['response'] = filtered_response
                    log_entry['response_truncated'] = False
                
                # Store full raw response for orchestration debugging (separate field)
                log_entry['raw_response_full'] = response_text
            else:
                log_entry['response_length'] = len(response_text) if response_text else 0
                log_entry['response_logged'] = self.log_raw_responses
                log_entry['response_truncated'] = False
            
            # Handle errors
            if response.get('error'):
                log_entry['error'] = response['error']
                log_entry['error_occurred'] = True
            else:
                log_entry['error_occurred'] = False
            
            # Log at appropriate level based on success
            if response.get('success', False):
                self.logger.info(
                    f"LLM response received by {self.agent_name}",
                    extra=log_entry
                )
            else:
                self.logger.warning(
                    f"LLM response failed for {self.agent_name}",
                    extra=log_entry
                )
            
            self.logging_stats['responses_logged'] += 1
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_llm_response', e, {
                'response_success': response.get('success', False) if response else False,
                'session_id': session_id,
                'iteration': iteration
            })
    
    def log_parsed_response(self, 
                           parsed_data: Dict[str, Any],
                           session_id: Optional[str] = None,
                           iteration: Optional[int] = None,
                           parsing_success: bool = True,
                           parsing_errors: Optional[List[str]] = None) -> None:
        """
        Log parsed LLM response data with parsing success indicators.
        
        Args:
            parsed_data: Dictionary containing parsed response data
            session_id: Session identifier for traceability
            iteration: Iteration number in the optimization process
            parsing_success: Whether parsing was successful
            parsing_errors: List of parsing errors if any occurred
        """
        try:
            log_entry = {
                'interaction_type': 'response_parsing',
                'agent_name': self.agent_name,
                'parsing_success': parsing_success,
                'parsed_components': list(parsed_data.keys()) if parsed_data else [],
                'component_count': len(parsed_data) if parsed_data else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if session_id:
                log_entry['session_id'] = session_id
            if iteration is not None:
                log_entry['iteration'] = iteration
            if parsing_errors:
                log_entry['parsing_errors'] = parsing_errors
                log_entry['error_count'] = len(parsing_errors)
            
            # Add parsed data details
            if parsed_data:
                # Log confidence if present
                if 'confidence' in parsed_data:
                    log_entry['confidence_score'] = parsed_data['confidence']
                
                # Log recommendations count if present
                if 'recommendations' in parsed_data:
                    recommendations = parsed_data['recommendations']
                    if isinstance(recommendations, list):
                        log_entry['recommendations_count'] = len(recommendations)
                    else:
                        log_entry['recommendations_count'] = 1
                
                # Log reasoning presence
                if 'reasoning' in parsed_data:
                    reasoning = parsed_data['reasoning']
                    log_entry['has_reasoning'] = bool(reasoning)
                    if reasoning:
                        log_entry['reasoning_length'] = len(str(reasoning))
                
                # Log analysis data if present
                if 'analysis' in parsed_data:
                    analysis = parsed_data['analysis']
                    if isinstance(analysis, dict):
                        log_entry['analysis_keys'] = list(analysis.keys())
                        log_entry['analysis_components'] = len(analysis)
            
            # Log at appropriate level based on success
            if parsing_success:
                self.logger.info(
                    f"Response parsing completed by {self.agent_name}",
                    extra=log_entry
                )
            else:
                self.logger.warning(
                    f"Response parsing failed for {self.agent_name}",
                    extra=log_entry
                )
            
            self.logging_stats['parsing_logs'] += 1
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_parsed_response', e, {
                'parsing_success': parsing_success,
                'component_count': len(parsed_data) if parsed_data else 0,
                'session_id': session_id,
                'iteration': iteration
            })
    
    def log_component_extraction(self, 
                               component_type: str,
                               extracted_data: Any,
                               success: bool,
                               extraction_method: Optional[str] = None,
                               confidence: Optional[float] = None) -> None:
        """
        Log component extraction results with success indicators.
        
        Args:
            component_type: Type of component being extracted
            extracted_data: The extracted data
            success: Whether extraction was successful
            extraction_method: Method used for extraction
            confidence: Confidence score for the extraction
        """
        try:
            log_entry = {
                'interaction_type': 'component_extraction',
                'agent_name': self.agent_name,
                'component_type': component_type,
                'extraction_success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if extraction_method:
                log_entry['extraction_method'] = extraction_method
            if confidence is not None:
                log_entry['extraction_confidence'] = confidence
            
            # Analyze extracted data
            if extracted_data is not None:
                log_entry['data_type'] = type(extracted_data).__name__
                
                if isinstance(extracted_data, (list, tuple)):
                    log_entry['data_count'] = len(extracted_data)
                elif isinstance(extracted_data, dict):
                    log_entry['data_keys'] = list(extracted_data.keys())
                    log_entry['data_count'] = len(extracted_data)
                elif isinstance(extracted_data, str):
                    log_entry['data_length'] = len(extracted_data)
                    log_entry['data_preview'] = extracted_data[:200] + "..." if len(extracted_data) > 200 else extracted_data
                else:
                    log_entry['data_value'] = str(extracted_data)
            else:
                log_entry['data_type'] = 'None'
                log_entry['data_count'] = 0
            
            # Log at appropriate level based on success
            if success:
                self.logger.info(
                    f"Component extraction successful: {component_type} by {self.agent_name}",
                    extra=log_entry
                )
            else:
                self.logger.warning(
                    f"Component extraction failed: {component_type} by {self.agent_name}",
                    extra=log_entry
                )
            
            self.logging_stats['extraction_logs'] += 1
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_component_extraction', e, {
                'component_type': component_type,
                'extraction_success': success,
                'data_type': type(extracted_data).__name__ if extracted_data is not None else 'None'
            })
    
    def log_confidence_calculation(self, 
                                 confidence_score: float,
                                 reasoning: Optional[str] = None,
                                 factors: Optional[Dict[str, float]] = None,
                                 calculation_method: Optional[str] = None) -> None:
        """
        Log confidence score calculation with reasoning and factors.
        
        Args:
            confidence_score: Calculated confidence score
            reasoning: Reasoning behind the confidence calculation
            factors: Dictionary of factors that influenced the score
            calculation_method: Method used for calculation
        """
        try:
            log_entry = {
                'interaction_type': 'confidence_calculation',
                'agent_name': self.agent_name,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if reasoning:
                log_entry['reasoning'] = reasoning
                log_entry['reasoning_length'] = len(reasoning)
            if factors:
                log_entry['confidence_factors'] = factors
                log_entry['factor_count'] = len(factors)
                log_entry['factor_average'] = sum(factors.values()) / len(factors)
            if calculation_method:
                log_entry['calculation_method'] = calculation_method
            
            # Categorize confidence level
            if confidence_score >= 0.8:
                log_entry['confidence_level'] = 'high'
            elif confidence_score >= 0.6:
                log_entry['confidence_level'] = 'medium'
            elif confidence_score >= 0.4:
                log_entry['confidence_level'] = 'low'
            else:
                log_entry['confidence_level'] = 'very_low'
            
            self.logger.info(
                f"Confidence calculated by {self.agent_name}: {confidence_score:.3f}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_confidence_calculation', e, {
                'confidence_score': confidence_score,
                'has_reasoning': reasoning is not None,
                'has_factors': factors is not None
            })
    
    def log_agent_reasoning(self, 
                          reasoning_type: str,
                          reasoning_text: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log agent-specific reasoning and analysis.
        
        Args:
            reasoning_type: Type of reasoning (e.g., 'analysis', 'refinement', 'validation')
            reasoning_text: The reasoning text from the agent
            metadata: Additional metadata about the reasoning
        """
        try:
            log_entry = {
                'interaction_type': 'agent_reasoning',
                'agent_name': self.agent_name,
                'reasoning_type': reasoning_type,
                'reasoning_length': len(reasoning_text),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add reasoning text with security filtering and truncation
            filtered_reasoning = reasoning_text
            if self.enable_security_filtering:
                filtered_reasoning = self._filter_sensitive_content(filtered_reasoning)
            
            if len(filtered_reasoning) > self.max_reasoning_length:
                log_entry['reasoning'] = filtered_reasoning[:self.max_reasoning_length] + self.truncation_indicator
                log_entry['reasoning_truncated'] = True
            else:
                log_entry['reasoning'] = filtered_reasoning
                log_entry['reasoning_truncated'] = False
            
            # Add metadata if provided
            if metadata:
                log_entry['metadata'] = metadata
                log_entry['metadata_keys'] = list(metadata.keys())
            
            self.logger.info(
                f"Agent reasoning logged: {reasoning_type} by {self.agent_name}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_agent_reasoning', e, {
                'reasoning_type': reasoning_type,
                'reasoning_length': len(reasoning_text) if reasoning_text else 0,
                'has_metadata': metadata is not None
            })
    
    def log_orchestration_raw_feedback(self, 
                                     agent_name: str,
                                     raw_llm_response: str,
                                     parsed_data: Dict[str, Any],
                                     session_id: Optional[str] = None,
                                     iteration: Optional[int] = None) -> None:
        """
        Log complete raw LLM feedback for orchestration debugging.
        
        Args:
            agent_name: Name of the agent that generated the response
            raw_llm_response: Complete raw LLM response text
            parsed_data: Parsed structured data from the response
            session_id: Session identifier for traceability
            iteration: Iteration number in the optimization process
        """
        try:
            log_entry = {
                'interaction_type': 'orchestration_raw_feedback',
                'agent_name': self.agent_name,
                'source_agent': agent_name,
                'raw_response_length': len(raw_llm_response),
                'parsed_components': list(parsed_data.keys()) if parsed_data else [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optional fields
            if session_id:
                log_entry['session_id'] = session_id
            if iteration is not None:
                log_entry['iteration'] = iteration
            
            # Store complete raw response for debugging (no truncation for orchestration debugging)
            log_entry['raw_llm_feedback'] = raw_llm_response
            
            # Add key parsed components for quick reference
            if parsed_data:
                if 'confidence' in parsed_data:
                    log_entry['extracted_confidence'] = parsed_data['confidence']
                if 'recommendations' in parsed_data:
                    log_entry['recommendations_count'] = len(parsed_data['recommendations']) if isinstance(parsed_data['recommendations'], list) else 1
                if 'reasoning' in parsed_data:
                    log_entry['has_reasoning'] = bool(parsed_data['reasoning'])
            
            self.logger.info(
                f"Raw LLM feedback captured for orchestration debugging: {agent_name} -> {self.agent_name}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_orchestration_raw_feedback', e, {
                'source_agent': agent_name,
                'raw_response_length': len(raw_llm_response) if raw_llm_response else 0,
                'session_id': session_id,
                'iteration': iteration
            })
    
    def log_error(self, 
                  error_type: str,
                  error_message: str,
                  context: Optional[Dict[str, Any]] = None,
                  exception: Optional[Exception] = None) -> None:
        """
        Log errors that occur during LLM agent processing.
        
        Args:
            error_type: Type of error (e.g., 'llm_call_failed', 'parsing_error')
            error_message: Error message
            context: Additional context about the error
            exception: Exception object if available
        """
        try:
            log_entry = {
                'interaction_type': 'error',
                'agent_name': self.agent_name,
                'error_type': error_type,
                'error_message': error_message,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add context if provided
            if context:
                log_entry['error_context'] = context
            
            # Add exception details if provided
            if exception:
                log_entry['exception_type'] = type(exception).__name__
                log_entry['exception_message'] = str(exception)
            
            self.logger.error(
                f"Error in {self.agent_name}: {error_type}",
                extra=log_entry,
                exc_info=exception is not None
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_error', e, {
                'original_error_type': error_type,
                'original_error_message': error_message
            })
    
    def log_fallback_usage(self, 
                          fallback_reason: str,
                          fallback_agent: Optional[str] = None,
                          original_error: Optional[str] = None,
                          fallback_type: str = "agent_fallback",
                          context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log when fallback mechanisms are used with enhanced error scenario tracking.
        
        Args:
            fallback_reason: Reason for using fallback
            fallback_agent: Name of fallback agent if applicable
            original_error: Original error that triggered fallback
            fallback_type: Type of fallback (agent_fallback, parsing_fallback, confidence_fallback)
            context: Additional context about the fallback scenario
        """
        try:
            log_entry = {
                'interaction_type': 'fallback_usage',
                'agent_name': self.agent_name,
                'fallback_reason': fallback_reason,
                'fallback_type': fallback_type,
                'timestamp': datetime.now().isoformat()
            }
            
            if fallback_agent:
                log_entry['fallback_agent'] = fallback_agent
            if original_error:
                log_entry['original_error'] = original_error
                log_entry['error_length'] = len(original_error)
            if context:
                log_entry['fallback_context'] = context
                log_entry['context_keys'] = list(context.keys())
            
            # Categorize fallback severity
            if any(keyword in fallback_reason.lower() for keyword in ['critical', 'severe', 'complete failure']):
                log_entry['fallback_severity'] = 'critical'
                log_level = 'error'
            elif any(keyword in fallback_reason.lower() for keyword in ['partial', 'degraded', 'low quality']):
                log_entry['fallback_severity'] = 'moderate'
                log_level = 'warning'
            else:
                log_entry['fallback_severity'] = 'minor'
                log_level = 'warning'
            
            # Log at appropriate level based on severity
            if log_level == 'error':
                self.logger.error(
                    f"Critical fallback used by {self.agent_name}: {fallback_reason}",
                    extra=log_entry
                )
            else:
                self.logger.warning(
                    f"Fallback used by {self.agent_name}: {fallback_reason}",
                    extra=log_entry
                )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_fallback_usage', e, {
                'fallback_reason': fallback_reason,
                'fallback_agent': fallback_agent,
                'fallback_type': fallback_type
            })
    
    def log_llm_service_failure(self, 
                               error_details: Dict[str, Any],
                               fallback_action: Optional[str] = None) -> None:
        """
        Log LLM service failures with detailed error information.
        
        Args:
            error_details: Dictionary containing error information
            fallback_action: Action taken as fallback
        """
        try:
            log_entry = {
                'interaction_type': 'llm_service_failure',
                'agent_name': self.agent_name,
                'timestamp': datetime.now().isoformat(),
                'error_details': error_details,
                'service_available': False
            }
            
            if fallback_action:
                log_entry['fallback_action'] = fallback_action
            
            # Extract specific error information
            if 'error_type' in error_details:
                log_entry['error_type'] = error_details['error_type']
            if 'status_code' in error_details:
                log_entry['status_code'] = error_details['status_code']
            if 'timeout' in error_details:
                log_entry['timeout_occurred'] = error_details['timeout']
            if 'retry_count' in error_details:
                log_entry['retry_attempts'] = error_details['retry_count']
            
            self.logger.error(
                f"LLM service failure for {self.agent_name}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_llm_service_failure', e, error_details)
    
    def log_parsing_failure(self, 
                           parsing_error: str,
                           raw_response: str,
                           partial_results: Optional[Dict[str, Any]] = None,
                           fallback_strategy: Optional[str] = None) -> None:
        """
        Log response parsing failures with partial results and fallback strategies.
        
        Args:
            parsing_error: Description of the parsing error
            raw_response: The raw response that failed to parse
            partial_results: Any partial data that was successfully extracted
            fallback_strategy: Strategy used to handle the parsing failure
        """
        try:
            log_entry = {
                'interaction_type': 'parsing_failure',
                'agent_name': self.agent_name,
                'parsing_error': parsing_error,
                'response_length': len(raw_response),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add truncated response for debugging
            max_response_log = 500
            if len(raw_response) > max_response_log:
                log_entry['raw_response_sample'] = raw_response[:max_response_log] + "... [truncated]"
                log_entry['response_truncated'] = True
            else:
                log_entry['raw_response_sample'] = raw_response
                log_entry['response_truncated'] = False
            
            # Log partial results if available
            if partial_results:
                log_entry['partial_results'] = partial_results
                log_entry['partial_extraction_success'] = True
                log_entry['partial_components'] = list(partial_results.keys())
            else:
                log_entry['partial_extraction_success'] = False
            
            if fallback_strategy:
                log_entry['fallback_strategy'] = fallback_strategy
            
            self.logger.warning(
                f"Response parsing failed for {self.agent_name}: {parsing_error}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_parsing_failure', e, {
                'parsing_error': parsing_error,
                'response_length': len(raw_response) if raw_response else 0
            })
    
    def log_extraction_failure(self, 
                              component_type: str,
                              extraction_error: str,
                              attempted_methods: List[str],
                              fallback_value: Any = None,
                              context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log component extraction failures with attempted methods and fallback values.
        
        Args:
            component_type: Type of component that failed to extract
            extraction_error: Description of the extraction error
            attempted_methods: List of extraction methods that were tried
            fallback_value: Default/fallback value used instead
            context: Additional context about the extraction attempt
        """
        try:
            log_entry = {
                'interaction_type': 'extraction_failure',
                'agent_name': self.agent_name,
                'component_type': component_type,
                'extraction_error': extraction_error,
                'attempted_methods': attempted_methods,
                'method_count': len(attempted_methods),
                'timestamp': datetime.now().isoformat()
            }
            
            if fallback_value is not None:
                log_entry['fallback_value'] = str(fallback_value)
                log_entry['fallback_value_type'] = type(fallback_value).__name__
                log_entry['has_fallback'] = True
            else:
                log_entry['has_fallback'] = False
            
            if context:
                log_entry['extraction_context'] = context
            
            self.logger.warning(
                f"Component extraction failed for {self.agent_name}: {component_type}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_extraction_failure', e, {
                'component_type': component_type,
                'extraction_error': extraction_error,
                'attempted_methods': attempted_methods
            })
    
    def log_confidence_calculation_fallback(self, 
                                          calculation_error: str,
                                          fallback_confidence: float,
                                          original_factors: Optional[Dict[str, float]] = None,
                                          fallback_reasoning: str = "Default confidence due to calculation error") -> None:
        """
        Log confidence calculation failures and fallback confidence values.
        
        Args:
            calculation_error: Description of the calculation error
            fallback_confidence: Fallback confidence value used
            original_factors: Original confidence factors that failed
            fallback_reasoning: Reasoning for the fallback confidence value
        """
        try:
            log_entry = {
                'interaction_type': 'confidence_calculation_fallback',
                'agent_name': self.agent_name,
                'calculation_error': calculation_error,
                'fallback_confidence': fallback_confidence,
                'fallback_reasoning': fallback_reasoning,
                'timestamp': datetime.now().isoformat()
            }
            
            if original_factors:
                log_entry['original_factors'] = original_factors
                log_entry['original_factor_count'] = len(original_factors)
                log_entry['had_original_factors'] = True
            else:
                log_entry['had_original_factors'] = False
            
            # Categorize confidence level
            if fallback_confidence >= 0.7:
                log_entry['confidence_category'] = 'high_fallback'
            elif fallback_confidence >= 0.5:
                log_entry['confidence_category'] = 'medium_fallback'
            else:
                log_entry['confidence_category'] = 'low_fallback'
            
            self.logger.warning(
                f"Confidence calculation fallback for {self.agent_name}: {fallback_confidence:.3f}",
                extra=log_entry
            )
            
            self.logging_stats['total_logs'] += 1
            
        except Exception as e:
            self._handle_logging_error('log_confidence_calculation_fallback', e, {
                'calculation_error': calculation_error,
                'fallback_confidence': fallback_confidence
            })
    
    def get_logging_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics for this agent logger.
        
        Returns:
            Dictionary containing logging statistics
        """
        return {
            'agent_name': self.agent_name,
            'logger_name': self.logger_name,
            'stats': self.logging_stats.copy(),
            'success_rate': (
                (self.logging_stats['total_logs'] - self.logging_stats['failed_logs']) / 
                max(self.logging_stats['total_logs'], 1)
            )
        }
    
    def _handle_logging_error(self, 
                            operation: str, 
                            error: Exception, 
                            context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle errors that occur during logging operations.
        
        Args:
            operation: Name of the logging operation that failed
            error: Exception that occurred
            context: Additional context about the failed operation
        """
        self.logging_stats['failed_logs'] += 1
        
        try:
            # Use fallback logger to log the logging error
            error_entry = {
                'agent_name': self.agent_name,
                'failed_operation': operation,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'timestamp': datetime.now().isoformat()
            }
            
            if context:
                error_entry['context'] = context
            
            self.fallback_logger.error(
                f"Logging failed for {self.agent_name} in operation {operation}",
                extra=error_entry,
                exc_info=True
            )
            
        except Exception as fallback_error:
            # If even fallback logging fails, use basic print as last resort
            print(f"CRITICAL: All logging failed for {self.agent_name}. "
                  f"Original error: {error}, Fallback error: {fallback_error}")
    
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