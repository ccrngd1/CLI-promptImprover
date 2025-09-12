"""
LLMAgent base class for LLM-enhanced prompt improvement agents.

This module provides the base class for agents that leverage Large Language Models
for intelligent analysis, refinement, and validation of prompts with embedded
best practices and reasoning capabilities.
"""

import json
import time
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Union
from agents.base import Agent, AgentResult
from agents.llm_agent_logger import LLMAgentLogger
from models import PromptIteration, UserFeedback
from bedrock.executor import BedrockExecutor, ModelConfig


class LLMAgent(Agent):
    """
    Base class for LLM-enhanced agents that use language models for intelligent
    prompt analysis and improvement with embedded best practices.
    
    This class extends the basic Agent with LLM integration capabilities,
    system prompt management, and reasoning frameworks.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM-enhanced agent.
        
        Args:
            name: The name of the agent
            config: Optional configuration dictionary including LLM settings
        """
        super().__init__(name, config)
        
        # LLM configuration
        self.llm_model = self.config.get('llm_model', 'claude-3-sonnet')
        self.llm_temperature = self.config.get('llm_temperature', 0.3)
        self.llm_max_tokens = self.config.get('llm_max_tokens', 2000)
        self.llm_timeout = self.config.get('llm_timeout', 30.0)
        
        # System prompt configuration
        self.system_prompt_template = self.config.get('system_prompt_template', '')
        self.best_practices_enabled = self.config.get('best_practices_enabled', True)
        self.reasoning_framework = self.config.get('reasoning_framework', 'structured')
        
        # Initialize LLM agent logger with configuration
        try:
            from config_loader import get_llm_logging_config, load_config
            llm_logging_config = get_llm_logging_config()
            full_config = load_config()
            bedrock_config = full_config.get('bedrock', {})
        except ImportError:
            llm_logging_config = {}
            bedrock_config = {}
        
        self.llm_logger = LLMAgentLogger(agent_name=name, config=llm_logging_config)
        
        # Initialize Bedrock executor
        self.bedrock_executor = BedrockExecutor(
            region_name=bedrock_config.get('region', 'us-east-1')
        )
        
        # Initialize system prompt
        self.system_prompt = self._build_system_prompt()
    
    @abstractmethod
    def _get_base_system_prompt(self) -> str:
        """
        Get the base system prompt for this agent type.
        
        Returns:
            Base system prompt string specific to the agent's role
        """
        pass
    
    @abstractmethod
    def _get_best_practices_prompt(self) -> str:
        """
        Get the best practices prompt section for this agent type.
        
        Returns:
            Best practices prompt string with domain-specific expertise
        """
        pass
    
    @abstractmethod
    def _get_reasoning_framework_prompt(self) -> str:
        """
        Get the reasoning framework prompt for structured analysis.
        
        Returns:
            Reasoning framework prompt string for systematic thinking
        """
        pass
    
    def _build_system_prompt(self) -> str:
        """
        Build the complete system prompt by combining base prompt,
        best practices, and reasoning framework.
        
        Returns:
            Complete system prompt for the LLM
        """
        prompt_parts = [self._get_base_system_prompt()]
        
        if self.best_practices_enabled:
            prompt_parts.append(self._get_best_practices_prompt())
        
        if self.reasoning_framework:
            prompt_parts.append(self._get_reasoning_framework_prompt())
        
        # Add custom template if provided
        if self.system_prompt_template:
            prompt_parts.append(self.system_prompt_template)
        
        return "\n\n".join(prompt_parts)
    
    def _call_llm(self, user_prompt: str, context: Optional[Dict[str, Any]] = None, 
                  session_id: Optional[str] = None, iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Make a call to the LLM with the system prompt and user input.
        
        Args:
            user_prompt: The user prompt to send to the LLM
            context: Optional context to include in the prompt
            session_id: Optional session identifier for logging
            iteration: Optional iteration number for logging
            
        Returns:
            Dictionary containing the LLM response and metadata
        """
        # Prepare the full prompt
        full_prompt = self._prepare_full_prompt(user_prompt, context)
        
        # Log the LLM call
        model_config = {
            'model': self.llm_model,
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens,
            'timeout': self.llm_timeout
        }
        
        self.llm_logger.log_llm_call(
            prompt=full_prompt,
            context=context,
            session_id=session_id,
            iteration=iteration,
            model_config=model_config
        )
        
        # Record start time for performance logging
        start_time = time.time()
        
        try:
            # Use actual Bedrock executor instead of simulation
            response = self._execute_bedrock_call(full_prompt)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            llm_response = {
                'success': True,
                'response': response,
                'model': self.llm_model,
                'tokens_used': len(full_prompt.split()) + len(response.split()),
                'temperature': self.llm_temperature,
                'error': None
            }
            
            # Log the LLM response
            self.llm_logger.log_llm_response(
                response=llm_response,
                session_id=session_id,
                iteration=iteration,
                processing_time=processing_time
            )
            
            return llm_response
            
        except Exception as e:
            # Calculate processing time even for errors
            processing_time = time.time() - start_time
            
            error_response = {
                'success': False,
                'response': '',
                'model': self.llm_model,
                'tokens_used': 0,
                'temperature': self.llm_temperature,
                'error': str(e)
            }
            
            # Log the failed LLM response
            self.llm_logger.log_llm_response(
                response=error_response,
                session_id=session_id,
                iteration=iteration,
                processing_time=processing_time
            )
            
            # Enhanced error logging with service failure details
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'model': self.llm_model,
                'timeout': processing_time > self.llm_timeout,
                'prompt_length': len(full_prompt),
                'processing_time': processing_time
            }
            
            # Log as LLM service failure with fallback information
            self.llm_logger.log_llm_service_failure(
                error_details=error_details,
                fallback_action="Will attempt fallback agent if enabled"
            )
            
            # Also log the general error
            self.llm_logger.log_error(
                error_type='llm_call_failed',
                error_message=str(e),
                context={'user_prompt_length': len(user_prompt), 'has_context': context is not None},
                exception=e
            )
            
            return error_response
    
    def _parse_llm_response(self, response: str, session_id: Optional[str] = None, 
                           iteration: Optional[int] = None) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response: Raw LLM response string
            session_id: Optional session identifier for logging
            iteration: Optional iteration number for logging
            
        Returns:
            Parsed response as structured dictionary
        """
        parsing_errors = []
        parsing_success = True
        
        try:
            # Try to extract structured information from the response
            parsed = {
                'raw_response': response,
                'analysis': {},
                'recommendations': [],
                'confidence': 0.8,  # Default confidence
                'reasoning': ''
            }
            
            # Extract recommendations (lines starting with numbers or bullets)
            import re
            try:
                recommendations = re.findall(r'^\s*[\d\-\*]\.\s*(.+)', response, re.MULTILINE)
                parsed['recommendations'] = recommendations
                
                # Log component extraction for recommendations
                self.llm_logger.log_component_extraction(
                    component_type='recommendations',
                    extracted_data=recommendations,
                    success=len(recommendations) > 0,
                    extraction_method='regex_pattern'
                )
                
            except Exception as e:
                parsing_errors.append(f"Failed to extract recommendations: {str(e)}")
                parsed['recommendations'] = []
                
                # Log extraction failure with attempted methods
                self.llm_logger.log_extraction_failure(
                    component_type='recommendations',
                    extraction_error=str(e),
                    attempted_methods=['regex_pattern'],
                    fallback_value=[],
                    context={'response_length': len(response)}
                )
            
            # Extract confidence if mentioned
            try:
                confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
                if confidence_match:
                    parsed['confidence'] = float(confidence_match.group(1))
                    if parsed['confidence'] > 1.0:  # Handle percentage format
                        parsed['confidence'] /= 100.0
                    
                    # Log component extraction for confidence
                    self.llm_logger.log_component_extraction(
                        component_type='confidence_score',
                        extracted_data=parsed['confidence'],
                        success=True,
                        extraction_method='regex_pattern',
                        confidence=1.0
                    )
                else:
                    # Log that no confidence was found, using default
                    self.llm_logger.log_component_extraction(
                        component_type='confidence_score',
                        extracted_data=parsed['confidence'],
                        success=False,
                        extraction_method='default_value'
                    )
                    
            except Exception as e:
                parsing_errors.append(f"Failed to extract confidence: {str(e)}")
                parsed['confidence'] = 0.8
                
                # Log extraction failure for confidence
                self.llm_logger.log_extraction_failure(
                    component_type='confidence_score',
                    extraction_error=str(e),
                    attempted_methods=['regex_pattern'],
                    fallback_value=0.8,
                    context={'response_length': len(response)}
                )
            
            # Extract reasoning sections
            try:
                reasoning_sections = re.findall(r'(reasoning|analysis|assessment)[:\s]*\n(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                              response, re.IGNORECASE | re.DOTALL)
                if reasoning_sections:
                    parsed['reasoning'] = '\n'.join([section[1].strip() for section in reasoning_sections])
                
                # Log component extraction for reasoning
                self.llm_logger.log_component_extraction(
                    component_type='reasoning',
                    extracted_data=parsed['reasoning'],
                    success=bool(parsed['reasoning']),
                    extraction_method='regex_pattern'
                )
                
            except Exception as e:
                parsing_errors.append(f"Failed to extract reasoning: {str(e)}")
                parsed['reasoning'] = ''
                
                # Log extraction failure for reasoning
                self.llm_logger.log_extraction_failure(
                    component_type='reasoning',
                    extraction_error=str(e),
                    attempted_methods=['regex_pattern'],
                    fallback_value='',
                    context={'response_length': len(response)}
                )
            
            # Determine overall parsing success and handle partial results
            if parsing_errors:
                parsing_success = False
                
                # Check if we have partial results despite errors
                partial_results = {}
                if parsed['recommendations']:
                    partial_results['recommendations'] = parsed['recommendations']
                if parsed['confidence'] > 0:
                    partial_results['confidence'] = parsed['confidence']
                if parsed['reasoning']:
                    partial_results['reasoning'] = parsed['reasoning']
                
                # Log partial results if any were extracted
                if partial_results:
                    self.llm_logger.log_parsing_failure(
                        parsing_error=f"Partial parsing failure: {'; '.join(parsing_errors)}",
                        raw_response=response,
                        partial_results=partial_results,
                        fallback_strategy="Using partial results with defaults for missing components"
                    )
            
        except Exception as e:
            # Complete parsing failure
            parsing_success = False
            parsing_errors.append(f"Complete parsing failure: {str(e)}")
            
            parsed = {
                'raw_response': response,
                'analysis': {},
                'recommendations': [],
                'confidence': 0.0,
                'reasoning': ''
            }
            
            # Enhanced parsing failure logging
            self.llm_logger.log_parsing_failure(
                parsing_error=str(e),
                raw_response=response,
                partial_results=None,
                fallback_strategy="Using default empty structure"
            )
            
            # Also log the general error
            self.llm_logger.log_error(
                error_type='response_parsing_failed',
                error_message=str(e),
                context={'response_length': len(response)},
                exception=e
            )
        
        # Log the parsing results
        self.llm_logger.log_parsed_response(
            parsed_data=parsed,
            session_id=session_id,
            iteration=iteration,
            parsing_success=parsing_success,
            parsing_errors=parsing_errors if parsing_errors else None
        )
        
        return parsed
    
    def _calculate_confidence_with_logging(self, factors: Dict[str, float], 
                                         reasoning: str = "", 
                                         method: str = "weighted_average") -> float:
        """
        Calculate confidence score with comprehensive logging.
        
        Args:
            factors: Dictionary of confidence factors and their weights
            reasoning: Reasoning behind the confidence calculation
            method: Method used for calculation
            
        Returns:
            Calculated confidence score
        """
        try:
            if not factors:
                confidence = 0.5  # Default when no factors provided
                reasoning = "No confidence factors provided, using default value"
            else:
                # Calculate weighted average
                total_weight = sum(factors.values())
                if total_weight > 0:
                    confidence = sum(weight for weight in factors.values()) / len(factors)
                else:
                    confidence = 0.5
                    reasoning = f"Zero total weight in factors, using default. Factors: {factors}"
            
            # Ensure confidence is within valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Log the confidence calculation
            self.llm_logger.log_confidence_calculation(
                confidence_score=confidence,
                reasoning=reasoning,
                factors=factors,
                calculation_method=method
            )
            
            return confidence
            
        except Exception as e:
            # Enhanced confidence calculation fallback logging
            fallback_confidence = 0.5
            self.llm_logger.log_confidence_calculation_fallback(
                calculation_error=str(e),
                fallback_confidence=fallback_confidence,
                original_factors=factors,
                fallback_reasoning=f"Calculation failed with {method}, using default confidence"
            )
            
            # Also log the general error
            self.llm_logger.log_error(
                error_type='confidence_calculation_failed',
                error_message=str(e),
                context={'factors': factors, 'method': method},
                exception=e
            )
            return fallback_confidence
    
    def _prepare_full_prompt(self, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare the full prompt including system prompt, context, and user input.
        
        Args:
            user_prompt: The main user prompt
            context: Optional context information
            
        Returns:
            Complete prompt string for the LLM
        """
        prompt_parts = [f"System: {self.system_prompt}"]
        
        if context:
            context_str = self._format_context(context)
            prompt_parts.append(f"Context: {context_str}")
        
        prompt_parts.append(f"User: {user_prompt}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context information for inclusion in the prompt.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        if 'intended_use' in context:
            context_parts.append(f"Intended Use: {context['intended_use']}")
        
        if 'target_audience' in context:
            context_parts.append(f"Target Audience: {context['target_audience']}")
        
        if 'domain' in context:
            context_parts.append(f"Domain: {context['domain']}")
        
        if 'constraints' in context:
            context_parts.append(f"Constraints: {context['constraints']}")
        
        return "\n".join(context_parts)
    
    def _execute_bedrock_call(self, full_prompt: str) -> str:
        """
        Execute the actual Bedrock API call.
        
        Args:
            full_prompt: The complete prompt to send to the model
            
        Returns:
            The response text from the model
            
        Raises:
            Exception: If the Bedrock call fails
        """
        # Create model configuration
        model_config = ModelConfig(
            model_id=self.llm_model,
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
            top_p=0.9,
            top_k=250
        )
        
        # Execute the prompt
        result = self.bedrock_executor.execute_prompt(full_prompt, model_config)
        
        if not result.success:
            raise Exception(f"Bedrock execution failed: {result.error_message}")
        
        return result.response_text
    

    
    def update_system_prompt(self, new_template: str) -> None:
        """
        Update the system prompt template and rebuild the full system prompt.
        
        Args:
            new_template: New system prompt template
        """
        self.system_prompt_template = new_template
        self.system_prompt = self._build_system_prompt()
    
    def get_system_prompt(self) -> str:
        """
        Get the current complete system prompt.
        
        Returns:
            Current system prompt string
        """
        return self.system_prompt
    
    def enable_best_practices(self, enabled: bool = True) -> None:
        """
        Enable or disable best practices integration.
        
        Args:
            enabled: Whether to include best practices in system prompt
        """
        self.best_practices_enabled = enabled
        self.system_prompt = self._build_system_prompt()
    
    def set_reasoning_framework(self, framework: str) -> None:
        """
        Set the reasoning framework for structured analysis.
        
        Args:
            framework: Reasoning framework type ('structured', 'chain_of_thought', 'step_by_step')
        """
        self.reasoning_framework = framework
        self.system_prompt = self._build_system_prompt()
    

    
    def _handle_llm_failure(self, 
                           prompt: str, 
                           context: Optional[Dict[str, Any]] = None,
                           history: Optional[List[PromptIteration]] = None,
                           feedback: Optional[UserFeedback] = None,
                           error_reason: str = "LLM service unavailable") -> None:
        """
        Handle LLM failure by logging and raising an appropriate exception.
        
        Args:
            prompt: The prompt that failed to process
            context: Optional context information
            history: Optional prompt history
            feedback: Optional user feedback
            error_reason: Reason for the failure
            
        Raises:
            Exception: Always raises an exception with detailed error information
        """
        # Enhanced failure logging
        failure_context = {
            'prompt_length': len(prompt),
            'has_context': context is not None,
            'has_history': history is not None and len(history) > 0 if history else False,
            'has_feedback': feedback is not None,
            'agent_type': self.name,
            'model': self.llm_model,
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens
        }
        
        # Log the LLM failure
        self.llm_logger.log_error(
            error_type='llm_service_failure',
            error_message=f"LLM service failed after retries: {error_reason}",
            context=failure_context
        )
        
        # Log service failure details
        self.llm_logger.log_llm_service_failure(
            error_details={
                'error_type': 'service_unavailable',
                'error_message': error_reason,
                'model': self.llm_model,
                'prompt_length': len(prompt),
                'agent_name': self.name
            },
            fallback_action="No fallback available - raising exception"
        )
        
        # Raise a detailed exception
        raise Exception(
            f"LLM service failure in {self.name}: {error_reason}. "
            f"Model: {self.llm_model}, Prompt length: {len(prompt)} characters. "
            f"No fallback available - ensure Bedrock service is accessible and properly configured."
        )