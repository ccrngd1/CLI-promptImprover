"""
LLMAgent base class for LLM-enhanced prompt improvement agents.

This module provides the base class for agents that leverage Large Language Models
for intelligent analysis, refinement, and validation of prompts with embedded
best practices and reasoning capabilities.
"""

import json
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Union
from agents.base import Agent, AgentResult
from models import PromptIteration, UserFeedback


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
    
    def _call_llm(self, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a call to the LLM with the system prompt and user input.
        
        Args:
            user_prompt: The user prompt to send to the LLM
            context: Optional context to include in the prompt
            
        Returns:
            Dictionary containing the LLM response and metadata
        """
        # In a real implementation, this would call an actual LLM API
        # For now, we'll simulate the response structure
        
        # Prepare the full prompt
        full_prompt = self._prepare_full_prompt(user_prompt, context)
        
        try:
            # Simulate LLM call - in practice, this would use an actual LLM client
            response = self._simulate_llm_response(full_prompt)
            
            return {
                'success': True,
                'response': response,
                'model': self.llm_model,
                'tokens_used': len(full_prompt.split()) + len(response.split()),
                'temperature': self.llm_temperature,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': '',
                'model': self.llm_model,
                'tokens_used': 0,
                'temperature': self.llm_temperature,
                'error': str(e)
            }
    
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
    
    def _simulate_llm_response(self, full_prompt: str) -> str:
        """
        Simulate an LLM response for testing purposes.
        
        In a real implementation, this would be replaced with actual LLM API calls.
        
        Args:
            full_prompt: The complete prompt sent to the LLM
            
        Returns:
            Simulated LLM response
        """
        # This is a placeholder that generates a structured response
        # based on the agent type and prompt content
        
        agent_type = self.name.lower()
        
        if 'analyzer' in agent_type:
            return self._simulate_analyzer_response(full_prompt)
        elif 'refiner' in agent_type:
            return self._simulate_refiner_response(full_prompt)
        elif 'validator' in agent_type:
            return self._simulate_validator_response(full_prompt)
        else:
            return "LLM analysis completed with structured reasoning and best practices applied."
    
    def _simulate_analyzer_response(self, prompt: str) -> str:
        """Simulate analyzer LLM response."""
        return """
        Analysis Results:
        
        Structure Assessment:
        - The prompt has a clear task definition
        - Context information could be enhanced
        - Examples would improve clarity
        
        Clarity Evaluation:
        - Language is generally clear and direct
        - Some technical terms may need explanation
        - Action verbs are well-defined
        
        Best Practices Applied:
        - Evaluated against prompt engineering standards
        - Checked for completeness and specificity
        - Assessed readability and actionability
        
        Recommendations:
        1. Add specific examples to illustrate expected output
        2. Include context section for background information
        3. Define success criteria more explicitly
        """
    
    def _simulate_refiner_response(self, prompt: str) -> str:
        """Simulate refiner LLM response."""
        return """
        Refined Prompt Suggestion:
        
        ## Task
        [Enhanced task definition with clearer instructions]
        
        ## Context
        [Added background information and use case details]
        
        ## Requirements
        - Specific requirement 1
        - Specific requirement 2
        - Output format specification
        
        ## Example
        [Concrete example of expected output]
        
        Improvements Made:
        - Enhanced structure with clear sections
        - Added specific examples and context
        - Clarified success criteria and constraints
        - Applied prompt engineering best practices
        """
    
    def _simulate_validator_response(self, prompt: str) -> str:
        """Simulate validator LLM response."""
        return """
        Validation Results:
        
        Syntax Check: PASS
        - No formatting issues detected
        - Proper structure and organization
        
        Logical Consistency: PASS
        - No contradictions found
        - Instructions flow logically
        
        Completeness Assessment: MINOR ISSUES
        - Task definition is clear
        - Output format could be more specific
        - Success criteria need enhancement
        
        Quality Score: 8.2/10
        
        Recommendations:
        1. Add more specific output format requirements
        2. Define measurable success criteria
        3. Consider adding edge case handling instructions
        """
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured data.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed response as structured dictionary
        """
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
        recommendations = re.findall(r'^\s*[\d\-\*]\.\s*(.+)$', response, re.MULTILINE)
        parsed['recommendations'] = recommendations
        
        # Extract confidence if mentioned
        confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if confidence_match:
            parsed['confidence'] = float(confidence_match.group(1))
            if parsed['confidence'] > 1.0:  # Handle percentage format
                parsed['confidence'] /= 100.0
        
        # Extract reasoning sections
        reasoning_sections = re.findall(r'(reasoning|analysis|assessment)[:\s]*\n(.+?)(?=\n\n|\n[A-Z]|\Z)', 
                                      response, re.IGNORECASE | re.DOTALL)
        if reasoning_sections:
            parsed['reasoning'] = '\n'.join([section[1].strip() for section in reasoning_sections])
        
        return parsed
    
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