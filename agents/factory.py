"""
AgentFactory for mode-based agent selection and creation.

This module implements the AgentFactory class that creates appropriate agent sets
based on the configuration mode (LLM-only vs hybrid), ensuring consistent
interfaces regardless of the creation mode.
"""

from typing import Dict, Any, Optional, List, Tuple
from agents.base import Agent
from agents.analyzer import AnalyzerAgent
from agents.refiner import RefinerAgent
from agents.validator import ValidatorAgent
from agents.llm_enhanced_analyzer import LLMAnalyzerAgent
from agents.llm_enhanced_refiner import LLMRefinerAgent
from agents.llm_enhanced_validator import LLMValidatorAgent
from logging_config import get_logger, orchestration_logger, mode_usage_tracker


class AgentFactory:
    """
    Factory class for creating agent sets based on configuration mode.
    
    Supports two modes:
    - Hybrid mode: Creates both heuristic and LLM-enhanced agents
    - LLM-only mode: Creates only LLM-enhanced agents
    
    Ensures consistent agent interfaces regardless of creation mode.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AgentFactory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('agent_factory')
        
        # Extract optimization configuration
        self.optimization_config = self.config.get('optimization', {})
        self.llm_only_mode = self.optimization_config.get('llm_only_mode', False)
        self.fallback_to_heuristic = self.optimization_config.get('fallback_to_heuristic', True)
        
        # Agent configuration
        self.agent_configs = self.config.get('agents', {})
        
        self.logger.info(
            f"AgentFactory initialized with mode: {'LLM-only' if self.llm_only_mode else 'Hybrid'}",
            extra={
                'llm_only_mode': self.llm_only_mode,
                'fallback_enabled': self.fallback_to_heuristic,
                'agent_configs_available': list(self.agent_configs.keys())
            }
        )
    
    def create_agents(self) -> Dict[str, Agent]:
        """
        Create agent set based on the current configuration mode.
        
        Returns:
            Dictionary mapping agent names to agent instances
        """
        if self.llm_only_mode:
            return self._create_llm_only_agents()
        else:
            return self._create_hybrid_agents()
    
    def _create_llm_only_agents(self) -> Dict[str, Agent]:
        """
        Create LLM-only agent set.
        
        Returns:
            Dictionary containing only LLM-enhanced agents
        """
        self.logger.info("Creating LLM-only agent set")
        
        agents = {}
        failed_agents = []
        
        try:
            # Create LLM-enhanced analyzer
            try:
                analyzer_config = self._get_agent_config('analyzer')
                agents['analyzer'] = LLMAnalyzerAgent(analyzer_config)
                self.logger.debug("Created LLMAnalyzerAgent")
            except Exception as e:
                self.logger.warning(f"Failed to create LLMAnalyzerAgent: {str(e)}")
                failed_agents.append(('analyzer', str(e)))
            
            # Create LLM-enhanced refiner
            try:
                refiner_config = self._get_agent_config('refiner')
                agents['refiner'] = LLMRefinerAgent(refiner_config)
                self.logger.debug("Created LLMRefinerAgent")
            except Exception as e:
                self.logger.warning(f"Failed to create LLMRefinerAgent: {str(e)}")
                failed_agents.append(('refiner', str(e)))
            
            # Create LLM-enhanced validator
            try:
                validator_config = self._get_agent_config('validator')
                agents['validator'] = LLMValidatorAgent(validator_config)
                self.logger.debug("Created LLMValidatorAgent")
            except Exception as e:
                self.logger.warning(f"Failed to create LLMValidatorAgent: {str(e)}")
                failed_agents.append(('validator', str(e)))
            
            # If some agents were created successfully, return them
            if agents:
                self.logger.info(
                    f"Successfully created {len(agents)} LLM-only agents",
                    extra={
                        'agent_types': list(agents.keys()),
                        'mode': 'llm_only',
                        'failed_agents': [name for name, _ in failed_agents]
                    }
                )
                
                # If fallback is enabled and some agents failed, create fallback agents for failed ones
                if failed_agents and self.fallback_to_heuristic:
                    self.logger.info(f"Creating fallback agents for {len(failed_agents)} failed LLM agents")
                    fallback_agents = self._create_fallback_agents_for_failed(failed_agents)
                    agents.update(fallback_agents)
                
                return agents
            
            # If no agents were created successfully, fall back to heuristic agents if enabled
            if self.fallback_to_heuristic:
                self.logger.warning("All LLM agents failed to create, falling back to heuristic agents")
                return self._create_heuristic_agents()
            else:
                raise Exception(f"Failed to create any LLM agents: {failed_agents}")
            
        except Exception as e:
            self.logger.error(
                f"Failed to create LLM-only agents: {str(e)}",
                extra={'error_type': type(e).__name__}
            )
            
            # If fallback is enabled and LLM agent creation fails, create heuristic agents
            if self.fallback_to_heuristic:
                self.logger.warning("Falling back to heuristic agents due to LLM agent creation failure")
                return self._create_heuristic_agents()
            else:
                raise
    
    def _create_hybrid_agents(self) -> Dict[str, Agent]:
        """
        Create hybrid agent set with both heuristic and LLM-enhanced agents.
        
        Returns:
            Dictionary containing both heuristic and LLM-enhanced agents
        """
        self.logger.info("Creating hybrid agent set")
        
        agents = {}
        
        try:
            # Create heuristic agents
            heuristic_agents = self._create_heuristic_agents()
            agents.update(heuristic_agents)
            
            # Create LLM-enhanced agents with different names to avoid conflicts
            llm_agents = self._create_llm_enhanced_agents()
            agents.update(llm_agents)
            
            self.logger.info(
                f"Successfully created {len(agents)} hybrid agents",
                extra={
                    'agent_types': list(agents.keys()),
                    'heuristic_count': len(heuristic_agents),
                    'llm_count': len(llm_agents),
                    'mode': 'hybrid'
                }
            )
            
            return agents
            
        except Exception as e:
            self.logger.error(
                f"Failed to create hybrid agents: {str(e)}",
                extra={'error_type': type(e).__name__}
            )
            raise
    
    def _create_heuristic_agents(self) -> Dict[str, Agent]:
        """
        Create heuristic agent set.
        
        Returns:
            Dictionary containing heuristic agents
        """
        self.logger.debug("Creating heuristic agents")
        
        agents = {}
        
        # Create heuristic analyzer
        analyzer_config = self._get_agent_config('analyzer')
        agents['analyzer'] = AnalyzerAgent(analyzer_config)
        
        # Create heuristic refiner
        refiner_config = self._get_agent_config('refiner')
        agents['refiner'] = RefinerAgent(refiner_config)
        
        # Create heuristic validator
        validator_config = self._get_agent_config('validator')
        agents['validator'] = ValidatorAgent(validator_config)
        
        self.logger.debug(f"Created {len(agents)} heuristic agents")
        
        return agents
    
    def _create_llm_enhanced_agents(self) -> Dict[str, Agent]:
        """
        Create LLM-enhanced agent set with prefixed names for hybrid mode.
        
        Returns:
            Dictionary containing LLM-enhanced agents with prefixed names
        """
        self.logger.debug("Creating LLM-enhanced agents")
        
        agents = {}
        
        # Create LLM-enhanced agents with 'llm_' prefix for hybrid mode
        analyzer_config = self._get_agent_config('analyzer')
        agents['llm_analyzer'] = LLMAnalyzerAgent(analyzer_config)
        
        refiner_config = self._get_agent_config('refiner')
        agents['llm_refiner'] = LLMRefinerAgent(refiner_config)
        
        validator_config = self._get_agent_config('validator')
        agents['llm_validator'] = LLMValidatorAgent(validator_config)
        
        self.logger.debug(f"Created {len(agents)} LLM-enhanced agents")
        
        return agents
    
    def _get_agent_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.
        
        Args:
            agent_type: Type of agent ('analyzer', 'refiner', 'validator')
            
        Returns:
            Configuration dictionary for the agent
        """
        agent_config = self.agent_configs.get(agent_type, {}).copy()
        
        # Add mode information to agent config
        agent_config['llm_only_mode'] = self.llm_only_mode
        agent_config['fallback_enabled'] = self.fallback_to_heuristic
        
        return agent_config
    
    def get_available_agents(self) -> List[str]:
        """
        Get list of available agent names for the current mode.
        
        Returns:
            List of agent names that will be created
        """
        if self.llm_only_mode:
            return ['analyzer', 'refiner', 'validator']
        else:
            return ['analyzer', 'refiner', 'validator', 'llm_analyzer', 'llm_refiner', 'llm_validator']
    
    def get_bypassed_agents(self) -> List[str]:
        """
        Get list of agent types that are bypassed in the current mode.
        
        Returns:
            List of agent types that are not used in current mode
        """
        if self.llm_only_mode:
            return ['heuristic_analyzer', 'heuristic_refiner', 'heuristic_validator']
        else:
            return []  # No agents are bypassed in hybrid mode
    
    def is_llm_only_mode(self) -> bool:
        """
        Check if factory is in LLM-only mode.
        
        Returns:
            True if in LLM-only mode, False otherwise
        """
        return self.llm_only_mode
    
    def is_hybrid_mode(self) -> bool:
        """
        Check if factory is in hybrid mode.
        
        Returns:
            True if in hybrid mode, False otherwise
        """
        return not self.llm_only_mode
    
    def get_mode_description(self) -> str:
        """
        Get human-readable description of current mode.
        
        Returns:
            String description of the current mode
        """
        if self.llm_only_mode:
            fallback_status = "with fallback" if self.fallback_to_heuristic else "without fallback"
            return f"LLM-only mode {fallback_status}"
        else:
            return "Hybrid mode (heuristic + LLM agents)"
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update factory configuration and reinitialize mode settings.
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update optimization configuration
        self.optimization_config = self.config.get('optimization', {})
        old_mode = self.llm_only_mode
        self.llm_only_mode = self.optimization_config.get('llm_only_mode', False)
        self.fallback_to_heuristic = self.optimization_config.get('fallback_to_heuristic', True)
        
        # Update agent configurations
        self.agent_configs = self.config.get('agents', {})
        
        # Log mode change if it occurred
        if old_mode != self.llm_only_mode:
            old_mode_str = 'llm_only' if old_mode else 'hybrid'
            new_mode_str = 'llm_only' if self.llm_only_mode else 'hybrid'
            
            self.logger.info(
                f"AgentFactory mode changed from {'LLM-only' if old_mode else 'Hybrid'} to {'LLM-only' if self.llm_only_mode else 'Hybrid'}",
                extra={
                    'old_mode': old_mode_str,
                    'new_mode': new_mode_str,
                    'fallback_enabled': self.fallback_to_heuristic,
                    'configuration_change': True
                }
            )
            
            # Log mode switch with orchestration logger
            orchestration_logger.log_mode_switch(
                old_mode=old_mode_str,
                new_mode=new_mode_str,
                trigger='configuration_update'
            )
            
            # Track mode switch with usage tracker
            mode_usage_tracker.track_mode_switch(
                old_mode=old_mode_str,
                new_mode=new_mode_str,
                trigger='configuration_update'
            )
    
    def _create_fallback_agents_for_failed(self, failed_agents: List[Tuple[str, str]]) -> Dict[str, Agent]:
        """
        Create fallback heuristic agents for failed LLM agents.
        
        Args:
            failed_agents: List of tuples containing (agent_name, error_message)
            
        Returns:
            Dictionary containing fallback heuristic agents
        """
        fallback_agents = {}
        
        for agent_name, error_message in failed_agents:
            try:
                agent_config = self._get_agent_config(agent_name)
                
                # Remove LLM-specific configuration for heuristic agent
                fallback_config = agent_config.copy()
                fallback_config.pop('model', None)
                fallback_config.pop('llm_model', None)
                fallback_config.pop('llm_temperature', None)
                fallback_config.pop('llm_max_tokens', None)
                fallback_config.pop('llm_timeout', None)
                
                # Add fallback metadata
                fallback_config['is_fallback'] = True
                fallback_config['llm_error'] = error_message
                
                if agent_name == 'analyzer':
                    fallback_agents[agent_name] = AnalyzerAgent(fallback_config)
                elif agent_name == 'refiner':
                    fallback_agents[agent_name] = RefinerAgent(fallback_config)
                elif agent_name == 'validator':
                    fallback_agents[agent_name] = ValidatorAgent(fallback_config)
                
                self.logger.info(f"Created fallback {agent_name} agent due to LLM failure: {error_message}")
                
            except Exception as e:
                self.logger.error(f"Failed to create fallback {agent_name} agent: {str(e)}")
        
        return fallback_agents
    
    def create_emergency_fallback_agents(self) -> Dict[str, Agent]:
        """
        Create emergency fallback agents when all LLM agents fail during runtime.
        
        Returns:
            Dictionary containing emergency fallback heuristic agents
        """
        self.logger.warning("Creating emergency fallback agents due to runtime LLM failures")
        
        emergency_agents = {}
        
        try:
            # Create basic heuristic agents with minimal configuration
            basic_config = {'is_emergency_fallback': True}
            
            emergency_agents['analyzer'] = AnalyzerAgent(basic_config)
            emergency_agents['refiner'] = RefinerAgent(basic_config)
            emergency_agents['validator'] = ValidatorAgent(basic_config)
            
            self.logger.info(f"Created {len(emergency_agents)} emergency fallback agents")
            
        except Exception as e:
            self.logger.error(f"Failed to create emergency fallback agents: {str(e)}")
        
        return emergency_agents
    
    def handle_llm_service_unavailable(self) -> Dict[str, Agent]:
        """
        Handle LLM service unavailability by creating appropriate fallback agents.
        
        Returns:
            Dictionary containing fallback agents
        """
        self.logger.warning("LLM service unavailable, handling fallback strategy")
        
        if self.fallback_to_heuristic:
            if self.llm_only_mode:
                # In LLM-only mode with fallback enabled, create heuristic agents
                self.logger.info("Creating heuristic fallback agents for LLM-only mode")
                return self._create_heuristic_agents()
            else:
                # In hybrid mode, return existing heuristic agents or create them
                return self._create_heuristic_agents()
        else:
            # No fallback enabled, return empty dict or raise exception
            self.logger.error("LLM service unavailable and fallback disabled")
            raise Exception("LLM service unavailable and fallback to heuristic agents is disabled")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration for agent creation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'mode': 'llm_only' if self.llm_only_mode else 'hybrid'
        }
        
        # Check if required agent configurations are present
        required_agents = ['analyzer', 'refiner', 'validator']
        for agent_type in required_agents:
            if agent_type not in self.agent_configs:
                validation_result['warnings'].append(
                    f"No configuration found for {agent_type} agent, using defaults"
                )
        
        # Validate LLM-only mode specific requirements
        if self.llm_only_mode:
            # Check if LLM model configurations are available
            for agent_type in required_agents:
                agent_config = self.agent_configs.get(agent_type, {})
                if 'model' not in agent_config:
                    validation_result['warnings'].append(
                        f"No LLM model specified for {agent_type} in LLM-only mode"
                    )
        
        # Check for conflicting configurations
        if self.llm_only_mode and not self.fallback_to_heuristic:
            validation_result['warnings'].append(
                "LLM-only mode without fallback may fail if LLM services are unavailable"
            )
        
        # Validate fallback configuration
        if self.llm_only_mode and self.fallback_to_heuristic:
            validation_result['warnings'].append(
                "LLM-only mode with fallback enabled - heuristic agents will be used if LLM fails"
            )
        
        return validation_result