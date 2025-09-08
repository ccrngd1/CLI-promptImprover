"""
Bedrock Executor for prompt execution against Amazon Bedrock models.

This module provides the BedrockExecutor class that handles all interactions
with the Amazon Bedrock API, including authentication, model configuration,
prompt execution, error handling, and rate limiting.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from models import ExecutionResult


@dataclass
class ModelConfig:
    """Configuration for Bedrock model execution."""
    
    model_id: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 250
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "topP": self.top_p,
        }
        
        # Add model-specific parameters
        if "claude" in self.model_id.lower():
            config["topK"] = self.top_k
        
        if self.stop_sequences:
            config["stopSequences"] = self.stop_sequences
            
        return config


class BedrockExecutor:
    """
    Handles execution of prompts against Amazon Bedrock models.
    
    Provides methods for model initialization, prompt execution with proper
    error handling, rate limiting, and support for multiple model types.
    """
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "name": "Claude 3 Sonnet",
            "provider": "anthropic",
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "name": "Claude 3 Haiku", 
            "provider": "anthropic",
            "input_cost_per_1k": 0.00025,
            "output_cost_per_1k": 0.00125
        },
        "amazon.titan-text-express-v1": {
            "name": "Titan Text Express",
            "provider": "amazon",
            "input_cost_per_1k": 0.0008,
            "output_cost_per_1k": 0.0016
        }
    }
    
    def __init__(self, region_name: str = "us-east-1", 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None):
        """
        Initialize the Bedrock executor.
        
        Args:
            region_name: AWS region for Bedrock service
            aws_access_key_id: AWS access key (optional, can use environment/IAM)
            aws_secret_access_key: AWS secret key (optional, can use environment/IAM)
            aws_session_token: AWS session token (optional, for temporary credentials)
        """
        self.region_name = region_name
        self.logger = logging.getLogger(__name__)
        
        # Configure boto3 with retry and timeout settings
        config = Config(
            region_name=region_name,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            read_timeout=60,
            connect_timeout=10
        )
        
        # Initialize Bedrock client
        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token
            
        try:
            session = boto3.Session(**session_kwargs)
            self.bedrock_client = session.client('bedrock-runtime', config=config)
            self.bedrock_info_client = session.client('bedrock', config=config)
            
            # Test authentication
            self._test_authentication()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
        
        # Rate limiting state
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
    def _test_authentication(self) -> None:
        """Test if authentication is working by listing available models."""
        try:
            self.bedrock_info_client.list_foundation_models()
            self.logger.info("Bedrock authentication successful")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['UnauthorizedOperation', 'AccessDenied']:
                raise ValueError(f"Authentication failed: {str(e)}")
            else:
                self.logger.warning(f"Authentication test inconclusive: {str(e)}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available Bedrock models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self.bedrock_info_client.list_foundation_models()
            models = []
            
            for model in response.get('modelSummaries', []):
                model_id = model.get('modelId', '')
                if model_id in self.SUPPORTED_MODELS:
                    model_info = {
                        'model_id': model_id,
                        'model_name': model.get('modelName', ''),
                        'provider_name': model.get('providerName', ''),
                        'supported': True,
                        **self.SUPPORTED_MODELS[model_id]
                    }
                    models.append(model_info)
            
            return models
            
        except ClientError as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            # Return supported models list as fallback
            return [
                {
                    'model_id': model_id,
                    'supported': True,
                    **info
                }
                for model_id, info in self.SUPPORTED_MODELS.items()
            ]
    
    def _handle_rate_limits(self) -> None:
        """Implement rate limiting to avoid API throttling."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _prepare_request_body(self, prompt: str, model_config: ModelConfig) -> str:
        """
        Prepare the request body based on the model provider.
        
        Args:
            prompt: The prompt text to execute
            model_config: Model configuration parameters
            
        Returns:
            JSON string of the request body
        """
        if "anthropic" in model_config.model_id:
            # Claude models format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "top_k": model_config.top_k,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            if model_config.stop_sequences:
                body["stop_sequences"] = model_config.stop_sequences
                
        elif "amazon.titan" in model_config.model_id:
            # Titan models format
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": model_config.max_tokens,
                    "temperature": model_config.temperature,
                    "topP": model_config.top_p,
                    "stopSequences": model_config.stop_sequences
                }
            }
        else:
            raise ValueError(f"Unsupported model: {model_config.model_id}")
        
        return json.dumps(body)
    
    def _parse_response(self, response_body: str, model_id: str) -> Dict[str, Any]:
        """
        Parse the response body based on the model provider.
        
        Args:
            response_body: Raw response body from Bedrock
            model_id: The model ID used for the request
            
        Returns:
            Parsed response with text and metadata
        """
        try:
            response_data = json.loads(response_body)
            
            if "anthropic" in model_id:
                # Claude response format
                content = response_data.get('content', [])
                if content and len(content) > 0:
                    response_text = content[0].get('text', '')
                else:
                    response_text = ''
                
                usage = response_data.get('usage', {})
                token_usage = {
                    'input_tokens': usage.get('input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0)
                }
                
            elif "amazon.titan" in model_id:
                # Titan response format
                results = response_data.get('results', [])
                if results and len(results) > 0:
                    response_text = results[0].get('outputText', '')
                else:
                    response_text = ''
                
                token_usage = {
                    'input_tokens': response_data.get('inputTextTokenCount', 0),
                    'output_tokens': response_data.get('results', [{}])[0].get('tokenCount', 0) if results else 0
                }
            else:
                raise ValueError(f"Unsupported model for response parsing: {model_id}")
            
            return {
                'response_text': response_text,
                'token_usage': token_usage,
                'raw_response': response_data
            }
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.logger.error(f"Failed to parse response: {str(e)}")
            return {
                'response_text': '',
                'token_usage': {'input_tokens': 0, 'output_tokens': 0},
                'raw_response': {}
            }
    
    def execute_prompt(self, prompt: str, model_config: ModelConfig) -> ExecutionResult:
        """
        Execute a prompt against a Bedrock model.
        
        Args:
            prompt: The prompt text to execute
            model_config: Configuration for the model execution
            
        Returns:
            ExecutionResult containing the response and metadata
        """
        if not prompt or not prompt.strip():
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text="",
                execution_time=0.0,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message="Empty prompt provided"
            )
        
        if model_config.model_id not in self.SUPPORTED_MODELS:
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text="",
                execution_time=0.0,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message=f"Unsupported model: {model_config.model_id}"
            )
        
        start_time = time.time()
        
        try:
            # Apply rate limiting
            self._handle_rate_limits()
            
            # Prepare request
            request_body = self._prepare_request_body(prompt, model_config)
            
            # Execute request
            response = self.bedrock_client.invoke_model(
                modelId=model_config.model_id,
                body=request_body,
                contentType='application/json',
                accept='application/json'
            )
            
            execution_time = time.time() - start_time
            
            # Parse response
            response_body = response['body'].read().decode('utf-8')
            parsed_response = self._parse_response(response_body, model_config.model_id)
            
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text=parsed_response['response_text'],
                execution_time=execution_time,
                token_usage=parsed_response['token_usage'],
                success=True,
                metadata={
                    'model_config': model_config.to_dict(),
                    'raw_response': parsed_response['raw_response']
                }
            )
            
        except ClientError as e:
            execution_time = time.time() - start_time
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            self.logger.error(f"Bedrock API error: {error_code} - {error_message}")
            
            # Handle specific error types
            if error_code == 'ThrottlingException':
                error_message = f"Rate limit exceeded: {error_message}"
            elif error_code == 'ValidationException':
                error_message = f"Invalid request: {error_message}"
            elif error_code == 'AccessDeniedException':
                error_message = f"Access denied: {error_message}"
            elif error_code == 'ModelNotReadyException':
                error_message = f"Model not ready: {error_message}"
            
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text="",
                execution_time=execution_time,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message=error_message,
                metadata={'error_code': error_code}
            )
            
        except BotoCoreError as e:
            execution_time = time.time() - start_time
            error_message = f"AWS SDK error: {str(e)}"
            self.logger.error(error_message)
            
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text="",
                execution_time=execution_time,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message=error_message
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Unexpected error: {str(e)}"
            self.logger.error(error_message)
            
            return ExecutionResult(
                model_name=model_config.model_id,
                response_text="",
                execution_time=execution_time,
                token_usage={'input_tokens': 0, 'output_tokens': 0},
                success=False,
                error_message=error_message
            )
    
    def validate_model_config(self, model_config: ModelConfig) -> bool:
        """
        Validate model configuration parameters.
        
        Args:
            model_config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not model_config.model_id:
            return False
        
        if model_config.model_id not in self.SUPPORTED_MODELS:
            return False
        
        if not (1 <= model_config.max_tokens <= 4096):
            return False
        
        if not (0.0 <= model_config.temperature <= 1.0):
            return False
        
        if not (0.0 <= model_config.top_p <= 1.0):
            return False
        
        if not (1 <= model_config.top_k <= 500):
            return False
        
        return True