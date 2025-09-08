"""
Unit tests for the BedrockExecutor class.

Tests cover AWS SDK integration, prompt execution, error handling,
rate limiting, and model configuration with mocked AWS responses.
"""

import json
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

from bedrock.executor import BedrockExecutor, ModelConfig
from models import ExecutionResult


class TestModelConfig:
    """Test cases for ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=500,
            temperature=0.8
        )
        
        assert config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.max_tokens == 500
        assert config.temperature == 0.8
        assert config.top_p == 0.9  # default
        assert config.stop_sequences == []  # default
    
    def test_model_config_to_dict(self):
        """Test ModelConfig to_dict conversion."""
        config = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7,
            stop_sequences=["Human:", "Assistant:"]
        )
        
        result = config.to_dict()
        
        assert result["maxTokens"] == 1000
        assert result["temperature"] == 0.7
        assert result["topP"] == 0.9
        assert result["topK"] == 250  # Claude model includes topK
        assert result["stopSequences"] == ["Human:", "Assistant:"]
    
    def test_model_config_titan_format(self):
        """Test ModelConfig for Titan models."""
        config = ModelConfig(
            model_id="amazon.titan-text-express-v1",
            max_tokens=800
        )
        
        result = config.to_dict()
        
        assert result["maxTokens"] == 800
        assert "topK" not in result  # Titan models don't use topK


class TestBedrockExecutor:
    """Test cases for BedrockExecutor class."""
    
    @patch('boto3.Session')
    def test_executor_initialization_success(self, mock_session):
        """Test successful BedrockExecutor initialization."""
        # Mock the boto3 session and clients
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        # Mock successful authentication test
        mock_bedrock_info_client.list_foundation_models.return_value = {
            'modelSummaries': []
        }
        
        executor = BedrockExecutor(region_name="us-west-2")
        
        assert executor.region_name == "us-west-2"
        assert executor.bedrock_client == mock_bedrock_client
        assert executor.bedrock_info_client == mock_bedrock_info_client
        
        # Verify client creation calls
        calls = mock_session_instance.client.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 'bedrock-runtime'
        assert calls[1][0][0] == 'bedrock'
    
    @patch('boto3.Session')
    def test_executor_initialization_with_credentials(self, mock_session):
        """Test BedrockExecutor initialization with explicit credentials."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {
            'modelSummaries': []
        }
        
        executor = BedrockExecutor(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
        
        # Verify session was created with credentials
        mock_session.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret"
        )
    
    @patch('boto3.Session')
    def test_executor_authentication_failure(self, mock_session):
        """Test BedrockExecutor initialization with authentication failure."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        # Mock authentication failure
        mock_bedrock_info_client.list_foundation_models.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='ListFoundationModels'
        )
        
        with pytest.raises(ValueError, match="Authentication failed"):
            BedrockExecutor()
    
    @patch('boto3.Session')
    def test_get_available_models_success(self, mock_session):
        """Test successful retrieval of available models."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        # Mock authentication success
        mock_bedrock_info_client.list_foundation_models.return_value = {
            'modelSummaries': [
                {
                    'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
                    'modelName': 'Claude 3 Sonnet',
                    'providerName': 'Anthropic'
                },
                {
                    'modelId': 'unsupported.model',
                    'modelName': 'Unsupported Model',
                    'providerName': 'Unknown'
                }
            ]
        }
        
        executor = BedrockExecutor()
        models = executor.get_available_models()
        
        # Should only return supported models
        assert len(models) == 1
        assert models[0]['model_id'] == 'anthropic.claude-3-sonnet-20240229-v1:0'
        assert models[0]['supported'] is True
        assert 'name' in models[0]
    
    @patch('boto3.Session')
    def test_get_available_models_api_failure(self, mock_session):
        """Test get_available_models with API failure fallback."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        # Mock authentication success but list_foundation_models failure
        mock_bedrock_info_client.list_foundation_models.side_effect = [
            {'modelSummaries': []},  # For authentication test
            ClientError(
                error_response={'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
                operation_name='ListFoundationModels'
            )
        ]
        
        executor = BedrockExecutor()
        models = executor.get_available_models()
        
        # Should return fallback list of supported models
        assert len(models) > 0
        assert all(model['supported'] for model in models)
    
    @patch('boto3.Session')
    def test_validate_model_config(self, mock_session):
        """Test model configuration validation."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        # Valid configuration
        valid_config = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=1000,
            temperature=0.7
        )
        assert executor.validate_model_config(valid_config) is True
        
        # Invalid model ID
        invalid_model = ModelConfig(model_id="invalid.model")
        assert executor.validate_model_config(invalid_model) is False
        
        # Invalid max_tokens
        invalid_tokens = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=5000  # Too high
        )
        assert executor.validate_model_config(invalid_tokens) is False
        
        # Invalid temperature
        invalid_temp = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=1.5  # Too high
        )
        assert executor.validate_model_config(invalid_temp) is False
    
    @patch('boto3.Session')
    def test_execute_prompt_success_claude(self, mock_session):
        """Test successful prompt execution with Claude model."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        # Mock successful API response
        mock_response_body = {
            'content': [{'text': 'This is a test response from Claude.'}],
            'usage': {'input_tokens': 10, 'output_tokens': 8}
        }
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode('utf-8')
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        executor = BedrockExecutor()
        
        config = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=100
        )
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is True
        assert result.response_text == "This is a test response from Claude."
        assert result.token_usage['input_tokens'] == 10
        assert result.token_usage['output_tokens'] == 8
        assert result.execution_time > 0
        assert result.model_name == config.model_id
        
        # Verify API call
        mock_bedrock_client.invoke_model.assert_called_once()
        call_args = mock_bedrock_client.invoke_model.call_args
        assert call_args[1]['modelId'] == config.model_id
        assert call_args[1]['contentType'] == 'application/json'
    
    @patch('boto3.Session')
    def test_execute_prompt_success_titan(self, mock_session):
        """Test successful prompt execution with Titan model."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        # Mock successful Titan API response
        mock_response_body = {
            'results': [{'outputText': 'This is a test response from Titan.', 'tokenCount': 8}],
            'inputTextTokenCount': 10
        }
        
        mock_response = {
            'body': Mock()
        }
        mock_response['body'].read.return_value = json.dumps(mock_response_body).encode('utf-8')
        mock_bedrock_client.invoke_model.return_value = mock_response
        
        executor = BedrockExecutor()
        
        config = ModelConfig(
            model_id="amazon.titan-text-express-v1",
            max_tokens=100
        )
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is True
        assert result.response_text == "This is a test response from Titan."
        assert result.token_usage['input_tokens'] == 10
        assert result.token_usage['output_tokens'] == 8
    
    @patch('boto3.Session')
    def test_execute_prompt_empty_prompt(self, mock_session):
        """Test prompt execution with empty prompt."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        result = executor.execute_prompt("", config)
        
        assert result.success is False
        assert "Empty prompt provided" in result.error_message
        assert result.execution_time == 0.0
    
    @patch('boto3.Session')
    def test_execute_prompt_unsupported_model(self, mock_session):
        """Test prompt execution with unsupported model."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        config = ModelConfig(model_id="unsupported.model")
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is False
        assert "Unsupported model" in result.error_message
    
    @patch('boto3.Session')
    def test_execute_prompt_throttling_error(self, mock_session):
        """Test prompt execution with throttling error."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        # Mock throttling error
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            error_response={
                'Error': {
                    'Code': 'ThrottlingException',
                    'Message': 'Rate exceeded'
                }
            },
            operation_name='InvokeModel'
        )
        
        executor = BedrockExecutor()
        
        config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is False
        assert "Rate limit exceeded" in result.error_message
        assert result.metadata['error_code'] == 'ThrottlingException'
    
    @patch('boto3.Session')
    def test_execute_prompt_validation_error(self, mock_session):
        """Test prompt execution with validation error."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        # Mock validation error
        mock_bedrock_client.invoke_model.side_effect = ClientError(
            error_response={
                'Error': {
                    'Code': 'ValidationException',
                    'Message': 'Invalid request parameters'
                }
            },
            operation_name='InvokeModel'
        )
        
        executor = BedrockExecutor()
        
        config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is False
        assert "Invalid request" in result.error_message
    
    @patch('boto3.Session')
    def test_execute_prompt_boto_core_error(self, mock_session):
        """Test prompt execution with BotoCoreError."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        # Mock BotoCoreError
        mock_bedrock_client.invoke_model.side_effect = BotoCoreError()
        
        executor = BedrockExecutor()
        
        config = ModelConfig(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
        
        result = executor.execute_prompt("Test prompt", config)
        
        assert result.success is False
        assert "AWS SDK error" in result.error_message
    
    @patch('boto3.Session')
    @patch('time.sleep')
    def test_rate_limiting(self, mock_sleep, mock_session):
        """Test rate limiting functionality."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        # Simulate rapid successive calls
        executor._last_request_time = time.time()
        executor._handle_rate_limits()
        
        # Should call sleep to enforce rate limiting
        mock_sleep.assert_called_once()
        assert mock_sleep.call_args[0][0] > 0  # Sleep time should be positive
    
    @patch('boto3.Session')
    def test_prepare_request_body_claude(self, mock_session):
        """Test request body preparation for Claude models."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        config = ModelConfig(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=500,
            temperature=0.8,
            stop_sequences=["Human:"]
        )
        
        body_str = executor._prepare_request_body("Test prompt", config)
        body = json.loads(body_str)
        
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 500
        assert body["temperature"] == 0.8
        assert body["stop_sequences"] == ["Human:"]
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Test prompt"
    
    @patch('boto3.Session')
    def test_prepare_request_body_titan(self, mock_session):
        """Test request body preparation for Titan models."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        config = ModelConfig(
            model_id="amazon.titan-text-express-v1",
            max_tokens=800,
            temperature=0.6
        )
        
        body_str = executor._prepare_request_body("Test prompt", config)
        body = json.loads(body_str)
        
        assert body["inputText"] == "Test prompt"
        assert body["textGenerationConfig"]["maxTokenCount"] == 800
        assert body["textGenerationConfig"]["temperature"] == 0.6
    
    @patch('boto3.Session')
    def test_parse_response_claude(self, mock_session):
        """Test response parsing for Claude models."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        response_body = json.dumps({
            'content': [{'text': 'Claude response'}],
            'usage': {'input_tokens': 15, 'output_tokens': 12}
        })
        
        result = executor._parse_response(response_body, "anthropic.claude-3-sonnet-20240229-v1:0")
        
        assert result['response_text'] == 'Claude response'
        assert result['token_usage']['input_tokens'] == 15
        assert result['token_usage']['output_tokens'] == 12
    
    @patch('boto3.Session')
    def test_parse_response_titan(self, mock_session):
        """Test response parsing for Titan models."""
        mock_bedrock_client = Mock()
        mock_bedrock_info_client = Mock()
        mock_session_instance = Mock()
        mock_session_instance.client.side_effect = [mock_bedrock_client, mock_bedrock_info_client]
        mock_session.return_value = mock_session_instance
        
        mock_bedrock_info_client.list_foundation_models.return_value = {'modelSummaries': []}
        
        executor = BedrockExecutor()
        
        response_body = json.dumps({
            'results': [{'outputText': 'Titan response', 'tokenCount': 10}],
            'inputTextTokenCount': 20
        })
        
        result = executor._parse_response(response_body, "amazon.titan-text-express-v1")
        
        assert result['response_text'] == 'Titan response'
        assert result['token_usage']['input_tokens'] == 20
        assert result['token_usage']['output_tokens'] == 10


if __name__ == "__main__":
    pytest.main([__file__])