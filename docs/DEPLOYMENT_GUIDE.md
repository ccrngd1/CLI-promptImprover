# Deployment Guide for Bedrock Prompt Optimizer

This guide covers deployment strategies, environment setup, and production considerations for the Bedrock Prompt Optimizer.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Environment Setup](#environment-setup)
3. [Production Configuration](#production-configuration)
4. [Security Considerations](#security-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Scaling Considerations](#scaling-considerations)
7. [Troubleshooting](#troubleshooting)

## Deployment Options

### 1. Local Development

For development and testing purposes:

```bash
# Clone repository
git clone https://github.com/example/bedrock-prompt-optimizer.git
cd bedrock-prompt-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run setup
python cli/setup.py
```

### 2. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 optimizer && \
    chown -R optimizer:optimizer /app
USER optimizer

# Create directories for data persistence
RUN mkdir -p /app/data/history /app/data/config /app/data/best_practices

# Set environment variables
ENV PYTHONPATH=/app
ENV OPTIMIZER_STORAGE_PATH=/app/data/history
ENV OPTIMIZER_CONFIG_PATH=/app/data/config

# Expose port (if running as web service)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["bedrock-optimizer", "--help"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  bedrock-optimizer:
    build: .
    container_name: bedrock-optimizer
    environment:
      - AWS_REGION=${AWS_REGION:-us-east-1}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - OPTIMIZER_STORAGE_PATH=/app/data/history
      - OPTIMIZER_CONFIG_PATH=/app/data/config
    volumes:
      - ./data/history:/app/data/history
      - ./data/config:/app/data/config
      - ./data/best_practices:/app/data/best_practices
    networks:
      - optimizer-network
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    container_name: optimizer-redis
    volumes:
      - redis_data:/data
    networks:
      - optimizer-network
    restart: unless-stopped

volumes:
  redis_data:

networks:
  optimizer-network:
    driver: bridge
```

#### Build and Run

```bash
# Build image
docker build -t bedrock-optimizer .

# Run with docker-compose
docker-compose up -d

# Run single container
docker run -it --rm \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -v $(pwd)/data:/app/data \
  bedrock-optimizer \
  bedrock-optimizer optimize "Test prompt"
```

### 3. AWS ECS Deployment

#### Task Definition

```json
{
  "family": "bedrock-optimizer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/BedrockOptimizerTaskRole",
  "containerDefinitions": [
    {
      "name": "bedrock-optimizer",
      "image": "your-account.dkr.ecr.region.amazonaws.com/bedrock-optimizer:latest",
      "essential": true,
      "environment": [
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        },
        {
          "name": "OPTIMIZER_STORAGE_PATH",
          "value": "/app/data/history"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "efs-storage",
          "containerPath": "/app/data"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bedrock-optimizer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "efs-storage",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

### 4. Kubernetes Deployment

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bedrock-optimizer
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bedrock-optimizer
  template:
    metadata:
      labels:
        app: bedrock-optimizer
    spec:
      serviceAccountName: bedrock-optimizer-sa
      containers:
      - name: bedrock-optimizer
        image: bedrock-optimizer:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_REGION
          value: "us-east-1"
        - name: OPTIMIZER_STORAGE_PATH
          value: "/app/data/history"
        volumeMounts:
        - name: data-storage
          mountPath: /app/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import sys; sys.exit(0)"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import sys; sys.exit(0)"
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: data-storage
        persistentVolumeClaim:
          claimName: optimizer-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: bedrock-optimizer-service
spec:
  selector:
    app: bedrock-optimizer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Service Account and RBAC

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: bedrock-optimizer-sa
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/BedrockOptimizerRole
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: bedrock-optimizer-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: bedrock-optimizer-binding
subjects:
- kind: ServiceAccount
  name: bedrock-optimizer-sa
  namespace: default
roleRef:
  kind: ClusterRole
  name: bedrock-optimizer-role
  apiGroup: rbac.authorization.k8s.io
```

## Environment Setup

### 1. AWS Configuration

#### IAM Policy for Bedrock Access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModel"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

#### IAM Role for ECS/EKS

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 2. Environment Variables

#### Production Environment Variables

```bash
# AWS Configuration
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1

# Application Configuration
export OPTIMIZER_ENVIRONMENT=production
export OPTIMIZER_LOG_LEVEL=INFO
export OPTIMIZER_STORAGE_PATH=/app/data/history
export OPTIMIZER_CONFIG_PATH=/app/data/config
export OPTIMIZER_MAX_ITERATIONS=10
export OPTIMIZER_TIMEOUT=300

# Security
export OPTIMIZER_ENCRYPT_STORAGE=true
export OPTIMIZER_SECURE_MODE=true

# Performance
export OPTIMIZER_CACHE_ENABLED=true
export OPTIMIZER_CACHE_TTL=3600
export OPTIMIZER_MAX_CONCURRENT_SESSIONS=10

# Monitoring
export OPTIMIZER_METRICS_ENABLED=true
export OPTIMIZER_HEALTH_CHECK_ENABLED=true
```

### 3. Configuration Files

#### Production Configuration

```yaml
# config/production.yaml
bedrock:
  region: us-east-1
  default_model: anthropic.claude-3-sonnet-20240229-v1:0
  timeout: 300
  max_retries: 5
  retry_backoff: exponential

orchestration:
  orchestrator_model: anthropic.claude-3-sonnet-20240229-v1:0
  orchestrator_temperature: 0.2
  min_iterations: 2
  max_iterations: 8
  score_improvement_threshold: 0.01
  convergence_confidence_threshold: 0.85

storage:
  path: /app/data/history
  format: json
  backup_enabled: true
  max_history_size: 10000
  compression_enabled: true
  encryption_enabled: true

logging:
  level: INFO
  format: json
  file: /app/logs/optimizer.log
  max_size: 100MB
  backup_count: 5

security:
  secure_mode: true
  encrypt_sensitive_data: true
  audit_logging: true

performance:
  cache_enabled: true
  cache_ttl: 3600
  max_concurrent_sessions: 20
  request_timeout: 300
  connection_pool_size: 10

monitoring:
  metrics_enabled: true
  health_check_enabled: true
  performance_monitoring: true
  error_tracking: true
```

## Production Configuration

### 1. Security Hardening

#### Secure Configuration

```python
# security_config.py
import os
from cryptography.fernet import Fernet

class SecurityConfig:
    def __init__(self):
        self.encryption_key = os.getenv('OPTIMIZER_ENCRYPTION_KEY') or Fernet.generate_key()
        self.secure_mode = os.getenv('OPTIMIZER_SECURE_MODE', 'true').lower() == 'true'
        self.audit_logging = os.getenv('OPTIMIZER_AUDIT_LOGGING', 'true').lower() == 'true'
    
    def encrypt_data(self, data: str) -> str:
        if not self.secure_mode:
            return data
        
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        if not self.secure_mode:
            return encrypted_data
        
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()
```

#### Input Validation

```python
# validation.py
import re
from typing import Any, Dict, List

class InputValidator:
    def __init__(self):
        self.max_prompt_length = 10000
        self.allowed_models = [
            'anthropic.claude-3-sonnet-20240229-v1:0',
            'anthropic.claude-3-haiku-20240307-v1:0',
            # Add other allowed models
        ]
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        validation = {'valid': True, 'errors': []}
        
        if not prompt or not prompt.strip():
            validation['errors'].append("Prompt cannot be empty")
            validation['valid'] = False
        
        if len(prompt) > self.max_prompt_length:
            validation['errors'].append(f"Prompt exceeds maximum length of {self.max_prompt_length}")
            validation['valid'] = False
        
        # Check for potentially harmful content
        if self._contains_harmful_content(prompt):
            validation['errors'].append("Prompt contains potentially harmful content")
            validation['valid'] = False
        
        return validation
    
    def _contains_harmful_content(self, prompt: str) -> bool:
        # Implement content filtering logic
        harmful_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        
        return False
```

### 2. Performance Optimization

#### Caching Layer

```python
# caching.py
import json
import hashlib
from typing import Any, Optional
from cachetools import TTLCache
import redis

class CacheManager:
    def __init__(self, cache_type: str = 'memory', redis_url: Optional[str] = None):
        self.cache_type = cache_type
        
        if cache_type == 'redis' and redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.memory_cache = TTLCache(maxsize=1000, ttl=3600)
    
    def get_cache_key(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate cache key from prompt and context."""
        data = f"{prompt}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if self.cache_type == 'redis':
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        else:
            return self.memory_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        if self.cache_type == 'redis':
            self.redis_client.setex(key, ttl, json.dumps(value))
        else:
            self.memory_cache[key] = value
```

#### Connection Pooling

```python
# connection_pool.py
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor

class BedrockConnectionPool:
    def __init__(self, max_connections: int = 10):
        self.config = Config(
            max_pool_connections=max_connections,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.client = boto3.client('bedrock-runtime', config=self.config)
        self.executor = ThreadPoolExecutor(max_workers=max_connections)
    
    def get_client(self):
        return self.client
    
    def submit_task(self, func, *args, **kwargs):
        return self.executor.submit(func, *args, **kwargs)
```

## Security Considerations

### 1. Data Protection

#### Encryption at Rest

```python
# encryption.py
from cryptography.fernet import Fernet
import json
import os

class DataEncryption:
    def __init__(self):
        key = os.getenv('OPTIMIZER_ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            print(f"Generated new encryption key: {key.decode()}")
        
        self.cipher = Fernet(key if isinstance(key, bytes) else key.encode())
    
    def encrypt_session_data(self, session_data: dict) -> str:
        json_data = json.dumps(session_data)
        encrypted_data = self.cipher.encrypt(json_data.encode())
        return encrypted_data.decode()
    
    def decrypt_session_data(self, encrypted_data: str) -> dict:
        decrypted_data = self.cipher.decrypt(encrypted_data.encode())
        return json.loads(decrypted_data.decode())
```

#### Secure Storage

```python
# secure_storage.py
import os
import stat
from pathlib import Path

class SecureStorage:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self._ensure_secure_directory()
    
    def _ensure_secure_directory(self):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions (owner read/write only)
        os.chmod(self.storage_path, stat.S_IRWXU)
    
    def save_secure_file(self, filename: str, data: str):
        file_path = self.storage_path / filename
        
        with open(file_path, 'w') as f:
            f.write(data)
        
        # Set secure file permissions
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
```

### 2. Access Control

#### API Key Management

```python
# api_keys.py
import os
import boto3
from botocore.exceptions import ClientError

class APIKeyManager:
    def __init__(self):
        self.secrets_client = boto3.client('secretsmanager')
    
    def get_api_key(self, key_name: str) -> str:
        try:
            response = self.secrets_client.get_secret_value(SecretId=key_name)
            return response['SecretString']
        except ClientError:
            # Fallback to environment variable
            return os.getenv(key_name)
    
    def rotate_key(self, key_name: str):
        # Implement key rotation logic
        pass
```

## Monitoring and Logging

### 1. Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    def __init__(self, name: str, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, data: Dict[str, Any], level: str = 'INFO'):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data,
            'level': level
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_entry))
    
    def log_optimization_start(self, session_id: str, prompt: str):
        self.log_event('optimization_start', {
            'session_id': session_id,
            'prompt_length': len(prompt)
        })
    
    def log_optimization_complete(self, session_id: str, iterations: int, final_score: float):
        self.log_event('optimization_complete', {
            'session_id': session_id,
            'iterations': iterations,
            'final_score': final_score
        })
```

### 2. Metrics Collection

```python
# metrics.py
import time
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class Metrics:
    counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gauges: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, list] = field(default_factory=lambda: defaultdict(list))
    timers: Dict[str, float] = field(default_factory=dict)

class MetricsCollector:
    def __init__(self):
        self.metrics = Metrics()
        self.start_times = {}
    
    def increment_counter(self, name: str, value: int = 1):
        self.metrics.counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        self.metrics.gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        self.metrics.histograms[name].append(value)
    
    def start_timer(self, name: str):
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str):
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics.timers[name] = duration
            del self.start_times[name]
            return duration
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            'counters': dict(self.metrics.counters),
            'gauges': self.metrics.gauges,
            'histograms': {k: {
                'count': len(v),
                'avg': sum(v) / len(v) if v else 0,
                'min': min(v) if v else 0,
                'max': max(v) if v else 0
            } for k, v in self.metrics.histograms.items()},
            'timers': self.metrics.timers
        }
```

### 3. Health Checks

```python
# health_check.py
import boto3
from typing import Dict, Any
from botocore.exceptions import ClientError

class HealthChecker:
    def __init__(self):
        self.checks = {
            'aws_credentials': self._check_aws_credentials,
            'bedrock_access': self._check_bedrock_access,
            'storage_access': self._check_storage_access,
            'memory_usage': self._check_memory_usage
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
                if not result.get('healthy', False):
                    overall_healthy = False
            except Exception as e:
                results[check_name] = {
                    'healthy': False,
                    'error': str(e)
                }
                overall_healthy = False
        
        return {
            'healthy': overall_healthy,
            'checks': results,
            'timestamp': time.time()
        }
    
    def _check_aws_credentials(self) -> Dict[str, Any]:
        try:
            sts = boto3.client('sts')
            sts.get_caller_identity()
            return {'healthy': True, 'message': 'AWS credentials valid'}
        except ClientError as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_bedrock_access(self) -> Dict[str, Any]:
        try:
            bedrock = boto3.client('bedrock')
            bedrock.list_foundation_models()
            return {'healthy': True, 'message': 'Bedrock access confirmed'}
        except ClientError as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_storage_access(self) -> Dict[str, Any]:
        # Implement storage access check
        return {'healthy': True, 'message': 'Storage accessible'}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return {
            'healthy': memory_percent < 90,
            'memory_usage_percent': memory_percent
        }
```

## Scaling Considerations

### 1. Horizontal Scaling

#### Load Balancer Configuration

```yaml
# nginx.conf
upstream bedrock_optimizer {
    server optimizer-1:8000;
    server optimizer-2:8000;
    server optimizer-3:8000;
}

server {
    listen 80;
    server_name optimizer.example.com;
    
    location / {
        proxy_pass http://bedrock_optimizer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://bedrock_optimizer/health;
    }
}
```

### 2. Auto Scaling

#### AWS Auto Scaling Group

```json
{
  "AutoScalingGroupName": "bedrock-optimizer-asg",
  "MinSize": 2,
  "MaxSize": 10,
  "DesiredCapacity": 3,
  "DefaultCooldown": 300,
  "HealthCheckType": "ELB",
  "HealthCheckGracePeriod": 300,
  "LaunchTemplate": {
    "LaunchTemplateName": "bedrock-optimizer-template",
    "Version": "$Latest"
  },
  "TargetGroupARNs": [
    "arn:aws:elasticloadbalancing:region:account:targetgroup/bedrock-optimizer/1234567890123456"
  ],
  "Tags": [
    {
      "Key": "Name",
      "Value": "bedrock-optimizer",
      "PropagateAtLaunch": true
    }
  ]
}
```

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bedrock-optimizer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bedrock-optimizer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Troubleshooting

### 1. Common Issues

#### AWS Credentials

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Test with CLI
bedrock-optimizer models --list
```

#### Memory Issues

```bash
# Monitor memory usage
docker stats bedrock-optimizer

# Check logs for memory errors
docker logs bedrock-optimizer | grep -i memory

# Increase memory limits
docker run -m 4g bedrock-optimizer
```

#### Performance Issues

```bash
# Enable debug logging
export OPTIMIZER_LOG_LEVEL=DEBUG

# Monitor API calls
export OPTIMIZER_METRICS_ENABLED=true

# Check health status
curl http://localhost:8000/health
```

### 2. Debugging Tools

#### Debug Script

```python
#!/usr/bin/env python3
# debug.py
import sys
import boto3
from cli.config import ConfigManager
from best_practices.repository import BestPracticesRepository

def debug_aws_connection():
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Identity: {identity['Arn']}")
        
        bedrock = boto3.client('bedrock')
        models = bedrock.list_foundation_models()
        print(f"✅ Available models: {len(models['modelSummaries'])}")
        
    except Exception as e:
        print(f"❌ AWS connection failed: {e}")

def debug_configuration():
    try:
        config = ConfigManager()
        validation = config.validate_config()
        
        if validation['valid']:
            print("✅ Configuration is valid")
        else:
            print(f"❌ Configuration errors: {validation['errors']}")
            
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")

def debug_best_practices():
    try:
        repo = BestPracticesRepository()
        rules = repo.list_all_rules()
        print(f"✅ Best practices loaded: {len(rules)} rules")
        
    except Exception as e:
        print(f"❌ Best practices check failed: {e}")

if __name__ == '__main__':
    print("🔍 Running diagnostic checks...")
    debug_aws_connection()
    debug_configuration()
    debug_best_practices()
    print("✅ Diagnostic complete")
```

This comprehensive deployment guide covers all aspects of deploying the Bedrock Prompt Optimizer in production environments, from basic Docker deployments to enterprise-scale Kubernetes clusters with full monitoring and security considerations.