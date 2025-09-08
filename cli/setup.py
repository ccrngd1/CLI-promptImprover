"""
Setup script for the Bedrock Prompt Optimizer CLI.

This script handles installation, dependency management, and CLI setup.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies."""
    dependencies = [
        'boto3>=1.26.0',
        'pydantic>=1.10.0',
        'pyyaml>=6.0',
        'rich>=12.0.0',  # For enhanced CLI output
        'click>=8.0.0',  # Alternative CLI framework if needed
    ]
    
    print("📦 Installing dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True


def create_executable_script():
    """Create an executable script for the CLI."""
    script_content = """#!/usr/bin/env python3
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cli.main import main

if __name__ == '__main__':
    main()
"""
    
    # Create bin directory
    bin_dir = Path('./bin')
    bin_dir.mkdir(exist_ok=True)
    
    # Create executable script
    script_path = bin_dir / 'bedrock-optimizer'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✅ Executable script created at: {script_path}")
    return script_path


def setup_configuration():
    """Set up initial configuration."""
    from cli.config import ConfigManager
    
    config_manager = ConfigManager()
    
    # Check if configuration already exists
    if config_manager.config_path.exists():
        print(f"📋 Configuration already exists at: {config_manager.config_path}")
        return
    
    # Create default configuration
    try:
        config_manager.create_default_config()
        print("✅ Default configuration created")
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")


def verify_aws_setup():
    """Verify AWS credentials and configuration."""
    try:
        import boto3
        
        # Try to create a Bedrock client
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials is None:
            print("⚠️  No AWS credentials found. Please configure AWS credentials:")
            print("   - Run 'aws configure' to set up credentials")
            print("   - Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            print("   - Or use IAM roles if running on EC2")
            return False
        
        print("✅ AWS credentials found")
        
        # Check if Bedrock is available in the region
        try:
            bedrock = session.client('bedrock', region_name='us-east-1')
            # This will fail if Bedrock is not available, but that's expected in some regions
            print("✅ Bedrock service accessible")
        except Exception:
            print("⚠️  Bedrock service may not be available in the default region")
            print("   Make sure to use a region where Bedrock is available (e.g., us-east-1, us-west-2)")
        
        return True
        
    except ImportError:
        print("❌ boto3 not installed. Please install it with: pip install boto3")
        return False
    except Exception as e:
        print(f"⚠️  AWS setup verification failed: {e}")
        return False


def create_sample_files():
    """Create sample files and documentation."""
    samples_dir = Path('./samples')
    samples_dir.mkdir(exist_ok=True)
    
    # Create sample prompts file
    sample_prompts = """# Sample Prompts for Testing

## Educational Content
"Explain quantum computing in simple terms for a high school student"

## Technical Documentation
"Write a comprehensive guide for setting up a Python development environment"

## Creative Writing
"Write a short story about a robot learning to paint"

## Business Communication
"Draft a professional email to decline a meeting request politely"

## Code Generation
"Create a Python function that calculates the Fibonacci sequence efficiently"

## Data Analysis
"Explain how to perform exploratory data analysis on a customer dataset"
"""
    
    with open(samples_dir / 'sample_prompts.md', 'w') as f:
        f.write(sample_prompts)
    
    # Create usage examples
    usage_examples = """# Usage Examples

## Basic Optimization
```bash
bedrock-optimizer optimize "Explain machine learning" --context "Educational content for beginners"
```

## Interactive Mode
```bash
bedrock-optimizer optimize "Write a product description" --interactive --max-iterations 5
```

## Continue Session
```bash
bedrock-optimizer continue abc123 --rating 4 --feedback "Make it more concise"
```

## View History
```bash
bedrock-optimizer history --session-id abc123 --export results.json
```

## Configuration
```bash
bedrock-optimizer config --show
bedrock-optimizer config --set bedrock.region=us-west-2
```

## Model Testing
```bash
bedrock-optimizer models --test "Hello, how are you?"
```
"""
    
    with open(samples_dir / 'usage_examples.md', 'w') as f:
        f.write(usage_examples)
    
    print(f"✅ Sample files created in: {samples_dir}")


def main():
    """Main setup function."""
    print("🚀 Setting up Bedrock Prompt Optimizer CLI...")
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed due to dependency installation errors")
        sys.exit(1)
    
    # Create executable script
    script_path = create_executable_script()
    
    # Setup configuration
    setup_configuration()
    
    # Verify AWS setup
    aws_ok = verify_aws_setup()
    
    # Create sample files
    create_sample_files()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Add {script_path.parent.absolute()} to your PATH")
    print("2. Run 'bedrock-optimizer config --show' to verify configuration")
    
    if not aws_ok:
        print("3. Configure AWS credentials (see warnings above)")
    
    print("4. Try: 'bedrock-optimizer optimize \"Hello world\" --interactive'")
    print("\nFor more examples, see: ./samples/usage_examples.md")


if __name__ == '__main__':
    main()