#!/usr/bin/env python3
"""
Demo script for the Bedrock Prompt Optimizer CLI.

This script demonstrates the CLI functionality without requiring AWS credentials.
It shows the help system, configuration management, and argument parsing.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a CLI command and display the results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Exit Code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
    except Exception as e:
        print(f"❌ Error running command: {e}")

def main():
    """Run CLI demonstration."""
    print("🎭 Bedrock Prompt Optimizer CLI Demo")
    print("This demo shows CLI functionality without requiring AWS credentials.")
    
    # Check if executable exists
    executable = "./bedrock-optimizer"
    if not Path(executable).exists():
        print(f"❌ Executable not found: {executable}")
        print("Run 'python cli/setup.py' first to create the executable.")
        return
    
    # Demo commands that don't require AWS
    demo_commands = [
        ([executable, "--help"], "Show main help"),
        ([executable, "config", "--help"], "Show config command help"),
        ([executable, "optimize", "--help"], "Show optimize command help"),
        ([executable, "history", "--help"], "Show history command help"),
        ([executable, "config", "--init"], "Initialize default configuration"),
        ([executable, "config", "--show"], "Show current configuration"),
        ([executable, "config", "--set", "cli.colored_output=false"], "Set configuration value"),
        ([executable, "config", "--get", "bedrock.region"], "Get configuration value"),
    ]
    
    for cmd, description in demo_commands:
        run_command(cmd, description)
    
    print(f"\n{'='*60}")
    print("🎉 CLI Demo Complete!")
    print("\nTo use the full functionality:")
    print("1. Configure AWS credentials: aws configure")
    print("2. Try: ./bedrock-optimizer optimize 'Hello world' --interactive")
    print("3. See cli/README.md for more examples")

if __name__ == '__main__':
    main()