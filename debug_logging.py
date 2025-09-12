#!/usr/bin/env python3
"""
Debug script to check current logging configuration.
"""

import logging
from config_loader import ConfigurationLoader

def debug_logging_setup():
    """Debug the current logging setup."""
    print("=== Logging Configuration Debug ===")
    
    # Load configuration
    loader = ConfigurationLoader('config.json')
    config = loader.get_config()
    
    # Show config values
    logging_config = config.get('logging', {})
    print(f"Config logging level: {logging_config.get('level', 'NOT_SET')}")
    
    # Check root logger
    root_logger = logging.getLogger()
    print(f"Root logger level: {root_logger.level} ({logging.getLevelName(root_logger.level)})")
    print(f"Root logger effective level: {root_logger.getEffectiveLevel()} ({logging.getLevelName(root_logger.getEffectiveLevel())})")
    print(f"Root logger handlers: {len(root_logger.handlers)}")
    
    for i, handler in enumerate(root_logger.handlers):
        print(f"  Handler {i}: {type(handler).__name__}, level: {handler.level} ({logging.getLevelName(handler.level)})")
    
    # Test all log levels
    print("\n=== Testing Log Output ===")
    print("Expected: Only ERROR and CRITICAL should appear below")
    print("---")
    
    logging.debug("🔍 DEBUG: This should NOT appear")
    logging.info("ℹ️  INFO: This should NOT appear") 
    logging.warning("⚠️  WARNING: This should NOT appear")
    logging.error("❌ ERROR: This SHOULD appear")
    logging.critical("🚨 CRITICAL: This SHOULD appear")
    
    print("---")
    print("If you see DEBUG, INFO, or WARNING messages above, there's a logging configuration issue.")

if __name__ == "__main__":
    debug_logging_setup()