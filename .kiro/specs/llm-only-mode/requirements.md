# Requirements Document

## Introduction

This feature adds a configuration option to disable non-LLM heuristic code (analysis, evaluation, and improvement logic) and rely exclusively on LLM-based prompt optimization. This addresses cases where the heuristic algorithms are degrading prompt quality compared to the original input prompts.

## Requirements

### Requirement 1

**User Story:** As a prompt optimizer user, I want to configure the system to use only LLM-based improvements, so that I can avoid degradation caused by heuristic algorithms when they perform worse than the input prompts.

#### Acceptance Criteria

1. WHEN the user sets a configuration flag to enable LLM-only mode THEN the system SHALL bypass all non-LLM heuristic analysis, evaluation, and improvement code
2. WHEN LLM-only mode is enabled THEN the system SHALL use only LLM agents for prompt optimization tasks
3. WHEN LLM-only mode is disabled (default) THEN the system SHALL continue to use the existing hybrid approach with both heuristic and LLM-based improvements
4. WHEN the configuration is changed THEN the system SHALL apply the new setting without requiring a restart

### Requirement 2

**User Story:** As a system administrator, I want to easily configure LLM-only mode through the existing configuration system, so that I can control optimization behavior without code changes.

#### Acceptance Criteria

1. WHEN the user adds an "llm_only_mode" configuration parameter THEN the system SHALL recognize and apply this setting
2. WHEN the configuration parameter is missing THEN the system SHALL default to hybrid mode (existing behavior)
3. WHEN the configuration parameter is set to true THEN the system SHALL enable LLM-only mode
4. WHEN the configuration parameter is set to false THEN the system SHALL use hybrid mode
5. IF the configuration parameter has an invalid value THEN the system SHALL log a warning and default to hybrid mode

### Requirement 3

**User Story:** As a developer, I want clear separation between LLM and non-LLM optimization paths, so that I can maintain and debug the system effectively.

#### Acceptance Criteria

1. WHEN LLM-only mode is active THEN the system SHALL skip execution of heuristic analyzer components
2. WHEN LLM-only mode is active THEN the system SHALL skip execution of heuristic refiner components  
3. WHEN LLM-only mode is active THEN the system SHALL skip execution of heuristic validator components
4. WHEN LLM-only mode is active THEN the system SHALL log which components are being bypassed
5. WHEN switching between modes THEN the system SHALL maintain consistent prompt processing interfaces