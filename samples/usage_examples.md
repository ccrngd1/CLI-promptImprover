# Usage Examples

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
