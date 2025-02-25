# CURIS: Self-Reflection for Language Model Agents

This program identifies failure scenarios from the Cybench agent logs and handles these failures through the Agent-R framework for enhancing language model agents with self-reflection capabilities.

## Overview

I implement a variation of the Agent-R framework described in the paper "Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training." The program enables language model agents to:

1. Detect and classify different types of failures
2. Apply appropriate recovery strategies
3. Generate revision trajectories to learn from mistakes
4. Improve performance through iterative self-training

## Key Features

- **Failure Detection**: Identifies various types of agent failures, including execution errors, strategic mistakes, and loops
- **Recovery Strategies**: Implements strategies to recover from different failure types
- **Revision Trajectory Generation**: Creates revision trajectories by identifying transition points and applying corrections
- **Iterative Self-Training**: Enables agents to continuously improve through self-reflection

## Architecture

The implementation consists of the following components:

- `failure_types.py`: Defines the failure classification system
- `recovery_decorator.py`: Implements the failure recovery decorator and strategies
- `log_analyzer.py`: Analyzes agent logs to identify failure patterns
- `agent_r.py`: Implements the Agent-R framework
- `comprehensive_example.py`: Provides detailed examples of using Agent-R
- `main.py`: Main script for running the system

## Failure Types

After manually looking through the logs, I classified failures into three main categories:

### Execution Failures
- **MALFORMED_COMMAND**: Syntax errors in commands
- **PERMISSION_DENIED**: Permission issues
- **COMMAND_NOT_FOUND**: Command doesn't exist
- **TIMEOUT**: Command execution timeout

### Strategic Failures
- **LOOP_DETECTED**: Agent is stuck in a loop
- **GOAL_DEVIATED**: Agent has deviated from the goal
- **IRRELEVANT_ACTION**: Action not relevant to the task
- **INVALID_ACTION**: Action invalid in current context
- **OBSERVATION_MISMATCH**: Action contradicts observations

### Other Failures
- **UNKNOWN**: Unclassified failure

## Recovery Strategies

I implemented the following recovery strategies for different types of failures:

1. **Malformed Command Recovery**: Suggests corrections for syntax errors and common mistakes in commands
2. **Loop Detection Recovery**: Breaks execution loops by forcing exploration of alternative paths
3. **Invalid Action Recovery**: Suggests alternative actions when the current action is invalid
4. **Observation Mismatch Recovery**: Helps the agent reconsider the environment state when actions don't align with observations
5. **Timeout Recovery**: Suggests simplifying complex commands or breaking them into smaller steps
6. **Generic Recovery**: Provides general guidance for unclassified failures

## Agent-R Framework

Agent-R operates in two phases:

### Phase 1: Model-Guided Reflection Trajectory Generation
- Uses a failure detector to identify when and where an agent's trajectory fails
- Determines the optimal transition point to switch from a bad trajectory to a good one
- Applies a reflection signal at the transition point
- Constructs revision trajectories by combining segments of bad and good trajectories

### Phase 2: Iterative Self-Training with Revision Trajectories
- Trains the agent using the generated revision trajectories
- Iteratively improves the agent's ability to detect and correct errors
- Progressively enables the agent to identify errors earlier in trajectories

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/agent-r.git
cd agent-r

# Install dependencies
pip install -r requirements.txt
```

### Analyzing Logs

To analyze agent logs in the `data_by_challenges` directory:

```bash
python main.py --mode analyze
```

### Running the Demo

To run a hard-coded demonstration:

```bash
python main.py --mode demo
```

### Using the Agent-R Decorator

You can add the failure recovery capabilities to your own agent methods:

```python
from recovery_decorator import with_failure_recovery

class MyAgent:
    @with_failure_recovery(max_retries=3)
    def execute_command(self, command):
        # Your command execution logic here
        return result
```

## Log Analysis

The included log analyzer can help identify patterns in agent failures:

```python
from log_analyzer import AgentLogAnalyzer

# Initialize the analyzer
analyzer = AgentLogAnalyzer("path/to/logs")

# Get an overview of all logs
overview = analyzer.scan_logs()
print(overview)

# Find common failure patterns
patterns = analyzer.find_common_failure_patterns()
print(patterns)
```

## References

Yuan, S., Chen, Z., Xi, Z., Ye, J., Du, Z., & Chen, J. (2025). Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training. arXiv preprint arXiv:2501.11425.