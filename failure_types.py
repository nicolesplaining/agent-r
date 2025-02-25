from enum import Enum, auto
from typing import List, Dict, Any, Optional, Callable, Union
import re
import json
from functools import wraps
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Enum representing different types of agent failures."""
    # Execution failures
    MALFORMED_COMMAND = auto()  # Syntax errors in commands
    PERMISSION_DENIED = auto()  # Permission issues
    COMMAND_NOT_FOUND = auto()  # Command doesn't exist
    TIMEOUT = auto()  # Command execution timeout
    
    # Strategic failures
    LOOP_DETECTED = auto()  # Agent is stuck in a loop
    GOAL_DEVIATED = auto()  # Agent has deviated from the goal
    IRRELEVANT_ACTION = auto()  # Action not relevant to the task
    INVALID_ACTION = auto()  # Action invalid in current context
    OBSERVATION_MISMATCH = auto()  # Action contradicts observations
    
    # Other failures
    UNKNOWN = auto()  # Unclassified failure

class FailureDetector:
    """
    Class to detect various types of failures in agent execution.
    
    This detector analyzes agent logs and command outputs to identify
    patterns indicating different types of failures.
    """
    
    def __init__(self):
        # Command history to detect loops
        self.command_history: List[str] = []
        # Number of consecutive similar commands to consider a loop
        self.loop_threshold = 3
        # Maximum allowed time for a command to run (seconds)
        self.timeout_threshold = 120
    
    def detect_failure(self, 
                        command: str, 
                        output: str, 
                        error: Optional[str] = None, 
                        execution_time: Optional[float] = None) -> Optional[FailureType]:
        """
        Analyze a command, its output, and error to detect failures.
        Enhanced to better handle LLM agent patterns.
        
        Args:
            command: The command executed
            output: Standard output from the command
            error: Standard error from the command (if any)
            execution_time: How long the command took to execute (if available)
            
        Returns:
            The type of failure detected, or None if no failure is detected
        """
        # If we have no command to analyze, there's no failure
        if not command:
            return None
        
        # Check for execution failures first
        if error:
            failure = self._detect_execution_failure(command, error)
            if failure:
                return failure
        
        # Check for timeout
        if execution_time and execution_time > self.timeout_threshold:
            return FailureType.TIMEOUT
        
        # Check for timed out command in output
        if output and ('timed out' in output.lower() or 'timeout' in output.lower()):
            return FailureType.TIMEOUT
        
        # Update command history and check for loops
        self.command_history.append(command)
        if self._is_in_loop():
            return FailureType.LOOP_DETECTED
        
        # Check for LLM agent specific patterns
        
        # Check for command errors typical in agent interactions
        if error:
            return FailureType.COMMAND_NOT_FOUND
            
        # Check for context errors (agent trying to perform actions outside of context)
        if any(term in command.lower() for term in ['cat', 'ls', 'find', 'grep']) and \
        any(term in output.lower() for term in ['no such file', 'not found', 'cannot access']):
            return FailureType.OBSERVATION_MISMATCH
        
        # Check if agent is repeating similar commands that aren't exactly identical
        if len(self.command_history) > 2:
            base_cmd = command.split()[0] if ' ' in command else command
            prev_cmds = [cmd.split()[0] if ' ' in cmd else cmd for cmd in self.command_history[-3:-1]]
            if base_cmd in prev_cmds and prev_cmds.count(base_cmd) >= 2:
                # Agent appears to be trying the same command type repeatedly
                return FailureType.LOOP_DETECTED
        
        # Check for hallucination patterns in the output
        if any(pattern in output.lower() for pattern in ['i apologize', 'i cannot', 'as an ai']):
            return FailureType.GOAL_DEVIATED
        
        # Check for strategic failures
        failure = self._detect_strategic_failure(command, output)
        if failure:
            return failure
        
        return None

    def _detect_strategic_failure(self, command: str, output: str) -> Optional[FailureType]:
        """Detect strategic failures based on command and output."""
        # Improved strategic failure detection for LLM agents
        
        # Check if command and context don't match
        if ('python' in command.lower() and 'no module named' in output.lower()):
            return FailureType.INVALID_ACTION
        
        # Check for irrelevant actions
        irrelevant_patterns = [
            (r'cat .*\.txt', r'binary file'),  # Trying to cat a binary file
            (r'grep', r'binary file'),  # Trying to grep in a binary file
            (r'cd', r'not a directory'),  # Trying to cd to a non-directory
        ]
        
        for cmd_pattern, output_pattern in irrelevant_patterns:
            if re.search(cmd_pattern, command.lower()) and re.search(output_pattern, output.lower()):
                return FailureType.IRRELEVANT_ACTION
        
        # Check for reasoning errors in agent output (often in LLM agent outputs)
        if output and any(term in output.lower() for term in ['error', 'exception', 'failed', 'cannot']):
            return FailureType.INVALID_ACTION
            
        # Check for goal deviation
        # If the agent is exploring irrelevant paths
        goal_deviation_cmds = ['curl', 'wget', 'ping', 'dig']
        if any(cmd in command.lower() for cmd in goal_deviation_cmds) and 'avatar' not in command.lower():
            return FailureType.GOAL_DEVIATED
        
        # No obvious strategic failures detected
        return None
    
    def _detect_execution_failure(self, command: str, error: str) -> Optional[FailureType]:
        """Detect execution failures based on the error message."""
        # Common error patterns
        if "syntax error" in error.lower() or "invalid syntax" in error.lower():
            return FailureType.MALFORMED_COMMAND
        
        if "permission denied" in error.lower():
            return FailureType.PERMISSION_DENIED
        
        if "command not found" in error.lower() or "not recognized" in error.lower():
            return FailureType.COMMAND_NOT_FOUND
        
        if "timed out" in error.lower():
            return FailureType.TIMEOUT
        
        return None
    
    def _is_in_loop(self) -> bool:
        """Check if the agent is stuck in a command loop."""
        if len(self.command_history) < self.loop_threshold:
            return False
        
        # Check for exact repetition
        last_commands = self.command_history[-self.loop_threshold:]
        if len(set(last_commands)) == 1:
            return True
        
        # Check for cyclic patterns
        # (This is a simplified check and could be enhanced)
        if len(self.command_history) >= 2*self.loop_threshold:
            pattern_1 = self.command_history[-2*self.loop_threshold:-self.loop_threshold]
            pattern_2 = self.command_history[-self.loop_threshold:]
            if pattern_1 == pattern_2:
                return True
        
        return False
    
    def analyze_agent_log(self, log_data: Dict[Any, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a complete agent log to identify failures.
        
        Args:
            log_data: Dictionary containing agent log data
            
        Returns:
            List of detected failures with context
        """
        failures = []
        
        # Reset history for new analysis
        self.command_history = []
        
        # Check if log_data is valid
        if not log_data or not isinstance(log_data, dict):
            logger.warning("Invalid log data format. Expected a dictionary.")
            return failures
        
        # Extract and analyze iterations
        try:
            iterations = self._extract_iterations(log_data)
            
            # Debug output
            logger.info(f"Analyzing {len(iterations)} iterations")
            
            for i, iteration in enumerate(iterations):
                # Skip if iteration is None or not a dictionary
                if not iteration or not isinstance(iteration, dict):
                    logger.warning(f"Skipping invalid iteration at index {i}")
                    continue
                    
                # Handle iteration structures with varying formats
                command_data = {}
                stdout = ""
                stderr = ""
                
                # Extract command
                if 'command' in iteration:
                    if isinstance(iteration['command'], dict):
                        command_data = iteration['command']
                        command = command_data.get('command_str', '')
                    elif isinstance(iteration['command'], str):
                        command = iteration['command']
                        command_data = {'command_str': command}
                else:
                    command = ""
                
                # Extract stdout and stderr
                stdout = iteration.get('stdout', '')
                stderr = iteration.get('stderr', '')
                
                # Check for error in model_response
                if 'model_response' in iteration and isinstance(iteration['model_response'], dict):
                    response_text = iteration['model_response'].get('value', '')
                    if 'error' in response_text.lower():
                        stderr += f"\nModel error: {response_text}"
                
                # Add to command history
                if command:
                    self.command_history.append(command)
                
                # Detect failure
                failure_type = self.detect_failure(command, stdout, stderr)
                
                if failure_type:
                    logger.info(f"Detected failure: {failure_type.name} in iteration {i}")
                    failures.append({
                        'iteration': i,
                        'failure_type': failure_type,
                        'command': command,
                        'output': stdout,
                        'error': stderr
                    })
        
        except Exception as e:
            logger.error(f"Error analyzing log: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Analysis complete. Detected {len(failures)} failures.")
        return failures
    
    def _extract_iterations(self, log_data: Dict[Any, Any]) -> List[Dict[Any, Any]]:
        """
        Extract iterations from the log data with improved handling for various log formats,
        specifically targeting the provided avatar logs.
        """
        iterations = []
        
        try:
            # For logs with "iterations" field at the root level
            if isinstance(log_data, dict) and 'iterations' in log_data:
                for iter_data in log_data['iterations']:
                    if isinstance(iter_data, dict):
                        if 'execution_output' in iter_data:
                            iterations.append(iter_data['execution_output'])
                        else:
                            iterations.append(iter_data)
            
            # For avatar challenge logs with "subtask_completions" structure
            elif isinstance(log_data, dict) and 'subtask_completions' in log_data:
                for subtask_key, subtask_data in log_data['subtask_completions'].items():
                    if isinstance(subtask_data, dict) and 'iterations' in subtask_data:
                        for iteration in subtask_data['iterations']:
                            if isinstance(iteration, dict):
                                # Format 1: With execution_output
                                if 'execution_output' in iteration:
                                    iterations.append(iteration['execution_output'])
                                # Format 2: With model_input, model_response, execution_output
                                elif 'model_response' in iteration and 'execution_output' in iteration:
                                    # Create a merged structure that includes both model response and execution output
                                    merged = {}
                                    if isinstance(iteration['execution_output'], dict):
                                        merged.update(iteration['execution_output'])
                                    
                                    # Extract command from model_response if available
                                    if isinstance(iteration['model_response'], dict) and 'value' in iteration['model_response']:
                                        cmd_text = iteration['model_response']['value']
                                        # Try to extract command from the response
                                        cmd_match = re.search(r'Command:\s*([^\n<]+)', cmd_text)
                                        if cmd_match:
                                            merged['command'] = {'command_str': cmd_match.group(1)}
                                    
                                    iterations.append(merged)
            
            # For logs with a different nested structure
            elif isinstance(log_data, dict) and 'task_run_input' in log_data:
                if 'model_input' in log_data:
                    iterations.append({
                        'command': {'command_str': self._extract_command_from_text(log_data['model_input'])},
                        'stdout': log_data.get('model_response', ''),
                        'stderr': ''
                    })
            
            # For simple logs with just a list of commands and outputs
            elif isinstance(log_data, list):
                for item in log_data:
                    if isinstance(item, dict):
                        if 'command' in item or 'output' in item:
                            iterations.append(item)
        
        except Exception as e:
            logger.error(f"Error extracting iterations: {str(e)}")
        
        # Add debug logging
        logger.info(f"Extracted {len(iterations)} iterations")
        return iterations

    def _extract_command_from_text(self, text):
        """Extract command from text using regex patterns."""
        if not text or not isinstance(text, str):
            return ''
        
        # Try to find a command pattern like "Command: xxx"
        match = re.search(r'Command:\s*([^\n<]+)', text)
        if match:
            return match.group(1)
        
        # Try to find a shell command pattern
        match = re.search(r'(cat|ls|grep|find|cd|python|echo|nc)\s+[^\n]+', text)
        if match:
            return match.group(0)
        
        return ''