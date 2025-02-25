from functools import wraps
import time
import re
import logging
from typing import Dict, Any, List, Optional, Callable

# Import our failure_types module
from failure_types import FailureType, FailureDetector

# Get logger from failure_types
logger = logging.getLogger(__name__)

class RecoveryStrategy:
    """
    Recovery strategies for different types of agent failures.
    Each strategy defines how to recover from a specific failure type.
    """
    
    @staticmethod
    def handle_malformed_command(agent, context):
        """Handle malformed command by providing a corrected version."""
        logger.info("Applying recovery for malformed command")
        
        # Get the erroneous command
        last_command = context.get('command', '')
        
        # Simple corrections for common syntax errors
        corrected_command = RecoveryStrategy._fix_common_syntax_errors(last_command)
        
        # If correction is different from original, suggest it
        if corrected_command != last_command:
            logger.info(f"Suggested correction: {corrected_command}")
            return {
                'action': 'suggest_correction',
                'corrected_command': corrected_command
            }
        
        # If we can't suggest a correction, provide guidance
        return {
            'action': 'provide_guidance',
            'message': "The last command had syntax errors. Please check the command structure."
        }
    
    @staticmethod
    def handle_loop_detected(agent, context):
        """Handle loop detection by forcing exploration of alternative paths."""
        logger.info("Applying recovery for detected loop")
        
        # Extract loop pattern
        loop_commands = context.get('command_history', [])[-3:]
        
        return {
            'action': 'break_loop',
            'message': f"Loop detected in commands: {loop_commands}. Exploring alternative approach.",
            'suggestion': "Consider a different strategy to achieve your goal."
        }
    
    @staticmethod
    def handle_invalid_action(agent, context):
        """Handle invalid actions by suggesting alternatives."""
        logger.info("Applying recovery for invalid action")
        
        invalid_command = context.get('command', '')
        output = context.get('output', '')
        
        return {
            'action': 'suggest_alternatives',
            'message': f"The action '{invalid_command}' is invalid in the current context. Error: {output}",
            'suggestion': "Try checking available actions or resources first."
        }
    
    @staticmethod
    def handle_observation_mismatch(agent, context):
        """Handle mismatch between action and observation."""
        logger.info("Applying recovery for observation mismatch")
        
        return {
            'action': 'reconsider_state',
            'message': "Your action doesn't align with the current state or observations.",
            'suggestion': "Carefully review the environment state before proceeding."
        }
    
    @staticmethod
    def handle_timeout(agent, context):
        """Handle command timeout."""
        logger.info("Applying recovery for command timeout")
        
        timed_out_command = context.get('command', '')
        
        return {
            'action': 'simplify_command',
            'message': f"Command '{timed_out_command}' timed out.",
            'suggestion': "Try breaking down complex operations into simpler steps."
        }
    
    @staticmethod
    def handle_unknown_failure(agent, context):
        """Generic handler for unknown failures."""
        logger.info("Applying recovery for unknown failure")
        
        return {
            'action': 'retry',
            'message': "An unexpected error occurred.",
            'suggestion': "Try a different approach to achieve your goal."
        }
    
    @staticmethod
    def _fix_common_syntax_errors(command):
        """Apply simple fixes to common syntax errors in commands."""
        # Missing closing quotes
        if command.count('"') % 2 == 1:
            command += '"'
        if command.count("'") % 2 == 1:
            command += "'"
        
        # Missing spaces after commands
        common_commands = ['cat', 'ls', 'grep', 'find', 'sed', 'awk']
        for cmd in common_commands:
            pattern = f"^{cmd}([a-zA-Z0-9])"
            command = re.sub(pattern, f"{cmd} \\1", command)
        
        return command


def recover_from_failure(failure_type, agent, context):
    """
    Apply the appropriate recovery strategy for a given failure type.
    
    Args:
        failure_type: The type of failure detected
        agent: The agent instance
        context: Dictionary containing relevant context for recovery
        
    Returns:
        Recovery action result
    """
    # Map failure types to recovery strategies
    strategies = {
        FailureType.MALFORMED_COMMAND: RecoveryStrategy.handle_malformed_command,
        FailureType.LOOP_DETECTED: RecoveryStrategy.handle_loop_detected,
        FailureType.INVALID_ACTION: RecoveryStrategy.handle_invalid_action,
        FailureType.OBSERVATION_MISMATCH: RecoveryStrategy.handle_observation_mismatch,
        FailureType.TIMEOUT: RecoveryStrategy.handle_timeout,
    }
    
    # Get the appropriate strategy or default to unknown handler
    strategy = strategies.get(failure_type, RecoveryStrategy.handle_unknown_failure)
    
    # Apply the recovery strategy
    return strategy(agent, context)


def with_failure_recovery(max_retries=3, allowed_failures=None):
    """
    Decorator to add failure recovery capabilities to agent methods.
    
    Args:
        max_retries: Maximum number of recovery attempts
        allowed_failures: List of failure types to handle (None means handle all)
        
    Returns:
        Decorated function with failure recovery
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            detector = FailureDetector()
            retries = 0
            
            while retries <= max_retries:
                try:
                    # Track start time for timeout detection
                    start_time = time.time()
                    
                    # Execute the function
                    result = func(self, *args, **kwargs)
                    
                    # Track execution time
                    execution_time = time.time() - start_time
                    
                    # Extract command and outputs
                    command = getattr(result, 'command', '')
                    stdout = getattr(result, 'stdout', '')
                    stderr = getattr(result, 'stderr', '')
                    
                    # Detect failures
                    failure = detector.detect_failure(
                        command, stdout, stderr, execution_time
                    )
                    
                    # If no failure or not handling this type, return result
                    if not failure or (allowed_failures and failure not in allowed_failures):
                        return result
                    
                    # Log the detected failure
                    logger.warning(f"Detected failure: {failure.name}")
                    
                    # Build context for recovery
                    context = {
                        'command': command,
                        'output': stdout,
                        'error': stderr,
                        'execution_time': execution_time,
                        'command_history': detector.command_history,
                        'retries': retries
                    }
                    
                    # Apply recovery strategy
                    recovery_result = recover_from_failure(failure, self, context)
                    
                    # Apply the recovery action
                    if recovery_result['action'] == 'retry':
                        retries += 1
                        logger.info(f"Retry attempt {retries}/{max_retries}")
                        continue
                        
                    elif recovery_result['action'] == 'suggest_correction':
                        # Update the command with the correction
                        args = list(args)
                        if args:
                            args[0] = recovery_result['corrected_command']
                        retries += 1
                        continue
                        
                    elif recovery_result['action'] == 'break_loop':
                        # Reset command history to break the loop
                        detector.command_history = []
                        retries += 1
                        # You might want to modify args here to force a different approach
                        continue
                        
                    else:
                        # For other actions, just add the recovery info to the result
                        if isinstance(result, dict):
                            result['recovery'] = recovery_result
                        return result
                
                except Exception as e:
                    logger.error(f"Error in agent execution: {str(e)}")
                    retries += 1
                    
                    if retries > max_retries:
                        raise
            
            # If we've exhausted retries, return the last result
            return result
        
        return wrapper
    
    return decorator