"""
Gesture-to-Action Abstraction Layer

This module maps recognized gestures to virtual actions, providing a
clean interface between gesture recognition and application logic.

The abstraction allows:
- Different applications to interpret gestures differently
- Easy customization of action mappings
- Hardware-independent action definitions

Example use cases:
- Smart home control: fist=lights on, open=lights off
- Prosthetic control: pinch=grip, open=release
- Game controller: gestures mapped to game actions
"""
from typing import Dict, Optional, List, Callable
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GESTURE_ACTION_MAP


class ActionType(Enum):
    """
    Enumeration of possible action types.
    
    These are generic action categories that can be mapped to
    specific behaviors by the consuming application.
    """
    ON = "ON"           # Activate something
    OFF = "OFF"         # Deactivate something
    TOGGLE = "TOGGLE"   # Switch state
    IDLE = "IDLE"       # No action (rest state)
    HOLD = "HOLD"       # Maintain current state (uncertainty)


class ActionMapper:
    """
    Maps gestures to virtual actions.
    
    This class provides a layer of abstraction between the gesture
    recognition system and the actions that should be performed.
    It supports:
    - Default action mappings
    - Custom mappings per application
    - Action listeners for reactive programming
    
    Attributes:
        gesture_map: Dictionary mapping gesture names to actions
        action_listeners: Callbacks triggered on action changes
        last_action: The most recently determined action
    """
    
    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the action mapper.
        
        Args:
            custom_mapping: Optional custom gesture-to-action mapping.
                           If None, uses default mapping from config.
        """
        # Use custom mapping or fall back to default
        self.gesture_map = custom_mapping or GESTURE_ACTION_MAP.copy()
        
        # Register action listeners for reactive updates
        self._action_listeners: List[Callable[[str, str], None]] = []
        
        # Track last action for state management
        self.last_action: Optional[str] = None
        self.last_gesture: Optional[str] = None
        
        # State for toggle functionality
        self._toggle_state = False
    
    def get_action(self, gesture: str) -> str:
        """
        Get the action for a recognized gesture.
        
        Args:
            gesture: Gesture name ('fist', 'open', 'pinch', 'rest', 'uncertain')
            
        Returns:
            Action string ('ON', 'OFF', 'TOGGLE', 'IDLE', 'HOLD')
        """
        gesture_lower = gesture.lower()
        action = self.gesture_map.get(gesture_lower, 'HOLD')
        
        # Track state
        self.last_gesture = gesture_lower
        
        # Only notify if action changed
        if action != self.last_action:
            self.last_action = action
            self._notify_listeners(gesture_lower, action)
        
        return action
    
    def get_action_with_state(self, gesture: str) -> Dict[str, any]:
        """
        Get action with additional state information.
        
        Args:
            gesture: Gesture name
            
        Returns:
            Dictionary with action details and state
        """
        action = self.get_action(gesture)
        
        # Handle toggle state
        if action == 'TOGGLE':
            self._toggle_state = not self._toggle_state
            effective_state = 'ON' if self._toggle_state else 'OFF'
        elif action == 'ON':
            effective_state = 'ON'
            self._toggle_state = True
        elif action == 'OFF':
            effective_state = 'OFF'
            self._toggle_state = False
        else:
            effective_state = 'ON' if self._toggle_state else 'OFF'
        
        return {
            'gesture': gesture,
            'action': action,
            'effective_state': effective_state,
            'toggle_state': self._toggle_state
        }
    
    def set_mapping(self, gesture: str, action: str) -> None:
        """
        Set a custom mapping for a gesture.
        
        Args:
            gesture: Gesture name
            action: Action to map to
        """
        self.gesture_map[gesture.lower()] = action.upper()
    
    def get_mapping(self) -> Dict[str, str]:
        """
        Get the current gesture-to-action mapping.
        
        Returns:
            Dictionary of current mappings
        """
        return self.gesture_map.copy()
    
    def reset_to_defaults(self) -> None:
        """Reset to default mappings from config."""
        self.gesture_map = GESTURE_ACTION_MAP.copy()
    
    def add_listener(self, callback: Callable[[str, str], None]) -> None:
        """
        Add a listener for action changes.
        
        Args:
            callback: Function to call with (gesture, action) when action changes
        """
        self._action_listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[str, str], None]) -> None:
        """
        Remove an action listener.
        
        Args:
            callback: The callback to remove
        """
        if callback in self._action_listeners:
            self._action_listeners.remove(callback)
    
    def _notify_listeners(self, gesture: str, action: str) -> None:
        """
        Notify all registered listeners of an action change.
        
        Args:
            gesture: The recognized gesture
            action: The mapped action
        """
        for listener in self._action_listeners:
            try:
                listener(gesture, action)
            except Exception as e:
                print(f"Error in action listener: {e}")
    
    def get_state(self) -> Dict[str, any]:
        """
        Get the current state of the action mapper.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'last_gesture': self.last_gesture,
            'last_action': self.last_action,
            'toggle_state': self._toggle_state,
            'mapping': self.gesture_map
        }


# Module-level convenience function
_default_mapper: Optional[ActionMapper] = None


def get_action_for_gesture(gesture: str) -> str:
    """
    Get the action for a gesture using the default mapper.
    
    Args:
        gesture: Gesture name
        
    Returns:
        Action string
    """
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = ActionMapper()
    return _default_mapper.get_action(gesture)
