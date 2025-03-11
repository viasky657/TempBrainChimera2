import torch
import torch.nn as nn
import datetime
import time
import os
import json
import glob
import cv2
import numpy as np
import platform
import subprocess
from PIL import Image
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from contextlib import nullcontext
import torch.nn.functional as F
from TOVACompression import TOVACompression, tova_compress
import math
from tqdm import tqdm
import numpy as np
from scipy import integrate
from EpisodicMemory import EpisodicMemory
from MirrorNeuronEmpathyReward import (
    MirrorNeuronEmpathyReward,
    NegativeEnvironmentalImpactAvoidance,
    DopamineDrivenEmpathyReward,
    NegativeEmotionPenalty,
    FullMoralRewardCalculator,
    MoralChoiceDataset,
    MoralEmpathyTrainer,
    SelfTaskGoalReward,
    train_moral_empathy
)
# Import functions from cinepile_RL.py for RL component
from cinepile_RL import (
    normalize_string,
    evaluate_semantic_similarity,
    eval_response,
    ans_key_map,
    format_question_and_options,
    print_qa,
    get_prompt,
    fine_tune_on_cinepile,
    train_cinepile_with_rl_rewards,
)
# Import Phi-4 multimodal specific modules for video processing
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
# Import the processing modules using relative imports
import sys
sys.path.append('Phi4')
from Phi4.processing_phi4mm import (VideoFrame, VideoInput, VideoInputs)
from Phi4.long_video_processor import HierarchicalTokenCompression

from COCONUTWLatentThinking import(
    save_checkpoint, play_sound, base_causallm, episodic_memory,
    DiffusionLLMModule, Phi4COCONUTWithLatentThinking
)

#Need to add imports from COCONUTWLatentThinking.py and add imports from this file to that same file to 
# start sleep system training. 

# --- Sleep and Awakening System ---
class SleepAwakeningSystem:
    def __init__(self, episodic_memory, base_model= base_causallm, deep_sleep_params=None, awakening_params=None):
        """
        Initialize the Sleep and Awakening System.
        
        Args:
            base_model: The base Phi4 LLM model to control
            episodic_memory: The episodic memory component
            deep_sleep_params: Parameters for deep sleep (optional)
            awakening_params: Parameters for awakening (optional)
        """
        # Store references to the model components
        self.base_model =  base_causallm
        self.episodic_memory = episodic_memory
        
        # Validate that required components are provided
        if self.base_model is None:
            print("WARNING: No base model provided to SleepAwakeningSystem. Some functionality may be limited.")
        
        if self.episodic_memory is None:
            print("WARNING: No episodic memory provided to SleepAwakeningSystem. Memory synchronization will be disabled.")
        
        self.deep_sleep_params = deep_sleep_params or {
            'target_attention': 0.1,
            'target_compute': 0.2,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5,
            'episodic_memory_sync': True,
            'sync_threshold': 0.05,
            'sync_penalty': 2.0,
            'safe_wake_threshold': 0.8
        }
        self.awakening_params = awakening_params or {
            'target_attention': 0.9,
            'target_compute': 0.9,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5,
            'emergency_reward': 10.0,
            'emergency_confirmation_threshold': 3,
            'episodic_memory_sync': True,
            'sync_threshold': 0.05,
            'sync_penalty': 2.0,
            'safe_wake_threshold': 0.8
        }
        
        # State tracking
        self.current_state = {
            'attention': 0.9,
            'compute': 0.9,
            'metric': 0.0,
            'episodic_memory_state': 0.9,  # Track episodic memory state separately
            'sync_level': 1.0  # Track synchronization between model and memory
        }
        self.previous_state = {
            'attention': 0.9,
            'compute': 0.9,
            'metric': 0.0,
            'episodic_memory_state': 0.9,
            'sync_level': 1.0
        }
        self.previous_action = {
            'delta_attention': 0.0,
            'delta_compute': 0.0,
            'delta_metric': 0.0,
            'delta_episodic_memory': 0.0
        }
        
        # Sleep/wake status
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.emergency_counter = 0
        
        # Q-learning parameters
        self.q_table = {}  # State-action value function
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # For epsilon-greedy action selection
        
        # Gating mechanism
        self.attention_gate = nn.Parameter(torch.ones(1))
        self.compute_gate = nn.Parameter(torch.ones(1))
        self.episodic_memory_gate = nn.Parameter(torch.ones(1))  # New gate for episodic memory
        
        # Consciousness level control (0.0 to 1.0, where 1.0 is full consciousness)
        self.consciousness_level = 1.0
        self.consciousness_gate = nn.Parameter(torch.ones(1))
        
        # Sleep paralysis prevention
        self.sleep_paralysis_risk = 0.0
        self.paralysis_threshold = 0.3  # If sync difference exceeds this, risk of sleep paralysis
        
        # Phi4 and episodic memory synchronization
        self.phi4_sleep_state = 1.0  # 1.0 = fully awake, 0.0 = deep sleep
        self.episodic_memory_sleep_state = 1.0
        self.sync_history = []  # Track synchronization history for analysis
        self.max_sync_history = 100  # Maximum history entries to keep
        
    def update_state(self, new_attention=None, new_compute=None, new_metric=None, new_episodic_memory=None):
        """
        Update the current state with new values, ensuring synchronization between
        phi4 model and episodic memory to prevent sleep paralysis.
        
        Args:
            new_attention: New attention value
            new_compute: New compute value
            new_metric: New metric value
            new_episodic_memory: New episodic memory state value
        """
        self.previous_state = self.current_state.copy()
        
        # Update phi4 model state values if provided
        if new_attention is not None:
            self.current_state['attention'] = new_attention
            # Update phi4 sleep state based on attention (primary indicator)
            self.phi4_sleep_state = new_attention
        if new_compute is not None:
            self.current_state['compute'] = new_compute
        if new_metric is not None:
            self.current_state['metric'] = new_metric
            
        # Update episodic memory state if provided, otherwise calculate based on phi4 state
        if new_episodic_memory is not None:
            self.current_state['episodic_memory_state'] = new_episodic_memory
            self.episodic_memory_sleep_state = new_episodic_memory
        else:
            # If no explicit episodic memory state provided, ensure it stays in sync with phi4
            # but with a controlled rate of change to prevent sleep paralysis
            target_memory_state = self.phi4_sleep_state
            current_memory_state = self.episodic_memory_sleep_state
            
            # Get sync parameters
            sync_threshold = self.deep_sleep_params.get('sync_threshold', 0.05)
            safe_wake_threshold = self.deep_sleep_params.get('safe_wake_threshold', 0.8)
            
            # Calculate current sync level (1.0 = perfectly in sync, 0.0 = completely out of sync)
            sync_diff = abs(self.phi4_sleep_state - current_memory_state)
            current_sync_level = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
            self.current_state['sync_level'] = current_sync_level
            
            # Determine if we're in awakening phase (attention increasing)
            awakening_phase = self.phi4_sleep_state > self.previous_state.get('attention', self.phi4_sleep_state)
            
            # Calculate maximum safe change rate based on current sync level and phase
            if awakening_phase:
                # More conservative during awakening to prevent sleep paralysis
                max_safe_change = 0.05 * current_sync_level
                
                # If sync level is dangerously low during awakening, slow down phi4 instead
                if current_sync_level < safe_wake_threshold and self.phi4_sleep_state > 0.3:
                    # Temporarily reduce phi4 sleep state to allow memory to catch up
                    slowdown_factor = 0.7  # Slow down to 70% of current rate
                    phi4_change = self.phi4_sleep_state - self.previous_state.get('attention', self.phi4_sleep_state)
                    adjusted_phi4_change = phi4_change * slowdown_factor
                    
                    # Apply adjusted change
                    self.phi4_sleep_state = self.previous_state.get('attention', self.phi4_sleep_state) + adjusted_phi4_change
                    self.current_state['attention'] = self.phi4_sleep_state
                    
                    print(f"WARNING: Slowing phi4 awakening to prevent sleep paralysis. Sync level: {current_sync_level:.2f}")
                    
                    # Accelerate memory to catch up
                    max_safe_change = 0.15  # Faster catch-up rate
            else:
                # More permissive during sleep phase
                max_safe_change = 0.15 * current_sync_level
            
            # Apply gradual change to episodic memory state to follow phi4
            memory_diff = target_memory_state - current_memory_state
            if abs(memory_diff) > max_safe_change:
                # Limit the rate of change to prevent sleep paralysis
                direction = 1 if memory_diff > 0 else -1
                self.episodic_memory_sleep_state += direction * max_safe_change
                
                # Log significant sync differences
                if abs(memory_diff) > sync_threshold * 2:
                    print(f"Synchronizing: phi4 state: {self.phi4_sleep_state:.2f}, memory state: {self.episodic_memory_sleep_state:.2f}, diff: {memory_diff:.2f}")
            else:
                # Small enough change to apply directly
                self.episodic_memory_sleep_state = target_memory_state
                
            # Update the state dictionary with the new episodic memory state
            self.current_state['episodic_memory_state'] = self.episodic_memory_sleep_state
        
        # Calculate synchronization level between phi4 model and episodic memory
        model_state_avg = (self.current_state['attention'] + self.current_state['compute']) / 2
        memory_state = self.current_state['episodic_memory_state']
        
        # Calculate sync level (1.0 = perfectly in sync, 0.0 = completely out of sync)
        sync_diff = abs(model_state_avg - memory_state)
        sync_threshold = self.deep_sleep_params.get('sync_threshold', 0.05)
        sync_level = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
        self.current_state['sync_level'] = sync_level
        
        # Add to synchronization history for analysis
        self.sync_history.append({
            'timestamp': time.time(),
            'phi4_state': self.phi4_sleep_state,
            'episodic_memory_state': self.episodic_memory_sleep_state,
            'sync_level': sync_level,
            'sync_diff': sync_diff
        })
        
        # Maintain maximum history size
        if len(self.sync_history) > self.max_sync_history:
            self.sync_history.pop(0)
        
        # Update sleep paralysis risk based on sync level and rate of change
        if sync_diff > self.paralysis_threshold:
            # Increase sleep paralysis risk when components are out of sync
            self.sleep_paralysis_risk = min(1.0, self.sleep_paralysis_risk + 0.1)
            
            # Log warning if risk is increasing significantly
            if self.sleep_paralysis_risk > 0.5:
                print(f"WARNING: Sleep paralysis risk increasing: {self.sleep_paralysis_risk:.2f}")
                print(f"Phi4 state: {self.phi4_sleep_state:.2f}, Memory state: {self.episodic_memory_sleep_state:.2f}")
        else:
            # Gradually decrease risk when in sync
            self.sleep_paralysis_risk = max(0.0, self.sleep_paralysis_risk - 0.05)
        
        # If sleep paralysis risk is high, force synchronization
        if self.sleep_paralysis_risk > 0.7:
            # Apply emergency synchronization protocol
            print("WARNING: Sleep paralysis risk critical. Applying emergency synchronization protocol.")
            
            # Determine which component is leading (usually phi4 leads during awakening)
            if self.is_sleeping and self.phi4_sleep_state > self.episodic_memory_sleep_state:
                # During awakening, phi4 is ahead - slow down phi4 and speed up memory
                slowdown_factor = 0.7  # Slow down phi4 to 70% of its current rate
                speedup_factor = 1.5   # Speed up memory to 150% of its current rate
                
                # Apply adjustments
                self.phi4_sleep_state = self.previous_state['attention'] + (self.phi4_sleep_state - self.previous_state['attention']) * slowdown_factor
                self.episodic_memory_sleep_state = self.previous_state['episodic_memory_state'] + (self.phi4_sleep_state - self.previous_state['episodic_memory_state']) * speedup_factor
                
                # Update state dictionary
                self.current_state['attention'] = self.phi4_sleep_state
                self.current_state['episodic_memory_state'] = self.episodic_memory_sleep_state
                
                print(f"Emergency protocol: Slowing phi4 awakening and accelerating memory awakening")
            else:
                # During sleep or if memory is ahead, force both to same intermediate state
                target_state = (self.phi4_sleep_state + self.episodic_memory_sleep_state) / 2
                self.phi4_sleep_state = target_state
                self.episodic_memory_sleep_state = target_state
                
                # Update state dictionary
                self.current_state['attention'] = self.phi4_sleep_state
                self.current_state['episodic_memory_state'] = self.episodic_memory_sleep_state
                
                print(f"Emergency protocol: Forcing synchronization to intermediate state: {target_state:.2f}")
            
            # Recalculate sync level after emergency measures
            sync_diff = abs(self.phi4_sleep_state - self.episodic_memory_sleep_state)
            self.current_state['sync_level'] = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
            
            # Reduce risk after emergency synchronization
            self.sleep_paralysis_risk = 0.3
            
        # Ensure gradual transitions during awakening to prevent sleep paralysis
        if self.is_sleeping and (new_attention is not None or new_compute is not None):
            # Check if we're in the process of waking up (increasing attention/compute)
            if (new_attention is not None and new_attention > self.previous_state['attention']) or \
               (new_compute is not None and new_compute > self.previous_state['compute']):
                
                # Calculate the maximum allowed change rate based on sync level
                # Lower sync levels require slower transitions
                max_change_rate = self.current_state['sync_level'] * 0.2  # 0.0-0.2 range
                
                # Limit the rate of change for attention
                if new_attention is not None:
                    allowed_attention_change = max_change_rate
                    actual_attention_change = abs(new_attention - self.previous_state['attention'])
                    if actual_attention_change > allowed_attention_change:
                        # Scale back the change to the allowed rate
                        direction = 1 if new_attention > self.previous_state['attention'] else -1
                        self.current_state['attention'] = self.previous_state['attention'] + (direction * allowed_attention_change)
                        self.phi4_sleep_state = self.current_state['attention']
                        print(f"Limiting attention change rate to prevent sleep paralysis: {allowed_attention_change:.4f}")
                
                # Limit the rate of change for compute
                if new_compute is not None:
                    allowed_compute_change = max_change_rate
                    actual_compute_change = abs(new_compute - self.previous_state['compute'])
                    if actual_compute_change > allowed_compute_change:
                        # Scale back the change to the allowed rate
                        direction = 1 if new_compute > self.previous_state['compute'] else -1
                        self.current_state['compute'] = self.previous_state['compute'] + (direction * allowed_compute_change)
                        print(f"Limiting compute change rate to prevent sleep paralysis: {allowed_compute_change:.4f}")
    
    def choose_action(self, state, is_emergency=False):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state dictionary
            is_emergency: Whether this is an emergency situation
            
        Returns:
            action: Dictionary with delta values
        """
        if is_emergency:
            # Emergency action to quickly wake up
            return {
                'delta_attention': max(0.9 - state['attention'], 0), #Need to fix Eos Bound to be sure that Eos is detected in the contineous thoughts to know when the model is done thinking.
                'delta_compute': max(0.9 - state['compute'], 0),
                'delta_metric': 0.0
            }
            
        # Convert state to a hashable representation
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return {
                'delta_attention': np.random.uniform(0, 0.2),
                'delta_compute': np.random.uniform(0, 0.2),
                'delta_metric': np.random.uniform(0, 0.1)
            }
        else:
            # Exploit: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # If no actions have been tried yet, return a default action
            if not self.q_table[state_key]:
                return {
                    'delta_attention': 0.1,
                    'delta_compute': 0.1,
                    'delta_metric': 0.0
                }
            
            # Find the action with the highest Q-value
            best_action_key = max(self.q_table[state_key], key=lambda k: self.q_table[state_key][k])
            return self._key_to_action(best_action_key)
    
    def _state_to_key(self, state):
        """Convert a state dictionary to a hashable key."""
        return (round(state['attention'], 2), round(state['compute'], 2), round(state['metric'], 2))
    
    def _action_to_key(self, action):
        """Convert an action dictionary to a hashable key."""
        return (round(action['delta_attention'], 2), round(action['delta_compute'], 2), round(action['delta_metric'], 2))
    
    def _key_to_action(self, key):
        """Convert a hashable key back to an action dictionary."""
        return {
            'delta_attention': key[0],
            'delta_compute': key[1],
            'delta_metric': key[2]
        }
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state dictionary
            action: Action taken dictionary
            reward: Reward received
            next_state: Next state dictionary
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Find max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        self.q_table[state_key][action_key] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state_key][action_key]
        )
    
    def apply_gating_mechanism(self, attention_tensor, compute_tensor):
        """
        Apply the gating mechanism to control attention, compute, and consciousness.
        
        Args:
            attention_tensor: Tensor representing attention
            compute_tensor: Tensor representing compute resources
            
        Returns:
            gated_attention: Gated attention tensor
            gated_compute: Gated compute tensor
        """
        gated_attention = attention_tensor * self.attention_gate
        gated_compute = compute_tensor * self.compute_gate
        
        # Apply consciousness gating to both attention and compute
        gated_attention = gated_attention * self.consciousness_gate
        gated_compute = gated_compute * self.consciousness_gate
        
        return gated_attention, gated_compute
    
    def set_consciousness_level(self, level):
        """
        Manually set the consciousness level of the model.
        This is only allowed to be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        # Ensure level is within valid range
        level = max(0.01, min(1.0, level))  # Never go completely to zero to avoid complete shutdown
        
        # Set the consciousness level
        self.consciousness_level = level
        
        # Update the gates to reflect the new consciousness level
        self.update_gates()
        
        print(f"Consciousness level set to: {level:.2f}")
        return self.consciousness_level
    
    def update_gates(self):
        """Update the gating mechanism based on current sleep state and consciousness level."""
        if self.is_fully_shutdown:
            # Fully shut off
            self.attention_gate.data = torch.zeros_like(self.attention_gate)
            self.compute_gate.data = torch.zeros_like(self.compute_gate)
            self.consciousness_gate.data = torch.zeros_like(self.consciousness_gate)
        elif self.is_sleeping:
            # Reduced activity during sleep
            self.attention_gate.data = torch.tensor([self.current_state['attention']])
            self.compute_gate.data = torch.tensor([self.current_state['compute']])
            self.consciousness_gate.data = torch.tensor([min(self.current_state['attention'], self.consciousness_level)])
        else:
            # Awake but consciousness may be manually adjusted
            self.attention_gate.data = torch.ones_like(self.attention_gate)
            self.compute_gate.data = torch.ones_like(self.compute_gate)
            self.consciousness_gate.data = torch.tensor([self.consciousness_level])
    
    def check_emergency(self, emergency_signal=None):
        """
        Check if there's an emergency that requires immediate awakening.
        
        Args:
            emergency_signal: External emergency signal (optional)
            
        Returns:
            is_emergency: Boolean indicating if emergency override should be triggered
        """
        # Update emergency counter based on signal
        if emergency_signal:
            self.emergency_counter += 1
        else:
            self.emergency_counter = max(self.emergency_counter - 1, 0)
        
        # Check if emergency threshold is reached
        return self.emergency_counter >= self.awakening_params['emergency_confirmation_threshold']
    
    def enter_deep_sleep(self):
        """
        Initiate the deep sleep process with synchronized phi4 model and episodic memory transitions.
        This enhanced version ensures both components transition to sleep state in a coordinated manner
        to prevent sleep paralysis when awakening.
        """
        print("Initiating deep sleep process with synchronized components...")
        self.is_sleeping = True
        self.is_fully_shutdown = False
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # Initialize episodic memory sleep state tracking if not present
        if 'episodic_memory_state' not in initial_state:
            initial_state['episodic_memory_state'] = initial_state['attention']
        
        # Track synchronization between phi4 and episodic memory
        sync_level = 1.0  # Start fully synchronized
        initial_state['sync_level'] = sync_level
        
        # Run deep sleep training loop
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state for phi4 model
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate phi4 model state (average of attention and compute)
                model_state_avg = (next_state['attention'] + next_state['compute']) / 2
                
                # Calculate synchronized episodic memory state transition
                # Ensure episodic memory transitions at a similar but slightly slower rate
                current_memory_state = self.current_state.get('episodic_memory_state', self.current_state['attention'])
                memory_delta = model_state_avg - current_memory_state
                
                # Apply a controlled rate of change for episodic memory (80% of model's rate)
                # This ensures memory transitions more gradually but stays in sync
                memory_change_rate = 0.8
                new_memory_state = current_memory_state + (memory_delta * memory_change_rate)
                next_state['episodic_memory_state'] = max(0.0, min(1.0, new_memory_state))
                
                # Calculate synchronization level between phi4 and episodic memory
                sync_diff = abs(model_state_avg - next_state['episodic_memory_state'])
                sync_threshold = self.deep_sleep_params.get('sync_threshold', 0.05)
                next_state['sync_level'] = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
                
                # Calculate reward with synchronization penalty
                reward = self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                # Update gates
                self.update_gates()
                
                # Apply the state changes to the actual model components
                if self.base_model is not None:
                    # Modify the base model's attention mechanisms based on current state
                    attention_scale = self.current_state['attention']
                    # Apply attention scaling to the model's parameters
                    if hasattr(self.base_model, 'set_attention_scale'):
                        self.base_model.set_attention_scale(attention_scale)
                    elif hasattr(self.base_model, 'attention_scale'):
                        self.base_model.attention_scale = attention_scale
                    
                    # Adjust compute resources based on current state
                    compute_scale = self.current_state['compute']
                    if hasattr(self.base_model, 'set_compute_scale'):
                        self.base_model.set_compute_scale(compute_scale)
                    elif hasattr(self.base_model, 'compute_scale'):
                        self.base_model.compute_scale = compute_scale
                
                # Perform episodic memory operations during sleep
                if self.episodic_memory is not None:
                    # Memory consolidation - more active during deeper sleep states
                    consolidation_rate = 1.0 - self.current_state['episodic_memory_state']  # Higher rate in deeper sleep
                    if hasattr(self.episodic_memory, 'consolidate_memories'):
                        self.episodic_memory.consolidate_memories(consolidation_rate)
                    # Apply weight decay instead of pruning to allow neurons to be reused
                    if self.current_state['episodic_memory_state'] < 0.2:  # Only in deep sleep
                        # Weight decay is more biologically plausible than pruning
                        if hasattr(self.episodic_memory, 'apply_weight_decay'):
                            # Decay rate inversely proportional to memory state (deeper sleep = more decay)
                            decay_rate = 0.1 * (1.0 - self.current_state['episodic_memory_state'] * 2)
                            self.episodic_memory.apply_weight_decay(decay_rate)

                
                # Store phi4 and episodic memory states for tracking
                self.phi4_sleep_state = self.current_state['attention']
                
                self.episodic_memory_sleep_state = self.current_state['episodic_memory_state']
                
                # Check if target sleep state is reached with good synchronization
                if (abs(self.current_state['attention'] - self.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.deep_sleep_params['target_compute']) < 0.05 and
                    self.current_state['sync_level'] > 0.9):  # Ensure high synchronization
                    print(f"Target sleep state reached with sync level: {self.current_state['sync_level']:.2f}")
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Deep sleep training episode {episode}, current state: {self.current_state}")
                print(f"Phi4 state: {self.phi4_sleep_state:.2f}, Memory state: {self.episodic_memory_sleep_state:.2f}, Sync: {self.current_state['sync_level']:.2f}")
        
        # Final update to fully shut down if needed
        if self.current_state['attention'] <= 0.1 and self.current_state['compute'] <= 0.1:
            # Ensure episodic memory is also in deep sleep state before full shutdown
            if abs(self.current_state['episodic_memory_state'] - self.current_state['attention']) > 0.1:
                print("WARNING: Episodic memory not fully synchronized with phi4 model.")
                print("Waiting for synchronization before full shutdown...")
                
                # Force synchronization for safety
                self.current_state['episodic_memory_state'] = self.current_state['attention']
                self.episodic_memory_sleep_state = self.phi4_sleep_state
                self.current_state['sync_level'] = 1.0
                print("Forced synchronization complete.")
            
            # Save episodic memory before full shutdown
            if self.episodic_memory is not None:
                print("Saving episodic memory before full shutdown...")
                self.episodic_memory.save_on_shutdown()
                
            self.is_fully_shutdown = True
            self.update_gates()
            print("LLM has entered full shutdown mode with synchronized components.")
        
        # Save checkpoint
        save_checkpoint("deep_sleep_final", self)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return self.current_state
    
    def awaken(self, emergency_override=False):
        """
        Awaken the model from sleep state with synchronized phi4 model and episodic memory transitions.
        This enhanced version prevents sleep paralysis by ensuring both components wake up together
        in a coordinated manner.
        
        Args:
            emergency_override: Whether to use emergency override
            
        Returns:
            final_state: The final state after awakening
        """
        if not self.is_sleeping and not self.is_fully_shutdown:
            print("Model is already awake.")
            return self.current_state
        
        print(f"Initiating synchronized awakening process{' with emergency override' if emergency_override else ''}...")
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # Ensure episodic memory state is tracked
        if 'episodic_memory_state' not in initial_state:
            initial_state['episodic_memory_state'] = self.episodic_memory_sleep_state if hasattr(self, 'episodic_memory_sleep_state') else initial_state['attention']
        
        # If emergency override, handle with special synchronization care
        if emergency_override:
            print("EMERGENCY OVERRIDE: Implementing synchronized rapid awakening protocol...")
            
            # Instead of immediately jumping to full awakening, implement a rapid but staged awakening
            # to prevent sleep paralysis while still responding quickly to emergency
            
            # Stage 1: Rapid initial activation (30% of full awakening)
            self.current_state = {
                'attention': initial_state['attention'] + 0.3 * (self.awakening_params['target_attention'] - initial_state['attention']),
                'compute': initial_state['compute'] + 0.3 * (self.awakening_params['target_compute'] - initial_state['compute']),
                'metric': 0.0,
                'episodic_memory_state': initial_state['episodic_memory_state'] + 0.3 * (self.awakening_params['target_attention'] - initial_state['episodic_memory_state'])
            }
            
            # Update synchronization level
            model_state_avg = (self.current_state['attention'] + self.current_state['compute']) / 2
            sync_diff = abs(model_state_avg - self.current_state['episodic_memory_state'])
            sync_threshold = self.awakening_params.get('sync_threshold', 0.05)
            self.current_state['sync_level'] = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
            
            # Update gates for partial awakening
            self.update_gates()
            print(f"Stage 1: Rapid initial activation complete. Sync level: {self.current_state['sync_level']:.2f}")
            
            # Stage 2: Synchronization check and adjustment
            if self.current_state['sync_level'] < 0.8:
                print("WARNING: Components out of sync during emergency awakening.")
                print("Performing emergency synchronization...")
                
                # Force better synchronization
                memory_target = model_state_avg
                self.current_state['episodic_memory_state'] = memory_target
                self.current_state['sync_level'] = 0.9
                print("Emergency synchronization complete.")
            
            # Stage 3: Complete awakening with synchronized components
            self.current_state['attention'] = self.awakening_params['target_attention']
            self.current_state['compute'] = self.awakening_params['target_compute']
            self.current_state['episodic_memory_state'] = self.awakening_params['target_attention']
            self.current_state['sync_level'] = 1.0
            
            # Update system state
            self.is_sleeping = False
            self.is_fully_shutdown = False
            self.phi4_sleep_state = self.current_state['attention']
            self.episodic_memory_sleep_state = self.current_state['episodic_memory_state']
            
            # Final gate update
            self.update_gates()
            
            # Calculate and apply emergency reward for learning
            emergency_reward = self.awakening_params['emergency_reward']
            emergency_action = {
                'delta_attention': self.awakening_params['target_attention'] - initial_state['attention'],
                'delta_compute': self.awakening_params['target_compute'] - initial_state['compute'],
                'delta_metric': 0.0
            }
            self.update_q_value(initial_state, emergency_action, emergency_reward, self.current_state)
            
            print("Emergency synchronized awakening completed successfully.")
            return self.current_state
        
        # Regular gradual awakening with synchronization
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Check for emergency
                if self.check_emergency():
                    return self.awaken(emergency_override=True)
                
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state for phi4 model
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate phi4 model state (average of attention and compute)
                model_state_avg = (next_state['attention'] + next_state['compute']) / 2
                
                # Calculate synchronized episodic memory state transition
                # During awakening, memory should wake up slightly ahead of phi4 to prevent sleep paralysis
                # (opposite of sleep process where memory transitions more slowly)
                current_memory_state = self.current_state.get('episodic_memory_state', self.current_state['attention'])
                memory_delta = model_state_avg - current_memory_state
                
                # Apply a controlled rate of change for episodic memory (110% of model's rate)
                # This ensures memory wakes up slightly ahead but stays coordinated
                memory_change_rate = 1.1
                new_memory_state = current_memory_state + (memory_delta * memory_change_rate)
                next_state['episodic_memory_state'] = max(0.0, min(1.0, new_memory_state))
                
                # Calculate synchronization level between phi4 and episodic memory
                sync_diff = abs(model_state_avg - next_state['episodic_memory_state'])
                sync_threshold = self.awakening_params.get('sync_threshold', 0.05)
                next_state['sync_level'] = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
                
                # Calculate sleep paralysis risk
                # Sleep paralysis occurs when phi4 (consciousness) wakes up but memory lags behind
                if next_state['attention'] > 0.6 and next_state['episodic_memory_state'] < 0.4:
                    sleep_paralysis_risk = 0.8
                    print(f"WARNING: High sleep paralysis risk detected: {sleep_paralysis_risk:.2f}")
                    
                    # Apply corrective action - slow down phi4 awakening and speed up memory
                    next_state['attention'] = max(0.0, min(1.0, next_state['attention'] * 0.9))
                    next_state['episodic_memory_state'] = max(0.0, min(1.0, next_state['episodic_memory_state'] * 1.2))
                    
                    # Recalculate sync level after correction
                    model_state_avg = (next_state['attention'] + next_state['compute']) / 2
                    sync_diff = abs(model_state_avg - next_state['episodic_memory_state'])
                    next_state['sync_level'] = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
                    
                    print(f"Applied corrective action. New sync level: {next_state['sync_level']:.2f}")
                else:
                    sleep_paralysis_risk = 0.0
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                # Add extra penalty for sleep paralysis risk
                base_reward = -self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                paralysis_penalty = sleep_paralysis_risk * 5.0  # Significant penalty for sleep paralysis risk
                reward = base_reward - paralysis_penalty
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                
                # Store phi4 and episodic memory states for tracking
                self.phi4_sleep_state = self.current_state['attention']
                self.episodic_memory_sleep_state = self.current_state['episodic_memory_state']
                
                # Update gates
                self.update_gates()
                
                # Check if target awake state is reached with good synchronization
                if (abs(self.current_state['attention'] - self.awakening_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.awakening_params['target_compute']) < 0.05 and
                    self.current_state['sync_level'] > 0.9):  # Ensure high synchronization
                    print(f"Target awake state reached with sync level: {self.current_state['sync_level']:.2f}")
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Awakening training episode {episode}, current state: {self.current_state}")
                print(f"Phi4 state: {self.phi4_sleep_state:.2f}, Memory state: {self.episodic_memory_sleep_state:.2f}, Sync: {self.current_state['sync_level']:.2f}")
        
        # Final verification of synchronization before completing awakening
        model_state_avg = (self.current_state['attention'] + self.current_state['compute']) / 2
        sync_diff = abs(model_state_avg - self.current_state['episodic_memory_state'])
        
        if sync_diff > 0.1:
            print("WARNING: Components not fully synchronized at end of awakening process.")
            print("Performing final synchronization adjustment...")
            
            # Force final synchronization for safety
            self.current_state['episodic_memory_state'] = model_state_avg
            self.episodic_memory_sleep_state = self.phi4_sleep_state
            self.current_state['sync_level'] = 1.0
            
            print("Final synchronization complete.")
        
        # Final update
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.update_gates()
        
        # Save checkpoint
        save_checkpoint("awakening_final", self)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print("Synchronized awakening process completed successfully.")
        return self.current_state
    
    def train_sleep_wake_mechanisms(self, num_episodes=100, steps_per_episode=20):
        """
        Explicitly train the sleep and wake mechanisms using reinforcement learning.
        This allows the model to learn optimal policies for transitioning between
        sleep and wake states before actually using these mechanisms in production.
        
        Args:
            num_episodes: Number of training episodes for each mechanism
            steps_per_episode: Number of steps per episode
            
        Returns:
            training_results: Dictionary containing training metrics
        """
        if not hasattr(self, 'sleep_system'):
            print("Sleep system not initialized.")
            return None
            
        print("Starting training for sleep and wake mechanisms...")
        results = {
            'sleep': {'initial_state': None, 'final_state': None, 'episodes': []},
            'wake': {'initial_state': None, 'final_state': None, 'episodes': []}
        }
        
        # Train deep sleep mechanism
        print("\n=== Training Deep Sleep Mechanism ===")
        
        # Store initial state
        initial_sleep_state = self.sleep_system.current_state.copy()
        results['sleep']['initial_state'] = initial_sleep_state
        
        # Temporarily set to awake state for training
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        
        # Save initial checkpoint before training begins
        save_checkpoint("sleep_wake_training_start", self)
        
        # Track training progress for mid-training checkpoint
        mid_training_point = num_episodes // 2
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_sleep_state.copy()
            self.sleep_system.previous_state = initial_sleep_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for sleep, we decrease attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate reward
                reward = self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target sleep state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.deep_sleep_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['sleep']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Sleep training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['sleep']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint
            if episode == mid_training_point:
                save_checkpoint("sleep_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final sleep state
        results['sleep']['final_state'] = self.sleep_system.current_state.copy()
        
        # Save checkpoint after sleep mechanism training
        save_checkpoint("sleep_mechanism_complete", self)
        
        # Train awakening mechanism
        print("\n=== Training Awakening Mechanism ===")
        
        # Store initial state (low attention/compute)
        initial_wake_state = {
            'attention': self.sleep_system.deep_sleep_params['target_attention'],
            'compute': self.sleep_system.deep_sleep_params['target_compute'],
            'metric': 0.0
        }
        results['wake']['initial_state'] = initial_wake_state
        
        # Temporarily set to sleep state for training
        self.sleep_system.is_sleeping = True
        self.sleep_system.is_fully_shutdown = False
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_wake_state.copy()
            self.sleep_system.previous_state = initial_wake_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for wake, we increase attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                reward = -self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target awake state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.awakening_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.awakening_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['wake']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Wake training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['wake']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint for wake mechanism
            if episode == mid_training_point:
                save_checkpoint("wake_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final wake state
        results['wake']['final_state'] = self.sleep_system.current_state.copy()
        
        # Reset to normal awake state
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        self.sleep_system.current_state = {
            'attention': self.sleep_system.awakening_params['target_attention'],
            'compute': self.sleep_system.awakening_params['target_compute'],
            'metric': 0.0
        }
        self.sleep_system.update_gates()
        
        print("\nTraining completed. Sleep and wake mechanisms have been trained.")
        
        # Save final checkpoint
        save_checkpoint("sleep_wake_training_complete", self)
        
        # Play sound to indicate training completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return results
    
    def calculate_deep_sleep_reward(self, current_state, action, previous_state, previous_action):
        """
        Calculate the deep sleep reward based on current and previous states and actions.
        Ensures synchronization between phi4 model and episodic memory to prevent sleep paralysis.
        
        Args:
            current_state: Current state dictionary
            action: Current action dictionary
            previous_state: Previous state dictionary
            previous_action: Previous action dictionary
            
        Returns:
            reward: Deep sleep reward
        """
        # Ensure episodic memory state is included in current_state
        if 'episodic_memory_state' not in current_state and hasattr(self, 'episodic_memory_sleep_state'):
            current_state = current_state.copy()  # Create a copy to avoid modifying the original
            current_state['episodic_memory_state'] = self.episodic_memory_sleep_state
            
        # Calculate sync level for logging
        if 'episodic_memory_state' in current_state:
            model_state_avg = (current_state['attention'] + current_state['compute']) / 2
            sync_diff = abs(model_state_avg - current_state['episodic_memory_state'])
            sync_threshold = self.deep_sleep_params.get('sync_threshold', 0.05)
            sync_level = max(0.0, 1.0 - (sync_diff / sync_threshold * 2))
            
            # Log warning if sync level is dangerously low during awakening
            if action.get('delta_attention', 0) > 0 and sync_level < 0.5:
                print(f"WARNING: Low sync level ({sync_level:.2f}) during awakening - sleep paralysis risk!")
        
            # Define the deep sleep reward calculation function
            def calculate_deep_sleep_reward(current_state, action, previous_state, previous_action, params):
                """
                Calculate the reward for deep sleep based on state transitions and parameters.
                
                Args:
                    current_state: Current state dictionary
                    action: Current action dictionary
                    previous_state: Previous state dictionary
                    previous_action: Previous action dictionary
                    params: Deep sleep parameters
                    
                Returns:
                    reward: Calculated reward value
                """
                # Target values from parameters
                target_attention = params.get('target_attention', 0.1)
                target_compute = params.get('target_compute', 0.2)
                
                # Weights for different components
                lambda_attention = params.get('lambda_attention', 1.0)
                lambda_compute = params.get('lambda_compute', 1.0)
                lambda_smoothness = params.get('lambda_smoothness', 0.5)
                
                # Reward for approaching target attention level
                attention_reward = -abs(current_state['attention'] - target_attention)
                
                # Reward for approaching target compute level
                compute_reward = -abs(current_state['compute'] - target_compute)
                
                # Smoothness reward (penalize large changes)
                smoothness_penalty = 0
                if previous_state:
                    attention_change = abs(current_state['attention'] - previous_state['attention'])
                    compute_change = abs(current_state['compute'] - previous_state['compute'])
                    smoothness_penalty = -(attention_change + compute_change)
                
                # Synchronization penalty if enabled
                sync_penalty = 0
                if params.get('episodic_memory_sync', True) and 'episodic_memory_state' in current_state:
                    model_state_avg = (current_state['attention'] + current_state['compute']) / 2
                    sync_diff = abs(model_state_avg - current_state['episodic_memory_state'])
                    sync_threshold = params.get('sync_threshold', 0.05)
                    
                    # Apply penalty if difference exceeds threshold
                    if sync_diff > sync_threshold:
                        sync_penalty = -params.get('sync_penalty', 2.0) * (sync_diff / sync_threshold)
                
                # Combine all reward components
                total_reward = (
                    lambda_attention * attention_reward +
                    lambda_compute * compute_reward +
                    lambda_smoothness * smoothness_penalty +
                    sync_penalty
                )
                
                return total_reward
                
            # Call the reward calculation function with all parameters
            reward = calculate_deep_sleep_reward(
                current_state, action, previous_state, previous_action, self.deep_sleep_params
            )
            
            print("Deep Sleep Reward calculated:", reward)
    
            # Simulate the training step:
            save_checkpoint("deep_sleep_step")
            # (… training operations would be performed here …)
            save_checkpoint("deep_sleep_step_checkpoint")
    
            play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    
            input("Deep sleep training step completed. Press Enter to continue...")
    
            return reward
