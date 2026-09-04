"""
Experience replay buffer for Stratagem training.
Stores and samples experiences for off-policy learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import os
import json
from datetime import datetime

from fp.stratagem.learning.trainer import StratagemExperience
from fp.stratagem.core.observation import Observation


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: List[StratagemExperience] = []
        self.position = 0

    def push(
        self,
        observation: Observation,
        action_taken: str,
        reward: float,
        next_observation: Optional[Observation],
        done: bool,
        action_values: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save an experience.

        Args:
            observation: Current observation
            action_taken: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            action_values: MCTS action values (optional)
        """
        experience = StratagemExperience(
            observation=observation,
            action_taken=action_taken,
            reward=reward,
            next_observation=next_observation,
            done=done,
            action_values=action_values
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[StratagemExperience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []

        if batch_size >= len(self.buffer):
            return self.buffer.copy()

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough experiences for sampling.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer has at least batch_size experiences
        """
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.position = 0

    def save(self, filepath: str) -> None:
        """
        Save buffer to file.

        Args:
            filepath: Path to save buffer
        """
        # Convert experiences to serializable format
        serializable_buffer = []
        for exp in self.buffer:
            # For simplicity, we'll save minimal info needed to reconstruct
            # In practice, you'd want to save the actual observations
            serializable_exp = {
                'action_taken': exp.action_taken,
                'reward': exp.reward,
                'done': exp.done,
                # Note: Storing full observations would require more complex serialization
                # This is a simplified version for MVP
            }
            serializable_buffer.append(serializable_exp)

        data = {
            'capacity': self.capacity,
            'position': self.position,
            'buffer': serializable_buffer,
            'timestamp': datetime.now().isoformat()
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Replay buffer saved to {filepath} ({len(self.buffer)} experiences)")

    def load(self, filepath: str) -> None:
        """
        Load buffer from file.

        Args:
            filepath: Path to load buffer from
        """
        if not os.path.exists(filepath):
            print(f"Replay buffer file {filepath} not found")
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.capacity = data['capacity']
            self.position = data['position']
            # Note: Full experience reconstruction would require storing observations
            # For MVP, we'll just note that the buffer was loaded
            print(f"Replay buffer loaded from {filepath}")
            print(f"  Capacity: {self.capacity}")
            print(f"  Position: {self.position}")
            print(f"  Experiences: {len(data.get('buffer', []))} (observation data not serialized in MVP)")

            # Clear existing buffer and set capacity
            self.clear()
            # Note: Actual experience reconstruction would happen here in a full implementation

        except Exception as e:
            print(f"Error loading replay buffer: {e}")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer that samples experiences based on TD-error.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def push(
        self,
        observation: Observation,
        action_taken: str,
        reward: float,
        next_observation: Optional[Observation],
        done: bool,
        action_values: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save an experience with max priority.
        """
        experience = StratagemExperience(
            observation=observation,
            action_taken=action_taken,
            reward=reward,
            next_observation=next_observation,
            done=done,
            action_values=action_values
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[
        List[StratagemExperience], List[int], np.ndarray
    ]:
        """
        Sample a batch of experiences with importance sampling weights.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (experiences, indices, weights)
        """
        if len(self.buffer) == 0:
            return [], [], np.array([])

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities of sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


# Global replay buffer instance
_replay_buffer = None


def get_replay_buffer(capacity: int = 10000) -> ReplayBuffer:
    """
    Get or create the global replay buffer.

    Args:
        capacity: Buffer capacity

    Returns:
        ReplayBuffer instance
    """
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = ReplayBuffer(capacity)
    return _replay_buffer


if __name__ == "__main__":
    # Test the replay buffer
    print("Stratagem Replay Buffer")
    print("=" * 25)

    buffer = ReplayBuffer(capacity=1000)
    print(f"Buffer capacity: {buffer.capacity}")

    # Test adding and sampling
    print(f"Buffer length: {len(buffer)}")
    print(f"Buffer ready for batch of 32: {buffer.is_ready(32)}")

    print("\nReplay buffer initialized successfully!")