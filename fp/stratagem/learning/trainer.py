"""
Training system for Stratagem's learned model.
Handles supervised learning from self-play data and experience replay.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import random

from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.features import extract_features
from fp.stratagem.learning.weights import CheckpointMetadata, load_training_checkpoint, save_training_checkpoint
from fp.stratagem.core.observation import Observation
from fp.stratagem.config import CONFIG


class StratagemExperience:
    """
    Represents a single experience tuple for training.
    (observation, action_taken, reward, next_observation, done, action_values)
    """

    def __init__(
        self,
        observation: Observation,
        action_taken: str,
        reward: float,
        next_observation: Optional[Observation],
        done: bool,
        action_values: Optional[Dict[str, float]] = None,
        model_prediction: Optional[Tuple[float, np.ndarray]] = None
    ):
        """
        Initialize experience tuple.

        Args:
            observation: Current battle observation
            action_taken: Action that was taken
            reward: Reward received after taking action
            next_observation: Observation after taking action (None if terminal)
            done: Whether the episode is done
            action_values: MCTS action values used for training target (if available)
            model_prediction: Model's prediction (value, action_probs) for this state
        """
        self.observation = observation
        self.action_taken = action_taken
        self.reward = reward
        self.next_observation = next_observation
        self.done = done
        self.action_values = action_values or {}
        self.model_prediction = model_prediction

        # Extract features for efficiency
        self.features = extract_features(observation)
        if next_observation is not None:
            self.next_features = extract_features(next_observation)
        else:
            self.next_features = None

        # Convert action to index (will be mapped by trainer)
        self.action_index = -1  # To be set by trainer


class StratagemExperienceDataset(Dataset):
    """
    PyTorch Dataset for Stratagem experiences.
    """

    def __init__(self, experiences: List[StratagemExperience], action_to_idx: Dict[str, int]):
        """
        Initialize dataset.

        Args:
            experiences: List of experience tuples
            action_to_idx: Mapping from action strings to indices
        """
        self.experiences = experiences
        self.action_to_idx = action_to_idx
        self.feature_size = len(experiences[0].features) if experiences else 0

        # Resolve each action against the fixed policy-head vocabulary.
        for exp in self.experiences:
            try:
                exp.action_index = self.action_to_idx[exp.action_taken]
            except KeyError as error:
                raise ValueError(
                    f"Experience action is not in the fixed model vocabulary: {exp.action_taken}"
                ) from error

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, float, Optional[np.ndarray], bool]:
        """
        Get a single experience.

        Returns:
            Tuple of (features, action_index, reward, next_features, done)
        """
        exp = self.experiences[idx]
        return (
            exp.features,
            exp.action_index,
            exp.reward,
            exp.next_features,
            exp.done
        )


class StratagemTrainer:
    """
    Trains the Stratagem learned model using experience replay.
    """

    def __init__(
        self,
        model_wrapper: StratagemModelWrapper,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        experience_buffer_size: int = 10000
    ):
        """
        Initialize the trainer.

        Args:
            model_wrapper: Wrapper containing the model to train
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            experience_buffer_size: Maximum size of experience replay buffer
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.experience_buffer_size = experience_buffer_size

        # Experience replay buffer
        self.experience_buffer: List[StratagemExperience] = []
        self.action_to_idx = dict(model_wrapper.action_to_idx)
        self.idx_to_action = {
            index: action for action, index in self.action_to_idx.items()
        }

        # Optimizer and loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.value_criterion = nn.MSELoss()
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')

        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0

    def add_experience(
        self,
        observation: Observation,
        action_taken: str,
        reward: float,
        next_observation: Optional[Observation],
        done: bool,
        action_values: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add an experience to the replay buffer.

        Args:
            observation: Current battle observation
            action_taken: Action that was taken
            reward: Reward received
            next_observation: Next observation (None if terminal)
            done: Whether episode is done
            action_values: MCTS action values for training target
        """
        experience = StratagemExperience(
            observation=observation,
            action_taken=action_taken,
            reward=reward,
            next_observation=next_observation,
            done=done,
            action_values=action_values
        )
        experience.action_index = self.model_wrapper.action_to_index(action_taken)

        self.experience_buffer.append(experience)

        # Keep buffer size within limits
        if len(self.experience_buffer) > self.experience_buffer_size:
            self.experience_buffer.pop(0)  # Remove oldest experience

    def add_episode_experiences(
        self,
        experiences: List[Tuple[Observation, str, float, bool]],
        final_reward: float = 0.0
    ) -> None:
        """
        Add a complete episode of experiences.

        Args:
            experiences: List of (observation, action, reward, done) tuples
            final_reward: Final reward for the episode (used for Monte Carlo returns)
        """
        # Convert to experience tuples with next_observation
        exp_tuples = []
        for i, (obs, action, reward, done) in enumerate(experiences):
            next_obs = experiences[i + 1][0] if i + 1 < len(experiences) else None
            exp_tuples.append((obs, action, reward, next_obs, done))

        # Calculate discounted returns if needed
        # For now, we'll use immediate rewards, but this could be enhanced
        # with Monte Carlo or TD(lambda) returns

        for obs, action, reward, next_obs, done in exp_tuples:
            self.add_experience(obs, action, reward, next_obs, done)

        self.episode_count += 1
        self.total_reward += sum(exp[2] for exp in experiences)

    def sample_batch(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[StratagemExperience],
    ]:
        """
        Sample a batch of experiences from the replay buffer.

        Returns:
            Tuple of (features, actions, rewards, next_features, dones)
        """
        if not self.experience_buffer:
            raise ValueError("Cannot sample an empty experience buffer")
        if len(self.experience_buffer) < self.batch_size:
            # Not enough experiences, sample with replacement
            batch = random.choices(self.experience_buffer, k=self.batch_size)
        else:
            batch = random.sample(self.experience_buffer, self.batch_size)

        self._validate_batch(batch)

        # Extract batch components
        features = np.array([exp.features for exp in batch])
        actions = np.array([exp.action_index for exp in batch])
        rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
        next_features = []
        dones = np.array([exp.done for exp in batch], dtype=np.float32)

        for exp in batch:
            if exp.next_features is not None:
                next_features.append(exp.next_features)
            else:
                # Use zeros for terminal states
                next_features.append(np.zeros_like(exp.features))

        next_features = np.array(next_features)

        # Convert to tensors
        features_tensor = torch.FloatTensor(features).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_features_tensor = torch.FloatTensor(next_features).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        return (
            features_tensor,
            actions_tensor,
            rewards_tensor,
            next_features_tensor,
            dones_tensor,
            batch,
        )

    def _validate_batch(self, batch: List[StratagemExperience]) -> None:
        """Reject malformed replay data before it reaches a tensor operation."""
        for experience in batch:
            if experience.features.shape != (self.model.feature_size,):
                raise ValueError("Experience feature size does not match the model")
            if not np.isfinite(experience.features).all() or not np.isfinite(experience.reward):
                raise ValueError("Experience contains non-finite features or reward")
            if not 0 <= experience.action_index < self.model.action_size:
                raise ValueError("Experience action index is outside the model vocabulary")
            for action, value in experience.action_values.items():
                self.model_wrapper.action_to_index(action)
                if not np.isfinite(value):
                    raise ValueError("Experience MCTS action values must be finite")
            if (
                experience.next_features is not None
                and (
                    experience.next_features.shape != (self.model.feature_size,)
                    or not np.isfinite(experience.next_features).all()
                )
            ):
                raise ValueError("Next experience features do not match the model")

    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.

        Returns:
            Dictionary of training losses
        """
        if not self.experience_buffer:
            return {"value_loss": 0.0, "policy_loss": 0.0, "total_loss": 0.0}

        # Sample batch
        features, actions, rewards, next_features, dones, batch = self.sample_batch()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        values, policy_logits = self.model(features)
        values = values.squeeze(-1)  # Remove last dimension

        # Compute target values using bootstrapped rewards (simple TD(0) target)
        with torch.no_grad():
            next_values, _ = self.model(next_features)
            next_values = next_values.squeeze(-1)
            target_values = rewards + (CONFIG.gamma * next_values * (1 - dones))

        # Value loss
        value_loss = self.value_criterion(values, target_values)

        target_policy = self._policy_targets(batch, actions)

        policy_loss = self.policy_criterion(
            torch.nn.functional.log_softmax(policy_logits, dim=-1),
            target_policy
        )

        # Total loss
        total_loss = value_loss + policy_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        self.training_step += 1

        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item()
        }

    def _policy_targets(
        self, batch: List[StratagemExperience], actions: torch.Tensor
    ) -> torch.Tensor:
        """Use normalized MCTS action values where available, otherwise action labels."""
        targets = torch.zeros(
            (len(batch), self.model.action_size), dtype=torch.float32, device=self.device
        )
        for row, experience in enumerate(batch):
            if experience.action_values:
                indexed_scores = [
                    (self.model_wrapper.action_to_index(action), value)
                    for action, value in experience.action_values.items()
                ]
                maximum_score = max(value for _, value in indexed_scores)
                weights = [np.exp(value - maximum_score) for _, value in indexed_scores]
                total_weight = sum(weights)
                for (index, _), weight in zip(indexed_scores, weights):
                    targets[row, index] = float(weight / total_weight)
            else:
                targets[row, actions[row]] = 1.0
        return targets

    def train_epoch(self, num_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            num_batches: Number of batches to train (if None, train until buffer is exhausted)

        Returns:
            Average losses for the epoch
        """
        if num_batches is None:
            num_batches = max(1, len(self.experience_buffer) // self.batch_size)

        total_losses = {"value_loss": 0.0, "policy_loss": 0.0, "total_loss": 0.0}

        for _ in range(num_batches):
            losses = self.train_step()
            for key in total_losses:
                total_losses[key] += losses[key]

        # Average losses
        for key in total_losses:
            total_losses[key] /= max(1, num_batches)

        return total_losses

    def save_checkpoint(self, filepath: str) -> CheckpointMetadata:
        """Atomically save a versioned checkpoint for safe later resumption."""
        return save_training_checkpoint(
            filepath,
            self.model_wrapper,
            self.optimizer,
            training_step=self.training_step,
            episode_count=self.episode_count,
            total_reward=self.total_reward,
        )

    def load_checkpoint(self, filepath: str) -> CheckpointMetadata:
        """Strictly validate and restore a compatible versioned checkpoint."""
        metadata, total_reward = load_training_checkpoint(
            filepath, self.model_wrapper, self.optimizer
        )
        self.training_step = metadata.training_step
        self.episode_count = metadata.episode_count
        self.total_reward = total_reward
        return metadata

    def get_stats(self) -> Dict[str, float | int]:
        """
        Get training statistics.

        Returns:
            Dictionary of training stats
        """
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.episode_count),
            'buffer_size': len(self.experience_buffer),
            'action_space_size': len(self.action_to_idx),
            'learning_rate': self.learning_rate
        }


# Global trainer instance
_trainer = None


def get_trainer(model_wrapper: Optional[StratagemModelWrapper] = None) -> StratagemTrainer:
    """
    Get or create the global trainer instance.

    Args:
        model_wrapper: Model wrapper to use (if None, creates new one)

    Returns:
        StratagemTrainer instance
    """
    global _trainer
    if _trainer is None:
        if model_wrapper is None:
            model_wrapper = StratagemModelWrapper()
        _trainer = StratagemTrainer(model_wrapper)
    return _trainer


if __name__ == "__main__":
    # Test the trainer
    print("Stratagem Trainer")
    print("=" * 20)

    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    # Create model wrapper and trainer
    model_wrapper = StratagemModelWrapper(device=device)
    trainer = StratagemTrainer(model_wrapper)

    print(f"Feature size: {model_wrapper.feature_extractor.feature_size}")
    print(f"Action size: {model_wrapper.action_size}")
    print(f"Device: {device}")

    # Print initial stats
    stats = trainer.get_stats()
    print("\nInitial stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTrainer initialized successfully!")