"""Unittest coverage for versioned Stratagem checkpoint persistence."""

import tempfile
import unittest
from pathlib import Path

import torch

from fp.stratagem.learning.model import StratagemModelWrapper
from fp.stratagem.learning.trainer import StratagemTrainer
from fp.stratagem.learning.weights import CHECKPOINT_VERSION


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_metadata_and_trainer_state(self):
        source_model = StratagemModelWrapper(hidden_sizes=(8,))
        source_trainer = StratagemTrainer(source_model, batch_size=1)
        source_trainer.training_step = 7
        source_trainer.episode_count = 3
        source_trainer.total_reward = 1.25
        with torch.no_grad():
            next(source_model.model.parameters()).fill_(0.125)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            metadata = source_trainer.save_checkpoint(path)

            self.assertTrue(path.is_file())
            self.assertEqual(metadata.checkpoint_version, CHECKPOINT_VERSION)
            self.assertEqual(metadata.training_step, 7)
            self.assertEqual(metadata.episode_count, 3)
            self.assertEqual(list(Path(temporary_directory).glob(".checkpoint.pt.*")), [])

            restored_model = StratagemModelWrapper(hidden_sizes=(8,))
            restored_trainer = StratagemTrainer(restored_model, batch_size=1)
            restored_metadata = restored_trainer.load_checkpoint(path)

            self.assertEqual(restored_metadata, metadata)
            self.assertEqual(restored_trainer.training_step, 7)
            self.assertEqual(restored_trainer.episode_count, 3)
            self.assertEqual(restored_trainer.total_reward, 1.25)
            self.assertTrue(
                torch.equal(
                    next(source_model.model.parameters()),
                    next(restored_model.model.parameters()),
                )
            )

    def test_incompatible_checkpoint_is_rejected(self):
        model = StratagemModelWrapper(hidden_sizes=(8,))
        trainer = StratagemTrainer(model, batch_size=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            trainer.save_checkpoint(path)
            payload = torch.load(path, weights_only=True)
            payload["metadata"]["checkpoint_version"] = CHECKPOINT_VERSION + 1
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "Checkpoint version"):
                trainer.load_checkpoint(path)

    def test_missing_checkpoint_fails_loudly(self):
        model = StratagemModelWrapper(hidden_sizes=(8,))
        trainer = StratagemTrainer(model, batch_size=1)
        with self.assertRaises(FileNotFoundError):
            trainer.load_checkpoint("does-not-exist.pt")

    def test_model_wrapper_loads_compatible_trainer_checkpoint_for_inference(self):
        source_model = StratagemModelWrapper(hidden_sizes=(8,))
        source_trainer = StratagemTrainer(source_model, batch_size=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            source_trainer.save_checkpoint(path)

            restored_model = StratagemModelWrapper(model_path=path)

        self.assertTrue(
            torch.equal(
                next(source_model.model.parameters()),
                next(restored_model.model.parameters()),
            )
        )


if __name__ == "__main__":
    unittest.main()