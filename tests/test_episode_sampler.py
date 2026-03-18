import torch
from torch.utils.data import DataLoader

from archdyn.config import FewShotConfig
from archdyn.data.episodic import EpisodeDataset, EpisodeSampler


class DummyEpisodeDataset:
    def __init__(self) -> None:
        self.samples = []
        self.items = []
        for label in range(10):
            for index in range(25):
                self.samples.append((f"item_{label}_{index}.png", label))
                self.items.append((torch.zeros(3, 32, 32), label))

    def __getitem__(self, index: int):
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)


def test_episode_sampler_shapes() -> None:
    dataset = DummyEpisodeDataset()
    config = FewShotConfig(n_way=10, k_shot=2, q_query=3, train_episodes=1, val_episodes=1, test_episodes=1)
    sampler = EpisodeSampler(dataset, config, seed=13)
    episode = sampler.sample_episode()
    assert episode["support_images"].shape[0] == 20
    assert episode["query_images"].shape[0] == 30
    assert episode["support_labels"].shape[0] == 20
    assert episode["query_labels"].shape[0] == 30


def test_episode_dataset_loader_returns_episode_dict() -> None:
    dataset = DummyEpisodeDataset()
    config = FewShotConfig(n_way=10, k_shot=2, q_query=3, train_episodes=1, val_episodes=1, test_episodes=1)
    sampler = EpisodeSampler(dataset, config, seed=13)
    episode_loader = DataLoader(EpisodeDataset(sampler, total_episodes=2), batch_size=None, num_workers=0)

    first_episode = next(iter(episode_loader))
    assert first_episode["support_images"].shape == (20, 3, 32, 32)
    assert first_episode["query_images"].shape == (30, 3, 32, 32)


def test_episode_dataset_is_deterministic_per_index() -> None:
    dataset = DummyEpisodeDataset()
    config = FewShotConfig(n_way=10, k_shot=2, q_query=3, train_episodes=1, val_episodes=1, test_episodes=1)
    sampler = EpisodeSampler(dataset, config, seed=13)
    episode_dataset = EpisodeDataset(sampler, total_episodes=2)

    first = episode_dataset[0]
    second = episode_dataset[0]
    assert torch.equal(first["support_images"], second["support_images"])
    assert torch.equal(first["support_labels"], second["support_labels"])
    assert torch.equal(first["query_images"], second["query_images"])
    assert torch.equal(first["query_labels"], second["query_labels"])
