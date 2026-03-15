import torch

from archdyn.config import FewShotConfig
from archdyn.data.episodic import EpisodeSampler


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


def test_episode_sampler_shapes() -> None:
    dataset = DummyEpisodeDataset()
    config = FewShotConfig(n_way=10, k_shot=2, q_query=3, train_episodes=1, val_episodes=1, test_episodes=1)
    sampler = EpisodeSampler(dataset, config, seed=13)
    episode = sampler.sample_episode()
    assert episode["support_images"].shape[0] == 20
    assert episode["query_images"].shape[0] == 30
    assert episode["support_labels"].shape[0] == 20
    assert episode["query_labels"].shape[0] == 30
