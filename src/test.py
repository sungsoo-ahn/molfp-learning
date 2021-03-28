from data.transform import AddRandomWalkSubstruct
from data.dataset import MoleculeDataset

if __name__ == "__main__":
    dataset = MoleculeDataset(
        "../resource/dataset/zinc_standard_agent",
        dataset="zinc_standard_agent",
        transform=AddRandomWalkSubstruct(min_walk_length=3, max_walk_length=5),
    )
    print(dataset[0])