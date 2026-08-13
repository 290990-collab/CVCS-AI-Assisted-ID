# src/vision/data/loader.py

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.vision.data.preprocess import get_transform


class RPLANDataset(Dataset):
    """
    Dataset per le immagini RPLAN.

    1 campione = 1 coppia (tensore_immagine, path_stringa)

    Due modalità d'uso:
      - 'data_dir':    raccoglie tutti i PNG ricorsivamente da una cartella
      - 'image_paths': usa esattamente la lista fornita

    Il preprocessing e' parametrico
    """

    def __init__(
        self,
        data_dir: str | None = None,
        image_size: int = 224,
        image_paths: list | None = None,
        transform=None,
    ):

        if data_dir is None and image_paths is None:
            raise ValueError("Specificare data_dir oppure image_paths (nessuno dei due fornito)")
        if data_dir is not None and image_paths is not None:
            raise ValueError("Specificare data_dir oppure image_paths, non entrambi")

        # transform specifico dell'encoder
        self.transform = transform if transform is not None else get_transform(image_size)

        if image_paths is not None:
            self.image_paths = [Path(p) for p in image_paths]
            source = f"lista di {len(self.image_paths)} path"
        else:
            self.data_dir    = Path(data_dir)
            self.image_paths = sorted(self.data_dir.rglob("*.png"))
            source           = str(data_dir)

        if len(self.image_paths) == 0:
            raise ValueError(f"Nessuna immagine trovata in: {source}")

        print(f"\n[RPLANDataset] Caricate {len(self.image_paths)} immagini da {source}\n")

    # len(dataset)
    def __len__(self) -> int:
        return len(self.image_paths)

    # dataset[idx]
    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(path)


def get_dataloader(
    data_dir: str,
    batch_size: int = 128,       # immagini passate in blocco sulla GPU
    prefetch_factor=4,
    image_size: int = 224,
    num_workers: int = 8,
    shuffle: bool = False,      
    transform=None,             
) -> DataLoader:
    """
    Crea un DataLoader per RPLAN.

    Args:
        data_dir:    cartella con i PNG
        batch_size:  immagini per batch
        image_size:  dimensione resize (usata solo se transform è None)
        num_workers: processi paralleli per il caricamento
        shuffle:     mescola i dati (False per retrieval, True per training)
        transform:   preprocessing specifico dell'encoder (None = default ImageNet)

    Returns:
        DataLoader configurato
    """
    dataset = RPLANDataset(data_dir=data_dir, image_size=image_size, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True   
    )
