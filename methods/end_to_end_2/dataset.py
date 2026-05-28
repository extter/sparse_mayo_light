from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

VALID_ANGLES = [45, 60, 90, 180]
VALIDATION_PATIENT = 'C030'   # paziente di train usato come validation


class MayoRecoCorruptedDataset(Dataset):
    """
    Carica triple (raw_image, sinogram_corrotto, reco) per un dato split e n_angles.

    Struttura attesa:
        data/
        ├── raw/
        │   ├── train/C001/0.png ...   (C030 qui → diventa validation)
        │   └── test/C002/0.png  ...
        │
        ├── reco/
        │   └── angles_<N>/
        │       ├── train/      train_C001_0.npy ...
        │       ├── test/       test_C002_0.npy  ...
        │       └── validation/ validation_C030_0.npy ...
        │
        └── sinogram_corrupted/
            ├── train/
            │   └── angles_<N>/  train_C001_0.npy ...
            ├── test/
            │   └── angles_<N>/  test_C002_0.npy  ...
            └── validation/
                └── angles_<N>/  train_C030_0.npy ...   ← prefisso 'train_', non 'validation_'
    """

    def __init__(self, data_root, split, n_angles, data_shape=256,
                 validation_patient=VALIDATION_PATIENT, verbose=True):
        super().__init__()

        assert split in ('train', 'test', 'validation'), f'Split non valido: {split}'
        assert n_angles in VALID_ANGLES, f'n_angles deve essere uno tra {VALID_ANGLES}'

        data_root = Path(data_root)
        angles_folder = f'angles_{n_angles}'

        self.reco_dir = data_root / 'reco' / angles_folder / split
        self.corr_dir = data_root / 'sinogram_corrupted' / split / angles_folder
        self.data_shape = data_shape
        self.split = split

        # Le raw di validation stanno fisicamente in raw/train/C030
        if split == 'validation':
            raw_root = data_root / 'raw' / 'train' / validation_patient
            self.input_files = sorted(raw_root.glob('*.png'))
        else:
            raw_root = data_root / 'raw' / split
            self.input_files = sorted(raw_root.glob('*/*.png'))

        # Entrambi reco e sinogram_corrupted usano prefisso 'train_' per i file di validation



        file_prefix = 'train' if split == 'validation' else split

        self.pairs = []
        for inp in self.input_files:
            patient_id = validation_patient if split == 'validation' else inp.parent.name
            slice_id = inp.stem

            corr = self.corr_dir / f'{file_prefix}_{patient_id}_{slice_id}.npy'
            tgt  = self.reco_dir / f'{file_prefix}_{patient_id}_{slice_id}.npy'

            if corr.exists() and tgt.exists():
                self.pairs.append((inp, corr, tgt))
            elif verbose:
                print(f'Missing corr: {corr.name}' if not corr.exists() else f'Missing reco: {tgt.name}')

        if len(self.pairs) == 0:
            raise RuntimeError(
                f'Nessuna coppia trovata per split={split}, angles={n_angles}.\n'
                f'  reco dir:  {self.reco_dir}\n'
                f'  corr dir:  {self.corr_dir}'
            )

        print(f'{split} | angles={n_angles}: {len(self.pairs)} triple trovate')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, corr_path, target_path = self.pairs[idx]

        # Ground truth raw
        x = Image.open(input_path).convert('L')
        x = transforms.ToTensor()(x)
        x = transforms.Resize((self.data_shape, self.data_shape))(x)

        # Sinogramma corrotto
        y_delta = np.load(corr_path).astype(np.float32)
        y_delta = torch.from_numpy(y_delta)
        if y_delta.ndim == 2:
            y_delta = y_delta.unsqueeze(0)

        # Reco target
        target = np.load(target_path).astype(np.float32)
        target = torch.from_numpy(target)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        target = transforms.Resize((self.data_shape, self.data_shape))(target)

        return x, y_delta, target


def get_dataloaders(data_root, n_angles, data_shape=256, batch_size=16,
                    validation_patient=VALIDATION_PATIENT, verbose=True):
    train_dataset = MayoRecoCorruptedDataset(
        data_root, 'train', n_angles, data_shape, validation_patient, verbose
    )
    val_dataset = MayoRecoCorruptedDataset(
        data_root, 'validation', n_angles, data_shape, validation_patient, verbose
    )
    test_dataset = MayoRecoCorruptedDataset(
        data_root, 'test', n_angles, data_shape, validation_patient, verbose
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader