import torch
import glob
import os
import numpy as np
import mrcfile as mf
import random
from torch.utils.data import Dataset

# Helper functions
def search_by_re(root_dir, re):
    return glob.glob(root_dir + re)

def split_file_path(full_path):
    path, name = os.path.split(full_path)
    prefix, post_fix = os.path.splitext(name)
    return path, prefix, post_fix

def read_mrc(mrc_path):
    with mf.open(mrc_path) as mp:
        return mp.data

def save_mrc(data, mrc_path):
    with mf.new(mrc_path) as mp:
        mp.set_data(data)

# class StructureProjPairs(Dataset):
#     def __init__(self, structure_paths, proj_paths):
#         super(StructureProjPairs, self).__init__()

#         self.proj_data, self.structure_paths = [], []
#         for sp, pp in zip(structure_paths, proj_paths):
#             proj = read_mrc(pp)
#             self.proj_data.append(proj)
#             for _ in range(proj.shape[0]):
#                 self.structure_paths.append(sp)

#         self.proj_data = np.concatenate(self.proj_data)

#     def __getitem__(self, idx):
#         # proj: [1, 64, 64]
#         # structure: [1, 64, 64, 64]
#         proj = torch.tensor(self.proj_data[idx], requires_grad = True).unsqueeze(0)
#         structure = torch.tensor(read_mrc(self.structure_paths[idx]), requires_grad = True).unsqueeze(0)
#         return proj, structure

#     def __len__(self):
#         return self.proj_data.shape[0]

class StructureProjPairs(Dataset):
    def __init__(self, structure_paths, proj_paths, shuffle = True):
        super(StructureProjPairs, self).__init__()

        self.proj_data, self.structure_data, self.idx_pairs = [], [], []
        for idx, (sp, pp) in enumerate(zip(structure_paths, proj_paths)):
            structure, proj = read_mrc(sp), read_mrc(pp)
            self.structure_data.append(structure)
            self.proj_data.append(proj)
            for j in range(proj.shape[0]):
                self.idx_pairs.append((idx, j))

        if shuffle:
            random.shuffle(self.idx_pairs)

    def __getitem__(self, idx):
        # proj: [1, 64, 64]
        # # structure: [1, 64, 64, 64]
        s_idx, p_idx = self.idx_pairs[idx]
        proj = torch.tensor(self.proj_data[s_idx][p_idx]).unsqueeze(0)
        structure = torch.tensor(self.structure_data[s_idx]).unsqueeze(0)
        return proj, structure

    def __len__(self):
        return len(self.idx_pairs)


if __name__ == '__main__':
    structures = sorted(search_by_re('../../Dataset/PDB/testSet/', '*/*_64.mrc'))
    projs = sorted(search_by_re('../../Dataset/PDB/testSet/', '*/*_64_projs.mrcs'))
    sp = StructureProjPairs(structures, projs)
    loader = torch.utils.data.DataLoader(sp, batch_size = 4, shuffle = True)

    for p, s in loader:
        print(p.size(), s.size())
        break
