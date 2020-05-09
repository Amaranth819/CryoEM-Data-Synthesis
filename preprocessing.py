import argparse
import os
import mrcfile as mf
import glob
import numpy as np
from scipy.ndimage.interpolation import rotate
from math import radians

# Relion
def resize(orig_structure_path, dest_structure_path, new_box_size):
    if not os.path.exists(orig_structure_path):
        print('Cannot open %s!' % orig_structure_path)
        return
    if os.path.exists(dest_structure_path):
        return

    with mf.open(orig_structure_path) as f:
        xsize, ysize, zsize = int(f.header['nx']), int(f.header['ny']), int(f.header['nz'])

        if xsize != ysize or xsize != zsize or ysize != zsize:
            return

        xangpix = float(f.voxel_size.x)
        rescale_xangpix = xangpix * xsize / new_box_size
        cmd = 'relion_image_handler --i %s --o %s --new_box %d --angpix %.3f --rescale_angpix %.3f' % (orig_structure_path, dest_structure_path, new_box_size, xangpix, rescale_xangpix)
        os.system(cmd)

def create_projections(orig_structure_path, dest_projs_path, num):
    if not os.path.exists(orig_structure_path):
        print('Cannot open %s!' % orig_structure_path)
        return
    if os.path.exists(dest_projs_path):
        return
        
    cmd = 'relion_project --i %s --o %s --nr_uniform %d' % (orig_structure_path, dest_projs_path, num)
    os.system(cmd)

# EMAN2
def normalize2d(orig_projs_path, dest_projs_path):
    if not os.path.exists(orig_projs_path):
        print('Cannot open %s!' % orig_projs_path)
        return
    if os.path.exists(dest_projs_path):
        return

    cmd = 'e2proc2d.py %s %s --process normalize.edgemean' % (orig_projs_path, dest_projs_path)
    os.system(cmd)

def normalize3d(orig_stru_path, dest_stru_path):
    if not os.path.exists(orig_stru_path):
        print('Cannot open %s!' %orig_stru_path)
        return
    if os.path.exists(dest_stru_path):
        return

    cmd = 'e2proc3d.py %s %s --process normalize.edgemean' % (orig_stru_path, dest_stru_path)
    os.system(cmd)

# Read/Write Mrc
def read_mrc(path):
    with mf.open(path) as f:
        return f.data

def write_mrc(path, data, overwrite = True):
    with mf.new(path, overwrite = overwrite) as f:
        f.set_data(data)

# Transformation
def random_angles(min_angle = 0, max_angle = 360):
    return [np.random.randint(min_angle, max_angle) for _ in range(3)]

def rotate_3d(data, x, y, z):
    dz = rotate(data, z, mode = 'nearest', axes = (0, 1), reshape = False)
    dy = rotate(dz, y, mode = 'nearest', axes = (0, 2), reshape = False)
    dx = rotate(dy, x, mode = 'nearest', axes = (1, 2), reshape = False)
    return dx

# Directory ops
def find_and_unzip_mapgz(mapgz_dir):
    cmd = 'gzip -d %s' % mapgz_dir
    os.system(cmd)

def convert_map_to_mrc(map_dir, mrc_dir):
    if not os.path.exists(map_dir):
        print('Cannot open %s!' % map_dir)
    if os.path.exists(mrc_dir):
        return

    with mf.new(mrc_dir) as mrc_file:
        with mf.open(map_dir) as map_file:
            mrc_file.set_data(map_file.data)
            mrc_file.set_extended_header(map_file.extended_header)
            mrc_file.voxel_size = map_file.voxel_size
            mrc_file.header['nxstart'] = map_file.header['nxstart']
            mrc_file.header['nystart'] = map_file.header['nystart']
            mrc_file.header['nzstart'] = map_file.header['nzstart']

def search_by_re(root_dir, re):
    return glob.glob(root_dir + re)

def split_file_path(path):
    file_dir, file_name = os.path.split(path)
    file_prefix, file_postfix = os.path.splitext(file_name)
    return file_dir, file_prefix, file_postfix

if __name__ == '__main__':
    # Steps for preprocessing:
    # 1. Unzip map.gz files
    # 2. Convert .map to .mrc
    # 3. Resize 3d structure files to 64x64x64
    # 4. Normalize 3d structure files
    # 5. (Optional) Rotate 3d structure files
    # 6. Create projections

    root_dir = '../../dataset/testSet/'
    norm_mrc_re = 'EMD-????/*_64_norm.mrc'
    norm_mrcs = sorted(search_by_re(root_dir, norm_mrc_re))

    for mp in norm_mrcs:
        create_projections(mp, mp[:-4] + '_24projs', 24)

    # Rotate
    for mp in norm_mrcs:
       fdir, fprefix, _ = split_file_path(mp)
       data = read_mrc(mp)
       for idx in range(2):
           rot_x, rot_y, rot_z = random_angles(30, 150)
           if np.random.rand() < 0.5:
               rot_x = -rot_x
           if np.random.rand() < 0.5:
               rot_y = -rot_y
           if np.random.rand() < 0.5:
               rot_z = -rot_z
           rot_mrc = rotate_3d(data, rot_x, rot_y, rot_z)
           model_type = fdir.split('/')[-1] + '_%d' % idx
           new_dir = root_dir + model_type + '/'

           if not os.path.exists(new_dir):
               os.makedirs(new_dir)
           mrc_name = model_type + '_64_norm.mrc'
           proj_name = model_type + '_64_norm_projs'
           write_mrc(new_dir + mrc_name, rot_mrc)
           create_projections(new_dir + mrc_name, new_dir + proj_name, 24)
