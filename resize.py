import argparse
import os
import mrcfile as mf

def resize(orig_structure_path, dest_structure_path, new_box_size):
    if not os.path.exists(orig_structure_path):
        # raise ValueError('Cannot open %s! --resize()' % orig_structure_path)
        return

    with mf.open(orig_structure_path) as f:
        xsize, ysize, zsize = int(f.header['nx']), int(f.header['ny']), int(f.header['nz'])

        if xsize != ysize or xsize != zsize or ysize != zsize:
            return

        xangpix = float(f.voxel_size.x)
        rescale_xangpix = xangpix * xsize / new_box_size
        cmd = 'relion_image_handler --i %s --o %s --new_box %d --angpix %.3f --rescale_angpix %.3f' % (orig_structure_path, dest_structure_path, new_box_size, xangpix, rescale_xangpix)

    os.system(cmd)

    print('Finish creating %s! -- resize()' % dest_structure_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str)
    parser.add_argument('--new_box_size', type = int)
    parser.add_argument('--output', type = str)

    # Start processing
    config = parser.parse_args()
    resize(config.input, config.output, config.new_box_size)


if __name__ == '__main__':
    main()