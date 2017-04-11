#!/usr/bin/env python

import os.path
import shutil

def main():
    version = 0.6
    release_dir = 'roots_ver_{}'.format(version)

    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)

    os.mkdir(release_dir)

    src_files = ['bound.py',
                 'config.py',
                 'fixer.py',
                 'prediction.py',
                 'rbfwarp2d.py',
                 'rectify.py',
                 'restore.py',
                 'utils.py']

    scripts = ['copy_to_rectify.py',
               'process_videos.py',
               'rectify_images.py',
               'rectify_single_image.py']

    for f in src_files:
        shutil.copy(f, release_dir)

    for f in scripts:
        shutil.copy(f, release_dir)

    # copy other files
    shutil.copy('changelog.txt', release_dir)


if __name__ == "__main__":
    main()