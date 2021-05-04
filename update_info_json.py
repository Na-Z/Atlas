# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

import argparse
import json
import os

from atlas.data import load_info_json


def update_scannet_info_json(path, path_meta, test_only=False, verbose=2):
    scenes = []
    if not test_only:
        scenes += sorted([os.path.join('scans', scene)
                          for scene in os.listdir(os.path.join(path, 'scans'))])
    scenes += sorted([os.path.join('scans_test', scene)
                      for scene in os.listdir(os.path.join(path, 'scans_test'))])

    for scene in scenes:
        if verbose > 0:
            print('update info json for %s' % scene)

        info_file = os.path.join(path_meta, scene, 'info.json')
        data = load_info_json(info_file)

        folder, scene = scene.split('/')
        data['path'] = path
        data['file_name_mesh_gt'] = os.path.join(path, folder, scene, scene + '_vh_clean_2.ply')
        data['file_name_seg_indices'] = os.path.join(path, folder, scene,
                                                     scene + '_vh_clean_2.0.010000.segs.json')
        data['file_name_seg_groups'] = os.path.join(path, folder, scene,
                                                    scene + '.aggregation.json')

        frames = data['frames']
        new_frames = []
        for frame_id, frame in enumerate(frames):
            frame['file_name_image'] = os.path.join(path, folder, scene, 'color', '%d.jpg' % frame_id)
            frame['file_name_depth'] = os.path.join(path, folder, scene, 'depth', '%d.png' % frame_id)
            if frame['file_name_instance'] != '':
                frame['file_name_instance'] = os.path.join(path, folder, scene, 'instance-filt', '%d.png' % frame_id)
            new_frames.append(frame)

        data['frames'] = new_frames

        for voxel_size in [4,8,16]:
            data['file_name_vol_%02d' % voxel_size] = os.path.join(path_meta, folder, scene, 'tsdf_%02d.npz' % voxel_size)

        json.dump(data, open(os.path.join(path_meta, folder, scene, 'info.json'), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf on Scannet')
    parser.add_argument("--path", default='./data/',
        help="path to raw dataset")
    parser.add_argument("--path_meta", default='./data_meta/',
        help="path to store processed (derived) dataset")
    parser.add_argument('--test', action='store_true',
        help='only prepare the test set (for rapid testing if you dont plan to train)')
    args = parser.parse_args()


    update_scannet_info_json(os.path.join(args.path, 'scannet'),
                              os.path.join(args.path_meta, 'scannet'),
                              args.test)