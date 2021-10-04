"""
Runs hmmr on a video.
Extracts 2D using OpenPose

Sample Usage:
python -m demo_video --out_dir demo_data/output
python -m demo_video --out_dir demo_data/output270k --load_path models/hmmr_model.ckpt-2699068
python3 demo_video.py --vid_path demo_data/penn_action-2278.mp4 --load_path models/hmmr_model.ckpt-1119816  --out_dir output/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from glob import glob
import json
import os
import os.path as osp
import pickle
import re
import sys

from typing import Union

from absl import flags
import ipdb
import numpy as np

from extract_tracks import compute_tracks
from src.config import get_config
from src.evaluation.run_video import (
    process_image,
    render_preds,
)
from src.evaluation.tester import Tester
from src.util.common import mkdir
from src.util.smooth_bbox import get_smooth_bbox_params


flags.DEFINE_string(
    'vid_path', 'penn_action-2278.mp4',
    'video to run on')
flags.DEFINE_integer(
    'track_id', 0,
    'PoseFlow generates a track for each detected person. This determines which'
    ' track index to use if using vid_path.'
)
flags.DEFINE_string('vid_dir', None, 'If set, runs on all video in directory.')
flags.DEFINE_string('out_dir', 'demo_output/',
                    'Where to save final HMMR results.')
flags.DEFINE_string('track_dir', 'demo_output/',
                    'Where to save intermediate tracking results.')
flags.DEFINE_string('pred_mode', 'pred',
                    'Which prediction track to use (Only pred supported now).')
flags.DEFINE_string('mesh_color', 'blue', 'Color of mesh.')
flags.DEFINE_integer(
    'sequence_length', 20,
    'Length of sequence during prediction. Larger will be faster for longer '
    'videos but use more memory.'
)
flags.DEFINE_boolean(
    'trim', False,
    'If True, trims the first and last couple of frames for which the temporal'
    'encoder doesn\'t see full fov.'
)

def read_json(json_path): #Openpose
    with open(json_path) as f1:
        data = json.load(f1)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps


def get_labels(json_path, num_frames, track_id, min_kp_count=20):
    """
    Args:
        json_path (str): Path to the json output from OpenPose.
        num_frames (int): Number of frames.
        min_kp_count (int): Minimum threshold length for a tracklet.
    Returns:
        List of length num_people. Each element in the list is another list of
        length num_frames containing the poses for each person.
    """
    print('reading %s' % json_path)
    # Opens json, smoothes the output
    json_paths = sorted(glob(osp.join(json_path, "*.json")))

    if len(json_paths) != num_frames:
        print('Not all frames were detected.')
        ipdb.set_trace()

    all_kps = []
    for i, json_path in enumerate(json_paths):
        kps = read_json(json_path)
        all_kps.append(kps)

    pVis = []; counts_list = []; nbox = []
    count = 0
    for fls in all_kps:
        for ps in fls:
            vis = ps[:, 2] > 0 #For all the positions (x, y) > 0 = True
            if np.sum(vis) >= min_kp_count: #Count if the sum of points are more than min acceptable
                pVis.append(ps)
                counts_list.append(len(pVis))

        sort_idx = np.argsort(counts_list)[::-1]
        kps_list_sorted = []
        for sort_id in sort_idx:
            kps_list_sorted.append(pVis[sort_id])

        count += 1

        """
        if len(kps_list_sorted) < 1:
            print('OpenPose - Person does not have the minimum points on the frame nº - ', count - 1)
            ipdb.set_trace() #Break in the first occurrence
        else:
            track_id = min(track_id, len(kps_list_sorted) - 1)
            kps = kps_list_sorted[track_id]
            newbox.append(kps)
        all_kps = newbox"""

        if len(kps_list_sorted) < 1:
            print('OpenPose - Person does not have the minimum points on the frame nº - ', count - 1)
            ipdb.set_trace() #Break in the first occurrence

        track_id = min(track_id, len(kps_list_sorted) - 1)
        kps = kps_list_sorted[track_id]
        nbox.append(kps)
    all_kps = nbox

    print('Len OpenPose kps', len(all_kps), '\n')

    if len(all_kps) != num_frames:
        print('Not all frames were detected.')
        ipdb.set_trace()

    return all_kps


def predict_on_tracks(model, img_dir, json_path, output_path, track_id,
                      trim_length):
    # Get all the images video_frames
    im_paths = sorted(glob(osp.join(img_dir, '*.png')))
    # Get all the images of openpose
    im_paths_open = sorted(glob(osp.join(json_path, '*.jpg')))
    #Get all the kps
    kps = get_labels(json_path, len(im_paths_open), track_id)

    # Here we set which track to use.
    #track_id = min(track_id, len(all_kps) - 1)
    #kps = all_kps[track_id]

    bbox_params_smooth, s, e = get_smooth_bbox_params(kps, vis_thresh=0.1)

    images = []
    images_orig = []
    min_f = max(s, 0)
    max_f = min(e, len(kps))

    print('----------')
    print('Preprocessing frames.')
    print('----------')

    for i in range(min_f, max_f):
        proc_params = process_image(
            im_path=im_paths[i],
            bbox_param=bbox_params_smooth[i],
        )
        images.append(proc_params.pop('image'))
        images_orig.append(proc_params)

    if track_id > 0:
        output_path += '_{}'.format(track_id)

    mkdir(output_path)


    #Json path to save the joints rot
    myjson_dir = osp.join(output_path, 'rot_output')
    myjson_path = osp.join(myjson_dir, 'joints_rot_output.json')
    myjson_path1 = osp.join(myjson_dir, 'joints1_rot_output.json')
    myjson_path2 = osp.join(myjson_dir, 'poses_rot_output.json')
    mkdir(myjson_dir)

    #print('myjson_dir', myjson_dir)


    pred_path = osp.join(output_path, 'hmmr_output.pkl')
    if osp.exists(pred_path):
        print('----------')
        print('Loading pre-computed prediction.')
        print('----------')

        with open(pred_path, 'rb') as f:
            preds = pickle.load(f)
    else:
        print('----------')
        print('Running prediction.')
        print('----------')

        preds = model.predict_all_images(images) #extract prediction --> tester.py

        """
        # Predictions.
        'cams', 'joints', 'kps', 'poses', shapes', 'verts', 'omegas'

        cams is 3D [s, tx, ty], numpy/cuda torch Variable or B x 3
        cams: N x 3, predicted camera
        joints: N x K x 3, predicted 3D joints for debug
        kp: N x K x 3, predicted 2D joints to figure out extent
        """

        # Save results
        print('Save results')


        with open(pred_path + '.txt', 'w') as f:
            f.write(str(preds) + "\n")


        with open(pred_path, 'wb') as f: #HERE
            print('Saving prediction results to', pred_path)
            pickle.dump(preds, f)

    # get the 3D joints
    myjoints = preds['joints']
    mydict = {}

    totaljointsdict = {}
    totaljointsdict['frame_Count'] = myjoints.shape[0]
    print("There are totally {} frames ".format(myjoints.shape[0]))

    for i in range(0, myjoints.shape[0]):
        rotmat = myjoints[i]
        rotlist = list(np.reshape(rotmat, (1, -1))[0])
        rotlist = [float(j) for j in rotlist]
        frame_index = 'frame_'+"%04d" %i
        mydict[frame_index] = rotlist
        framedict = {}
        #############################

        for j in range(0, myjoints.shape[1]):
            _joints = myjoints[i][j]
            jointslist = [float(j) for j in _joints]
            joints_index = 'joint_' +"%02d" % j
            framedict[joints_index]=jointslist
        totaljointsdict[frame_index] = framedict

    print('Saving joints file fomat results to', myjson_path)
    with open(myjson_path, 'w') as jf:
        json.dump(mydict, jf, sort_keys=True)


    print('Saving joints file fomat 1 results to', myjson_path1)
    with open(myjson_path1, 'w') as j:
        json.dump(totaljointsdict, j, sort_keys=True)

    # get the poses
    myposes = preds['poses']
    totaldict = {}
    totaldict['frame_Count'] = myposes.shape[0]
    for i in range(0, myposes.shape[0]):
        frame_index = "frame_" + "%04d" % i
        framedict = {}
        for j in range(0, myposes.shape[1]):
            rotmat = myposes[i][j]
            rotlist = list(np.reshape(rotmat, (1, -1))[0])
            rotlist = [float(j) for j in rotlist]
            rot_index = 'rot_'+"%02d" % j
            framedict[rot_index] = rotlist
        totaldict[frame_index] = framedict

    print('Saving rot results to', myjson_path2)
    with open(myjson_path2, 'w') as f:
        json.dump(totaldict, f, sort_keys=True)


    if trim_length > 0:
        output_path += '_trim'

    print('----------')
    print('Rendering results to {}.'.format(output_path))
    print('----------')
    render_preds(
        output_path=output_path,
        config=config,
        preds=preds,
        images=images,
        images_orig=images_orig,
        trim_length=trim_length,
    )


def run_on_video(model, vid_path, trim_length, openpose):
    """
    Main driver.
    First extracts alphapose/posetrack in track_dir
    Then runs HMMR.
    """
    print('----------')
    print('Computing tracks on {}.'.format(vid_path))
    print('----------')

    # See extract_tracks.py
    #poseflow_path, img_dir = compute_tracks(vid_path, config.track_dir)

    json_path, img_dir = compute_tracks(vid_path, config.track_dir, openpose)

    vid_name = osp.basename(vid_path).split('.')[0]
    out_dir = osp.join(config.out_dir, vid_name, 'hmmr_output_openpose')

    predict_on_tracks(
        model=model,
        img_dir=img_dir,
        json_path=json_path,
        output_path=out_dir,
        track_id=config.track_id,
        trim_length=trim_length
    )


def main(model):
    # Make output directory.
    mkdir(config.out_dir)

    if config.trim:
        trim_length = model.fov // 2
    else:
        trim_length = 0

    openpose = True

    if config.vid_dir:
        vid_paths = sorted(glob(config.vid_dir + '/*.mp4'))
        for vid_path in vid_paths:
            run_on_video(model, vid_path, trim_length, openpose)
    else:
        run_on_video(model, config.vid_path, trim_length, openpose)


if __name__ == '__main__':
    config = get_config()

    # Set up model:
    model_hmmr = Tester(
        config,
        pretrained_resnet_path='models/hmr_noS5.ckpt-642561'
    )

    main(model_hmmr)
