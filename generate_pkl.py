import pickle

"""
'jointPositions': array of shape P x N x J x 3.
This should contain the 3D joint location of each SMPL joint. The joint positions must be in meters.
'orientations': array of shape P x N x K x 3 x 3

The 14 common joints

{"0": "Right_heel", "1": "Right_knee", "2": "Right_hip", "3": "Left_hip", "4": "Left_knee", "5": "Left_heel", 
"6": "Right_wrist", "7": "Right_elbow", "8": "Right_shoulder", "9": "Left_shoulder", "10": "Left_elbow", "11": "Left_wrist",
"12": "Neck", "13": "Head_top"}
"""







def main():



    with open(path_in, 'rb') as f:
        preds = pickle.load(f)
    f.close()

def test(data, preds, eval_path, pred_mode='pred', has_3d=False,
                  min_visible=6, compute_mesh=False):
    """
    Tests one tube.

    Args:
        data (dict).
        model (Tester).
        pred_mode (str).
        has_3d (bool).

    Returns:
        Dictionary of lists. Keys are accel and kp. If 3d, also has pose
        and mesh.
    """
    images = (np.array(data['images']) / 255) * 2 - 1

    if pred_mode == 'hal':
        # The keys have a '_hal' suffix in them.
        preds = {k.replace('_hal', ''): v[:, 1]  # Only want center prediction.
                 for k, v in preds.items() if '_hal' in k}


    t0 = time()

    errors = compute_errors_batched(
        kps_gt=data['kps'],
        kps_pred=preds['kps'],
        joints_gt=data['gt3ds'],
        joints_pred=preds['joints'][:, :14],
        poses_gt=data['poses'],
        poses_pred=preds['poses'],
        shape_gt=data['shape'],
        shapes_pred=preds['shapes'],
        img_size=images.shape[1],
        has_3d=has_3d,
        min_visible=min_visible,
        compute_mesh=compute_mesh,
    )

    with open(eval_path, 'wb') as f:
        print('Saving eval to', eval_path)
        pickle.dump(errors, f)
    print('Eval time:', time() - t0)
    return errors








if __name__ == '__main__':



    name_file = '14'

    demo_output = '/demo_output/'

    path
