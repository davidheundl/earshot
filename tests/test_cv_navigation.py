import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('transformers')
pytest.importorskip('torch')

spec = importlib.util.spec_from_file_location(
    'cv_navigation', Path(__file__).resolve().parents[1]/'tools/cv_navigation.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_backproject_metric_depth():
    K = np.array([[100.,0,50],[0,100,30],[0,0,1]])
    xyz = demo.backproject(np.array([[50.,30],[60,50]]),np.array([10.,20]),K)
    np.testing.assert_allclose(xyz,[[0,0,10],[2,4,20]])


def test_pose_recovers_translation_without_query_truth():
    cv2.setRNGSeed(42)
    rng = np.random.default_rng(42)
    K = np.array([[300.,0,320],[0,300,192],[0,0,1]])
    uv = np.array([(x,y) for x in range(150,501,50) for y in range(90,301,50)],float)
    depth = rng.uniform(15,50,len(uv))
    xyz = demo.backproject(uv,depth,K)
    camera_delta = np.array([2.,-1.,.5])
    query_xyz = xyz-camera_delta
    q = query_xyz @ K.T
    q = q[:,:2]/q[:,2:]
    keys_ref = [cv2.KeyPoint(float(x),float(y),1) for x,y in uv]
    keys_query = [cv2.KeyPoint(float(x),float(y),1) for x,y in q]
    descriptors = rng.random((len(uv),128)).astype('float32')
    z = np.full((384,640),np.inf,dtype='float32')
    for (x,y),d in zip(uv,depth,strict=True):
        z[int(y),int(x)] = d
    ref = {'id':'test','position':[10,20,30],'K':K.tolist(),
           'R_camera_to_world':np.eye(3).tolist()}
    result = demo.estimate_pose(ref,keys_ref,descriptors,keys_query,descriptors,z,K)
    assert result is not None
    np.testing.assert_allclose(result['position'],np.array(ref['position'])+camera_delta,atol=.01)


def test_no_features_returns_no_pose():
    assert demo.estimate_pose({},[],None,[],None,None,None) is None
    assert demo.estimate_pose({},[],np.zeros((1,128), dtype='float32'),
                              [],np.zeros((1,128), dtype='float32'),None,None) is None


def candidate(ref, x, count=20, rotation=None):
    return {'reference':ref, 'position':[x, 0., 0.], 'inliers':count,
            'R_camera_to_world':np.eye(3).tolist() if rotation is None else rotation}


def test_agreement_beats_many_wrong_window_matches():
    wrong = candidate('a', 16., 200)
    correct = candidate('a', 0., 30)
    other = candidate('b', .1, 25)
    assert demo.select_consistent_pose([wrong, correct, other]) is correct


def test_same_reference_cannot_vote_twice():
    assert demo.select_consistent_pose([candidate('a', 0), candidate('a', .1)]) is None


def test_disagreement_and_no_candidates_abstain():
    assert demo.select_consistent_pose([]) is None
    assert demo.select_consistent_pose([candidate('a', 0), candidate('b', 8)]) is None


def test_rotational_disagreement_abstains():
    turned = cv2.Rodrigues(np.array([0., 0., .2]))[0].tolist()
    assert demo.select_consistent_pose([candidate('a', 0), candidate('b', 0, rotation=turned)]) is None


def test_two_competing_consensuses_abstain():
    poses = [candidate('a', 0), candidate('b', .1),
             candidate('a', 8, 100), candidate('b', 8.1, 100)]
    assert demo.select_consistent_pose(poses) is None


def test_hypotheses_without_descriptors():
    assert demo.pose_hypotheses({}, [], None, [], None, None, None) == []


def test_peeling_preserves_original_feature_indices(monkeypatch):
    calls = []
    def fake_pose(ref, keys, descriptors, *unused):
        calls.append(keys)
        if len(calls) > 2:
            return None
        return {'_matches':[cv2.DMatch(i, i, 0.) for i in range(12)]}
    monkeypatch.setattr(demo, 'estimate_pose', fake_pose)
    hypotheses = demo.pose_hypotheses({}, list(range(40)), np.zeros((40,128)),
                                     [], np.zeros((40,128)), None, None)
    assert [m.queryIdx for m in hypotheses[0]['_matches']] == list(range(12))
    assert [m.queryIdx for m in hypotheses[1]['_matches']] == list(range(12,24))
    assert calls[2] == list(range(24,40))
