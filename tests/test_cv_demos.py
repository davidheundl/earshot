import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'cv_demos', Path(__file__).resolve().parents[1] / 'tools/cv_demos.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_missing_detection_is_inconclusive():
    assert demo.verification_result({},3,[],1,'quadcopter')['camera_evidence']=='INCONCLUSIVE'


def test_one_image_cannot_confirm():
    assert demo.verification_result({1:1},3,[0.99],1,'quadcopter')['camera_evidence']=='INCONCLUSIVE'


def test_repeated_same_track_confirms():
    assert demo.verification_result({1:3},3,[0.9],3,'quadcopter')['camera_evidence']=='CONFIRM'


def test_different_tracks_do_not_accumulate():
    assert demo.verification_result({1:1,2:1,3:1},3,[0.9],3,'quadcopter')['camera_evidence']=='INCONCLUSIVE'
