from earshot.agent.belief import cell_id
from earshot.agent.policy import LISTEN_A, LISTEN_B, VERIFY, decode_task, task_id, task_kind

VALLEY = (-2000.0, -2000.0, 2000.0, 2000.0)  # 40 x 40 cells of 100 m


def test_task_ids_stay_distinct_on_a_4_km_city():
    north = cell_id(0.0, 1900.0, VALLEY)  # beyond the first 1024 cells
    assert north >= 1024
    ids = {task_id(kind, north) for kind in (VERIFY, LISTEN_A, LISTEN_B)}
    assert len(ids) == 3
    assert task_id(VERIFY, north) not in {task_id(LISTEN_A, c) for c in range(1600)}
    for kind in (VERIFY, LISTEN_A, LISTEN_B):
        tid = task_id(kind, north)
        assert task_kind(tid) == kind
        name, x, y = decode_task(tid, VALLEY)
        assert (name == "verify") == (kind == VERIFY)
        assert (x, y) == (50.0, 1950.0)
