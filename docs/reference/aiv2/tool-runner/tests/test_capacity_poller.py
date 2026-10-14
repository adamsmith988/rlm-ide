"""tool-runner capacity poller applies proxy resource_limits."""

from src.capacity_poller import apply_capacity_payload
from src.job_admission import configure_admission, reset_admission_for_tests


def setup_function():
    reset_admission_for_tests()
    configure_admission(
        max_weight=64,
        max_weight_per_tenant=32,
        resource_limits={"llm": 48, "cpu": 16, "gpu": 4},
    )


def teardown_function():
    reset_admission_for_tests()


def test_apply_capacity_merges_and_keeps_cpu():
    from src.job_admission import get_admission

    apply_capacity_payload({"resource_limits": {"llm": 20, "gpu": 40}})
    limits = get_admission().resource_limits
    assert limits["llm"] == 20
    assert limits["gpu"] == 40
    assert limits["cpu"] == 16  # preserved (merge)


def test_saturation_payload_lowers_gpu():
    from src.job_admission import get_admission

    apply_capacity_payload({"resource_limits": {"gpu": 2, "llm": 8}})
    assert get_admission().resource_limits["gpu"] == 2
    assert not get_admission().has_capacity("gpu", weight=3)
