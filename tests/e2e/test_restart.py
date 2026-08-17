# ABOUTME: E2E restart persistence tests for energy accumulator.
# ABOUTME: Verifies accumulated energy survives real HA container restarts.

import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from .conftest import _docker, CONTAINER_NAME, HA_IMAGE, HA_URL
from .ha_client import HAClient
from .seed import read_storage_file, write_storage_file


pytestmark = [pytest.mark.e2e]


def _backup_upgrade_gap_storage(config_dir: Path) -> Path:
    """Backup just the two files that the upgrade-gap test mutates."""
    backup_dir = config_dir / ".storage_upgrade_gap_backup"
    backup_dir.mkdir(exist_ok=True)
    storage_dir = config_dir / ".storage"

    for name in ("device_role_accumulators.json", "core.restore_state"):
        source = storage_dir / name
        target = backup_dir / name
        if source.exists():
            shutil.copyfile(source, target)
    return backup_dir


def _restore_upgrade_gap_storage(config_dir: Path, backup_dir: Path) -> None:
    """Restore the exact upgrade-gap files and remove the temporary backup."""
    storage_dir = config_dir / ".storage"
    storage_dir.mkdir(exist_ok=True)

    for name in ("device_role_accumulators.json", "core.restore_state"):
        source = backup_dir / name
        destination = storage_dir / name
        if source.exists():
            shutil.copyfile(source, destination)
        elif destination.exists():
            destination.unlink()

    shutil.rmtree(backup_dir, ignore_errors=True)


def _assert_recorder_has_raw_state(config_dir: Path, entity_id: str, expected: str) -> None:
    """Verify HA recorder actually contains the raw pre-restart state."""
    db_path = config_dir / "home-assistant_v2.db"
    assert db_path.exists(), f"Recorder DB missing: {db_path}"

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT states.state
            FROM states
            JOIN states_meta ON states.metadata_id = states_meta.metadata_id
            WHERE states_meta.entity_id = ?
            ORDER BY states.last_updated_ts DESC, states.state_id DESC
            LIMIT 1
            """,
            (entity_id,),
        ).fetchone()

    assert row is not None, f"Recorder has no state for {entity_id}"
    assert row[0] == expected, (
        f"Recorder raw state mismatch for {entity_id}: expected {expected}, got {row[0]}"
    )


@pytest.mark.usefixtures("ha_bootstrap")
def test_energy_accumulation_survives_restart(ha_client, restart_ha):
    """Energy accumulated before restart is preserved after restart."""
    # Set initial energy reading
    ha_client.call_service("fake_device", "set_value", {
        "entity_id": "sensor.test_plug_energy",
        "value": 100.0,
    })
    # Wait for the role energy sensor to pick it up.
    # session_start deferred from 0.0 to first non-zero reading (100.0),
    # so role_value = 0 + (100 - 100) = 0.0
    ha_client.wait_for_state(
        "sensor.e2e_role_energy", "0.0", timeout=15,
    )

    # Accumulate 50 kWh delta
    ha_client.call_service("fake_device", "set_value", {
        "entity_id": "sensor.test_plug_energy",
        "value": 150.0,
    })
    state = ha_client.wait_for_state(
        "sensor.e2e_role_energy", "50.0", timeout=15,
    )
    assert state is not None

    # Restart HA container (graceful stop preserves .storage/)
    restart_ha()

    # After restart, fake_device's energy resets to 0.0.
    # The accumulator detects the reset (drop 150→0 > threshold),
    # commits the pre-restart delta (50) to historical_sum,
    # and starts a new session from 0.
    # Set energy to 50 → role_value = 50 (historical) + (50 - 0) = 100
    ha_client.call_service("fake_device", "set_value", {
        "entity_id": "sensor.test_plug_energy",
        "value": 50.0,
    })

    state = ha_client.wait_for_state(
        "sensor.e2e_role_energy", "100.0", timeout=30,
    )
    assert state is not None


@pytest.mark.usefixtures("ha_bootstrap")
def test_upgrade_gap_uses_recorder_floor_on_restart(ha_client, ha_bootstrap):
    """A v0.4-like restart should restore from the last raw recorder value."""
    config_dir = ha_bootstrap["config_dir"]
    backup_dir = _backup_upgrade_gap_storage(config_dir)

    try:
        client = HAClient(HA_URL)
        try:
            client.wait_for_ready(timeout=120)
            client.onboard_and_authenticate()
            client.wait_for_entity("sensor.e2e_role_energy", timeout=60)

            client.call_service("fake_device", "set_value", {
                "entity_id": "sensor.test_plug_energy",
                "value": 100.0,
            })
            client.wait_for_state("sensor.e2e_role_energy", "0.0", timeout=15)

            client.call_service("fake_device", "set_value", {
                "entity_id": "sensor.test_plug_energy",
                "value": 140.0,
            })
            state = client.wait_for_state("sensor.e2e_role_energy", "40.0", timeout=15)
            assert state is not None
            assert float(state["state"]) == 40.0

            # This is the real guardrail: the last raw numeric role state must be in
            # the recorder DB before the upgrade-gap files disappear.
            _assert_recorder_has_raw_state(config_dir, "sensor.e2e_role_energy", "40.0")

            # Simulate the v0.4 upgrade gap: custom accumulator store and restore state are
            # absent, while the last raw role value remains in Recorder.
            _docker("stop", CONTAINER_NAME)
            _docker(
                "run", "--rm",
                "-v", f"{config_dir}:/config",
                HA_IMAGE,
                "bash", "-c",
                "rm -f /config/.storage/device_role_accumulators.json /config/.storage/core.restore_state && chmod -R a+rw /config/.storage",
            )
            _docker("start", CONTAINER_NAME)

            restart_client = HAClient(HA_URL)
            try:
                restart_client.wait_for_ready(timeout=120)
                restart_client.onboard_and_authenticate()
                restart_client.wait_for_entity("sensor.e2e_role_energy", timeout=60)

                state = restart_client.wait_for_state(
                    "sensor.e2e_role_energy", "40.0", timeout=30
                )
                assert state is not None
                assert float(state["state"]) == 40.0, (
                    "Role should restore from the last raw recorder value when store and "
                    "restore data are absent"
                )

                # Controlled startup baseline: a new session starts at the current source value.
                restart_client.call_service("fake_device", "set_value", {
                    "entity_id": "sensor.test_plug_energy",
                    "value": 100.0,
                })
                state = restart_client.wait_for_state(
                    "sensor.e2e_role_energy", "40.0", timeout=15
                )
                assert state is not None
                assert float(state["state"]) == 40.0

                restart_client.call_service("fake_device", "set_value", {
                    "entity_id": "sensor.test_plug_energy",
                    "value": 160.0,
                })
                state = restart_client.wait_for_state(
                    "sensor.e2e_role_energy", "100.0", timeout=30
                )
                assert state is not None
                assert float(state["state"]) == 100.0
            finally:
                restart_client.close()
        finally:
            client.close()
    finally:
        _docker("stop", CONTAINER_NAME)
        _docker(
            "run", "--rm",
            "-v", f"{config_dir}:/config",
            HA_IMAGE,
            "bash", "-c",
            "chmod -R a+rw /config/.storage && for f in device_role_accumulators.json core.restore_state; do if [ -f /config/.storage_upgrade_gap_backup/$f ]; then cp /config/.storage_upgrade_gap_backup/$f /config/.storage/$f; else rm -f /config/.storage/$f; fi; done",
        )
        _docker("start", CONTAINER_NAME)
        _restore_upgrade_gap_storage(config_dir, backup_dir)


@pytest.mark.usefixtures("ha_bootstrap")
def test_deactivate_commit_survives_restart(ha_client, ha_bootstrap, restart_ha):
    """Energy committed on deactivation is preserved across restart."""
    import time

    # Read the current accumulated energy from prior tests
    state = ha_client.get_state("sensor.e2e_role_energy")
    baseline = float(state["state"])

    # Accumulate some delta on top of baseline
    ha_client.call_service("fake_device", "set_value", {
        "entity_id": "sensor.test_plug_energy",
        "value": 300.0,
    })
    time.sleep(3)
    state = ha_client.get_state("sensor.e2e_role_energy")
    pre_deactivate = float(state["state"])
    assert pre_deactivate > baseline, (
        f"Energy should increase: was {baseline}, now {pre_deactivate}"
    )

    config_dir = ha_bootstrap["config_dir"]

    def _reactivate_role():
        """Re-activate the role so subsequent tests aren't broken."""
        _docker("stop", CONTAINER_NAME)
        _docker(
            "run", "--rm",
            "-v", f"{config_dir}:/config",
            HA_IMAGE,
            "bash", "-c", "chmod -R a+rw /config/.storage",
        )
        entries = read_storage_file(config_dir, "core.config_entries")
        for e in entries["data"]["entries"]:
            if e["entry_id"] == "device_role_e2e":
                e["data"]["active"] = True
        write_storage_file(config_dir, "core.config_entries", entries)
        _docker("start", CONTAINER_NAME)
        cleanup_client = HAClient(HA_URL)
        try:
            cleanup_client.wait_for_ready(timeout=120)
            cleanup_client.onboard_and_authenticate()
        finally:
            cleanup_client.close()

    try:
        # Deactivate the role by updating config entry storage and restarting.
        _docker("stop", CONTAINER_NAME)
        _docker(
            "run", "--rm",
            "-v", f"{config_dir}:/config",
            HA_IMAGE,
            "bash", "-c", "chmod -R a+rw /config/.storage",
        )
        config_entries = read_storage_file(config_dir, "core.config_entries")
        for entry in config_entries["data"]["entries"]:
            if entry["entry_id"] == "device_role_e2e":
                entry["data"]["active"] = False
        write_storage_file(config_dir, "core.config_entries", config_entries)

        _docker("start", CONTAINER_NAME)
        client = HAClient(HA_URL)
        try:
            client.wait_for_ready(timeout=120)
            client.onboard_and_authenticate()

            # Wait for the device_role integration to finish setting up
            # entities after restart. The API is reachable before custom
            # integrations have loaded.
            client.wait_for_entity("sensor.e2e_role_energy", timeout=60)

            # Energy sensor should be frozen (available but not updating).
            state = client.get_state("sensor.e2e_role_energy")
            assert state is not None
            frozen_value = float(state["state"])
            assert frozen_value > 0.0, (
                f"Frozen energy should be positive, got {frozen_value}"
            )
        finally:
            client.close()

        # Restart again to verify committed value persists
        restart_ha()

        state = ha_client.get_state("sensor.e2e_role_energy")
        assert state is not None
        after_restart = float(state["state"])
        assert after_restart == frozen_value, (
            f"Energy should be frozen at {frozen_value} after restart, "
            f"got {after_restart}"
        )
    finally:
        _reactivate_role()
