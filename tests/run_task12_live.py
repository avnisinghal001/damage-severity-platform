"""Opt-in native serving check with real checkpoints and a disposable migrated PostgreSQL DB.

This launches two local Uvicorn processes and always stops them. Docker is tested separately.
The database must be dedicated to these tests and have a name starting with task12_test.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from damage_analytics.api.repository import PostgresRepository
from damage_analytics.io import write_json
from damage_analytics.serving.migrate import migrate
from damage_analytics.serving.verify import verify


def wait_ready(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service {port} exited; inspect Task 12 verification logs")
        try:
            if (
                httpx.get(
                    f"http://127.0.0.1:{port}/ready", timeout=1.0, trust_env=False
                ).status_code
                == 200
            ):
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Service {port} did not become ready")


def run(image: Path, database_url: str, output: Path) -> None:
    if not urlsplit(database_url).path.lstrip("/").startswith("task12_test"):
        raise ValueError("Only a dedicated task12_test* database is allowed")
    output.mkdir(parents=True, exist_ok=False)
    migrate(database_url)
    config = tomllib.loads(Path("config/task12.toml").read_text())
    paths = {role: str((Path("config") / config[role]["path"]).resolve()) for role in config}
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "SEVERITY_CHECKPOINT": paths["severity"],
        "SEVERITY_CHECKPOINT_SHA256": config["severity"]["sha256"],
        "DAMAGE_API_DATABASE_URL": database_url,
        "DAMAGE_API_CHECKPOINT": paths["segmentation"],
        "DAMAGE_API_CHECKPOINT_SHA256": config["segmentation"]["sha256"],
        "DAMAGE_API_CRACK_CHECKPOINT": paths["crack"],
        "DAMAGE_API_CRACK_CHECKPOINT_SHA256": config["crack"]["sha256"],
        "DAMAGE_API_SEVERITY_URL": "http://127.0.0.1:18851",
        "DAMAGE_API_SEVERITY_CHECKPOINT_SHA256": config["severity"]["sha256"],
        "DAMAGE_API_STORAGE_DIR": str((output / "artifacts").resolve()),
        "DAMAGE_API_DEVICE": "cpu",
        "DAMAGE_API_CPU_THREADS": "1",
        "SEVERITY_DEVICE": "cpu",
    }
    processes, logs = [], []
    try:
        for name, module, port in (
            ("severity", "damage_analytics.serving.severity_app", 18851),
            ("backend", "damage_analytics.api.app", 18850),
        ):
            log = (output / f"{name}.log").open("w")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    f"{module}:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            processes.append(process)
            wait_ready(port, process)
        result = verify(
            "http://127.0.0.1:18850", image, Path("config/task12.toml"), output / "upload.json"
        )
        repository = PostgresRepository(database_url)
        from uuid import UUID

        record = repository.get(UUID(result["inspection_id"]))
        assert (
            record.features and record.classification and record.severity["status"] == "available"
        )
        # Stop the independent service, then verify honest degraded behavior and persistence.
        processes[0].terminate()
        processes[0].wait(timeout=10)
        with httpx.Client(
            base_url="http://127.0.0.1:18850", timeout=120.0, trust_env=False
        ) as client:
            assert client.get("/ready").status_code == 503
            response = client.post(
                "/inspections",
                files={
                    "file": (
                        image.name,
                        image.read_bytes(),
                        "image/png" if image.suffix.lower() == ".png" else "image/jpeg",
                    )
                },
            )
            response.raise_for_status()
            partial = response.json()
            assert (
                partial["severity"]["status"] == "unavailable"
                and "label" not in partial["severity"]
            )
            assert client.get(f"/inspections/{partial['id']}").json() == partial
            # A full successful result remains retrievable while severity is down.
            assert client.get(f"/inspections/{result['inspection_id']}").status_code == 200
        write_json(
            output / "native_result.json",
            {
                "status": "passed",
                "real_checkpoints": True,
                "real_http_services": True,
                "real_postgresql": True,
                "docker_tested": False,
                "upload": result,
                "outage_record_id": partial["id"],
                "outage_handling_verified": True,
                "test_performance_evaluated": False,
            },
        )
        print(json.dumps({"status": "passed", "report": str(output / "native_result.json")}))
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.image, os.environ["DAMAGE_TASK12_TEST_DATABASE_URL"], args.output)
