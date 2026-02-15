# Test-time Scaling Local

Local benchmark runner for testing LLM agents on AndroidWorld.

## Quick Start

Prerequisite:

- Android Studio with AVD named `AndroidWorldAvd` (Pixel 6, API 33)
- SSH tunnel to remote server: `ssh -L 9000:localhost:80 ubuntu@<SERVER_IP>`
- Conda environment with dependencies

```bash
# Run the emulator
bat\launch_emulator.bat
# Connect to server with ssh
bat\start_ssh_tunnel.bat
# Activate environment
conda activate $ENV_NAME
# First run
python run_benchmark.py --perform_emulator_setup
# Subsequent runs
python run_benchmark.py
```

Results are saved to `~/android_world/runs/` by default.

## Configuration

Edit `config.yaml` to change the scaling technique and the corresponding parameters. For example, if we want to run the baseline (only gelab-zero-4b-preview):

```yaml
server:
  url: "http://localhost:9000"
  timeout: 300.0

model:
  name: "gelab-zero-4b-preview"
  temperature: 0.5
  max_tokens: 512
  max_retry: 3

benchmark:
  adb_path: "%LOCALAPPDATA%\\Android\\Sdk\\platform-tools\\adb.exe"
  console_port: 5554
  perform_emulator_setup: false
  n_task_combinations: 1
  task_random_seed: 30
  output_path: "~/android_world/runs"
  checkpoint_dir: ""
  tasks: null

scaling:
  strategy: "baseline"
  best_of_n_weighted:
    # ........
```

## Client Usage

```python
import asyncio
from client import get_client

async def main():
    client = get_client()
    
    # Health check
    health = await client.health()
    
    # Chat
    response = await client.chat(
        model="gelab-zero-4b-preview",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)

asyncio.run(main())
```

## Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | None | Comma-separated list of tasks to run |
| `--n_task_combinations` | 1 | Number of task instances per template |
| `--perform_emulator_setup` | False | First-time setup (install apps) |
| `--model_name` | gelab-zero-4b-preview | Model to use |
| `--temperature` | 0.0 | Sampling temperature |
| `--output_path` | ~/android_world/runs | Results directory |