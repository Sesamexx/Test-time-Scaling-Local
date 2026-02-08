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
ssh -L 9000:localhost:80 ubuntu@$SERVER_IP
# Activate environment
conda activate $ENV_NAME
# First run
python run_benchmark.py --perform_emulator_setup
# Subsequent runs
python run_benchmark.py
# Or run specific tasks
python run_benchmark.py --tasks=ContactsAddContact,ClockStopWatch
```

Results are saved to `~/android_world/runs/` by default.

## Configuration

Edit `config.yaml` to change server URL and timeout:

```yaml
server:
  url: "http://localhost:9000"
  timeout: 300.0
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