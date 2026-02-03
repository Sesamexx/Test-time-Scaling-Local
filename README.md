# Test-time Scaling Local

## Usage

```python
import asyncio
from client import get_client

async def main():
    client = get_client()
    
    # Health check
    health = await client.health()
    
    # Chat
    response = await client.chat(
        model="qwen3-vl",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)

asyncio.run(main())
```