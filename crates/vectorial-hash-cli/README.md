# vectorial-hash-cli

Command-line tools for the [`vectorial-hash`](../vectorial-hash) workspace. Installs as the `vh` binary.

## Subcommands

| Subcommand | Storage | Notes |
| --- | --- | --- |
| `generate` | in-memory | Default. Single process, `HashMap`-backed dedup. |
| `compare` | in-memory | Benchmark across grids 16 + 32. |
| `heavy` | in-memory | Benchmark across grids 16 + 32 + 64 + 128. |
| `generate-redis` | Redis | Requires feature `redis-store`. Multi-process via per-task locks. |

## Usage

```bash
# default in-memory generation
cargo run -p vectorial-hash-cli -- generate

# benchmarks
cargo run -p vectorial-hash-cli -- compare
cargo run -p vectorial-hash-cli -- heavy

# Redis-backed (requires the feature)
cargo run -p vectorial-hash-cli --features redis-store -- generate-redis \
    --redis-host 127.0.0.1 --redis-port 6379
```

## Features

| Feature | Effect |
| --- | --- |
| `redis-store` | Enables the `generate-redis` subcommand. Propagates to `vectorial-hash-templates/redis-store`. |
