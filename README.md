# Visor

A homegrown ROS 2 viewer (an RViz2 alternative) written in Rust that connects
directly to a zenoh router as a zenoh client.

## Requirements

- Rust 1.92 or newer (rustup recommended)
- A ROS 2 (Jazzy) environment running `rmw_zenoh` with a zenoh router
- Linux, macOS or Windows (on Windows: the MSVC toolchain with Visual Studio Build Tools)

## Quick start

```bash
cargo run                                                          # connect to tcp/localhost:7447
cargo run --release -- --endpoint tcp/<host>:7447 --domain-id 32   # a remote robot
cargo run -- --bag path/to/log.bag                                 # replay a ROS 1 .bag or a ROS 2 bag directory
```

`--bag` cannot be combined with `--endpoint` / `--domain-id`; the `Source` menu switches at runtime.

## Install

```bash
cargo install --path apps/visor --force   # the GUI, with the in-repo plugins
cargo install --path .                    # the probe and baginfo CLIs
```

On Windows, install the `.msi` attached to each GitHub Release instead.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ZENOH_ENDPOINT` | `tcp/localhost:7447` | zenoh router endpoint |
| `ROS_DOMAIN_ID` | `0` | ROS 2 domain ID |

`--endpoint` / `--domain-id` override these, and `--config <path>` loads a viewer configuration file.

## Plugins

Display types and panels are added by implementing `visor::plugin::Plugin` in a crate;
`plugins/sample` with `cargo run -p visor --example visor_with_sample` is a working example.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`--workspace` is required: a bare `cargo` command only covers `apps/visor`.
