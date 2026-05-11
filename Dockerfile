# pcq — Model Context Protocol server image
#
# This image packages the published reference Python implementation of
# the pcq contract as an MCP server that speaks stdio JSON-RPC.
#
# Build:
#   docker build -t pcq .
#
# Run (stdio MCP, attach a client to stdin/stdout):
#   docker run -i --rm pcq
#
# Glama / awesome-mcp-servers context:
#   The image installs `pcq[mcp]` from PyPI (not the local source tree),
#   so it always tracks the published release. Glama's verification
#   pipeline builds this Dockerfile, starts the container, sends an MCP
#   `initialize` request on stdin, and inspects the tool list it gets
#   back on stdout.
#
# Other surfaces of pcq (CLI subcommands like `pcq run`,
# `pcq describe-run`, `pcq agent install`) are intentionally not the
# default ENTRYPOINT of this image — for those, install pcq directly
# (`uv add pcq` / `pip install pcq`).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 비루트 user (uid 1000) — Docker security 표준
RUN useradd --create-home --uid 1000 pcq
USER pcq
WORKDIR /home/pcq

# pcq[mcp] = pcq + MCP server dependency group (Anthropic mcp SDK 포함)
RUN pip install --user --no-cache-dir 'pcq[mcp]'

# user-site bin이 PATH에 들어와야 `pcq` 명령이 보임
ENV PATH="/home/pcq/.local/bin:${PATH}"

# stdio MCP: stdin/stdout JSON-RPC. SSE는 옵션이며 그건 별도 build target에서.
ENTRYPOINT ["pcq", "mcp", "serve"]
