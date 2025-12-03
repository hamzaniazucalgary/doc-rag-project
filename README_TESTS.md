# Testing Guide

This project uses `pytest` for comprehensive testing.

## Prerequisites

Ensure you have the testing dependencies installed:

```bash
pip install -r requirements.txt
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

## Test Structure

- **`tests/unit/`**: Isolated tests for individual modules (`storage`, `ingestion`, `agent`, etc.). Mocks external dependencies like OpenAI and ChromaDB.
- **`tests/integration/`**: Tests the interaction between modules (e.g., Ingestion -> Storage -> Retrieval).
- **`tests/e2e/`**: Simulates full user workflows (Upload -> Agent -> Answer).

## Coverage

The test suite covers:
- PDF Loading & Chunking
- Vector Store Operations (Add, Query, Delete)
- Retrieval Logic & Query Rewriting
- Agent Reasoning & Tool Usage
- Generation & Streaming
- Utility Functions

## Mocks

External services (OpenAI API, ChromaDB persistence) are mocked in `tests/conftest.py` to ensure tests are:
- **Fast**: No network latency.
- **Free**: No API costs.
- **Reliable**: No dependency on external service status.
