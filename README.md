# RoomEQ

**Simple Room Correction Wizard for RME Audio Interfaces**

RoomEQ is a macOS desktop application that simplifies room acoustic measurement and correction for RME audio interfaces (UCX II, 802 FS, UFX+, etc.).

## Features

- **Guided Wizard Workflow**: Step-by-step process for room measurement
- **Auto-Detection**: Automatically detects RME audio interfaces
- **Visual Guides**: Clear diagrams showing microphone placement positions
- **Smart Analysis**: Identifies room problems (peaks, dips, modes)
- **RME Integration**: Direct export to TotalMix Room EQ format

## Requirements

- macOS 11.0+ (Big Sur or later)
- Python 3.11+
- RME audio interface with TotalMix FX 1.96+
- Measurement microphone

## Installation

```bash
# Clone the repository
git clone https://github.com/user/roomeq.git
cd roomeq

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Usage

```bash
# Run the application
roomeq

# Or run as module
python -m roomeq
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=roomeq

# Run linter
ruff check src tests
```

## License

MIT License
