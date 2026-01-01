# EQilibrium

**A guided room correction wizard for audio interfaces with parametric EQ**

EQilibrium simplifies room acoustic measurement and correction by providing a step-by-step wizard experience. Instead of complex configuration screens, you get clear guidance through each stage of the measurement and correction process.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-240%20passing-brightgreen.svg)]()

## Features

- **Guided Wizard Workflow** - Step-by-step process eliminates guesswork
- **Works with Any Interface** - Compatible with any CoreAudio audio interface
- **Visual Mic Placement** - Clear diagrams show exactly where to position your microphone
- **Multi-Position Averaging** - Measures from 5-9 positions for accurate room response
- **Smart Analysis** - Identifies peaks, dips, and room modes with severity ratings
- **Best Practices Built-In** - Prioritizes cuts over boosts, warns about uncorrectable issues
- **Configurable EQ Profiles** - Adapts to different hardware constraints
- **Multiple Export Formats** - RME TotalMix (`.tmreq`), REW-compatible text, or manual entry tables

## Why EQilibrium?

Room correction software like REW (Room EQ Wizard) is powerful but complex. EQilibrium takes a different approach:

| Feature | REW | EQilibrium |
|---------|-----|------------|
| Learning curve | Steep | Minimal |
| Configuration options | Hundreds | Just what you need |
| Workflow | Manual | Guided wizard |
| Purpose | General measurement | Room correction only |
| Export formats | Many (requires setup) | One-click (RME, REW, manual) |

## Requirements

- **Operating System**: macOS 11.0+ (Big Sur or later)
- **Python**: 3.11 or higher
- **Audio Interface**: Any CoreAudio-compatible interface
- **Measurement Microphone**: Calibrated measurement mic (e.g., miniDSP UMIK-1, Dayton EMM-6)

## Supported Interfaces

EQilibrium works with any CoreAudio-compatible audio interface. It automatically detects devices from:

- **RME** - Full support with direct `.tmreq` export to TotalMix Room EQ
- **Focusrite** - Scarlett, Clarett series
- **MOTU** - M2, M4, UltraLite, 828 series
- **Universal Audio** - Apollo, Volt series
- **PreSonus** - AudioBox, Studio series
- **Audient** - iD series
- **And many more** - Any device that appears in macOS Audio MIDI Setup

### Export Format Options

| Interface | Recommended Export | Notes |
|-----------|-------------------|-------|
| RME with Room EQ | TotalMix (`.tmreq`) | Direct import to Room EQ panel |
| Other interfaces | REW Text (`.txt`) | Import into most EQ software |
| Any interface | Manual Display | Copy values to any EQ manually |

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/robfosterdotnet/EQilibrium.git
cd EQilibrium

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Running the Application

```bash
# Using the entry point
eqilibrium

# Or as a Python module
python -m roomeq
```

## Quick Start

1. **Connect your equipment**
   - Audio interface connected and powered on
   - Measurement microphone plugged into an input
   - Speakers connected to outputs

2. **Launch EQilibrium**
   ```bash
   eqilibrium
   ```

3. **Follow the wizard**
   - Select your audio interface and channels
   - Position your microphone as shown in the diagrams
   - Let the measurements run (about 15-20 minutes for full measurement)
   - Review the analysis and export your correction EQ

## How It Works

### The 6-Step Wizard

#### Step 1: Welcome
Review the checklist and optionally load an existing project or microphone calibration file.

#### Step 2: Device Setup
- Auto-detects your audio interface
- Select microphone input channel
- Select speaker outputs (left/right/both)
- Test levels with real-time meter

#### Step 3: Listening Position
- Choose number of measurement positions (5, 7, or 9)
- View the position diagram showing where to place the mic
- Positions cover the "sweet spot" and surrounding area

#### Step 4: Measurement
- Separate measurement rounds for left and right speakers
- Visual guide shows current mic position
- Real-time level monitoring
- Option to redo any measurement
- 3-second countdown before each sweep

#### Step 5: Analysis
- View averaged frequency response graph
- See detected problems with severity ratings
- Warnings about issues EQ cannot fix (deep nulls)
- Toggle smoothing for clearer view

#### Step 6: Export
- Generated parametric EQ (up to 9 bands for RME, 31 for generic)
- Before/after comparison graph
- Choose export format based on your interface:
  - **RME TotalMix**: Direct `.tmreq` export
  - **REW Text**: Compatible with most EQ software
  - **Manual Display**: Human-readable table for manual entry
- Step-by-step import instructions for each format

### Measurement Positions

The default 7-position measurement covers the listening area:

```
       [L]           [R]      <- Speakers
          \         /
           \   1   /          <- Center (ear height)
            \ 2 3 /           <- 30cm left/right of center
             \4 5/            <- 30cm forward
              \6/             <- 30cm back
               7              <- 30cm up

      [Listening Position]
```

This spatial averaging captures how the room behaves across the listening area, not just at a single point.

## Technical Details

### Measurement Method

EQilibrium uses the **Farina logarithmic sine sweep method**:

1. **Sweep Generation**: 5-second exponential sweep from 20 Hz to 20 kHz
2. **Synchronized Capture**: Play sweep through speakers while recording through mic
3. **Deconvolution**: Convolve recording with inverse filter to extract impulse response
4. **FFT Analysis**: Convert to frequency domain for response curve

### Analysis Pipeline

1. **Power Averaging**: Combine multi-position measurements (preserves magnitude accuracy)
2. **Smoothing**: 1/24 octave fractional smoothing reduces noise
3. **Normalization**: Reference to 200-2000 Hz average (standard practice)
4. **Problem Detection**: `scipy.signal.find_peaks` identifies peaks and dips
5. **Q Estimation**: Calculate Q factor from -3dB bandwidth
6. **Severity Classification**: Minor (3-5 dB), Moderate (5-8 dB), Severe (8+ dB)

### EQ Optimization

The optimizer follows **room correction best practices**:

- **Cuts over boosts**: Peaks are cut (effective), dips get minimal boost (often ineffective)
- **Maximum boost limit**: 3 dB max (boosting dips wastes power and can cause distortion)
- **Deep null detection**: Nulls >10 dB are flagged as uncorrectable (phase cancellation)
- **Frequency range**: Focus on 30 Hz - 10 kHz (where EQ is most effective)
- **Constraint-aware**: Respects hardware limits (9 bands, Q 0.4-9.9, gain +/-20 dB)

The optimization uses `scipy.optimize.minimize` with L-BFGS-B to minimize weighted RMS error from target response.

### Biquad Filters

Filter coefficients calculated using the [Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html) formulas for peaking EQ:

```
H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)
```

Where A = 10^(dBgain/40) for peaking filters.

## Project Structure

```
EQilibrium/
├── src/roomeq/
│   ├── core/                    # DSP engine (UI-independent)
│   │   ├── audio_device.py      # Device detection and management
│   │   ├── sweep.py             # Log sweep generation, deconvolution
│   │   ├── measurement.py       # Capture orchestration
│   │   ├── averaging.py         # Multi-position averaging, smoothing
│   │   ├── analysis.py          # Problem detection, severity classification
│   │   ├── eq_optimizer.py      # Parametric EQ fitting algorithm
│   │   ├── biquad.py            # Filter coefficient calculation
│   │   ├── interface_profiles.py # Hardware profiles and EQ constraints
│   │   ├── export_formats.py    # Multi-format export (RME, REW, manual)
│   │   └── rme_export.py        # TotalMix .tmreq export (backward compat)
│   ├── ui/
│   │   ├── wizard.py            # Main QWizard framework
│   │   ├── pages/               # 6 wizard step pages
│   │   └── widgets/             # Level meter, frequency plot, position diagram
│   └── session/                 # Project save/load
├── tests/                       # 240 tests covering all modules
├── pyproject.toml
└── README.md
```

## Development

### Setup

```bash
# Clone and setup
git clone https://github.com/robfosterdotnet/EQilibrium.git
cd EQilibrium
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=roomeq --cov-report=html

# Run specific test file
pytest tests/test_analysis.py -v
```

### Code Quality

```bash
# Lint with ruff
ruff check src tests

# Type checking with mypy
mypy src
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Array operations, FFT |
| scipy | Signal processing, optimization |
| sounddevice | Audio I/O via PortAudio |
| PyQt6 | Desktop UI framework |
| pyqtgraph | High-performance plotting |

## Importing EQ Settings

### RME TotalMix (`.tmreq` files)

1. Open **TotalMix FX**
2. Click the **Room EQ** button (or press F8)
3. Click **Options** (gear icon)
4. Select **Load Preset...**
5. Navigate to your exported `.tmreq` file
6. Click **Open**

The EQ settings will be loaded into the Room EQ section. Enable Room EQ to hear the correction.

### REW Text Format (`.txt` files)

The REW-compatible text format can be imported into various EQ applications:
- **Equalizer APO** (Windows): Import as Filter Settings file
- **AU Lab** (macOS): Configure parametric EQ manually from values
- **DAW plugins**: Most parametric EQ plugins accept these values

### Manual Entry

The Manual Display format provides a clean table you can use to configure any parametric EQ:

```
Band | Type | Frequency | Gain   | Q
-----|------|-----------|--------|------
1    | Peak | 63 Hz     | -6.0dB | 4.3
2    | Peak | 125 Hz    | +4.5dB | 3.2
...
```

## Limitations

- **Deep nulls cannot be fixed**: Dips greater than ~10 dB are typically caused by phase cancellation at your listening position. No amount of EQ boost will fix this—the sound waves are literally canceling each other out. Consider speaker/listener repositioning or acoustic treatment.

- **9-band limit**: Most interfaces have a limited number of parametric EQ bands. EQilibrium prioritizes the most impactful corrections.

- **Not a replacement for acoustic treatment**: EQ corrects frequency response but cannot fix issues like excessive reverb, flutter echo, or early reflections.

## References

- [Farina, A. - Simultaneous Measurement of Impulse Response and Distortion](https://www.angelofarina.it/Public/Presentations/AES122-Farina.pdf)
- [Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)
- [HouseCurve - Equalization Best Practices](https://housecurve.com/docs/tuning/equalization)
- [PS Audio - Using EQ with Speakers: Some Limitations](https://www.psaudio.com/blogs/copper/using-eq-with-speakers-some-limitations)
- [RME TotalMix Room EQ](https://rme-audio.de/totalmix-fx-room-eq.html)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The audio engineering community for documenting room correction best practices
- Angelo Farina for the exponential sine sweep measurement method
- Robert Bristow-Johnson for the Audio EQ Cookbook
- The RME team for TotalMix FX and its Room EQ feature
