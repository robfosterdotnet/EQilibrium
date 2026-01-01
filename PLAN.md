# RoomEQ - Simple Room Correction Wizard for RME Interfaces

## Overview

A macOS desktop application that simplifies room acoustic measurement and correction for RME audio interfaces (UCX II, 802 FS, etc.). Competes with REW by providing a guided wizard experience instead of complex configuration.

**Key Differentiators from REW:**
- Single-purpose: Room correction only (not a general measurement tool)
- Guided wizard workflow (no configuration overload)
- Auto-detection of RME interfaces
- Visual mic placement guides
- Direct export to RME TotalMix Room EQ format

## Tech Stack

- **Python 3.11+** with **PyQt6** for UI
- **python-sounddevice** for audio I/O (synchronized playback/recording)
- **numpy/scipy** for DSP (FFT, signal processing, optimization)
- **pyqtgraph** for efficient frequency response plotting

## Project Structure

```
RoomEQ/
├── pyproject.toml
├── PLAN.md                      # This file
├── src/roomeq/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
│   ├── core/                    # DSP engine (UI-independent)
│   │   ├── audio_device.py      # RME detection, device management
│   │   ├── sweep.py             # Log sweep generation, deconvolution
│   │   ├── measurement.py       # Capture orchestration
│   │   ├── averaging.py         # Multi-position averaging
│   │   ├── analysis.py          # Frequency response analysis
│   │   ├── eq_optimizer.py      # Parametric EQ fitting
│   │   ├── biquad.py            # Filter coefficient calculation
│   │   └── rme_export.py        # TotalMix export format
│   ├── session/                 # Project save/load
│   │   ├── project.py
│   │   └── calibration.py       # Mic calibration file support
│   ├── ui/
│   │   ├── wizard.py            # QWizard main flow
│   │   ├── pages/               # Wizard pages (6 steps)
│   │   │   ├── welcome.py
│   │   │   ├── device_setup.py
│   │   │   ├── listening_position.py
│   │   │   ├── measurement.py
│   │   │   ├── analysis.py
│   │   │   └── export.py
│   │   └── widgets/             # Reusable components
│   │       ├── level_meter.py
│   │       ├── frequency_plot.py
│   │       └── position_diagram.py
│   └── utils/
└── tests/
    ├── conftest.py
    ├── test_sweep.py
    ├── test_audio_device.py
    ├── test_measurement.py
    ├── test_averaging.py
    ├── test_analysis.py
    ├── test_biquad.py
    ├── test_eq_optimizer.py
    └── test_rme_export.py
```

## Wizard Flow (6 Steps)

### Step 1: Welcome
- Brief intro, checklist (RME connected, mic ready, speakers positioned)
- Optional: Load existing project, import mic calibration file

### Step 2: Device Setup
- Auto-detect RME interface
- Select mic input channel
- Select speaker output (left/right/both)
- Test button with real-time level meter

### Step 3: Listening Position
- Visual diagram of measurement positions
- Select number of positions (5/7/9 - default 7)
- Shows where mic will be placed relative to sweet spot

### Step 4: Measurement (Main Work)
- **Separate L/R measurement rounds** (more accurate per-channel correction)
  - First: All 7 positions with LEFT speaker only
  - Then: All 7 positions with RIGHT speaker only
  - Total: 14 measurements (~15-20 minutes)
- Visual diagram showing current mic position
- Real-time level meter
- Progress indicator (e.g., "Left Speaker: Position 3 of 7")
- "Redo last" option
- Countdown before each sweep plays

### Step 5: Analysis
- Frequency response graph (averaged measurements)
- List of detected room problems (peaks/dips with severity)
- Overall deviation metric
- Smoothing toggle

### Step 6: Export
- Generated 9-band parametric EQ
- Before/after comparison graph
- Export for left/right/both channels
- Clear instructions for TotalMix import

## Core Algorithms

### 1. Measurement: Logarithmic Sine Sweep (Farina Method)
- Generate 5-second exponential sweep (20Hz-20kHz)
- Play through speakers, record through mic simultaneously
- Deconvolve with inverse filter to extract impulse response
- FFT to get frequency response

### 2. Averaging
- Complex averaging of multiple position measurements
- 1/24 octave smoothing for display
- Preserves phase information for accurate correction

### 3. Problem Detection
- Peak/dip detection using scipy.signal.find_peaks
- Estimate Q from peak width
- Classify severity (minor/moderate/severe based on dB deviation)

### 4. EQ Optimization
- Initialize filters at detected problem frequencies
- Iterative optimization (scipy.optimize.minimize)
- Respect RME constraints: 9 bands, Q 0.4-9.9, gain +/-20dB
- Round to RME precision (freq 1Hz, gain 0.1dB, Q 0.1)

### 5. RME Export Format
REW-compatible text format that TotalMix can import:
```
Filter Settings file

Room EQ Wizard V5.20

Equaliser: Generic

Filter  1: ON  PK       Fc    63.0 Hz  Gain  -6.0 dB  Q  4.32
Filter  2: ON  PK       Fc   125.0 Hz  Gain  -4.5 dB  Q  3.20
...
```

## Measurement Positions (7-position default)

```
       [L]           [R]      <- Speakers
          \         /
           \   1   /          <- Center, ear height
            \ 2 3 /           <- 30cm left/right
             \4 5/            <- 30cm forward
              \6/             <- 30cm back
               7              <- 30cm up

      [Listening Position]
```

## Development Phases & Progress

### Phase 1: Core Foundation
- [x] Project setup, dependencies (pyproject.toml)
- [x] Audio device management (RME detection)
- [x] Sweep generation and deconvolution
- [x] **Tests pass for Phase 1** (53 tests passing)

### Phase 2: Measurement Pipeline
- [x] Synchronized playback/recording
- [x] Frequency response calculation
- [x] Multi-position averaging and smoothing
- [x] **Tests pass for Phase 2** (134 total tests passing)

### Phase 3: EQ Optimization
- [x] Biquad filter coefficients (Audio EQ Cookbook)
- [x] Optimization algorithm
- [x] RME export format
- [x] **Tests pass for Phase 3** (214 total tests passing)

### Phase 4: User Interface
- [x] PyQt6 wizard framework
- [x] All 6 wizard pages
- [x] Custom widgets (level meter, frequency plot, position diagram)
- [x] **Tests pass for Phase 4** (214 tests still passing)

### Phase 5: Polish
- [ ] Error handling
- [ ] Testing with real hardware
- [ ] User guide

### Phase 6: Distribution
- [ ] py2app packaging for macOS
- [ ] DMG creation

## Dependencies

```toml
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "sounddevice>=0.4.6",
    "PyQt6>=6.5.0",
    "pyqtgraph>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
```

## Key Sources

- [Farina's Exponential Sine Sweep Method](https://www.angelofarina.it/Public/Presentations/AES122-Farina.pdf)
- [python-sounddevice](https://python-sounddevice.readthedocs.io/)
- [RME TotalMix Room EQ](https://rme-audio.de/totalmix-fx-room-eq.html)
- [Audio EQ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html)
- [RME Forum - Import to Room EQ](https://forum.rme-audio.de/viewtopic.php?id=38604)
