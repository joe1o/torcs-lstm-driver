# TORCS AI Driver — LSTM-Based Autonomous Racing Agent

An end-to-end autonomous driving agent for the [TORCS](http://torcs.sourceforge.net/) (The Open Racing Car Simulator) racing simulator, built as part of an AI course project. The agent collects real-time telemetry data through manual play, trains an LSTM neural network on that data, and then drives the car autonomously using the learned model.

---

## Project Overview

The pipeline has two phases:

1. **Data Collection** — A human driver plays TORCS while the client logs all sensor readings and control inputs to a CSV file in real time.
2. **Model Training & Inference** — The logged data is used to train an LSTM model that learns to predict steering, acceleration, braking, gear, and reverse mode from a sequence of sensor inputs. The trained model is then deployed back into the client to drive autonomously.

---

## Architecture

The model is a multi-output LSTM network (`TorcsLSTM`) with separate output branches per control signal:

```
Input Sequence (seq_len=10, features=58)
        │
   LSTM (2 layers, hidden=128)
        │
   Shared FC (128 → 64, ReLU, Dropout)
        │
   ┌────┬────┬────┬──────┬─────────┐
Steer Accel Brake Gear  Reverse
                (Sigmoid)
```

- **Input features (58 total):** SpeedX/Y/Z, Angle, TrackPos, RPM, 19 track sensors, 36 opponent sensors
- **Output targets (5):** Steer, Accel, Brake, Gear, ReverseMode
- **Loss:** Weighted MSE — Brake has a 10× weight to compensate for class imbalance (braking is rare)
- **Optimizer:** Adam (lr=0.001) with `ReduceLROnPlateau` scheduler

---

## Project Structure

```
├── pyclient.py          # UDP client — connects to TORCS server, main entry point
├── driver.py            # Core driver logic — data collection + AI inference
├── model.py             # TorcsLSTM model definition
├── train_model.py       # Training script with weighted loss and LR scheduling
├── data_processor.py    # Data loading, normalization, sequence creation
├── carState.py          # Parses TORCS sensor messages into Python object
├── carControl.py        # Builds TORCS control messages from Python object
├── msgParser.py         # UDP message parser/serializer
├── telemetry_log.csv    # Collected training data (~384k timesteps)
└── training_history.png # Loss curves from training run
```

---

## Inputs & Outputs

### Sensor Inputs (Features)
| Feature | Description |
|---|---|
| `SpeedX`, `SpeedY`, `SpeedZ` | Car velocity along 3 axes (km/h) |
| `Angle` | Car angle relative to track axis (radians) |
| `TrackPos` | Lateral position on track (−1 to +1) |
| `RPM` | Engine RPM |
| `track[1..19]` | Distance to track edge in 19 directions (meters) |
| `opponent[1..36]` | Distance to nearest opponent in 36 directions (200 = none) |

### Control Outputs (Targets)
| Output | Range | Description |
|---|---|---|
| `Steer` | −1.0 to +1.0 | Steering angle |
| `Accel` | 0.0 to 1.0 | Throttle |
| `Brake` | 0.0 to 1.0 | Brake pressure |
| `Gear` | −1 to 4 | Gear selection |
| `ReverseMode` | 0 or 1 | Reverse flag |

---

## Key Design Decisions

- **Sequence modeling with LSTM** — Rather than a single-frame feedforward net, the LSTM receives a window of the last 10 timesteps, allowing it to capture dynamics like acceleration ramps and cornering intent.
- **Weighted brake loss** — Braking events are rare (<5% of timesteps) but critical. A 10× MSE weight on the brake output forces the model to take braking seriously instead of predicting near-zero always.
- **Separate output branches** — Each control signal (steer, accel, brake, gear, reverse) has its own output head. The brake branch adds an extra hidden layer + Sigmoid to improve sensitivity.
- **Opponent sensors included** — All 36 opponent distance sensors are included as features to allow the model to eventually learn overtaking or avoidance behavior.

---

## Dataset

The telemetry CSV (`telemetry_log.csv`) contains **~384,000 timesteps** of human-driven data with 66 columns (6 core sensors + 19 track sensors + 36 opponent sensors + 5 control outputs).

```
SpeedX, SpeedY, SpeedZ, Angle, TrackPos, Steer, Gear, Accel, RPM, Brake, ReverseMode,
track[1..19], opponent[1..36]
```

---

## References

- [TORCS — The Open Racing Car Simulator](http://torcs.sourceforge.net/)
- [SCR Championship Patch](https://github.com/fmirus/torcs-1.3.7)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Base client structure adapted from `lanquarden/raceconfig` (2012)

---
