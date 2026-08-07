# ENERGY_DRIFT_CONFIRMED

Corrected energy was reconstructed offline from the existing lossless-float64 state CSVs.

## Statistics

| Lane | Method | Maximum | Worst epoch (yr) | RMS | P99 | Fitted change over 1 Myr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m0_conv_0p5d_1myr_s12345 | recorded | 2.96666364955e-10 | 998700 | 1.63116276543e-10 | 2.80748636934e-10 | 2.89548318171e-10 |
| m0_conv_0p5d_1myr_s12345 | float64 | 2.96666364955e-10 | 998700 | 1.63116276543e-10 | 2.80748636934e-10 | 2.89548318171e-10 |
| m0_conv_0p5d_1myr_s12345 | compensated | 2.96666923766e-10 | 998700 | 1.63116755325e-10 | 2.80750313366e-10 | 2.89548287063e-10 |
| m0_conv_0p5d_1myr_s12345 | decimal | 2.96666496845e-10 | 998700 | 1.63116515219e-10 | 2.80749895559e-10 | 2.89548283388e-10 |
| m0_conv_0p25d_1myr_s12345 | recorded | 5.77876207139e-10 | 998700 | 3.32581788951e-10 | 5.69616050926e-10 | 5.77548936344e-10 |
| m0_conv_0p25d_1myr_s12345 | float64 | 5.77876207139e-10 | 998700 | 3.32581788951e-10 | 5.69616050926e-10 | 5.77548936344e-10 |
| m0_conv_0p25d_1myr_s12345 | compensated | 5.77877511031e-10 | 998700 | 3.32582287068e-10 | 5.6961847244e-10 | 5.77548934483e-10 |
| m0_conv_0p25d_1myr_s12345 | decimal | 5.77877144974e-10 | 998700 | 3.32582043383e-10 | 5.696177853e-10 | 5.77548927044e-10 |

## Method Checks

- Historical artifact audit: `21` Step 3 entries and `28` Step 3b entries verified (`28` unique artifacts).
- m0_conv_0p5d_1myr_s12345: telemetry reproduction `True`; worst tolerance ratio `0` at `0` years.
- m0_conv_0p5d_1myr_s12345: compensated/Decimal agreement `True`; worst drift difference `1.0233876108e-15` at `866300` years.
- m0_conv_0p5d_1myr_s12345 float64: worst corrected-energy composition cancellation `3.22210412225` at `336300` years; worst drift-subtraction cancellation `9.85053676786e+13` at `20900` years.
- m0_conv_0p5d_1myr_s12345 compensated: worst corrected-energy composition cancellation `3.22210412225` at `336300` years; worst drift-subtraction cancellation `1.02257953114e+14` at `20900` years.
- m0_conv_0p5d_1myr_s12345 decimal: worst corrected-energy composition cancellation `3.22210412225` at `336300` years; worst drift-subtraction cancellation `9.92823164462e+13` at `20900` years.
- m0_conv_0p25d_1myr_s12345: telemetry reproduction `True`; worst tolerance ratio `0` at `0` years.
- m0_conv_0p25d_1myr_s12345: compensated/Decimal agreement `True`; worst drift difference `1.00092458576e-15` at `777000` years.
- m0_conv_0p25d_1myr_s12345 float64: worst corrected-energy composition cancellation `3.22210414461` at `336300` years; worst drift-subtraction cancellation `2.90191488567e+14` at `1300` years.
- m0_conv_0p25d_1myr_s12345 compensated: worst corrected-energy composition cancellation `3.22210414461` at `336300` years; worst drift-subtraction cancellation `2.90191488567e+14` at `1300` years.
- m0_conv_0p25d_1myr_s12345 decimal: worst corrected-energy composition cancellation `3.22210414461` at `336300` years; worst drift-subtraction cancellation `2.84152642669e+14` at `1300` years.

## Classification

- Telemetry reconstruction passed: `True`.
- Compensated/Decimal agreement passed: `True`.
- Confirmed-drift rule passed: `True`.
- Evidence supports future telemetry evaluation changes: `False`.
- Historical Step 3 and Step 3b statuses remain unchanged.
- No integration or Stage 4 command was run or produced.

## Next Action

Before any further timestep halving, preregister a bounded integrator/roundoff diagnosis that separates WHFast truncation, synchronization, and accumulated state roundoff over shorter fixed horizons.
