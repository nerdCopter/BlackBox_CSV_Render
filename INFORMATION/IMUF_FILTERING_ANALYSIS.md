# IMU-F Filtering Implementation Analysis

## Issue Identified
The filtered gyro data from EmuFlight HELIOSPRING IMU-F does not appear to have the expected 90Hz cutoff frequency, despite the header parameters indicating `IMUF_lowpass_*=90` and `IMUF_ptn_order=2`.

## Potential Root Causes in IMU-F Firmware

### 1. PTN Filter Scaling Factor Issue
**Location**: `src/filter/ptnFilter.c` in IMU-F repository
```c
const float ScaleF[] = { 1.0f, 1.553773974f, 1.961459177f, 2.298959223f };
Adj_f_cut = (float)f_cut * ScaleF[filter->order - 1];
```

**Problem**: For PT2 filters (order=2), the actual cutoff becomes:
- Configured: 90Hz
- Actual: `90Hz * 1.553773974f ≈ 140Hz`

This scaling is supposedly for Butterworth filter correction, but may be incorrectly applied or the wrong coefficients used.

### 2. Refresh Rate Mismatch
**Location**: `src/filter/ptnFilter.h`
```c
#define REFRESH_RATE  0.00003125f  // 32kHz assumed
```

**Problem**: If the actual gyro sampling rate differs from 32kHz, all filter calculations become incorrect. The filter gain calculation:
```c
filter->k = REFRESH_RATE / ((1.0f / (2.0f * M_PI_FLOAT * Adj_f_cut)) + REFRESH_RATE);
```
Would produce wrong cutoff frequencies if `REFRESH_RATE` doesn't match reality.

### 3. Filter Chain Complexity
**Processing Order**:
1. Raw Gyro → IMUF lowpass (per-axis)
2. → PTN filter (based on `IMUF_ptn_order`) 
3. → Kalman filter with adaptive parameters
4. → Flight Controller

**Problem**: Each stage can introduce:
- Phase delays that shift effective cutoff
- Non-linear behavior from Kalman adaptive Q-factor
- Cumulative errors from multiple filtering stages

### 4. Kalman Filter Interference
**Location**: `src/filter/kalman.c`
```c
float e = CONSTRAIN(kalmanState->r / 45.0f + 0.005f, 0.005f, 0.9f);
e = -SQUARE(e - 1.0f) * 0.7f + (e - 1.0f) * (1.0f - 0.7f) + 1.0f;
kalmanState->p = kalmanState->p + (kalmanState->q * kalmanState->e);
```

**Problem**: The pseudo-Kalman filter runs after PTN filtering and:
- Uses adaptive noise parameters that change with flight conditions
- May significantly alter frequency response beyond simple lowpass
- Has non-linear behavior that could mask the intended cutoff

### 5. Parameter Conversion Issues
**Multiple conversion points**:
- Headers store integer values (Q-factors scaled by 1000)
- Firmware converts to floats with potential precision loss
- Multiple filter stages each do their own conversions

**Problem**: Quantization errors and floating-point precision issues could accumulate across the filter chain.

### 6. Configuration Application Timing
**Location**: `src/filter/filter.c`
```c
if (allowFilterInit) {
    allowFilterInit = 0;
    filterConfig.roll_q = (float)filterConfig.i_roll_q;
    // ... conversions
    filter_init();
}
```

**Problem**: Filter parameters are applied once at initialization. If there are issues with:
- Parameter parsing from flight controller
- Timing of when parameters are applied
- Default fallbacks not matching expectations

## Diagnostic Recommendations

### 1. Verify Actual IMU-F Parameters
Check if the IMU-F firmware is actually receiving and applying the expected parameters:
- Are the 90Hz lowpass values actually being used?
- Is the PT2 order being applied correctly?
- Are there default overrides or limits being applied?

### 2. Measure Actual Filter Performance
Compare theoretical vs actual response:
- Generate test signals at known frequencies
- Measure actual attenuation at 90Hz, 180Hz, etc.
- Check for phase delays and group delay issues

### 3. Isolate Filter Stages
Test each filter stage independently:
- Raw gyro → IMUF lowpass only
- → PTN filter only  
- → Full chain including Kalman

### 4. Check Sampling Rate Assumptions
Verify the actual gyro sampling rate matches the 32kHz assumption in the PTN filter calculations.

## Implications for BlackBox Analysis

The filter response curves we generate are mathematically correct for the *intended* filtering behavior based on header parameters. However, if the IMU-F firmware has implementation bugs or design issues, the actual filtering may not match these theoretical curves.

This could explain why:
- The filtered gyro spectrum doesn't show the expected 90Hz rolloff
- There may be unexpected peaks or resonances
- The filtering delay measurements may not match theoretical predictions

## Recommendations

1. **Document the discrepancy**: Note that our filter response curves represent intended behavior, not necessarily actual IMU-F performance
2. **Add warning messages**: When IMUF filters are detected, include a note about potential firmware implementation issues
3. **Consider empirical curve fitting**: If possible, derive actual filter response from the filtered vs unfiltered gyro data rather than using theoretical curves

This analysis suggests the IMU-F firmware may have bugs or design flaws that prevent it from achieving the filtering performance indicated by its configuration parameters.
