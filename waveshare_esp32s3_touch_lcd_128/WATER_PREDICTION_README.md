# Water Consumption Prediction on ESP32-S3

This project implements a water consumption prediction model using TensorFlow Lite Micro on an ESP32-S3 microcontroller. The model analyzes audio signals (MFCC features) to predict water consumption in liters.

## Features

- **TensorFlow Lite Micro Integration**: Uses the Chirale TensorFlow Lite library optimized for microcontrollers
- **MFCC Feature Extraction**: Simplified MFCC feature extraction for audio analysis
- **INT8 Quantized Model**: Uses an optimized INT8 quantized model for efficient inference
- **Mock Audio Generation**: Includes mock audio generation for testing without real microphone
- **Serial Interface**: Interactive serial commands for testing and prediction

## Hardware Requirements

- ESP32-S3 development board
- Waveshare ESP32-S3 Touch LCD 1.28" (optional for display)
- Microphone (for real audio input - currently using mock data)

## Model Information

- **Model**: `model_improved_int8_3_2_1.tflite`
- **Input**: MFCC features (13 coefficients × 200 time frames)
- **Output**: Water consumption prediction in liters
- **Quantization**: INT8 (8-bit integer)
- **Model Size**: ~11KB

## Usage

### Serial Commands

1. **'p' or 'P'**: Perform water prediction with mock audio
2. **'t' or 'T'**: Test with different mock audio patterns
3. **'h' or 'H'**: Show help message

### Example Output

```
Water Consumption Prediction Model
Initializing TensorFlow Lite Micro Interpreter...
Model loaded successfully. Size: 11304 bytes
Model version: 3, Library version: 3
Tensor allocation successful
Input tensor dimensions: 1x200x13 (type: 2)
Output tensor dimensions: 1x1 (type: 2)
Initialization done.

Water prediction model ready. Press 'p' to predict or 't' to test with mock audio.
```

## Code Structure

### Key Components

1. **MFCC Processing**: Simplified MFCC feature extraction functions
2. **Model Inference**: TensorFlow Lite Micro interpreter setup and inference
3. **Audio Simulation**: Mock audio generation for testing
4. **Serial Interface**: Command-based interaction system

### Main Functions

- `extractMFCCFeatures()`: Extracts MFCC features from audio data
- `normalizeMFCC()`: Normalizes MFCC features
- `quantizeMFCC()`: Quantizes features for INT8 model
- `generateMockAudio()`: Generates test audio data
- `performWaterPrediction()`: Main prediction function
- `testWithMockAudio()`: Testing function with multiple patterns

## Configuration

### Audio Processing Parameters

```cpp
const int CHUNK_DURATION = 3;  // seconds
const int SAMPLE_RATE = 16000;
const int SAMPLES_PER_CHUNK = CHUNK_DURATION * SAMPLE_RATE;
const int N_MFCC = 13;
const int MAX_SEQUENCE_LENGTH = 200;
```

### Memory Configuration

```cpp
constexpr int kTensorArenaSize = 50000;  // Increased for larger model
```

## Integration with Real Microphone

To integrate with a real microphone:

1. Replace `generateMockAudio()` with actual microphone input
2. Implement proper audio sampling at 16kHz
3. Add audio preprocessing (filtering, windowing)
4. Consider implementing full MFCC extraction with FFT

## Model Training Context

This model was trained on:
- Audio samples of water flow sounds
- MFCC features extracted from 3-second audio chunks
- Ground truth water consumption measurements
- INT8 quantization for microcontroller deployment

## Performance Considerations

- **Memory Usage**: ~50KB tensor arena for model inference
- **Processing Time**: Depends on MFCC extraction complexity
- **Accuracy**: Model accuracy depends on training data quality
- **Real-time**: Current implementation suitable for batch processing

## Future Improvements

1. **Real MFCC Implementation**: Full FFT-based MFCC extraction
2. **Real-time Audio**: Continuous audio processing
3. **Model Optimization**: Further quantization and pruning
4. **Display Integration**: Use LCD for visual feedback
5. **Data Logging**: Store predictions and audio samples

## Troubleshooting

### Common Issues

1. **Tensor Allocation Failed**: Increase `kTensorArenaSize`
2. **Model Version Mismatch**: Check TensorFlow Lite library version
3. **Memory Issues**: Monitor heap and PSRAM usage
4. **Inference Errors**: Verify input tensor dimensions and data types

### Debug Information

The code includes extensive debug output:
- Model loading status
- Tensor dimensions and types
- Quantization parameters
- Memory usage statistics

## License

This project uses the TensorFlow Lite Micro library and follows the Apache 2.0 license.
