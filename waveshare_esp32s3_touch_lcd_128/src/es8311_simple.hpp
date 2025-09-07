#pragma once
#include <Arduino.h>
#include <Wire.h>

class ES8311 {
public:
    ES8311(uint8_t i2c_addr = 0x18, TwoWire &wire = Wire);
    bool begin(uint8_t sda, uint8_t scl, uint32_t sampleRate = 16000, uint8_t volume = 90, uint8_t micGain = 3, uint32_t mclk_hz = 6144000);
    void setVolume(uint8_t volume); // 0-100
    void setMicGain(uint8_t gain); // 0-7 (0=0dB, 7=42dB)
    void power(bool enable);
private:
    uint8_t _addr;
    TwoWire *_wire;
    uint32_t _mclk_hz;
    bool writeReg(uint8_t reg, uint8_t val);
    bool readReg(uint8_t reg, uint8_t &val);
    void reset();
    void configureClock(uint32_t sampleRate);
    void configureFormat();
    void configurePower();
    void configureMic(uint8_t gain);
    void configureVolume(uint8_t volume);
};
