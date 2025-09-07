#include "es8311_simple.hpp"

// ES8311 register map (minimal, add more as needed)
#define ES8311_RESET_REG00      0x00
#define ES8311_CLK_MANAGER_REG01 0x01
#define ES8311_CLK_MANAGER_REG02 0x02
#define ES8311_CLK_MANAGER_REG03 0x03
#define ES8311_CLK_MANAGER_REG04 0x04
#define ES8311_CLK_MANAGER_REG05 0x05
#define ES8311_CLK_MANAGER_REG06 0x06
#define ES8311_CLK_MANAGER_REG07 0x07
#define ES8311_CLK_MANAGER_REG08 0x08
#define ES8311_SDPIN_REG09      0x09
#define ES8311_SDPOUT_REG0A     0x0A
#define ES8311_SYSTEM_REG0D     0x0D
#define ES8311_SYSTEM_REG0E     0x0E
#define ES8311_SYSTEM_REG12     0x12
#define ES8311_SYSTEM_REG13     0x13
#define ES8311_SYSTEM_REG14     0x14
#define ES8311_ADC_REG16        0x16
#define ES8311_DAC_REG32        0x32

ES8311::ES8311(uint8_t i2c_addr, TwoWire &wire) : _addr(i2c_addr), _wire(&wire), _mclk_hz(6144000) {}

bool ES8311::begin(uint8_t sda, uint8_t scl, uint32_t sampleRate, uint8_t volume, uint8_t micGain, uint32_t mclk_hz) {
    Serial.println("[ES8311] begin()");
    _wire->begin(sda, scl);
    _mclk_hz = mclk_hz;
    Serial.printf("[ES8311] Using MCLK: %lu Hz\n", (unsigned long)_mclk_hz);
    reset();
    configureClock(sampleRate);
    configureFormat();
    configurePower();
    configureMic(micGain);
    configureVolume(volume);
    Serial.println("[ES8311] begin() complete");
    return true;
}

void ES8311::reset() {
    Serial.println("[ES8311] reset()");
    writeReg(ES8311_RESET_REG00, 0x1F); // Reset
    delay(10);
    writeReg(ES8311_RESET_REG00, 0x00);
    delay(10);
    writeReg(ES8311_RESET_REG00, 0x80); // Power-on
    delay(10);
    Serial.println("[ES8311] reset() done");
}

void ES8311::configureClock(uint32_t sampleRate) {
    Serial.println("[ES8311] configureClock()");
    // Minimal: set all clocks enabled, MCLK from MCLK pin, not inverted
    writeReg(ES8311_CLK_MANAGER_REG01, 0x3F);
    // Set dividers for 16kHz, 256fs (adjust for other rates as needed)
    writeReg(ES8311_CLK_MANAGER_REG02, 0x60); // pre_div=3 (0x60), pre_multi=0
    writeReg(ES8311_CLK_MANAGER_REG03, 0x10); // fs_mode=0, adc_osr=0x10
    writeReg(ES8311_CLK_MANAGER_REG04, 0x10); // dac_osr=0x10
    writeReg(ES8311_CLK_MANAGER_REG05, 0x00); // adc_div=1, dac_div=1
    writeReg(ES8311_CLK_MANAGER_REG06, 0x00); // bclk_div=1
    writeReg(ES8311_CLK_MANAGER_REG07, 0x00); // lrck_h=0
    writeReg(ES8311_CLK_MANAGER_REG08, 0xFF); // lrck_l=0xFF
    Serial.println("[ES8311] configureClock() done");
}

void ES8311::configureFormat() {
    Serial.println("[ES8311] configureFormat()");
    // I2S, 16-bit, slave mode
    writeReg(ES8311_SDPIN_REG09, 0x0C); // 16-bit
    writeReg(ES8311_SDPOUT_REG0A, 0x0C); // 16-bit
    Serial.println("[ES8311] configureFormat() done");
}

void ES8311::configurePower() {
    Serial.println("[ES8311] configurePower()");
    writeReg(ES8311_SYSTEM_REG0D, 0x01); // Power up analog
    writeReg(ES8311_SYSTEM_REG0E, 0x02); // Enable analog PGA, ADC modulator
    writeReg(ES8311_SYSTEM_REG12, 0x00); // Power up DAC
    writeReg(ES8311_SYSTEM_REG13, 0x10); // Enable output to HP drive
    Serial.println("[ES8311] configurePower() done");
}

void ES8311::configureMic(uint8_t gain) {
    Serial.println("[ES8311] configureMic()");
    writeReg(ES8311_SYSTEM_REG14, 0x1A); // Analog mic, max PGA gain
    setMicGain(gain);
    Serial.println("[ES8311] configureMic() done");
}

void ES8311::configureVolume(uint8_t volume) {
    Serial.println("[ES8311] configureVolume()");
    setVolume(volume);
    Serial.println("[ES8311] configureVolume() done");
}

void ES8311::setVolume(uint8_t volume) {
    if (volume > 100) volume = 100;
    uint8_t reg = (volume == 0) ? 0 : ((volume * 256) / 100) - 1;
    Serial.printf("[ES8311] setVolume: %d (reg=0x%02X)\n", volume, reg);
    writeReg(ES8311_DAC_REG32, reg);
}

void ES8311::setMicGain(uint8_t gain) {
    if (gain > 7) gain = 7;
    Serial.printf("[ES8311] setMicGain: %d\n", gain);
    writeReg(ES8311_ADC_REG16, gain);
}

void ES8311::power(bool enable) {
    Serial.printf("[ES8311] power: %s\n", enable ? "ON" : "OFF");
    if (enable) {
        writeReg(ES8311_SYSTEM_REG0D, 0x01);
    } else {
        writeReg(ES8311_SYSTEM_REG0D, 0x00);
    }
}

bool ES8311::writeReg(uint8_t reg, uint8_t val) {
    _wire->beginTransmission(_addr);
    _wire->write(reg);
    _wire->write(val);
    bool ok = (_wire->endTransmission() == 0);
    Serial.printf("[ES8311] writeReg: 0x%02X = 0x%02X %s\n", reg, val, ok ? "OK" : "FAIL");
    return ok;
}

bool ES8311::readReg(uint8_t reg, uint8_t &val) {
    _wire->beginTransmission(_addr);
    _wire->write(reg);
    if (_wire->endTransmission(false) != 0) {
        Serial.printf("[ES8311] readReg: 0x%02X NACK\n", reg);
        return false;
    }
    if (_wire->requestFrom(_addr, (uint8_t)1) != 1) {
        Serial.printf("[ES8311] readReg: 0x%02X no data\n", reg);
        return false;
    }
    val = _wire->read();
    Serial.printf("[ES8311] readReg: 0x%02X = 0x%02X\n", reg, val);
    return true;
}
