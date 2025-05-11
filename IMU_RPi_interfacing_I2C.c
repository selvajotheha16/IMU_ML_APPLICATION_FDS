#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <stdint.h>

#define ICM20948_ADDR     0x68
#define REG_PWR_MGMT_1    0x06
#define REG_ACCEL_XOUT_H  0x2D
#define REG_GYRO_XOUT_H   0x33

int i2c_fd;

// Write 8-bit value to register
void write_register(uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    write(i2c_fd, buffer, 2);
}

// Read 8-bit value from register
uint8_t read_register(uint8_t reg) {
    write(i2c_fd, &reg, 1);
    uint8_t value;
    read(i2c_fd, &value, 1);
    return value;
}

// Read 16-bit signed value from register pair
int16_t read_word(uint8_t regH) {
    write(i2c_fd, &regH, 1);
    uint8_t data[2];
    read(i2c_fd, data, 2);
    return (int16_t)((data[0] << 8) | data[1]);
}

// Convert accelerometer raw value (±16g range)
float convert_accel(int16_t raw) {
    return raw / 2048.0;
}

// Convert gyroscope raw value (±2000 dps range)
float convert_gyro(int16_t raw) {
    return raw / 16.4;
}

// Setup I2C communication and wake up sensor
int setup_icm20948() {
    const char *device = "/dev/i2c-1";  // I2C bus on Raspberry Pi
    i2c_fd = open(device, O_RDWR);
    if (i2c_fd < 0) {
        perror("Failed to open I2C bus");
        return -1;
    }

    if (ioctl(i2c_fd, I2C_SLAVE, ICM20948_ADDR) < 0) {
        perror("Failed to connect to ICM20948");
        return -1;
    }

    write_register(REG_PWR_MGMT_1, 0x01); // Wake up device
    usleep(100000);
    return 0;
}

// Read and print sensor values
void read_sensor_data() {
    int16_t ax = read_word(REG_ACCEL_XOUT_H);
    int16_t ay = read_word(REG_ACCEL_XOUT_H + 2);
    int16_t az = read_word(REG_ACCEL_XOUT_H + 4);

    int16_t gx = read_word(REG_GYRO_XOUT_H);
    int16_t gy = read_word(REG_GYRO_XOUT_H + 2);
    int16_t gz = read_word(REG_GYRO_XOUT_H + 4);

    printf("Accel X: %.2f m/s^2, Y: %.2f m/s^2, Z: %.2f m/s^2\n",
           convert_accel(ax), convert_accel(ay), convert_accel(az));

    printf("Gyro X: %.2f rad/s, Y: %.2f rad/s, Z: %.2f rad/s\n",
           convert_gyro(gx), convert_gyro(gy), convert_gyro(gz));
}

int main() {
    if (setup_icm20948() != 0) return 1;

    while (1) {
        read_sensor_data();
        sleep(1);
    }

    close(i2c_fd);
    return 0;
}
