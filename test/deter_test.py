import lgpio
import time

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, 18)

try:
    while True:
        # Buzz up
        for freq in range(900, 10000, 10):
            lgpio.tx_pwm(h, 18, freq, 50)
            time.sleep(0.02)
        # Buzz down
        for freq in range(400, 200, -10):
            lgpio.tx_pwm(h, 18, freq, 50)
            time.sleep(0.02)

except KeyboardInterrupt:
    pass

finally:
    lgpio.tx_pwm(h, 18, 0, 0)
    lgpio.gpiochip_close(h)
    print("Stopped.")