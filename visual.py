# Файл: minimal_bch_test.py
import bchlib
import math

m = 8
t = 4
n_bits = (1 << m) - 1
ecc_bits = m * t # Предполагаемый расчет
k_bits = n_bits - ecc_bits
n_bytes = math.ceil(n_bits / 8.0)

print(f"Testing with m={m}, t={t} => n={n_bits}, k={k_bits}, ecc={ecc_bits}")
print(f"Expected packet length: {n_bytes} bytes")

try:
    bch = bchlib.BCH(m=m, t=t)
    print("BCH object created.")

    # Создаем фиктивный пакет правильной длины
    dummy_packet = bytearray([0] * n_bytes)
    print(f"Dummy packet created with length {len(dummy_packet)}")

    print("Calling bch.decode...")
    flips, data, ecc = bch.decode(dummy_packet) # Вызываем decode
    print(f"bch.decode returned: flips={flips}")
    print("Test successful!")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()