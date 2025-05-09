import av
import logging

def check_hw_accel_support():
    """
    Проверяет доступность некоторых распространенных аппаратных кодеков через PyAV.
    """
    print("-" * 30)
    print("Checking PyAV Hardware Acceleration Support...")
    print("-" * 30)

    hw_support = {
        "nvenc": {"encoder": False, "decoder": False},
        "qsv":   {"encoder": False, "decoder": False},
        "vaapi": {"encoder": False, "decoder": False},
        "amf":   {"encoder": False, "decoder": False},
        "dxva2": {"decoder": False},
        "d3d11va": {"decoder": False},
    }

    encoders = [
        ('nvenc', 'h264_nvenc'), ('nvenc', 'hevc_nvenc'),
        ('qsv', 'h264_qsv'), ('qsv', 'hevc_qsv'),
        ('vaapi', 'h264_vaapi'), ('vaapi', 'hevc_vaapi'),
        ('amf', 'h264_amf'), ('amf', 'hevc_amf'),
    ]
    print("\n--- Checking Encoders ---")
    for api, name in encoders:
        try:
            codec = av.codec.Codec(name, mode='w')
            print(f"[ OK ] Encoder '{name}' ({api.upper()}) found.")
            logging.info(f"HW Encoder Check: '{name}' found.")
            if api in hw_support:
                hw_support[api]["encoder"] = True
        except (av.FFmpegError, ValueError):
            print(f"[FAIL] Encoder '{name}' ({api.upper()}) NOT found.")
            logging.warning(f"HW Encoder Check: '{name}' not found.")
        except Exception as e:
             print(f"[ERROR] Checking encoder '{name}': {e}")
             logging.error(f"Error checking encoder '{name}': {e}", exc_info=True)


    decoders = [
        ('nvenc', 'h264_cuvid'), ('nvenc', 'hevc_cuvid'),
        ('qsv', 'h264_qsv'), ('qsv', 'hevc_qsv'),
        ('vaapi', 'h264_vaapi'), ('vaapi', 'hevc_vaapi'),
        ('amf', 'h264_amf'), ('amf', 'hevc_amf'),
        ('dxva2', 'h264_dxva2'), ('dxva2', 'hevc_dxva2'),
        ('d3d11va', 'h264_d3d11va'), ('d3d11va', 'hevc_d3d11va'),
    ]
    print("\n--- Checking Decoders ---")
    for api, name in decoders:
        try:
            codec = av.codec.Codec(name, mode='r')
            print(f"[ OK ] Decoder '{name}' ({api.upper()}) found.")
            logging.info(f"HW Decoder Check: '{name}' found.")
            if api in hw_support:
                hw_support[api]["decoder"] = True
        except (av.FFmpegError, ValueError):
            print(f"[FAIL] Decoder '{name}' ({api.upper()}) NOT found.")
            logging.warning(f"HW Decoder Check: '{name}' not found.")
        except Exception as e:
             print(f"[ERROR] Checking decoder '{name}': {e}")
             logging.error(f"Error checking decoder '{name}': {e}", exc_info=True)

    print("-" * 30)
    print("Summary:")
    for api, support in hw_support.items():
        enc_stat = "Supported" if support.get("encoder", False) else "NOT Supported"
        dec_stat = "Supported" if support.get("decoder", False) else "NOT Supported"
        if api in ['dxva2', 'd3d11va']:
             print(f"  {api.upper()}: Decoder={dec_stat}")
        else:
             print(f"  {api.upper()}: Encoder={enc_stat}, Decoder={dec_stat}")
    print("-" * 30)

    try:
         h264_codec = av.codec.Codec('h264', 'r')
         hw_configs = h264_codec.get_hw_configs()
         if hw_configs:
              print("\n--- Available HW Configs for H.264 Decoder ---")
              for config in hw_configs:
                   print(f"  Method: {config.method}, Device Type: {config.device_type}, Pixel Format: {config.pix_fmt}")
              print("-" * 30)
         else:
              print("\nCodec 'h264' reported no specific HW configs via get_hw_configs().")
    except (AttributeError, av.FFmpegError, ValueError):
         print("\nCould not retrieve HW configs via get_hw_configs() (may not be supported).")
    except Exception as e:
         print(f"\nError retrieving HW configs: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    check_hw_accel_support()
