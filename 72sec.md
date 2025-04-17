[2025-04-16 22:40:37,301] I MainThread - <module>:88 - --- Запуск Скрипта Извлечения (ThreadPool + Batches + OpenCL Attempt) ---
[2025-04-16 22:40:37,302] I MainThread - <module>:89 - Метод выбора колец: pool_entropy_selection, Pool: 4, Select: 2
[2025-04-16 22:40:37,302] I MainThread - <module>:90 - Ожид. Payload: 64bit, Ожид. ECC: True (Galois BCH m=8, t=4), Доступно/Работает: True
[2025-04-16 22:40:37,302] I MainThread - <module>:91 - Ожид. Альфа для логирования: MIN=1.005, MAX=1.12, N_RINGS_Total=8
[2025-04-16 22:40:37,302] I MainThread - <module>:92 - Компонент: Cb
[2025-04-16 22:40:37,302] I MainThread - <module>:93 - Параллелизм: ThreadPoolExecutor (max_workers=default) с батчингом.
[2025-04-16 22:40:37,302] I MainThread - <module>:94 - DTCWT Бэкенд: Попытка использовать OpenCL (иначе NumPy).
[2025-04-16 22:40:37,303] I MainThread - <module>:772 - Original dtcwt backend: numpy
[2025-04-16 22:40:37,303] I MainThread - <module>:773 - Attempting to switch dtcwt backend to OpenCL...
[2025-04-16 22:40:37,303] I MainThread - <module>:776 - DTCWT backend switched to: opencl
[2025-04-16 22:40:37,303] I MainThread - <module>:779 - Initializing OpenCL backend via Transform2d()...
[2025-04-16 22:40:37,919] I MainThread - <module>:784 - OpenCL backend initialized and test transform successful.
[2025-04-16 22:40:37,920] I MainThread - <module>:802 - Active DTCWT backend before calling main: opencl
[2025-04-16 22:40:37,921] I MainThread - main:727 - Original ID loaded: 48a49976264bdf27
[2025-04-16 22:40:37,921] I MainThread - main:731 - --- Starting Extraction Main Process (ThreadPool + Batches + OPENCL DTCWT) ---
[2025-04-16 22:40:37,921] I MainThread - read_video:288 - Reading: watermarked_galois_t4_opencl_thr_batched.avi
[2025-04-16 22:40:37,925] I MainThread - read_video:292 - Props: 842x720@30.00~907f
[2025-04-16 22:40:39,460] I MainThread - read_video:299 - Read loop finished. V:907,N:0,I:0
[2025-04-16 22:40:39,460] D MainThread - read_video:302 - Releasing capture
[2025-04-16 22:40:39,469] I MainThread - main:736 - Read 907 frames.
[2025-04-16 22:40:39,469] I MainThread - extract_watermark_from_video:581 - Starting extraction (ThreadPool+Batches, Bits/Pair:2)
[2025-04-16 22:40:39,469] I MainThread - extract_watermark_from_video:598 - Galois BCH OK: n=255, k=223, t=4. Expecting ECC packets (255b).
[2025-04-16 22:40:39,469] I MainThread - extract_watermark_from_video:607 - Extracting from 453 pairs (max repeats:5).
[2025-04-16 22:40:39,470] I MainThread - extract_watermark_from_video:628 - Launching 16 batches (453 pairs) using ThreadPool (mw=16, batch_size\u224830)...
[2025-04-16 22:40:39,471] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:0] Candidates: [2, 7, 1, 6]
[2025-04-16 22:40:39,472] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:30] Candidates: [2, 4, 0, 3]
[2025-04-16 22:40:39,473] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:60] Candidates: [3, 0, 2, 6]
[2025-04-16 22:40:39,475] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:90] Candidates: [3, 5, 6, 2]
[2025-04-16 22:40:39,477] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:120] Candidates: [6, 7, 2, 3]
[2025-04-16 22:40:39,479] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:150] Candidates: [6, 2, 0, 1]
[2025-04-16 22:40:39,481] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:180] Candidates: [2, 5, 3, 7]
[2025-04-16 22:40:39,484] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:210] Candidates: [3, 6, 0, 1]
[2025-04-16 22:40:39,485] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:240] Candidates: [5, 7, 1, 6]
[2025-04-16 22:40:39,487] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:270] Candidates: [3, 6, 0, 1]
[2025-04-16 22:40:39,490] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:300] Candidates: [7, 1, 6, 4]
[2025-04-16 22:40:39,491] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:330] Candidates: [1, 2, 0, 6]
[2025-04-16 22:40:39,493] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:360] Candidates: [4, 7, 1, 2]
[2025-04-16 22:40:39,496] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:390] Candidates: [7, 4, 2, 6]
[2025-04-16 22:40:39,497] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:420] Candidates: [5, 4, 3, 7]
[2025-04-16 22:40:39,500] D ThreadPoolExecutor-0_15 - get_fixed_pseudo_random_rings:200 - [P:450] Candidates: [6, 2, 4, 7]
[2025-04-16 22:40:39,661] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:0] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:39,719] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:30] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:39,728] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:60] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:39,770] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:300] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:39,877] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:120] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:40,031] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:150] Selected rings for extraction: [6, 2]
[2025-04-16 22:40:40,103] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:210] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:40,123] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:270] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:40,143] I ThreadPoolExecutor-0_15 - _extract_single_pair_task:476 - [P:450] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:40,146] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:240] Selected rings for extraction: [5, 7]
[2025-04-16 22:40:40,169] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:90] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:40,178] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:330] Selected rings for extraction: [6, 2]
[2025-04-16 22:40:40,179] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:360] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:40,181] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:180] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:40,186] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:420] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:40,196] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:390] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:40,358] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:0, R:6] s1=147.4999, s2=146.9180, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,418] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:30, R:4] s1=189.8147, s2=188.8782, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,534] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:60, R:6] s1=146.3745, s2=145.7388, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,739] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:150, R:6] s1=145.6977, s2=145.1456, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,747] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:210, R:6] s1=145.5534, s2=145.0220, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,750] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:300, R:4] s1=192.5773, s2=191.5446, ratio=1.0054 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,797] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:270, R:6] s1=144.5167, s2=145.0514, ratio=0.9963 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:40,840] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:120, R:6] s1=145.8175, s2=145.2931, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,844] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:240, R:5] s1=191.2004, s2=191.9764, ratio=0.9960 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:40,861] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:450, R:4] s1=192.5739, s2=191.6546, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,938] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:360, R:4] s1=192.4755, s2=191.5517, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,969] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:390, R:4] s1=192.4780, s2=191.5340, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:40,975] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:330, R:6] s1=146.0503, s2=145.5789, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,068] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:420, R:4] s1=192.4577, s2=191.5239, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,072] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:180, R:7] s1=77.7526, s2=77.4529, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,082] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:90, R:5] s1=194.0962, s2=193.1192, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,092] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:0, R:7] s1=77.6263, s2=78.0145, ratio=0.9950 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,098] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:1] Candidates: [0, 4, 5, 1]
[2025-04-16 22:40:41,225] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:30, R:3] s1=165.5875, s2=166.3480, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,227] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:31] Candidates: [7, 2, 0, 3]
[2025-04-16 22:40:41,328] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:60, R:3] s1=166.6587, s2=165.8605, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,331] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:61] Candidates: [7, 2, 4, 1]
[2025-04-16 22:40:41,525] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:150, R:2] s1=141.2617, s2=141.9290, ratio=0.9953 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,529] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:151] Candidates: [3, 2, 0, 1]
[2025-04-16 22:40:41,539] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:210, R:3] s1=168.2868, s2=167.3767, ratio=1.0054 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,542] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:211] Candidates: [1, 0, 2, 5]
[2025-04-16 22:40:41,558] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:300, R:6] s1=145.7169, s2=145.7654, ratio=0.9997 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,561] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:301] Candidates: [2, 6, 7, 4]
[2025-04-16 22:40:41,571] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:270, R:3] s1=167.0568, s2=166.2444, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,573] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:271] Candidates: [1, 0, 4, 7]
[2025-04-16 22:40:41,578] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:1] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:41,612] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:450, R:7] s1=78.3983, s2=78.1376, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,614] D ThreadPoolExecutor-0_15 - get_fixed_pseudo_random_rings:200 - [P:451] Candidates: [7, 5, 0, 2]
[2025-04-16 22:40:41,718] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:120, R:7] s1=77.4973, s2=77.7880, ratio=0.9963 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,722] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:121] Candidates: [3, 7, 5, 1]
[2025-04-16 22:40:41,729] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:240, R:7] s1=77.3674, s2=77.6510, ratio=0.9963 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:41,731] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:241] Candidates: [2, 7, 6, 0]
[2025-04-16 22:40:41,732] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:360, R:7] s1=78.3672, s2=78.0731, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,739] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:361] Candidates: [2, 3, 5, 6]
[2025-04-16 22:40:41,758] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:390, R:6] s1=145.9818, s2=145.9783, ratio=1.0000 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,762] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:391] Candidates: [3, 2, 0, 5]
[2025-04-16 22:40:41,791] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:31] Selected rings for extraction: [7, 3]
[2025-04-16 22:40:41,924] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:180, R:5] s1=195.7498, s2=194.8784, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,927] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:181] Candidates: [5, 2, 7, 0]
[2025-04-16 22:40:41,939] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:90, R:6] s1=146.0946, s2=145.4807, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,941] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:91] Candidates: [3, 2, 5, 6]
[2025-04-16 22:40:41,944] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:420, R:7] s1=78.5003, s2=78.2353, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,948] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:421] Candidates: [4, 6, 0, 5]
[2025-04-16 22:40:41,951] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:330, R:2] s1=142.3601, s2=141.6141, ratio=1.0053 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:41,954] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:331] Candidates: [6, 7, 2, 0]
[2025-04-16 22:40:41,981] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:61] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:42,067] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:151] Selected rings for extraction: [3, 2]
[2025-04-16 22:40:42,124] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:271] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:42,145] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:211] Selected rings for extraction: [5, 2]
[2025-04-16 22:40:42,156] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:301] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:42,243] I ThreadPoolExecutor-0_15 - _extract_single_pair_task:476 - [P:451] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:42,268] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:391] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:42,288] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:241] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:42,323] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:121] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:42,333] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:361] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:42,435] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:1, R:4] s1=189.0975, s2=188.5340, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:42,494] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:181] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:42,521] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:331] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:42,549] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:91] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:42,593] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:421] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:42,705] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:31, R:7] s1=78.1088, s2=78.2405, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:42,866] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:151, R:3] s1=166.7593, s2=167.2939, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:42,950] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:271, R:4] s1=187.9035, s2=187.1030, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:42,960] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:61, R:4] s1=191.4883, s2=190.2833, ratio=1.0063 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:42,971] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:301, R:4] s1=192.2431, s2=191.5742, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:42,982] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:211, R:5] s1=195.5548, s2=194.9510, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,045] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:451, R:7] s1=78.3493, s2=78.1954, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,109] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:121, R:7] s1=77.6361, s2=77.6458, ratio=0.9999 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:43,136] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:241, R:6] s1=145.2836, s2=145.5346, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:43,152] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:391, R:5] s1=193.2355, s2=192.6716, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,152] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:361, R:6] s1=145.9238, s2=145.5790, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,197] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:1, R:5] s1=192.3528, s2=191.6677, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,199] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:2] Candidates: [5, 1, 0, 6]
[2025-04-16 22:40:43,239] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:331, R:6] s1=145.8476, s2=145.5772, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,278] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:181, R:7] s1=77.6024, s2=77.4495, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,293] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:91, R:5] s1=193.6908, s2=193.1462, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,446] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:421, R:4] s1=192.1208, s2=191.5488, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,525] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:31, R:3] s1=165.8275, s2=166.3879, ratio=0.9966 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:43,526] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:32] Candidates: [1, 7, 5, 4]
[2025-04-16 22:40:43,574] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:2] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:43,632] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:151, R:2] s1=141.4136, s2=141.9446, ratio=0.9963 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:43,637] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:152] Candidates: [1, 5, 7, 2]
[2025-04-16 22:40:43,717] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:451, R:5] s1=193.2362, s2=192.6746, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,719] D ThreadPoolExecutor-0_15 - get_fixed_pseudo_random_rings:200 - [P:452] Candidates: [0, 5, 4, 1]
[2025-04-16 22:40:43,722] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:211, R:2] s1=142.1889, s2=141.5226, ratio=1.0047 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,724] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:212] Candidates: [3, 0, 2, 1]
[2025-04-16 22:40:43,725] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:271, R:7] s1=78.1830, s2=77.9704, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,730] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:272] Candidates: [4, 2, 1, 6]
[2025-04-16 22:40:43,743] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:301, R:6] s1=145.7708, s2=145.7729, ratio=1.0000 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:43,757] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:302] Candidates: [1, 4, 5, 3]
[2025-04-16 22:40:43,816] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:61, R:7] s1=78.0513, s2=77.8649, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,817] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:62] Candidates: [0, 6, 2, 4]
[2025-04-16 22:40:43,847] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:391, R:3] s1=167.8188, s2=167.0993, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,852] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:392] Candidates: [6, 0, 5, 2]
[2025-04-16 22:40:43,893] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:331, R:7] s1=78.2895, s2=78.0551, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,895] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:332] Candidates: [6, 1, 5, 4]
[2025-04-16 22:40:43,897] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:241, R:7] s1=77.6522, s2=77.5010, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,901] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:242] Candidates: [4, 0, 7, 1]
[2025-04-16 22:40:43,949] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:121, R:5] s1=194.5201, s2=194.1254, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,952] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:122] Candidates: [3, 2, 0, 5]
[2025-04-16 22:40:43,971] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:361, R:5] s1=193.2372, s2=192.6292, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:43,974] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:362] Candidates: [5, 1, 6, 2]
[2025-04-16 22:40:43,985] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:32] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:44,053] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:181, R:5] s1=195.4550, s2=194.9046, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:44,055] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:182] Candidates: [0, 1, 5, 7]
[2025-04-16 22:40:44,071] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:91, R:6] s1=145.7580, s2=145.4917, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:44,073] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:92] Candidates: [7, 4, 6, 2]
[2025-04-16 22:40:44,182] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:152] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:44,195] I ThreadPoolExecutor-0_15 - _extract_single_pair_task:476 - [P:452] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:44,212] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:212] Selected rings for extraction: [3, 2]
[2025-04-16 22:40:44,239] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:421, R:6] s1=145.9346, s2=145.6128, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:44,240] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:422] Candidates: [5, 1, 2, 3]
[2025-04-16 22:40:44,276] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:302] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:44,281] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:272] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:44,317] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:62] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:44,372] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:392] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:44,381] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:242] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:44,410] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:332] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:44,455] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:2, R:5] s1=191.6631, s2=192.1454, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:44,464] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:122] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:44,475] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:362] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:44,511] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:182] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:44,620] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:92] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:44,741] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:32, R:4] s1=189.3633, s2=188.8607, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:44,840] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:422] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:45,039] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:152, R:7] s1=77.3219, s2=77.5171, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:45,096] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:452, R:4] s1=192.2309, s2=191.4947, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,121] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:272, R:4] s1=186.7106, s2=187.5031, ratio=0.9958 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:45,130] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:302, R:4] s1=192.2488, s2=191.5466, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,169] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:212, R:3] s1=168.0303, s2=167.3762, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,259] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:62, R:4] s1=190.7990, s2=190.2861, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,298] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:2, R:6] s1=147.3203, s2=147.0599, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,304] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:3] Candidates: [0, 5, 7, 4]
[2025-04-16 22:40:45,335] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:332, R:4] s1=192.2619, s2=191.5177, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,391] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:242, R:4] s1=189.0506, s2=188.3591, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,457] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:392, R:6] s1=145.6743, s2=145.9656, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:45,560] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:122, R:5] s1=194.7254, s2=194.1953, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,568] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:182, R:7] s1=77.5992, s2=77.4517, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,666] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:362, R:6] s1=145.8719, s2=145.5788, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:45,946] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:92, R:4] s1=190.8568, s2=190.1507, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,199] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:32, R:5] s1=192.4949, s2=191.8338, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,206] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:33] Candidates: [5, 1, 4, 7]
[2025-04-16 22:40:46,227] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:3] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:46,491] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:422, R:5] s1=193.2478, s2=192.6682, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,722] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:302, R:5] s1=195.0897, s2=194.4500, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,725] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:303] Candidates: [2, 6, 0, 5]
[2025-04-16 22:40:46,739] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:272, R:6] s1=145.9510, s2=145.6568, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,742] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:273] Candidates: [4, 6, 3, 0]
[2025-04-16 22:40:46,774] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:152, R:5] s1=195.0800, s2=194.3584, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,777] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:153] Candidates: [6, 1, 5, 7]
[2025-04-16 22:40:46,833] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:62, R:6] s1=145.7701, s2=145.5295, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,845] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:63] Candidates: [7, 1, 2, 5]
[2025-04-16 22:40:46,850] D ThreadPoolExecutor-0_15 - extract_single_bit:349 - [P:452, R:5] s1=193.1225, s2=192.5147, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,925] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:212, R:2] s1=142.1060, s2=141.5099, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:46,927] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:213] Candidates: [4, 3, 0, 2]
[2025-04-16 22:40:47,046] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:242, R:7] s1=77.6528, s2=77.5029, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,050] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:243] Candidates: [2, 3, 4, 6]
[2025-04-16 22:40:47,126] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:392, R:5] s1=192.6362, s2=193.1096, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:47,136] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:393] Candidates: [2, 7, 6, 5]
[2025-04-16 22:40:47,199] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:332, R:6] s1=145.8524, s2=145.5884, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,201] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:333] Candidates: [5, 3, 6, 2]
[2025-04-16 22:40:47,246] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:33] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:47,290] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:182, R:5] s1=195.4530, s2=194.9046, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,294] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:183] Candidates: [1, 6, 5, 3]
[2025-04-16 22:40:47,337] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:122, R:3] s1=166.8169, s2=166.2394, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,345] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:123] Candidates: [6, 1, 3, 5]
[2025-04-16 22:40:47,397] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:362, R:5] s1=193.1051, s2=192.6308, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,400] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:363] Candidates: [3, 0, 1, 6]
[2025-04-16 22:40:47,476] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:153] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:47,486] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:273] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:47,523] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:303] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:47,589] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:63] Selected rings for extraction: [5, 7]
[2025-04-16 22:40:47,594] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:3, R:4] s1=188.9696, s2=188.5430, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,653] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:92, R:6] s1=145.7496, s2=145.5099, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,657] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:93] Candidates: [1, 5, 6, 3]
[2025-04-16 22:40:47,746] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:213] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:47,806] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:422, R:3] s1=167.7990, s2=167.0787, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:47,809] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:423] Candidates: [2, 7, 5, 0]
[2025-04-16 22:40:47,825] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:393] Selected rings for extraction: [7, 6]
[2025-04-16 22:40:47,830] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:243] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:47,888] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:333] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:47,917] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:183] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:47,994] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:363] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:48,026] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:123] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:48,256] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:93] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:48,266] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:33, R:4] s1=189.3530, s2=188.8617, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,378] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:423] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:48,438] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:273, R:4] s1=186.5862, s2=185.8150, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,453] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:153, R:6] s1=145.0579, s2=145.3098, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:48,489] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:303, R:6] s1=145.7618, s2=145.3958, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,585] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:63, R:5] s1=193.8120, s2=193.1384, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,591] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:3, R:5] s1=192.1711, s2=191.6873, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,594] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:4] Candidates: [2, 3, 5, 1]
[2025-04-16 22:40:48,646] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:213, R:4] s1=193.4039, s2=192.5590, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,682] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:243, R:4] s1=188.9160, s2=188.3630, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,777] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:183, R:6] s1=145.3311, s2=145.0257, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,785] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:393, R:7] s1=78.2298, s2=78.0494, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,793] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:333, R:6] s1=145.8821, s2=145.5687, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:48,918] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:123, R:6] s1=145.2103, s2=145.4240, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:48,930] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:363, R:6] s1=145.8553, s2=145.6121, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,061] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:33, R:5] s1=192.3984, s2=191.8380, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,063] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:34] Candidates: [7, 1, 5, 3]
[2025-04-16 22:40:49,113] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:93, R:5] s1=193.6786, s2=193.1553, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,130] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:423, R:7] s1=78.3433, s2=78.1442, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,135] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:4] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:49,262] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:273, R:3] s1=164.4869, s2=164.9493, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:49,264] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:274] Candidates: [3, 7, 2, 0]
[2025-04-16 22:40:49,277] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:153, R:7] s1=77.3057, s2=77.5094, ratio=0.9974 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:49,280] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:154] Candidates: [2, 6, 1, 4]
[2025-04-16 22:40:49,285] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:303, R:5] s1=194.8964, s2=194.2125, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,286] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:304] Candidates: [2, 5, 3, 6]
[2025-04-16 22:40:49,414] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:213, R:3] s1=168.0127, s2=167.3130, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,415] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:214] Candidates: [1, 0, 3, 4]
[2025-04-16 22:40:49,466] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:393, R:6] s1=145.9563, s2=145.6920, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,466] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:63, R:7] s1=77.9979, s2=77.8636, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,482] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:394] Candidates: [1, 5, 7, 3]
[2025-04-16 22:40:49,489] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:243, R:3] s1=165.7728, s2=166.2049, ratio=0.9974 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:49,491] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:64] Candidates: [1, 6, 0, 5]
[2025-04-16 22:40:49,493] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:244] Candidates: [3, 6, 5, 1]
[2025-04-16 22:40:49,596] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:34] Selected rings for extraction: [5, 7]
[2025-04-16 22:40:49,626] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:183, R:5] s1=195.4529, s2=194.8641, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,628] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:184] Candidates: [0, 2, 1, 4]
[2025-04-16 22:40:49,629] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:333, R:5] s1=193.2155, s2=192.5946, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,630] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:334] Candidates: [7, 5, 2, 4]
[2025-04-16 22:40:49,708] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:154] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:49,768] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:274] Selected rings for extraction: [3, 7]
[2025-04-16 22:40:49,799] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:123, R:5] s1=194.1645, s2=194.7162, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:49,801] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:124] Candidates: [7, 0, 1, 4]
[2025-04-16 22:40:49,804] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:363, R:3] s1=167.8115, s2=167.1012, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,807] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:364] Candidates: [5, 2, 1, 4]
[2025-04-16 22:40:49,815] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:304] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:49,816] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:214] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:49,973] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:93, R:6] s1=145.7499, s2=145.4862, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,975] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:94] Candidates: [5, 0, 1, 3]
[2025-04-16 22:40:49,977] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:4, R:5] s1=191.6924, s2=192.1461, ratio=0.9976 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:49,982] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:423, R:5] s1=193.1022, s2=192.6743, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:49,985] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:424] Candidates: [7, 6, 0, 5]
[2025-04-16 22:40:50,066] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:394] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:50,067] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:64] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:50,088] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:244] Selected rings for extraction: [3, 5]
[2025-04-16 22:40:50,120] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:184] Selected rings for extraction: [4, 2]
[2025-04-16 22:40:50,124] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:334] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:50,361] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:124] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:50,406] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:364] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:50,432] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:34, R:5] s1=192.3861, s2=191.8614, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,450] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:424] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:50,455] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:94] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:50,613] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:154, R:4] s1=191.9103, s2=192.5904, ratio=0.9965 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:50,636] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:304, R:6] s1=145.6822, s2=145.4228, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,678] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:274, R:3] s1=164.8225, s2=165.1533, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:50,738] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:214, R:4] s1=193.2852, s2=192.5588, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,783] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:4, R:3] s1=166.2665, s2=165.6274, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,787] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:5] Candidates: [5, 2, 3, 1]
[2025-04-16 22:40:50,841] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:64, R:5] s1=193.6610, s2=193.1113, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,892] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:394, R:7] s1=78.0502, s2=78.1933, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:50,934] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:334, R:4] s1=192.0852, s2=191.4903, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,958] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:184, R:4] s1=193.1604, s2=192.3333, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:50,969] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:244, R:3] s1=165.7785, s2=166.0264, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,091] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:124, R:4] s1=191.7233, s2=192.3025, ratio=0.9970 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,161] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:34, R:7] s1=78.2430, s2=78.1157, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,163] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:35] Candidates: [2, 3, 7, 4]
[2025-04-16 22:40:51,166] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:364, R:4] s1=192.1279, s2=191.5021, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,282] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:424, R:6] s1=145.8636, s2=145.5244, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,327] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:94, R:5] s1=193.6755, s2=193.1770, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,401] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:154, R:6] s1=145.0723, s2=145.3062, ratio=0.9984 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,403] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:155] Candidates: [7, 0, 5, 2]
[2025-04-16 22:40:51,404] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:5] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:51,409] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:274, R:7] s1=77.9702, s2=78.1189, ratio=0.9981 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,411] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:275] Candidates: [5, 3, 1, 2]
[2025-04-16 22:40:51,530] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:304, R:5] s1=194.7900, s2=194.2139, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,542] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:305] Candidates: [3, 0, 2, 4]
[2025-04-16 22:40:51,546] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:214, R:3] s1=167.9978, s2=167.3180, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,548] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:215] Candidates: [0, 4, 1, 2]
[2025-04-16 22:40:51,582] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:64, R:6] s1=145.7306, s2=145.4935, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,585] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:65] Candidates: [6, 5, 7, 2]
[2025-04-16 22:40:51,671] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:334, R:7] s1=78.2467, s2=78.0914, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,673] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:394, R:5] s1=193.0761, s2=192.6201, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,674] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:335] Candidates: [5, 2, 3, 1]
[2025-04-16 22:40:51,678] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:395] Candidates: [1, 6, 7, 3]
[2025-04-16 22:40:51,705] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:35] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:51,825] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:244, R:5] s1=191.9889, s2=192.5096, ratio=0.9973 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,828] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:245] Candidates: [1, 7, 3, 4]
[2025-04-16 22:40:51,840] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:184, R:2] s1=141.9889, s2=141.3540, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,842] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:185] Candidates: [3, 0, 2, 6]
[2025-04-16 22:40:51,875] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:124, R:7] s1=77.4217, s2=77.5736, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:51,877] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:125] Candidates: [6, 5, 0, 2]
[2025-04-16 22:40:51,885] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:155] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:51,934] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:275] Selected rings for extraction: [3, 5]
[2025-04-16 22:40:51,989] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:364, R:5] s1=193.0987, s2=192.6049, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:51,991] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:365] Candidates: [7, 5, 4, 0]
[2025-04-16 22:40:52,006] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:424, R:7] s1=78.3071, s2=78.0615, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:52,009] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:425] Candidates: [7, 0, 4, 5]
[2025-04-16 22:40:52,034] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:215] Selected rings for extraction: [4, 2]
[2025-04-16 22:40:52,063] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:305] Selected rings for extraction: [4, 3]
[2025-04-16 22:40:52,139] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:94, R:3] s1=166.0353, s2=165.4063, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:52,142] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:95] Candidates: [4, 5, 1, 0]
[2025-04-16 22:40:52,204] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:5, R:5] s1=191.7022, s2=192.1478, ratio=0.9977 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:52,207] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:65] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:52,255] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:395] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:52,276] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:335] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:52,330] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:245] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:52,368] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:185] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:52,440] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:125] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:52,581] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:365] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:52,586] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:425] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:52,610] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:35, R:4] s1=189.3416, s2=188.8838, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:52,640] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:95] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:52,806] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:155, R:7] s1=77.3353, s2=77.5084, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:52,836] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:275, R:3] s1=164.3911, s2=165.6705, ratio=0.9923 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:52,935] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:305, R:4] s1=192.1715, s2=191.4305, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:52,945] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:215, R:4] s1=193.2762, s2=192.5835, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,087] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:395, R:6] s1=145.6568, s2=145.9172, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:53,126] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:335, R:5] s1=193.0523, s2=192.6070, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,173] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:185, R:6] s1=145.2003, s2=144.9732, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,177] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:5, R:3] s1=166.1476, s2=165.6297, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,191] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:6] Candidates: [5, 3, 2, 6]
[2025-04-16 22:40:53,201] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:65, R:5] s1=193.6564, s2=193.1131, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,243] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:245, R:4] s1=189.6656, s2=190.3835, ratio=0.9962 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:53,279] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:125, R:6] s1=145.2480, s2=145.4014, ratio=0.9989 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:53,339] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:425, R:4] s1=192.1005, s2=191.5604, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,368] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:365, R:4] s1=192.1037, s2=191.5254, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,444] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:35, R:7] s1=78.2470, s2=78.1197, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,446] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:36] Candidates: [5, 2, 1, 6]
[2025-04-16 22:40:53,545] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:95, R:4] s1=190.7310, s2=190.1052, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,686] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:155, R:5] s1=194.9453, s2=194.3740, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,690] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:156] Candidates: [2, 4, 7, 1]
[2025-04-16 22:40:53,746] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:275, R:5] s1=190.6075, s2=187.6263, ratio=1.0159 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,749] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:276] Candidates: [7, 5, 3, 4]
[2025-04-16 22:40:53,862] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:6] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:53,899] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:215, R:2] s1=142.0997, s2=141.5180, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,901] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:216] Candidates: [2, 6, 0, 3]
[2025-04-16 22:40:53,904] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:305, R:3] s1=167.1879, s2=166.3589, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:53,908] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:306] Candidates: [4, 6, 1, 2]
[2025-04-16 22:40:53,958] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:395, R:7] s1=78.0286, s2=78.1684, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:53,960] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:396] Candidates: [2, 6, 5, 4]
[2025-04-16 22:40:54,022] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:335, R:3] s1=167.8109, s2=167.0981, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,026] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:336] Candidates: [6, 0, 4, 3]
[2025-04-16 22:40:54,044] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:65, R:6] s1=145.7132, s2=145.4933, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,046] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:66] Candidates: [2, 3, 7, 0]
[2025-04-16 22:40:54,058] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:36] Selected rings for extraction: [5, 6]
[2025-04-16 22:40:54,174] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:185, R:3] s1=167.6309, s2=167.0071, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,177] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:186] Candidates: [4, 7, 3, 1]
[2025-04-16 22:40:54,223] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:245, R:7] s1=77.6443, s2=77.5105, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,227] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:246] Candidates: [3, 2, 4, 7]
[2025-04-16 22:40:54,255] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:125, R:5] s1=194.7064, s2=194.1884, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,260] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:126] Candidates: [1, 3, 7, 2]
[2025-04-16 22:40:54,278] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:425, R:7] s1=78.2422, s2=78.0748, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,280] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:426] Candidates: [7, 2, 1, 6]
[2025-04-16 22:40:54,301] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:365, R:7] s1=78.3309, s2=78.1889, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,305] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:366] Candidates: [5, 1, 0, 3]
[2025-04-16 22:40:54,324] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:276] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:54,363] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:156] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:54,551] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:95, R:5] s1=193.6830, s2=193.1362, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:54,553] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:96] Candidates: [4, 2, 0, 5]
[2025-04-16 22:40:54,558] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:306] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:54,602] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:216] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:54,629] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:396] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:54,689] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:66] Selected rings for extraction: [7, 3]
[2025-04-16 22:40:54,692] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:336] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:54,779] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:186] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:54,813] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:246] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:54,917] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:126] Selected rings for extraction: [7, 3]
[2025-04-16 22:40:54,920] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:426] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:54,964] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:6, R:5] s1=192.4992, s2=191.7119, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,002] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:366] Selected rings for extraction: [5, 3]
[2025-04-16 22:40:55,114] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:96] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:55,189] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:36, R:5] s1=192.8155, s2=191.8656, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,301] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:276, R:4] s1=187.1715, s2=185.9635, ratio=1.0065 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,331] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:156, R:4] s1=193.0786, s2=191.8853, ratio=1.0062 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,486] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:306, R:4] s1=192.5582, s2=191.4749, ratio=1.0057 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,611] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:336, R:4] s1=192.4687, s2=191.5386, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,615] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:396, R:4] s1=191.1807, s2=191.9980, ratio=0.9957 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:55,644] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:216, R:6] s1=145.5465, s2=145.0357, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,699] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:66, R:7] s1=78.1992, s2=77.8991, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,747] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:246, R:4] s1=189.0174, s2=189.9277, ratio=0.9952 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:55,820] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:186, R:4] s1=193.4932, s2=192.3451, ratio=1.0060 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,842] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:6, R:6] s1=146.6745, s2=147.2168, ratio=0.9963 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:55,843] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:7] Candidates: [4, 3, 6, 2]
[2025-04-16 22:40:55,868] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:366, R:5] s1=193.4357, s2=192.7003, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,901] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:126, R:7] s1=77.7598, s2=77.4297, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:55,927] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:426, R:6] s1=146.0232, s2=145.5274, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,046] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:36, R:6] s1=148.1741, s2=147.3993, ratio=1.0053 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,048] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:37] Candidates: [4, 2, 5, 0]
[2025-04-16 22:40:56,054] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:96, R:4] s1=191.2666, s2=190.0759, ratio=1.0063 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,248] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:276, R:5] s1=188.2035, s2=187.2372, ratio=1.0052 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,250] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:277] Candidates: [0, 2, 6, 4]
[2025-04-16 22:40:56,256] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:156, R:7] s1=77.1660, s2=77.5114, ratio=0.9955 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:56,258] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:157] Candidates: [4, 5, 1, 2]
[2025-04-16 22:40:56,359] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:7] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:56,360] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:306, R:6] s1=146.0379, s2=145.4796, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,362] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:307] Candidates: [6, 1, 0, 2]
[2025-04-16 22:40:56,462] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:336, R:6] s1=146.0957, s2=145.6083, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,478] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:337] Candidates: [6, 5, 3, 0]
[2025-04-16 22:40:56,531] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:396, R:6] s1=146.1504, s2=145.6620, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,535] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:397] Candidates: [5, 4, 7, 1]
[2025-04-16 22:40:56,576] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:216, R:3] s1=168.2892, s2=167.3854, ratio=1.0054 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,579] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:217] Candidates: [4, 5, 1, 7]
[2025-04-16 22:40:56,594] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:246, R:7] s1=77.6505, s2=77.6505, ratio=1.0000 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:56,596] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:247] Candidates: [1, 4, 3, 6]
[2025-04-16 22:40:56,624] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:66, R:3] s1=166.2637, s2=165.5121, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,631] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:67] Candidates: [1, 2, 3, 6]
[2025-04-16 22:40:56,665] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:37] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:56,712] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:186, R:7] s1=77.7145, s2=77.3337, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,715] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:187] Candidates: [1, 5, 4, 0]
[2025-04-16 22:40:56,751] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:366, R:3] s1=167.9898, s2=167.1473, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,754] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:367] Candidates: [3, 4, 7, 5]
[2025-04-16 22:40:56,828] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:126, R:3] s1=165.8911, s2=166.6705, ratio=0.9953 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:56,832] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:127] Candidates: [1, 6, 5, 4]
[2025-04-16 22:40:56,839] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:157] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:56,849] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:277] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:56,862] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:426, R:7] s1=78.3676, s2=78.0589, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:56,864] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:427] Candidates: [1, 0, 6, 4]
[2025-04-16 22:40:56,880] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:307] Selected rings for extraction: [6, 2]
[2025-04-16 22:40:57,013] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:96, R:5] s1=194.1030, s2=193.1147, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:57,015] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:97] Candidates: [4, 5, 7, 1]
[2025-04-16 22:40:57,117] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:397] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:57,123] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:247] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:57,138] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:337] Selected rings for extraction: [6, 5]
[2025-04-16 22:40:57,158] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:217] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:57,234] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:67] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:57,251] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:7, R:4] s1=189.0701, s2=188.5306, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:57,353] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:187] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:57,412] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:367] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:57,427] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:127] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:57,450] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:427] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:57,581] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:37, R:4] s1=189.4787, s2=188.8263, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:57,727] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:97] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:57,860] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:157, R:4] s1=192.6096, s2=191.8587, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:57,866] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:277, R:4] s1=185.8755, s2=186.9292, ratio=0.9944 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:57,901] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:307, R:6] s1=145.7539, s2=145.6909, ratio=1.0004 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,071] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:247, R:4] s1=189.3093, s2=189.8942, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:58,088] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:397, R:4] s1=191.4028, s2=191.9839, ratio=0.9970 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:58,156] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:67, R:6] s1=145.7743, s2=145.5045, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,194] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:7, R:6] s1=147.2402, s2=146.9104, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,197] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:8] Candidates: [5, 4, 3, 1]
[2025-04-16 22:40:58,244] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:217, R:4] s1=193.4065, s2=192.6150, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,270] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:187, R:4] s1=193.0719, s2=192.3324, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,341] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:337, R:6] s1=145.8734, s2=145.5119, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,364] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:367, R:4] s1=192.2496, s2=191.5100, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,371] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:127, R:4] s1=191.6885, s2=192.3414, ratio=0.9966 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:58,471] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:427, R:4] s1=192.2206, s2=191.4903, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,493] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:37, R:5] s1=192.4321, s2=191.8791, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,496] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:38] Candidates: [4, 3, 6, 0]
[2025-04-16 22:40:58,610] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:97, R:4] s1=190.7581, s2=190.1109, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,760] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:307, R:2] s1=141.7468, s2=141.5974, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,763] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:308] Candidates: [6, 3, 0, 1]
[2025-04-16 22:40:58,812] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:277, R:6] s1=145.4944, s2=145.3284, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,814] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:278] Candidates: [4, 2, 0, 1]
[2025-04-16 22:40:58,823] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:157, R:5] s1=195.0633, s2=194.3060, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,826] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:158] Candidates: [0, 7, 5, 3]
[2025-04-16 22:40:58,863] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:8] Selected rings for extraction: [4, 5]
[2025-04-16 22:40:58,980] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:247, R:6] s1=145.0343, s2=144.7152, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:58,981] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:248] Candidates: [1, 6, 7, 2]
[2025-04-16 22:40:59,073] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:38] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:59,129] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:217, R:7] s1=77.6169, s2=77.4306, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,132] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:218] Candidates: [3, 7, 0, 5]
[2025-04-16 22:40:59,134] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:67, R:3] s1=166.0463, s2=165.5327, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,136] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:68] Candidates: [3, 2, 6, 4]
[2025-04-16 22:40:59,218] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:397, R:7] s1=78.0629, s2=78.2232, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:59,225] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:187, R:5] s1=195.4568, s2=194.7109, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,227] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:398] Candidates: [6, 7, 5, 1]
[2025-04-16 22:40:59,230] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:188] Candidates: [5, 6, 4, 1]
[2025-04-16 22:40:59,301] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:337, R:5] s1=193.1921, s2=192.5746, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,317] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:338] Candidates: [1, 2, 7, 4]
[2025-04-16 22:40:59,372] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:308] Selected rings for extraction: [6, 3]
[2025-04-16 22:40:59,382] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:127, R:6] s1=145.4896, s2=145.2141, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,384] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:128] Candidates: [3, 6, 7, 5]
[2025-04-16 22:40:59,391] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:278] Selected rings for extraction: [4, 2]
[2025-04-16 22:40:59,400] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:367, R:7] s1=78.1494, s2=78.3203, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:59,405] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:368] Candidates: [1, 6, 2, 3]
[2025-04-16 22:40:59,471] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:158] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:59,487] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:427, R:6] s1=145.8375, s2=145.5745, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,489] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:428] Candidates: [0, 4, 5, 1]
[2025-04-16 22:40:59,498] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:248] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:59,649] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:97, R:5] s1=193.6942, s2=193.1344, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:40:59,653] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:98] Candidates: [5, 1, 2, 6]
[2025-04-16 22:40:59,736] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:218] Selected rings for extraction: [7, 5]
[2025-04-16 22:40:59,738] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:68] Selected rings for extraction: [4, 6]
[2025-04-16 22:40:59,739] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:8, R:4] s1=188.5326, s2=188.9606, ratio=0.9977 vs thr=1.0 -> Bit=1
[2025-04-16 22:40:59,852] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:398] Selected rings for extraction: [6, 7]
[2025-04-16 22:40:59,878] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:338] Selected rings for extraction: [4, 7]
[2025-04-16 22:40:59,879] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:188] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:00,030] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:128] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:00,032] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:368] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:00,071] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:38, R:4] s1=189.3327, s2=188.8601, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:00,074] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:428] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:00,289] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:98] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:00,345] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:308, R:6] s1=145.6758, s2=145.2901, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:00,350] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:278, R:4] s1=185.5815, s2=186.4780, ratio=0.9952 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:00,586] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:158, R:7] s1=77.3219, s2=77.5232, ratio=0.9974 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:00,633] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:248, R:6] s1=144.6973, s2=144.9555, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:00,907] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:68, R:4] s1=190.9256, s2=190.2989, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:00,944] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:398, R:6] s1=145.9753, s2=145.6961, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,132] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:218, R:7] s1=77.5856, s2=77.4308, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,325] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:188, R:4] s1=193.0344, s2=192.3667, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,384] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:8, R:5] s1=192.1752, s2=191.7074, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,410] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:9] Candidates: [2, 7, 5, 4]
[2025-04-16 22:41:01,486] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:38, R:6] s1=147.8316, s2=147.4635, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,490] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:39] Candidates: [7, 5, 4, 6]
[2025-04-16 22:41:01,496] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:128, R:6] s1=145.2136, s2=145.4093, ratio=0.9987 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:01,504] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:338, R:4] s1=192.1277, s2=191.5658, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,556] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:368, R:6] s1=145.6812, s2=145.9239, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:01,670] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:428, R:4] s1=192.0867, s2=191.4779, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,717] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:308, R:3] s1=168.4385, s2=167.4561, ratio=1.0059 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:01,720] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:309] Candidates: [2, 7, 3, 4]
[2025-04-16 22:41:01,829] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:278, R:2] s1=141.4974, s2=142.1081, ratio=0.9957 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:01,843] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:279] Candidates: [1, 2, 4, 6]
[2025-04-16 22:41:02,013] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:98, R:5] s1=193.6727, s2=193.1312, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,014] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:158, R:5] s1=194.3060, s2=194.9356, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:02,029] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:159] Candidates: [3, 2, 5, 6]
[2025-04-16 22:41:02,226] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:248, R:7] s1=77.4992, s2=77.6758, ratio=0.9977 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:02,227] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:249] Candidates: [7, 6, 2, 4]
[2025-04-16 22:41:02,248] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:9] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:02,270] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:398, R:7] s1=78.2430, s2=78.0439, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,271] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:68, R:6] s1=145.7074, s2=145.5045, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,273] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:399] Candidates: [1, 2, 4, 7]
[2025-04-16 22:41:02,274] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:69] Candidates: [2, 1, 5, 7]
[2025-04-16 22:41:02,297] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:39] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:02,327] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:309] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:02,385] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:188, R:6] s1=145.2844, s2=144.9625, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,387] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:189] Candidates: [5, 2, 7, 1]
[2025-04-16 22:41:02,439] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:218, R:5] s1=195.5378, s2=194.9348, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,441] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:219] Candidates: [2, 1, 0, 3]
[2025-04-16 22:41:02,458] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:279] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:02,568] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:338, R:7] s1=78.3510, s2=78.1546, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,570] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:339] Candidates: [2, 3, 0, 6]
[2025-04-16 22:41:02,606] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:428, R:5] s1=193.2791, s2=192.6566, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,607] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:128, R:7] s1=77.5839, s2=77.4320, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,611] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:429] Candidates: [7, 5, 4, 3]
[2025-04-16 22:41:02,617] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:129] Candidates: [0, 5, 6, 4]
[2025-04-16 22:41:02,631] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:368, R:3] s1=167.7362, s2=167.1448, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,634] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:369] Candidates: [3, 0, 1, 2]
[2025-04-16 22:41:02,705] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:159] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:02,734] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:249] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:02,736] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:69] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:02,767] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:399] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:02,833] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:189] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:02,883] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:9, R:4] s1=189.0557, s2=188.5491, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:02,957] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:219] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:03,021] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:98, R:6] s1=145.8011, s2=145.4707, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,022] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:99] Candidates: [4, 3, 2, 7]
[2025-04-16 22:41:03,129] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:39, R:4] s1=189.3363, s2=188.8362, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,147] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:129] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:03,147] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:369] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:03,150] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:429] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:03,159] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:339] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:03,187] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:309, R:4] s1=192.6946, s2=191.9870, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,288] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:279, R:4] s1=183.8424, s2=184.2198, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:03,415] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:159, R:6] s1=145.0787, s2=145.2880, ratio=0.9986 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:03,438] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:249, R:4] s1=189.8643, s2=189.3104, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,465] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:69, R:5] s1=193.6258, s2=192.9200, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,579] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:99] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:03,591] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:399, R:4] s1=191.9724, s2=191.3860, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,683] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:189, R:7] s1=77.5399, s2=77.3381, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,719] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:9, R:5] s1=191.5250, s2=192.1078, ratio=0.9970 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:03,721] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:10] Candidates: [1, 3, 7, 4]
[2025-04-16 22:41:03,753] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:219, R:3] s1=168.0464, s2=167.3656, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:03,975] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:279, R:6] s1=145.8046, s2=146.0984, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:03,977] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:280] Candidates: [2, 3, 1, 0]
[2025-04-16 22:41:03,984] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:129, R:4] s1=192.3322, s2=191.6847, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,016] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:339, R:6] s1=145.7848, s2=145.5583, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,026] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:39, R:5] s1=192.4194, s2=191.8853, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,028] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:40] Candidates: [4, 6, 3, 0]
[2025-04-16 22:41:04,035] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:429, R:4] s1=192.0923, s2=191.5057, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,065] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:309, R:7] s1=78.2104, s2=77.9629, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,068] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:310] Candidates: [3, 7, 2, 0]
[2025-04-16 22:41:04,146] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:369, R:3] s1=167.1448, s2=167.7015, ratio=0.9967 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:04,203] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:10] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:04,203] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:249, R:6] s1=144.9555, s2=144.7255, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,205] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:250] Candidates: [5, 1, 2, 3]
[2025-04-16 22:41:04,218] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:159, R:5] s1=194.9276, s2=194.3435, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,222] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:160] Candidates: [2, 7, 1, 6]
[2025-04-16 22:41:04,225] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:69, R:7] s1=78.0187, s2=77.8345, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,227] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:70] Candidates: [1, 3, 4, 2]
[2025-04-16 22:41:04,280] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:219, R:2] s1=142.1996, s2=141.5089, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,288] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:220] Candidates: [5, 6, 0, 1]
[2025-04-16 22:41:04,296] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:280] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:04,359] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:40] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:04,437] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:399, R:7] s1=78.0457, s2=78.2223, ratio=0.9977 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:04,439] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:400] Candidates: [6, 0, 4, 1]
[2025-04-16 22:41:04,472] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:99, R:4] s1=190.7189, s2=190.1373, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,546] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:310] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:04,610] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:189, R:5] s1=195.3171, s2=194.7492, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,612] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:190] Candidates: [0, 7, 2, 4]
[2025-04-16 22:41:04,736] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:129, R:6] s1=145.2146, s2=145.4174, ratio=0.9986 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:04,737] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:130] Candidates: [4, 7, 6, 0]
[2025-04-16 22:41:04,749] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:220] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:04,763] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:160] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:04,764] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:250] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:04,768] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:70] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:04,887] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:400] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:04,947] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:429, R:7] s1=78.2414, s2=78.0773, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,950] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:430] Candidates: [4, 5, 2, 6]
[2025-04-16 22:41:04,955] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:339, R:3] s1=167.8083, s2=167.0868, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:04,957] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:340] Candidates: [0, 5, 6, 2]
[2025-04-16 22:41:05,004] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:369, R:2] s1=142.2255, s2=141.5788, ratio=1.0046 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,006] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:370] Candidates: [0, 2, 1, 5]
[2025-04-16 22:41:05,030] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:190] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:05,039] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:280, R:3] s1=164.7218, s2=163.7392, ratio=1.0060 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,205] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:10, R:4] s1=188.4569, s2=188.9975, ratio=0.9971 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:05,212] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:130] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:05,327] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:40, R:4] s1=189.3313, s2=188.8616, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,350] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:99, R:7] s1=78.0445, s2=77.8771, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,352] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:100] Candidates: [4, 2, 0, 7]
[2025-04-16 22:41:05,406] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:310, R:7] s1=78.1973, s2=77.9988, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,479] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:220, R:6] s1=145.2325, s2=144.9936, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,484] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:430] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:05,508] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:370] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:05,518] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:340] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:05,639] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:250, R:5] s1=192.2974, s2=191.6957, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,669] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:160, R:6] s1=145.3056, s2=145.0590, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,683] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:70, R:4] s1=190.7222, s2=190.0855, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,709] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:400, R:4] s1=191.9586, s2=191.3895, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,823] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:100] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:05,836] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:280, R:2] s1=141.5042, s2=142.0050, ratio=0.9965 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:05,837] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:281] Candidates: [4, 1, 3, 6]
[2025-04-16 22:41:05,912] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:190, R:4] s1=193.0288, s2=192.3786, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:05,998] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:10, R:7] s1=77.9151, s2=77.6728, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,000] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:11] Candidates: [4, 2, 5, 7]
[2025-04-16 22:41:06,045] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:130, R:4] s1=192.3458, s2=191.6877, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,109] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:40, R:6] s1=147.8373, s2=147.4641, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,110] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:41] Candidates: [6, 2, 7, 4]
[2025-04-16 22:41:06,174] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:310, R:3] s1=167.9124, s2=167.3490, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,175] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:311] Candidates: [1, 2, 5, 7]
[2025-04-16 22:41:06,220] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:220, R:5] s1=195.4569, s2=194.9208, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,222] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:221] Candidates: [1, 6, 2, 7]
[2025-04-16 22:41:06,280] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:340, R:6] s1=145.8465, s2=145.5840, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,315] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:281] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:06,318] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:430, R:4] s1=192.0847, s2=191.4918, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,348] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:370, R:5] s1=193.1258, s2=192.6823, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,404] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:250, R:3] s1=165.4201, s2=165.9580, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:06,406] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:251] Candidates: [6, 4, 1, 3]
[2025-04-16 22:41:06,428] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:400, R:6] s1=145.9435, s2=145.6570, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,430] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:401] Candidates: [3, 0, 7, 2]
[2025-04-16 22:41:06,535] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:160, R:7] s1=77.5252, s2=77.2869, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,538] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:161] Candidates: [4, 2, 1, 7]
[2025-04-16 22:41:06,552] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:70, R:3] s1=165.9588, s2=165.3681, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,555] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:71] Candidates: [3, 2, 5, 0]
[2025-04-16 22:41:06,565] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:11] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:06,635] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:41] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:06,673] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:100, R:4] s1=190.7370, s2=190.1427, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,723] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:190, R:7] s1=77.5414, s2=77.3457, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,724] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:191] Candidates: [4, 7, 5, 2]
[2025-04-16 22:41:06,735] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:311] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:06,754] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:221] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:06,881] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:130, R:6] s1=145.4174, s2=145.2159, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:06,882] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:131] Candidates: [1, 5, 4, 6]
[2025-04-16 22:41:06,893] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:401] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:06,925] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:251] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:06,938] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:161] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:06,986] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:71] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:07,112] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:281, R:4] s1=183.5411, s2=185.9002, ratio=0.9873 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:07,175] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:340, R:5] s1=193.1204, s2=192.6365, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,184] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:341] Candidates: [1, 7, 3, 0]
[2025-04-16 22:41:07,245] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:430, R:6] s1=145.8490, s2=145.5979, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,247] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:431] Candidates: [7, 2, 3, 1]
[2025-04-16 22:41:07,320] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:370, R:2] s1=142.1988, s2=141.5780, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,322] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:371] Candidates: [2, 3, 1, 0]
[2025-04-16 22:41:07,328] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:191] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:07,405] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:11, R:4] s1=189.1749, s2=188.7293, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,406] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:131] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:07,477] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:100, R:7] s1=78.0097, s2=77.8771, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,479] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:101] Candidates: [5, 3, 6, 7]
[2025-04-16 22:41:07,571] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:41, R:4] s1=189.3314, s2=188.8616, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,604] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:221, R:6] s1=145.1998, s2=144.9684, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,623] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:401, R:7] s1=78.0568, s2=78.2246, ratio=0.9979 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:07,640] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:251, R:4] s1=189.3043, s2=189.8386, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:07,670] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:311, R:5] s1=193.4165, s2=192.8026, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,747] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:341] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:07,763] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:431] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:07,774] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:161, R:4] s1=192.5670, s2=191.9063, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:07,823] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:371] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:07,895] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:281, R:6] s1=145.1691, s2=145.4682, ratio=0.9979 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:07,897] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:282] Candidates: [7, 2, 1, 5]
[2025-04-16 22:41:07,940] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:101] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:07,963] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:71, R:5] s1=193.5160, s2=192.9808, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,135] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:191, R:4] s1=193.0403, s2=192.3814, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,221] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:11, R:5] s1=191.5079, s2=192.1057, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:08,223] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:12] Candidates: [3, 0, 2, 7]
[2025-04-16 22:41:08,249] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:282] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:08,265] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:131, R:4] s1=192.3493, s2=191.6877, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,366] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:41, R:7] s1=78.2790, s2=78.1363, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,371] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:42] Candidates: [2, 3, 7, 6]
[2025-04-16 22:41:08,381] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:401, R:3] s1=167.0686, s2=167.6697, ratio=0.9964 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:08,393] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:402] Candidates: [4, 7, 6, 2]
[2025-04-16 22:41:08,416] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:221, R:7] s1=77.5909, s2=77.4071, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,423] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:222] Candidates: [4, 0, 7, 6]
[2025-04-16 22:41:08,432] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:431, R:7] s1=78.3102, s2=78.1669, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,472] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:311, R:7] s1=78.2152, s2=78.0014, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,473] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:312] Candidates: [6, 1, 7, 2]
[2025-04-16 22:41:08,474] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:251, R:6] s1=144.7181, s2=144.9334, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:08,478] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:252] Candidates: [2, 3, 1, 4]
[2025-04-16 22:41:08,554] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:341, R:7] s1=78.3438, s2=78.1951, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,592] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:161, R:7] s1=77.5114, s2=77.3100, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,594] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:162] Candidates: [1, 7, 0, 4]
[2025-04-16 22:41:08,651] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:371, R:3] s1=167.1158, s2=167.7215, ratio=0.9964 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:08,695] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:12] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:08,847] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:101, R:5] s1=193.6636, s2=193.1362, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,937] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:42] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:08,962] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:191, R:7] s1=77.5428, s2=77.3472, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,963] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:192] Candidates: [4, 2, 6, 5]
[2025-04-16 22:41:08,985] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:71, R:3] s1=165.8982, s2=165.4031, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:08,988] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:72] Candidates: [5, 3, 4, 0]
[2025-04-16 22:41:08,989] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:402] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:09,009] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:222] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:09,036] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:312] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:09,043] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:252] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:09,047] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:162] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:09,163] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:131, R:6] s1=145.2155, s2=145.4194, ratio=0.9986 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:09,167] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:132] Candidates: [6, 2, 3, 0]
[2025-04-16 22:41:09,197] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:282, R:5] s1=188.7671, s2=189.6464, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:09,298] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:431, R:3] s1=167.8075, s2=167.0771, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,301] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:432] Candidates: [2, 0, 1, 5]
[2025-04-16 22:41:09,353] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:341, R:3] s1=167.6746, s2=167.0863, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,381] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:342] Candidates: [3, 4, 5, 7]
[2025-04-16 22:41:09,541] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:371, R:2] s1=141.5621, s2=142.1875, ratio=0.9956 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:09,543] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:372] Candidates: [1, 0, 2, 5]
[2025-04-16 22:41:09,564] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:192] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:09,572] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:72] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:09,574] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:12, R:7] s1=78.1149, s2=77.7300, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,686] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:101, R:6] s1=145.7219, s2=145.4710, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,688] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:102] Candidates: [7, 5, 3, 4]
[2025-04-16 22:41:09,706] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:132] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:09,739] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:432] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:09,803] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:402, R:4] s1=192.4532, s2=191.4875, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,879] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:162, R:4] s1=193.0895, s2=191.8893, ratio=1.0063 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,882] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:222, R:4] s1=193.5994, s2=192.1775, ratio=1.0074 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,925] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:42, R:7] s1=78.4734, s2=78.1080, ratio=1.0047 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,925] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:312, R:6] s1=146.0226, s2=145.4404, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:09,941] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:252, R:4] s1=188.9769, s2=189.8821, ratio=0.9952 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:10,013] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:342] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:10,025] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:282, R:7] s1=77.6696, s2=78.0778, ratio=0.9948 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:10,027] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:283] Candidates: [4, 3, 1, 2]
[2025-04-16 22:41:10,041] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:372] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:10,158] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:102] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:10,332] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:72, R:4] s1=191.1958, s2=190.1015, ratio=1.0058 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,356] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:192, R:4] s1=193.4932, s2=192.3473, ratio=1.0060 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,415] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:12, R:3] s1=165.4269, s2=166.2176, ratio=0.9952 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:10,425] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:13] Candidates: [3, 7, 2, 6]
[2025-04-16 22:41:10,471] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:132, R:6] s1=145.7663, s2=145.2180, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,480] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:283] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:10,567] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:432, R:5] s1=193.4537, s2=192.7379, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,586] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:402, R:7] s1=78.3733, s2=78.0965, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,588] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:403] Candidates: [2, 5, 4, 3]
[2025-04-16 22:41:10,589] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:162, R:7] s1=77.7161, s2=77.3329, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,591] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:163] Candidates: [1, 7, 0, 5]
[2025-04-16 22:41:10,594] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:222, R:6] s1=145.5045, s2=144.9483, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,596] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:223] Candidates: [2, 5, 7, 6]
[2025-04-16 22:41:10,639] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:312, R:7] s1=78.3725, s2=77.9720, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,641] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:313] Candidates: [4, 3, 6, 0]
[2025-04-16 22:41:10,739] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:42, R:6] s1=148.1590, s2=147.4321, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,744] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:43] Candidates: [2, 0, 5, 6]
[2025-04-16 22:41:10,762] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:252, R:3] s1=165.1618, s2=165.9968, ratio=0.9950 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:10,765] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:253] Candidates: [3, 5, 2, 1]
[2025-04-16 22:41:10,792] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:342, R:4] s1=192.5689, s2=191.6281, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,931] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:372, R:5] s1=193.4136, s2=192.6705, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:10,932] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:13] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:10,967] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:102, R:4] s1=191.2878, s2=190.5286, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,041] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:223] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:11,051] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:403] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:11,060] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:163] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:11,175] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:192, R:6] s1=145.2754, s2=145.2658, ratio=1.0001 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,177] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:193] Candidates: [2, 0, 6, 4]
[2025-04-16 22:41:11,181] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:313] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:11,235] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:72, R:5] s1=193.9441, s2=192.9790, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,237] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:73] Candidates: [6, 4, 0, 2]
[2025-04-16 22:41:11,250] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:253] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:11,284] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:283, R:4] s1=187.5235, s2=186.7838, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,287] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:132, R:3] s1=165.8911, s2=166.6705, ratio=0.9953 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:11,289] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:133] Candidates: [2, 0, 7, 3]
[2025-04-16 22:41:11,299] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:43] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:11,341] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:432, R:2] s1=142.3676, s2=141.6180, ratio=1.0053 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,344] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:433] Candidates: [0, 4, 5, 6]
[2025-04-16 22:41:11,622] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:342, R:7] s1=78.4849, s2=78.1959, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,630] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:343] Candidates: [0, 6, 7, 1]
[2025-04-16 22:41:11,687] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:193] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:11,710] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:73] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:11,812] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:372, R:2] s1=141.3978, s2=142.1559, ratio=0.9947 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:11,814] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:373] Candidates: [7, 0, 4, 5]
[2025-04-16 22:41:11,831] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:13, R:7] s1=77.6893, s2=77.9306, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:11,857] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:133] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:11,880] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:102, R:5] s1=194.1068, s2=193.2789, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,881] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:103] Candidates: [2, 0, 5, 1]
[2025-04-16 22:41:11,888] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:433] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:11,901] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:223, R:6] s1=145.2833, s2=145.0693, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:11,994] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:403, R:4] s1=191.6512, s2=192.4602, ratio=0.9958 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:12,032] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:253, R:5] s1=192.2998, s2=191.7168, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,040] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:313, R:4] s1=192.6650, s2=191.9997, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,070] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:163, R:7] s1=77.5484, s2=77.3344, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,108] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:283, R:3] s1=166.6105, s2=165.7370, ratio=1.0053 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,110] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:284] Candidates: [3, 7, 2, 1]
[2025-04-16 22:41:12,126] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:343] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:12,228] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:43, R:5] s1=192.9904, s2=192.3136, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,281] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:373] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:12,297] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:103] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:12,456] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:73, R:4] s1=190.7030, s2=190.1568, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,547] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:193, R:4] s1=193.0594, s2=192.3829, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,562] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:284] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:12,605] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:13, R:6] s1=147.0954, s2=147.4679, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:12,607] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:14] Candidates: [7, 6, 5, 0]
[2025-04-16 22:41:12,611] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:133, R:7] s1=77.6192, s2=77.4409, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,638] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:223, R:7] s1=77.6399, s2=77.4374, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,640] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:224] Candidates: [0, 5, 2, 1]
[2025-04-16 22:41:12,671] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:433, R:4] s1=192.2282, s2=191.4884, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,730] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:313, R:6] s1=145.7298, s2=145.4637, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,732] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:314] Candidates: [3, 2, 7, 6]
[2025-04-16 22:41:12,825] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:403, R:5] s1=194.0796, s2=193.2343, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,828] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:404] Candidates: [2, 1, 4, 3]
[2025-04-16 22:41:12,845] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:253, R:3] s1=165.9965, s2=165.4117, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,847] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:254] Candidates: [5, 3, 2, 7]
[2025-04-16 22:41:12,868] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:163, R:5] s1=195.0605, s2=194.3560, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,870] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:164] Candidates: [1, 0, 5, 7]
[2025-04-16 22:41:12,921] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:343, R:6] s1=145.9400, s2=145.5999, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,955] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:43, R:6] s1=147.9002, s2=147.3963, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:12,963] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:44] Candidates: [4, 7, 6, 2]
[2025-04-16 22:41:13,040] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:14] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:13,094] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:224] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:13,121] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:373, R:4] s1=192.2221, s2=191.4862, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,136] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:103, R:5] s1=194.1169, s2=193.7390, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,248] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:73, R:6] s1=145.7468, s2=145.4232, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,250] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:74] Candidates: [1, 6, 5, 2]
[2025-04-16 22:41:13,280] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:284, R:7] s1=77.8381, s2=78.0091, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:13,293] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:314] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:13,318] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:164] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:13,324] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:254] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:13,333] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:404] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:13,428] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:193, R:6] s1=145.2818, s2=144.9634, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,429] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:194] Candidates: [1, 6, 2, 3]
[2025-04-16 22:41:13,436] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:433, R:6] s1=145.9697, s2=145.6426, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,438] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:434] Candidates: [0, 6, 5, 7]
[2025-04-16 22:41:13,441] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:133, R:3] s1=166.6721, s2=166.1451, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,442] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:134] Candidates: [6, 1, 3, 0]
[2025-04-16 22:41:13,444] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:44] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:13,621] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:343, R:7] s1=78.3695, s2=78.1614, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,623] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:344] Candidates: [0, 2, 5, 6]
[2025-04-16 22:41:13,717] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:74] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:13,820] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:14, R:5] s1=192.1819, s2=191.5958, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,927] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:373, R:7] s1=78.1188, s2=78.3121, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:13,928] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:374] Candidates: [6, 5, 1, 2]
[2025-04-16 22:41:13,935] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:434] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:13,949] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:134] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:13,954] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:103, R:2] s1=142.0701, s2=141.4051, ratio=1.0047 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:13,956] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:104] Candidates: [5, 2, 0, 4]
[2025-04-16 22:41:14,007] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:194] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:14,013] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:314, R:6] s1=145.7849, s2=145.5014, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,048] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:224, R:5] s1=195.7740, s2=194.9445, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,120] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:164, R:7] s1=77.5446, s2=77.3323, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,128] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:284, R:3] s1=166.2088, s2=165.7406, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,130] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:285] Candidates: [1, 5, 6, 0]
[2025-04-16 22:41:14,160] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:254, R:5] s1=191.7080, s2=192.1680, ratio=0.9976 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:14,222] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:404, R:4] s1=192.8375, s2=192.3016, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,227] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:344] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:14,279] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:44, R:4] s1=190.7675, s2=190.1366, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,457] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:104] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:14,489] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:374] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:14,555] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:14, R:7] s1=77.7184, s2=77.9216, ratio=0.9974 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:14,556] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:15] Candidates: [6, 1, 7, 2]
[2025-04-16 22:41:14,585] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:285] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:14,621] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:74, R:5] s1=193.5685, s2=193.0110, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,717] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:434, R:7] s1=78.3682, s2=78.1421, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,720] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:134, R:6] s1=145.2222, s2=145.4465, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:14,758] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:194, R:6] s1=145.1785, s2=144.9678, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,763] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:314, R:7] s1=78.2020, s2=77.9083, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,767] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:315] Candidates: [0, 7, 6, 5]
[2025-04-16 22:41:14,844] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:164, R:5] s1=194.9553, s2=194.3560, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,847] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:165] Candidates: [0, 5, 1, 6]
[2025-04-16 22:41:14,926] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:254, R:7] s1=77.5219, s2=77.6772, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:14,931] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:255] Candidates: [4, 6, 1, 5]
[2025-04-16 22:41:14,942] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:224, R:2] s1=142.1765, s2=141.5042, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:14,943] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:225] Candidates: [4, 2, 1, 7]
[2025-04-16 22:41:14,969] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:15] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:15,054] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:404, R:3] s1=167.0987, s2=167.6743, ratio=0.9966 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:15,057] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:405] Candidates: [1, 7, 4, 6]
[2025-04-16 22:41:15,078] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:344, R:6] s1=145.8865, s2=145.5658, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,130] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:44, R:7] s1=77.8766, s2=77.7394, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,133] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:45] Candidates: [0, 5, 6, 2]
[2025-04-16 22:41:15,260] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:374, R:6] s1=145.5535, s2=145.8331, ratio=0.9981 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:15,298] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:104, R:4] s1=191.8782, s2=191.2773, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,336] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:315] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:15,435] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:225] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:15,440] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:74, R:6] s1=145.6818, s2=145.3911, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,443] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:75] Candidates: [7, 5, 6, 3]
[2025-04-16 22:41:15,451] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:255] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:15,474] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:165] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:15,475] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:285, R:5] s1=189.9671, s2=189.4244, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,565] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:434, R:6] s1=145.9141, s2=145.6177, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,570] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:435] Candidates: [3, 2, 4, 7]
[2025-04-16 22:41:15,571] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:405] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:15,603] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:134, R:3] s1=166.6763, s2=166.1483, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,611] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:135] Candidates: [0, 4, 6, 5]
[2025-04-16 22:41:15,667] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:45] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:15,714] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:194, R:3] s1=167.6415, s2=167.0271, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:15,724] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:195] Candidates: [5, 6, 3, 4]
[2025-04-16 22:41:15,936] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:15, R:7] s1=77.9735, s2=77.7884, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,317] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:75] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:16,419] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:344, R:5] s1=193.2760, s2=192.6443, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,423] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:345] Candidates: [5, 0, 3, 4]
[2025-04-16 22:41:16,508] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:435] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:16,529] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:135] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:16,842] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:315, R:6] s1=145.8883, s2=145.5810, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,867] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:104, R:5] s1=194.3192, s2=193.8096, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,869] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:105] Candidates: [1, 4, 6, 3]
[2025-04-16 22:41:16,883] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:374, R:5] s1=192.6430, s2=193.1263, ratio=0.9975 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:16,887] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:375] Candidates: [4, 6, 0, 7]
[2025-04-16 22:41:16,887] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:255, R:4] s1=184.8659, s2=184.0325, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,907] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:165, R:6] s1=145.4405, s2=145.0972, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:16,920] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:285, R:6] s1=145.2775, s2=145.5477, ratio=0.9981 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:16,922] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:286] Candidates: [6, 4, 1, 0]
[2025-04-16 22:41:16,951] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:225, R:4] s1=192.6685, s2=191.9115, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,003] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:405, R:4] s1=192.8141, s2=192.2518, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,017] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:195] Selected rings for extraction: [6, 4]
[2025-04-16 22:41:17,165] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:45, R:5] s1=193.2270, s2=192.6417, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,187] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:345] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:17,379] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:15, R:6] s1=147.0374, s2=147.4567, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:17,387] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:16] Candidates: [1, 7, 3, 4]
[2025-04-16 22:41:17,522] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:75, R:5] s1=193.6359, s2=193.0599, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,542] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:375] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:17,562] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:105] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:17,589] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:286] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:17,651] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:435, R:4] s1=192.0717, s2=191.4917, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,684] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:135, R:4] s1=192.4840, s2=191.6785, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,749] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:315, R:7] s1=78.1637, s2=77.9922, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,753] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:316] Candidates: [7, 6, 0, 3]
[2025-04-16 22:41:17,797] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:255, R:5] s1=190.7114, s2=191.1627, ratio=0.9976 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:17,805] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:256] Candidates: [0, 4, 2, 7]
[2025-04-16 22:41:17,849] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:405, R:7] s1=78.1510, s2=78.3215, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:17,851] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:406] Candidates: [2, 5, 3, 4]
[2025-04-16 22:41:17,854] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:225, R:7] s1=77.6473, s2=77.5045, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,856] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:226] Candidates: [6, 7, 0, 4]
[2025-04-16 22:41:17,891] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:165, R:5] s1=194.9999, s2=194.4041, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:17,894] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:166] Candidates: [5, 3, 4, 6]
[2025-04-16 22:41:17,947] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:16] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:18,004] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:195, R:6] s1=145.2178, s2=145.0493, ratio=1.0012 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,039] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:45, R:6] s1=147.6572, s2=147.3288, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,042] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:46] Candidates: [0, 6, 5, 2]
[2025-04-16 22:41:18,103] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:345, R:4] s1=192.1522, s2=191.5385, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,209] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:256] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:18,257] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:316] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:18,294] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:105, R:4] s1=192.0539, s2=191.3463, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,297] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:226] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:18,349] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:406] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:18,351] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:375, R:4] s1=192.0939, s2=191.5098, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,388] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:135, R:6] s1=145.1752, s2=145.3945, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:18,392] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:136] Candidates: [1, 4, 6, 0]
[2025-04-16 22:41:18,454] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:166] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:18,462] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:75, R:6] s1=145.7467, s2=145.4627, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,465] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:76] Candidates: [0, 5, 6, 7]
[2025-04-16 22:41:18,488] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:286, R:4] s1=186.0385, s2=186.8905, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:18,516] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:435, R:7] s1=78.3060, s2=78.1561, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,518] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:436] Candidates: [1, 6, 3, 2]
[2025-04-16 22:41:18,530] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:46] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:18,701] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:16, R:4] s1=189.3708, s2=188.7131, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,795] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:195, R:4] s1=193.1422, s2=192.4721, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,801] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:196] Candidates: [2, 0, 1, 6]
[2025-04-16 22:41:18,917] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:345, R:5] s1=193.1311, s2=192.6150, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,919] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:346] Candidates: [4, 1, 6, 0]
[2025-04-16 22:41:18,940] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:76] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:18,949] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:136] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:18,972] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:256, R:4] s1=184.7138, s2=183.9345, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:18,987] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:436] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:18,995] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:316, R:6] s1=145.8983, s2=145.4870, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,098] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:226, R:4] s1=189.5643, s2=188.2760, ratio=1.0068 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,132] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:105, R:6] s1=145.5796, s2=145.2639, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,138] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:106] Candidates: [6, 0, 5, 4]
[2025-04-16 22:41:19,154] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:375, R:6] s1=145.8333, s2=145.8363, ratio=1.0000 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:19,157] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:376] Candidates: [7, 1, 2, 4]
[2025-04-16 22:41:19,197] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:166, R:4] s1=192.7044, s2=192.0352, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,201] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:406, R:4] s1=192.2131, s2=192.7609, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:19,242] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:286, R:6] s1=145.3896, s2=145.7039, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:19,244] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:287] Candidates: [3, 7, 6, 4]
[2025-04-16 22:41:19,325] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:46, R:5] s1=193.1950, s2=192.6447, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,380] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:196] Selected rings for extraction: [6, 2]
[2025-04-16 22:41:19,415] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:346] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:19,521] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:16, R:7] s1=77.9729, s2=77.7968, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,524] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:17] Candidates: [7, 6, 0, 2]
[2025-04-16 22:41:19,618] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:376] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:19,718] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:106] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:19,771] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:136, R:4] s1=192.3291, s2=191.6762, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,771] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:76, R:5] s1=193.5801, s2=193.0667, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,825] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:256, R:7] s1=77.9313, s2=77.7555, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,827] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:257] Candidates: [0, 1, 2, 3]
[2025-04-16 22:41:19,830] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:436, R:6] s1=145.9181, s2=145.6599, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,856] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:287] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:19,874] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:316, R:7] s1=78.1711, s2=77.8869, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,875] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:317] Candidates: [0, 5, 2, 3]
[2025-04-16 22:41:19,928] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:226, R:7] s1=77.9421, s2=77.9381, ratio=1.0001 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:19,930] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:227] Candidates: [0, 3, 1, 2]
[2025-04-16 22:41:19,994] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:17] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:20,010] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:406, R:5] s1=192.8246, s2=193.3226, ratio=0.9974 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:20,014] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:407] Candidates: [1, 0, 6, 7]
[2025-04-16 22:41:20,064] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:166, R:6] s1=145.3231, s2=145.1288, ratio=1.0013 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,066] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:167] Candidates: [5, 7, 0, 2]
[2025-04-16 22:41:20,100] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:196, R:6] s1=145.2189, s2=145.0564, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,231] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:46, R:6] s1=147.6524, s2=147.3298, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,233] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:47] Candidates: [4, 7, 6, 0]
[2025-04-16 22:41:20,254] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:346, R:4] s1=192.1147, s2=191.5579, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,292] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:317] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:20,366] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:257] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:20,376] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:227] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:20,385] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:376, R:4] s1=191.5133, s2=192.1104, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:20,509] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:106, R:4] s1=191.9336, s2=191.3531, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,519] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:76, R:6] s1=145.7150, s2=145.4705, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,520] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:77] Candidates: [5, 0, 4, 6]
[2025-04-16 22:41:20,537] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:136, R:6] s1=145.3945, s2=145.1876, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,539] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:137] Candidates: [6, 1, 4, 7]
[2025-04-16 22:41:20,567] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:167] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:20,571] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:407] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:20,601] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:287, R:4] s1=186.9412, s2=186.7489, ratio=1.0010 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,657] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:436, R:3] s1=167.8046, s2=167.0919, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,659] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:437] Candidates: [0, 3, 6, 7]
[2025-04-16 22:41:20,763] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:47] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:20,785] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:17, R:7] s1=77.7217, s2=77.9395, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:20,893] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:196, R:2] s1=141.9556, s2=141.4054, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:20,894] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:197] Candidates: [5, 1, 7, 6]
[2025-04-16 22:41:20,972] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:137] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:20,996] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:77] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:21,087] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:317, R:5] s1=193.1417, s2=192.4518, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,104] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:346, R:6] s1=145.8335, s2=145.5951, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,106] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:347] Candidates: [4, 3, 5, 6]
[2025-04-16 22:41:21,134] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:227, R:3] s1=166.4648, s2=165.7309, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,142] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:257, R:3] s1=162.4840, s2=163.6352, ratio=0.9930 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:21,195] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:437] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:21,208] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:376, R:7] s1=78.3257, s2=78.1806, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,210] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:377] Candidates: [6, 7, 5, 0]
[2025-04-16 22:41:21,312] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:106, R:6] s1=145.4530, s2=145.2724, ratio=1.0012 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,318] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:107] Candidates: [7, 5, 3, 4]
[2025-04-16 22:41:21,341] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:287, R:3] s1=164.0486, s2=163.2099, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,343] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:288] Candidates: [7, 5, 0, 3]
[2025-04-16 22:41:21,361] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:167, R:7] s1=77.5703, s2=77.4260, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,369] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:407, R:7] s1=78.1816, s2=78.3230, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:21,433] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:197] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:21,582] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:17, R:6] s1=147.4385, s2=146.9815, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,584] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:18] Candidates: [2, 1, 5, 0]
[2025-04-16 22:41:21,620] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:47, R:4] s1=190.7341, s2=190.1857, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,659] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:347] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:21,694] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:377] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:21,763] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:107] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:21,808] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:288] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:21,831] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:137, R:4] s1=191.6760, s2=192.3258, ratio=0.9966 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:21,962] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:77, R:4] s1=190.7776, s2=190.1504, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:21,997] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:437, R:7] s1=78.3239, s2=78.1325, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,018] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:317, R:3] s1=167.7487, s2=167.0184, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,019] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:318] Candidates: [6, 1, 3, 2]
[2025-04-16 22:41:22,021] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:227, R:2] s1=142.4971, s2=141.8074, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,025] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:228] Candidates: [7, 1, 0, 4]
[2025-04-16 22:41:22,036] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:257, R:2] s1=142.1254, s2=141.4847, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,038] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:258] Candidates: [4, 6, 1, 5]
[2025-04-16 22:41:22,072] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:18] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:22,166] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:167, R:5] s1=195.0397, s2=194.4839, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,168] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:168] Candidates: [6, 1, 4, 2]
[2025-04-16 22:41:22,235] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:407, R:6] s1=145.7856, s2=145.5213, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,237] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:408] Candidates: [0, 4, 2, 3]
[2025-04-16 22:41:22,281] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:197, R:6] s1=145.2245, s2=145.0401, ratio=1.0013 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,308] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:47, R:6] s1=147.4591, s2=147.4586, ratio=1.0000 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,309] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:48] Candidates: [5, 6, 3, 1]
[2025-04-16 22:41:22,454] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:347, R:4] s1=192.1057, s2=191.5277, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,473] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:107, R:4] s1=191.9358, s2=191.3578, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,475] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:377, R:6] s1=145.8949, s2=145.6776, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,507] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:228] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:22,539] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:258] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:22,582] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:318] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:22,609] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:288, R:7] s1=78.2371, s2=77.9661, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,624] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:137, R:6] s1=145.1850, s2=145.3919, ratio=0.9986 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:22,627] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:138] Candidates: [0, 5, 3, 7]
[2025-04-16 22:41:22,658] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:168] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:22,763] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:408] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:22,774] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:77, R:5] s1=193.5817, s2=193.0492, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,776] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:78] Candidates: [5, 7, 6, 1]
[2025-04-16 22:41:22,786] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:18, R:5] s1=192.4478, s2=191.4979, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,923] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:437, R:6] s1=145.9280, s2=145.6802, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:22,925] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:438] Candidates: [0, 1, 5, 2]
[2025-04-16 22:41:22,929] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:48] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:23,064] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:138] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:23,109] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:197, R:7] s1=77.6136, s2=77.3966, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,111] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:198] Candidates: [1, 4, 6, 5]
[2025-04-16 22:41:23,169] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:78] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:23,232] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:347, R:6] s1=145.8278, s2=145.5947, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,236] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:348] Candidates: [1, 0, 5, 4]
[2025-04-16 22:41:23,289] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:228, R:4] s1=189.5487, s2=188.6098, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,293] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:107, R:7] s1=77.8525, s2=77.6992, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,296] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:108] Candidates: [6, 4, 7, 0]
[2025-04-16 22:41:23,300] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:377, R:7] s1=78.3604, s2=78.1961, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,304] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:378] Candidates: [4, 3, 2, 5]
[2025-04-16 22:41:23,337] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:168, R:4] s1=193.1847, s2=192.0383, ratio=1.0060 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,370] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:318, R:6] s1=146.0594, s2=145.5243, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,439] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:258, R:4] s1=187.0882, s2=185.8516, ratio=1.0067 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,462] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:288, R:3] s1=164.3826, s2=164.1268, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,465] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:289] Candidates: [7, 2, 6, 0]
[2025-04-16 22:41:23,471] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:438] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:23,606] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:408, R:4] s1=191.8004, s2=192.6833, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:23,613] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:18, R:2] s1=141.5855, s2=142.2985, ratio=0.9950 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:23,620] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:19] Candidates: [3, 1, 7, 2]
[2025-04-16 22:41:23,785] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:198] Selected rings for extraction: [6, 4]
[2025-04-16 22:41:23,832] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:348] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:23,867] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:48, R:6] s1=148.0049, s2=147.3017, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,925] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:108] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:23,935] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:138, R:7] s1=77.7638, s2=77.4545, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:23,940] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:378] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:23,949] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:289] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:24,084] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:78, R:5] s1=194.0323, s2=193.0723, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,107] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:19] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:24,138] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:168, R:6] s1=145.6841, s2=145.1432, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,142] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:169] Candidates: [6, 1, 3, 4]
[2025-04-16 22:41:24,160] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:228, R:7] s1=78.2326, s2=77.9312, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,165] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:229] Candidates: [3, 5, 1, 4]
[2025-04-16 22:41:24,207] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:258, R:5] s1=190.5925, s2=189.7632, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,214] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:259] Candidates: [7, 4, 2, 3]
[2025-04-16 22:41:24,322] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:438, R:5] s1=193.4105, s2=192.6857, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,324] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:318, R:3] s1=167.9398, s2=167.0881, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,326] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:319] Candidates: [2, 6, 1, 4]
[2025-04-16 22:41:24,476] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:408, R:3] s1=166.7952, s2=167.6200, ratio=0.9951 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:24,479] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:409] Candidates: [4, 3, 5, 0]
[2025-04-16 22:41:24,525] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:198, R:6] s1=145.5768, s2=145.0501, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,636] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:48, R:5] s1=193.6212, s2=192.6412, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,644] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:49] Candidates: [3, 2, 4, 6]
[2025-04-16 22:41:24,664] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:108, R:4] s1=192.4496, s2=191.3188, ratio=1.0059 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,707] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:169] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:24,738] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:378, R:4] s1=191.1695, s2=192.0389, ratio=0.9955 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:24,747] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:259] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:24,757] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:289, R:6] s1=145.6289, s2=145.2738, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,779] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:348, R:4] s1=192.4532, s2=191.4836, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,787] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:138, R:5] s1=195.1480, s2=194.2520, ratio=1.0046 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,789] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:139] Candidates: [6, 1, 5, 4]
[2025-04-16 22:41:24,800] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:229] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:24,812] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:319] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:24,920] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:19, R:7] s1=77.7352, s2=77.9848, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:24,926] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:78, R:6] s1=146.0934, s2=145.4719, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:24,929] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:79] Candidates: [5, 6, 1, 4]
[2025-04-16 22:41:25,000] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:438, R:2] s1=142.3768, s2=141.6198, ratio=1.0053 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,002] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:439] Candidates: [4, 7, 3, 1]
[2025-04-16 22:41:25,012] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:409] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:25,194] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:49] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:25,211] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:139] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:25,283] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:198, R:4] s1=193.5880, s2=192.4758, ratio=1.0058 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,285] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:199] Candidates: [6, 3, 5, 2]
[2025-04-16 22:41:25,353] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:108, R:6] s1=145.8409, s2=145.2840, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,355] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:109] Candidates: [3, 1, 2, 4]
[2025-04-16 22:41:25,493] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:378, R:5] s1=192.2529, s2=193.0218, ratio=0.9960 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:25,495] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:379] Candidates: [5, 0, 4, 3]
[2025-04-16 22:41:25,508] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:439] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:25,539] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:319, R:4] s1=192.2300, s2=191.4931, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,543] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:348, R:5] s1=193.3950, s2=192.6279, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,545] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:349] Candidates: [6, 0, 5, 7]
[2025-04-16 22:41:25,561] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:169, R:4] s1=192.7279, s2=192.0510, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,581] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:259, R:4] s1=185.8515, s2=187.3959, ratio=0.9918 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:25,624] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:289, R:7] s1=78.3423, s2=78.1480, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,626] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:290] Candidates: [5, 6, 7, 1]
[2025-04-16 22:41:25,662] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:79] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:25,672] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:229, R:4] s1=189.2162, s2=188.5963, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,822] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:409, R:4] s1=191.8259, s2=192.1850, ratio=0.9981 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:25,826] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:19, R:3] s1=166.3462, s2=165.7637, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:25,828] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:20] Candidates: [3, 1, 5, 7]
[2025-04-16 22:41:25,877] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:109] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:25,878] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:199] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:25,935] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:379] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:26,180] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:139, R:4] s1=191.6938, s2=192.3327, ratio=0.9967 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:26,192] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:49, R:4] s1=190.8482, s2=190.1790, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,224] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:290] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:26,226] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:349] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:26,329] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:439, R:4] s1=192.2300, s2=191.4850, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,363] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:20] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:26,471] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:169, R:6] s1=145.3803, s2=145.1620, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,473] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:170] Candidates: [6, 0, 5, 7]
[2025-04-16 22:41:26,481] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:259, R:3] s1=164.6402, s2=164.7975, ratio=0.9990 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:26,485] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:260] Candidates: [1, 0, 7, 4]
[2025-04-16 22:41:26,519] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:319, R:6] s1=145.8277, s2=145.4947, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,526] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:320] Candidates: [0, 5, 2, 4]
[2025-04-16 22:41:26,553] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:229, R:5] s1=192.4805, s2=191.8595, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,555] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:230] Candidates: [3, 1, 4, 0]
[2025-04-16 22:41:26,558] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:79, R:4] s1=190.8664, s2=190.1233, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,614] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:409, R:5] s1=193.0735, s2=193.9266, ratio=0.9956 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:26,621] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:410] Candidates: [5, 2, 4, 6]
[2025-04-16 22:41:26,675] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:199, R:6] s1=145.2823, s2=145.0248, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,719] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:109, R:4] s1=191.9822, s2=191.2987, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,799] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:379, R:4] s1=191.4364, s2=192.0440, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:26,968] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:139, R:6] s1=145.4846, s2=145.2058, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:26,970] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:140] Candidates: [3, 4, 2, 5]
[2025-04-16 22:41:26,972] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:349, R:6] s1=145.8911, s2=145.5878, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,000] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:49, R:6] s1=147.6501, s2=147.3542, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,007] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:50] Candidates: [7, 6, 3, 1]
[2025-04-16 22:41:27,011] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:290, R:5] s1=192.6103, s2=191.9829, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,020] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:260] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:27,050] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:170] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:27,084] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:230] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:27,130] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:320] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:27,135] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:439, R:7] s1=78.3157, s2=78.1235, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,137] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:440] Candidates: [5, 2, 3, 1]
[2025-04-16 22:41:27,163] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:20, R:5] s1=192.1021, s2=191.5584, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,166] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:410] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:27,275] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:79, R:5] s1=193.6393, s2=193.0694, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,280] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:80] Candidates: [4, 5, 2, 0]
[2025-04-16 22:41:27,442] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:199, R:5] s1=195.5198, s2=194.8269, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,446] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:200] Candidates: [7, 6, 1, 3]
[2025-04-16 22:41:27,456] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:50] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:27,525] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:140] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:27,577] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:109, R:3] s1=166.4937, s2=165.8017, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,579] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:110] Candidates: [0, 4, 1, 5]
[2025-04-16 22:41:27,621] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:379, R:5] s1=192.5397, s2=193.0060, ratio=0.9976 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:27,623] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:380] Candidates: [6, 1, 0, 2]
[2025-04-16 22:41:27,627] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:440] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:27,637] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:349, R:7] s1=78.3278, s2=78.1098, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,641] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:350] Candidates: [6, 0, 7, 2]
[2025-04-16 22:41:27,734] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:260, R:4] s1=185.8627, s2=186.5804, ratio=0.9962 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:27,756] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:320, R:4] s1=192.0876, s2=191.4411, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,760] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:80] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:27,764] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:230, R:4] s1=188.5194, s2=189.3847, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:27,837] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:20, R:7] s1=77.7561, s2=77.9762, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:27,839] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:21] Candidates: [5, 7, 1, 6]
[2025-04-16 22:41:27,941] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:200] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:27,941] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:170, R:6] s1=145.3799, s2=145.1570, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,965] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:410, R:4] s1=191.4898, s2=192.0145, ratio=0.9973 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:27,980] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:290, R:6] s1=145.5162, s2=145.2145, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:27,984] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:291] Candidates: [6, 1, 4, 5]
[2025-04-16 22:41:28,054] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:110] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:28,064] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:350] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:28,066] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:380] Selected rings for extraction: [6, 2]
[2025-04-16 22:41:28,186] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:50, R:7] s1=77.8777, s2=77.6948, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,271] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:21] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:28,320] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:140, R:4] s1=191.6370, s2=192.3333, ratio=0.9964 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:28,523] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:291] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:28,532] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:260, R:7] s1=77.9387, s2=77.7920, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,536] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:261] Candidates: [3, 5, 1, 0]
[2025-04-16 22:41:28,554] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:440, R:5] s1=193.1117, s2=192.6777, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,602] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:80, R:4] s1=190.7423, s2=190.1161, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,608] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:230, R:3] s1=166.2088, s2=165.8456, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,610] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:231] Candidates: [7, 6, 3, 0]
[2025-04-16 22:41:28,741] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:320, R:5] s1=193.1025, s2=192.4403, ratio=1.0034 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,743] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:321] Candidates: [0, 2, 1, 5]
[2025-04-16 22:41:28,774] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:170, R:7] s1=77.6164, s2=77.4252, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,776] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:171] Candidates: [2, 5, 3, 4]
[2025-04-16 22:41:28,777] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:200, R:6] s1=145.2412, s2=145.0419, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,793] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:410, R:6] s1=145.8972, s2=145.6816, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,795] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:411] Candidates: [4, 1, 3, 0]
[2025-04-16 22:41:28,864] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:110, R:4] s1=191.9518, s2=191.2699, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,907] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:380, R:6] s1=145.6604, s2=145.8816, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:28,929] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:350, R:6] s1=145.8538, s2=145.5955, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:28,978] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:261] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:29,055] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:50, R:6] s1=147.6521, s2=147.3363, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,064] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:51] Candidates: [7, 6, 4, 2]
[2025-04-16 22:41:29,081] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:231] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:29,120] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:21, R:5] s1=192.0970, s2=191.5576, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,168] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:140, R:5] s1=194.2077, s2=194.8150, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:29,176] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:141] Candidates: [1, 3, 5, 2]
[2025-04-16 22:41:29,265] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:321] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:29,270] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:411] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:29,283] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:291, R:4] s1=189.3153, s2=186.8448, ratio=1.0132 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,331] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:171] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:29,366] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:440, R:3] s1=167.7959, s2=167.0938, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,368] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:441] Candidates: [3, 5, 6, 7]
[2025-04-16 22:41:29,473] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:80, R:5] s1=193.6192, s2=193.0694, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,475] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:81] Candidates: [2, 4, 7, 0]
[2025-04-16 22:41:29,610] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:200, R:7] s1=77.6071, s2=77.4054, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,612] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:201] Candidates: [5, 3, 7, 1]
[2025-04-16 22:41:29,612] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:51] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:29,732] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:380, R:2] s1=142.2408, s2=141.5907, ratio=1.0046 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,738] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:110, R:5] s1=194.4964, s2=193.7980, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,739] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:381] Candidates: [1, 4, 2, 6]
[2025-04-16 22:41:29,742] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:111] Candidates: [5, 7, 0, 6]
[2025-04-16 22:41:29,771] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:141] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:29,826] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:350, R:7] s1=78.2801, s2=78.1125, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,829] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:351] Candidates: [5, 6, 1, 2]
[2025-04-16 22:41:29,883] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:441] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:29,887] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:261, R:5] s1=189.5438, s2=190.1810, ratio=0.9966 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:29,890] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:21, R:7] s1=77.9762, s2=77.7561, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:29,891] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:22] Candidates: [2, 0, 1, 3]
[2025-04-16 22:41:29,943] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:231, R:6] s1=145.0295, s2=145.2265, ratio=0.9986 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:29,963] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:81] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:30,005] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:321, R:5] s1=192.9786, s2=192.4876, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,064] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:411, R:4] s1=192.0515, s2=191.4877, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,087] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:291, R:6] s1=145.4826, s2=145.1932, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,089] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:292] Candidates: [2, 6, 1, 0]
[2025-04-16 22:41:30,142] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:171, R:4] s1=192.7188, s2=192.0043, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,179] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:201] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:30,217] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:381] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:30,228] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:111] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:30,304] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:351] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:30,402] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:22] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:30,470] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:51, R:4] s1=190.7273, s2=190.1890, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,603] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:292] Selected rings for extraction: [6, 2]
[2025-04-16 22:41:30,652] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:141, R:5] s1=194.2746, s2=194.8806, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:30,704] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:231, R:7] s1=77.8243, s2=77.6796, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,706] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:232] Candidates: [0, 7, 1, 2]
[2025-04-16 22:41:30,760] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:441, R:6] s1=145.9388, s2=145.5903, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,781] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:261, R:3] s1=164.0327, s2=163.1434, ratio=1.0055 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,843] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:262] Candidates: [2, 0, 4, 6]
[2025-04-16 22:41:30,940] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:321, R:2] s1=142.1484, s2=141.5122, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:30,948] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:322] Candidates: [1, 6, 5, 2]
[2025-04-16 22:41:30,980] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:81, R:4] s1=190.7256, s2=190.1403, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,096] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:411, R:3] s1=167.0532, s2=167.6009, ratio=0.9967 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:31,106] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:412] Candidates: [6, 2, 0, 7]
[2025-04-16 22:41:31,297] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:171, R:5] s1=195.1906, s2=194.4563, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,304] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:172] Candidates: [2, 5, 7, 6]
[2025-04-16 22:41:31,308] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:111, R:6] s1=145.4918, s2=145.2908, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,334] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:381, R:4] s1=192.0242, s2=191.4640, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,342] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:201, R:7] s1=77.5511, s2=77.4204, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,649] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:22, R:3] s1=165.7869, s2=166.2999, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:31,657] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:232] Selected rings for extraction: [7, 2]
[2025-04-16 22:41:31,680] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:351, R:6] s1=145.8595, s2=145.5875, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,726] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:322] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:31,791] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:262] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:31,836] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:51, R:6] s1=147.4705, s2=147.4690, ratio=1.0000 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,838] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:52] Candidates: [1, 2, 4, 5]
[2025-04-16 22:41:31,846] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:412] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:31,871] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:141, R:3] s1=166.7821, s2=166.2778, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:31,879] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:142] Candidates: [3, 0, 7, 2]
[2025-04-16 22:41:31,993] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:292, R:6] s1=145.4335, s2=145.1584, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,027] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:172] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:32,060] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:441, R:7] s1=78.2639, s2=78.0736, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,062] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:442] Candidates: [6, 7, 5, 2]
[2025-04-16 22:41:32,098] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:81, R:7] s1=78.0338, s2=77.8754, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,102] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:82] Candidates: [6, 0, 1, 3]
[2025-04-16 22:41:32,310] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:111, R:7] s1=77.8565, s2=77.6496, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,313] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:112] Candidates: [5, 0, 6, 2]
[2025-04-16 22:41:32,341] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:201, R:5] s1=195.3911, s2=194.8666, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,342] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:202] Candidates: [0, 1, 3, 4]
[2025-04-16 22:41:32,439] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:52] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:32,498] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:381, R:6] s1=145.6353, s2=145.8629, ratio=0.9984 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:32,504] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:382] Candidates: [2, 5, 3, 0]
[2025-04-16 22:41:32,555] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:142] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:32,594] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:232, R:7] s1=77.8018, s2=77.6902, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,600] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:322, R:6] s1=145.7636, s2=145.4521, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,710] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:442] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:32,749] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:22, R:2] s1=142.3032, s2=141.7316, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,751] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:23] Candidates: [3, 2, 4, 5]
[2025-04-16 22:41:32,753] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:351, R:5] s1=193.0781, s2=192.6099, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,755] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:352] Candidates: [4, 7, 5, 0]
[2025-04-16 22:41:32,761] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:262, R:4] s1=186.7611, s2=186.0902, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,772] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:82] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:32,781] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:412, R:7] s1=78.3890, s2=78.1372, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,954] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:292, R:2] s1=142.1764, s2=141.5378, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:32,956] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:293] Candidates: [0, 6, 2, 7]
[2025-04-16 22:41:33,033] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:112] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:33,093] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:202] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:33,106] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:172, R:6] s1=145.3889, s2=145.1746, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,134] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:382] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:33,296] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:352] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:33,301] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:52, R:4] s1=190.7300, s2=190.1501, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,303] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:23] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:33,430] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:322, R:5] s1=192.9993, s2=192.4591, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,431] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:323] Candidates: [1, 0, 4, 5]
[2025-04-16 22:41:33,434] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:142, R:7] s1=77.4557, s2=77.5962, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:33,477] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:293] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:33,489] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:232, R:2] s1=142.3298, s2=141.6699, ratio=1.0047 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,493] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:233] Candidates: [7, 5, 3, 1]
[2025-04-16 22:41:33,530] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:442, R:6] s1=145.8688, s2=145.5597, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,631] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:262, R:6] s1=145.0588, s2=144.7194, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,633] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:263] Candidates: [4, 1, 2, 5]
[2025-04-16 22:41:33,643] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:82, R:6] s1=145.7523, s2=145.5120, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,664] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:412, R:6] s1=145.9061, s2=145.5959, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,665] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:413] Candidates: [6, 1, 4, 2]
[2025-04-16 22:41:33,819] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:112, R:6] s1=145.4858, s2=145.1417, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,827] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:172, R:7] s1=77.5802, s2=77.4126, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,830] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:173] Candidates: [4, 7, 1, 6]
[2025-04-16 22:41:33,940] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:323] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:33,954] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:382, R:5] s1=192.5486, s2=193.0031, ratio=0.9976 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:33,959] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:202, R:4] s1=193.1626, s2=192.4321, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:33,966] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:233] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:34,097] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:23, R:4] s1=188.5660, s2=189.1164, ratio=0.9971 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,135] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:263] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:34,138] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:413] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:34,161] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:52, R:5] s1=193.2068, s2=192.6423, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,164] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:53] Candidates: [1, 3, 4, 2]
[2025-04-16 22:41:34,170] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:352, R:4] s1=192.0785, s2=191.5218, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,210] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:142, R:3] s1=166.6496, s2=167.2408, ratio=0.9965 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,212] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:143] Candidates: [0, 2, 1, 4]
[2025-04-16 22:41:34,279] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:293, R:6] s1=145.4243, s2=145.1547, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,402] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:442, R:7] s1=78.2375, s2=78.0562, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,405] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:443] Candidates: [6, 4, 1, 2]
[2025-04-16 22:41:34,406] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:173] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:34,431] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:82, R:3] s1=166.0325, s2=165.3861, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,434] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:83] Candidates: [3, 6, 4, 1]
[2025-04-16 22:41:34,636] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:143] Selected rings for extraction: [4, 2]
[2025-04-16 22:41:34,663] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:202, R:3] s1=167.6644, s2=166.9892, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,665] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:203] Candidates: [3, 1, 7, 4]
[2025-04-16 22:41:34,687] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:53] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:34,754] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:233, R:5] s1=191.8397, s2=191.3555, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,772] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:112, R:5] s1=193.8418, s2=194.2690, ratio=0.9978 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,774] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:113] Candidates: [3, 5, 6, 2]
[2025-04-16 22:41:34,787] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:382, R:3] s1=167.7953, s2=167.0965, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,789] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:383] Candidates: [0, 6, 4, 3]
[2025-04-16 22:41:34,820] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:323, R:4] s1=192.0621, s2=191.4593, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,887] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:23, R:5] s1=191.5329, s2=192.0956, ratio=0.9971 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,889] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:24] Candidates: [3, 6, 1, 4]
[2025-04-16 22:41:34,899] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:443] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:34,923] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:413, R:4] s1=191.4949, s2=192.0388, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,967] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:263, R:4] s1=186.2689, s2=186.9933, ratio=0.9961 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:34,987] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:83] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:34,989] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:352, R:7] s1=78.2966, s2=78.1588, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:34,990] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:353] Candidates: [6, 5, 1, 3]
[2025-04-16 22:41:35,202] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:293, R:7] s1=78.3219, s2=78.1463, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,208] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:294] Candidates: [5, 2, 1, 4]
[2025-04-16 22:41:35,318] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:203] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:35,350] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:173, R:4] s1=192.6778, s2=192.0399, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,411] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:113] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:35,418] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:383] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:35,423] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:24] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:35,514] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:353] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:35,556] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:233, R:7] s1=77.8018, s2=77.6902, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,563] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:234] Candidates: [7, 0, 6, 1]
[2025-04-16 22:41:35,618] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:53, R:4] s1=190.6555, s2=190.0480, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,643] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:143, R:4] s1=192.6761, s2=192.0038, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,674] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:323, R:5] s1=193.0339, s2=192.4838, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,676] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:324] Candidates: [6, 5, 1, 4]
[2025-04-16 22:41:35,727] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:443, R:4] s1=192.0301, s2=191.4652, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,775] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:413, R:6] s1=145.5704, s2=145.8158, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:35,776] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:414] Candidates: [3, 4, 0, 2]
[2025-04-16 22:41:35,793] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:263, R:5] s1=191.4686, s2=190.9880, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:35,795] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:264] Candidates: [5, 7, 2, 6]
[2025-04-16 22:41:35,840] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:294] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:35,901] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:83, R:4] s1=190.7193, s2=190.1414, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,010] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:234] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:36,021] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:173, R:6] s1=145.3771, s2=145.1844, ratio=1.0013 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,023] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:174] Candidates: [0, 7, 5, 3]
[2025-04-16 22:41:36,121] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:324] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:36,211] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:203, R:4] s1=193.1386, s2=192.4717, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,223] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:24, R:4] s1=188.1915, s2=189.0509, ratio=0.9955 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:36,230] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:353, R:6] s1=145.9190, s2=145.6605, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,258] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:113, R:6] s1=145.1640, s2=145.4177, ratio=0.9983 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:36,295] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:264] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:36,315] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:383, R:4] s1=191.4482, s2=191.9882, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:36,332] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:414] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:36,421] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:53, R:3] s1=166.4342, s2=165.7529, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,426] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:54] Candidates: [5, 4, 2, 3]
[2025-04-16 22:41:36,469] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:443, R:6] s1=145.8374, s2=145.5544, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,471] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:444] Candidates: [0, 1, 4, 7]
[2025-04-16 22:41:36,473] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:174] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:36,482] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:143, R:2] s1=142.0770, s2=141.4646, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,484] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:144] Candidates: [4, 7, 5, 1]
[2025-04-16 22:41:36,732] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:83, R:6] s1=145.7464, s2=145.5118, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,734] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:84] Candidates: [5, 7, 4, 3]
[2025-04-16 22:41:36,737] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:294, R:4] s1=185.3661, s2=186.5478, ratio=0.9937 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:36,739] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:234, R:6] s1=145.8014, s2=145.2656, ratio=1.0037 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,892] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:324, R:4] s1=192.4507, s2=191.5362, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,918] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:24, R:6] s1=146.6765, s2=147.3908, ratio=0.9952 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:36,920] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:25] Candidates: [4, 2, 0, 6]
[2025-04-16 22:41:36,980] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:54] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:36,981] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:444] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:36,983] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:144] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:36,985] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:203, R:7] s1=77.5545, s2=77.4249, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:36,988] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:204] Candidates: [2, 0, 7, 3]
[2025-04-16 22:41:37,069] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:353, R:5] s1=193.0696, s2=192.6133, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,070] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:383, R:6] s1=145.8520, s2=145.5939, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,071] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:354] Candidates: [5, 3, 7, 6]
[2025-04-16 22:41:37,074] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:384] Candidates: [3, 1, 0, 4]
[2025-04-16 22:41:37,177] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:264, R:5] s1=191.7264, s2=191.2831, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,214] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:414, R:4] s1=191.2183, s2=192.0127, ratio=0.9959 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:37,218] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:113, R:5] s1=194.3117, s2=193.7628, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,220] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:114] Candidates: [2, 1, 0, 5]
[2025-04-16 22:41:37,301] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:84] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:37,326] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:174, R:7] s1=77.7653, s2=77.4496, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,351] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:25] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:37,529] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:234, R:7] s1=78.0369, s2=77.6575, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,531] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:235] Candidates: [7, 5, 2, 6]
[2025-04-16 22:41:37,611] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:294, R:5] s1=191.3712, s2=190.5422, ratio=1.0044 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,613] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:295] Candidates: [1, 4, 6, 5]
[2025-04-16 22:41:37,623] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:204] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:37,645] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:354] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:37,662] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:384] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:37,758] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:324, R:6] s1=146.0375, s2=145.5098, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,760] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:325] Candidates: [4, 0, 7, 1]
[2025-04-16 22:41:37,772] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:114] Selected rings for extraction: [5, 2]
[2025-04-16 22:41:37,885] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:54, R:4] s1=191.1262, s2=190.0633, ratio=1.0056 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,951] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:144, R:4] s1=193.1782, s2=192.0906, ratio=1.0057 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,980] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:235] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:37,987] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:444, R:4] s1=192.4406, s2=191.5153, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:37,994] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:264, R:6] s1=144.6589, s2=145.2749, ratio=0.9958 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:37,995] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:265] Candidates: [4, 1, 7, 2]
[2025-04-16 22:41:38,093] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:414, R:3] s1=167.9658, s2=167.1321, ratio=1.0050 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,097] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:415] Candidates: [1, 5, 0, 7]
[2025-04-16 22:41:38,117] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:295] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:38,172] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:84, R:4] s1=191.2616, s2=190.0899, ratio=1.0062 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,189] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:325] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:38,208] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:174, R:5] s1=195.5352, s2=194.7120, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,210] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:175] Candidates: [1, 3, 0, 6]
[2025-04-16 22:41:38,233] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:25, R:4] s1=189.0576, s2=188.5558, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,375] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:204, R:7] s1=77.7660, s2=77.4395, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,446] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:265] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:38,454] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:354, R:7] s1=78.5004, s2=78.1510, ratio=1.0045 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,460] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:384, R:4] s1=192.4572, s2=191.5303, ratio=1.0048 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,585] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:114, R:5] s1=193.3260, s2=194.2156, ratio=0.9954 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:38,649] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:54, R:5] s1=193.5280, s2=192.4881, ratio=1.0054 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,652] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:55] Candidates: [4, 3, 2, 1]
[2025-04-16 22:41:38,660] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:415] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:38,671] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:144, R:7] s1=77.2867, s2=77.5563, ratio=0.9965 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:38,672] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:145] Candidates: [3, 6, 7, 5]
[2025-04-16 22:41:38,679] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:175] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:38,717] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:235, R:6] s1=144.9748, s2=145.0717, ratio=0.9993 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:38,792] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:444, R:7] s1=78.2990, s2=77.9977, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,793] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:445] Candidates: [0, 6, 4, 2]
[2025-04-16 22:41:38,873] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:295, R:4] s1=189.9017, s2=189.2291, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,889] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:325, R:4] s1=192.1207, s2=191.5296, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:38,995] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:25, R:6] s1=146.9471, s2=147.3924, ratio=0.9970 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:38,997] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:26] Candidates: [1, 4, 5, 7]
[2025-04-16 22:41:39,031] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:84, R:5] s1=194.0381, s2=193.0588, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,033] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:85] Candidates: [1, 5, 3, 6]
[2025-04-16 22:41:39,054] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:55] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:39,119] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:145] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:39,259] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:265, R:4] s1=186.9537, s2=186.7774, ratio=1.0009 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,281] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:204, R:3] s1=167.8702, s2=167.0842, ratio=1.0047 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,283] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:205] Candidates: [3, 1, 2, 0]
[2025-04-16 22:41:39,294] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:384, R:3] s1=166.8432, s2=167.5612, ratio=0.9957 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:39,298] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:385] Candidates: [4, 5, 1, 6]
[2025-04-16 22:41:39,364] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:114, R:2] s1=142.0370, s2=141.3389, ratio=1.0049 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,370] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:115] Candidates: [7, 0, 6, 3]
[2025-04-16 22:41:39,388] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:354, R:6] s1=146.1620, s2=145.6506, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,391] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:355] Candidates: [3, 7, 2, 6]
[2025-04-16 22:41:39,464] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:445] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:39,538] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:175, R:6] s1=145.3949, s2=145.0731, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,557] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:415, R:7] s1=78.2772, s2=78.1131, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,588] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:85] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:39,614] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:26] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:39,688] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:235, R:5] s1=192.3014, s2=191.3231, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,690] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:236] Candidates: [4, 6, 1, 7]
[2025-04-16 22:41:39,858] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:325, R:7] s1=78.3176, s2=78.0771, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,864] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:326] Candidates: [0, 3, 5, 4]
[2025-04-16 22:41:39,953] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:295, R:5] s1=192.4893, s2=191.9638, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:39,955] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:296] Candidates: [0, 5, 4, 6]
[2025-04-16 22:41:39,973] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:205] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:39,987] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:55, R:4] s1=190.6526, s2=190.0569, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,037] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:355] Selected rings for extraction: [7, 6]
[2025-04-16 22:41:40,075] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:115] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:40,090] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:145, R:6] s1=145.4318, s2=145.1552, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,094] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:385] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:40,190] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:265, R:7] s1=78.0859, s2=77.9302, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,191] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:266] Candidates: [4, 6, 5, 0]
[2025-04-16 22:41:40,228] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:236] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:40,275] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:175, R:3] s1=167.6793, s2=167.0850, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,277] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:176] Candidates: [5, 2, 4, 7]
[2025-04-16 22:41:40,318] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:326] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:40,327] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:445, R:4] s1=192.1109, s2=191.5558, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,381] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:415, R:5] s1=193.3081, s2=192.6981, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,391] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:416] Candidates: [0, 1, 3, 6]
[2025-04-16 22:41:40,490] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:85, R:5] s1=193.6301, s2=193.0807, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,502] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:296] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:40,502] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:26, R:4] s1=188.5246, s2=189.0569, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:40,631] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:266] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:40,646] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:176] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:40,746] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:205, R:3] s1=167.6109, s2=167.0775, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,755] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:115, R:6] s1=145.4800, s2=145.1759, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,901] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:355, R:7] s1=78.3695, s2=78.1508, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,912] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:416] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:40,919] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:145, R:7] s1=77.5723, s2=77.3983, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,934] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:146] Candidates: [6, 1, 4, 5]
[2025-04-16 22:41:40,937] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:55, R:3] s1=166.4399, s2=165.7639, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,940] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:56] Candidates: [3, 4, 1, 2]
[2025-04-16 22:41:40,956] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:236, R:4] s1=188.9336, s2=188.1919, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:40,966] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:385, R:4] s1=192.1093, s2=191.5670, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,035] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:326, R:4] s1=192.1276, s2=191.4934, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,109] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:445, R:6] s1=145.8556, s2=145.5548, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,110] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:446] Candidates: [7, 0, 4, 5]
[2025-04-16 22:41:41,163] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:85, R:6] s1=145.8050, s2=145.4674, ratio=1.0023 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,165] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:86] Candidates: [6, 4, 7, 2]
[2025-04-16 22:41:41,218] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:26, R:5] s1=191.4718, s2=192.0606, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:41,226] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:27] Candidates: [5, 2, 1, 4]
[2025-04-16 22:41:41,415] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:176, R:4] s1=193.2861, s2=192.4924, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,438] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:296, R:4] s1=189.7923, s2=189.1863, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,457] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:266, R:4] s1=186.7872, s2=184.7601, ratio=1.0110 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,490] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:115, R:7] s1=77.8450, s2=77.5856, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,492] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:116] Candidates: [0, 6, 1, 2]
[2025-04-16 22:41:41,498] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:56] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:41,534] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:146] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:41,575] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:205, R:2] s1=142.0173, s2=141.3637, ratio=1.0046 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,577] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:206] Candidates: [5, 2, 6, 0]
[2025-04-16 22:41:41,581] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:446] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:41,599] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:86] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:41,692] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:355, R:6] s1=145.9375, s2=145.6696, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,693] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:356] Candidates: [6, 4, 0, 2]
[2025-04-16 22:41:41,696] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:416, R:6] s1=145.9592, s2=145.6512, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,726] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:236, R:6] s1=145.4470, s2=145.1402, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,728] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:237] Candidates: [0, 5, 3, 1]
[2025-04-16 22:41:41,781] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:385, R:6] s1=145.8948, s2=145.5781, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,783] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:386] Candidates: [3, 1, 0, 4]
[2025-04-16 22:41:41,830] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:326, R:5] s1=193.1976, s2=192.5145, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:41,832] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:327] Candidates: [2, 0, 5, 7]
[2025-04-16 22:41:41,898] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:27] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:41,965] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:116] Selected rings for extraction: [6, 2]
[2025-04-16 22:41:42,009] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:206] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:42,110] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:176, R:7] s1=77.6007, s2=77.4525, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,113] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:177] Candidates: [6, 3, 0, 7]
[2025-04-16 22:41:42,202] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:237] Selected rings for extraction: [3, 5]
[2025-04-16 22:41:42,249] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:146, R:4] s1=192.1225, s2=192.7409, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:42,249] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:266, R:5] s1=192.0036, s2=191.7980, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,269] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:267] Candidates: [0, 1, 3, 4]
[2025-04-16 22:41:42,352] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:356] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:42,368] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:386] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:42,407] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:296, R:5] s1=192.4166, s2=191.9912, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,408] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:297] Candidates: [3, 4, 0, 6]
[2025-04-16 22:41:42,411] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:86, R:4] s1=190.7325, s2=190.1465, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,433] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:56, R:4] s1=190.6372, s2=190.0615, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,458] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:327] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:42,468] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:446, R:4] s1=192.1409, s2=191.5866, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,571] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:416, R:3] s1=167.7342, s2=167.1402, ratio=1.0036 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,574] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:417] Candidates: [0, 7, 4, 2]
[2025-04-16 22:41:42,673] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:177] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:42,706] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:116, R:6] s1=145.1761, s2=145.4037, ratio=0.9984 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:42,789] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:27, R:4] s1=188.5266, s2=189.0567, ratio=0.9972 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:42,823] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:267] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:42,875] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:206, R:6] s1=145.3211, s2=145.0161, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:42,894] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:297] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:43,003] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:146, R:6] s1=145.1554, s2=145.3383, ratio=0.9987 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:43,005] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:147] Candidates: [3, 6, 0, 1]
[2025-04-16 22:41:43,048] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:237, R:3] s1=166.2545, s2=165.6762, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,102] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:417] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:43,106] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:356, R:4] s1=192.2944, s2=191.5676, ratio=1.0038 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,116] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:386, R:4] s1=192.0995, s2=191.5688, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,234] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:327, R:7] s1=78.3195, s2=78.0803, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,272] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:86, R:6] s1=145.7127, s2=145.4925, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,274] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:87] Candidates: [6, 1, 0, 3]
[2025-04-16 22:41:43,275] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:446, R:7] s1=78.1870, s2=78.0450, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,277] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:447] Candidates: [1, 4, 6, 3]
[2025-04-16 22:41:43,280] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:56, R:3] s1=166.3242, s2=165.7741, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,282] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:57] Candidates: [1, 3, 5, 4]
[2025-04-16 22:41:43,371] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:177, R:6] s1=145.2668, s2=145.0688, ratio=1.0014 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,395] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:147] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:43,478] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:116, R:2] s1=141.3451, s2=141.8871, ratio=0.9962 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:43,480] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:117] Candidates: [5, 6, 7, 2]
[2025-04-16 22:41:43,532] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:27, R:5] s1=191.4725, s2=192.0573, ratio=0.9970 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:43,534] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:28] Candidates: [5, 1, 6, 7]
[2025-04-16 22:41:43,641] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:267, R:4] s1=184.7351, s2=183.9631, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,661] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:297, R:4] s1=189.6876, s2=188.9142, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,703] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:206, R:5] s1=195.5143, s2=194.8309, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,705] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:207] Candidates: [1, 2, 0, 4]
[2025-04-16 22:41:43,725] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:237, R:5] s1=192.0145, s2=191.4645, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,727] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:238] Candidates: [2, 3, 1, 6]
[2025-04-16 22:41:43,822] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:57] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:43,825] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:87] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:43,833] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:447] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:43,878] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:417, R:4] s1=192.0609, s2=191.4548, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,936] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:28] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:43,942] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:117] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:43,975] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:356, R:6] s1=145.9380, s2=145.6890, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:43,977] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:357] Candidates: [6, 0, 4, 5]
[2025-04-16 22:41:44,060] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:386, R:3] s1=167.0269, s2=167.5879, ratio=0.9967 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:44,061] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:387] Candidates: [6, 7, 2, 1]
[2025-04-16 22:41:44,143] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:327, R:5] s1=193.0707, s2=192.5459, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,146] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:177, R:7] s1=77.6101, s2=77.4413, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,147] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:328] Candidates: [3, 4, 5, 7]
[2025-04-16 22:41:44,152] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:178] Candidates: [3, 5, 1, 6]
[2025-04-16 22:41:44,185] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:147, R:6] s1=145.3391, s2=145.1735, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,191] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:207] Selected rings for extraction: [4, 2]
[2025-04-16 22:41:44,221] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:238] Selected rings for extraction: [3, 6]
[2025-04-16 22:41:44,420] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:297, R:6] s1=145.7366, s2=145.2526, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,423] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:298] Candidates: [3, 0, 1, 5]
[2025-04-16 22:41:44,427] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:267, R:3] s1=162.1114, s2=162.6393, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:44,431] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:268] Candidates: [3, 7, 4, 1]
[2025-04-16 22:41:44,553] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:57, R:4] s1=190.6487, s2=190.0728, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,675] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:357] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:44,715] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:87, R:6] s1=145.7177, s2=145.4980, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,724] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:417, R:7] s1=78.3091, s2=78.1393, ratio=1.0022 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,726] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:418] Candidates: [0, 4, 1, 6]
[2025-04-16 22:41:44,746] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:28, R:5] s1=192.0595, s2=191.5020, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,754] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:387] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:44,758] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:447, R:4] s1=192.1616, s2=191.5968, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,770] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:178] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:44,773] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:328] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:44,855] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:117, R:6] s1=145.4024, s2=145.1848, ratio=1.0015 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:44,926] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:298] Selected rings for extraction: [5, 3]
[2025-04-16 22:41:44,927] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:268] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:44,956] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:207, R:4] s1=193.2170, s2=192.4567, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,048] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:147, R:3] s1=167.3827, s2=166.7959, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,050] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:148] Candidates: [4, 0, 7, 3]
[2025-04-16 22:41:45,111] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:238, R:3] s1=166.0932, s2=165.6630, ratio=1.0026 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,216] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:418] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:45,300] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:57, R:5] s1=193.1139, s2=192.5310, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,302] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:58] Candidates: [7, 6, 0, 5]
[2025-04-16 22:41:45,423] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:87, R:3] s1=166.0355, s2=165.3860, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,427] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:88] Candidates: [1, 3, 7, 2]
[2025-04-16 22:41:45,456] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:357, R:4] s1=192.1500, s2=191.5691, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,468] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:148] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:45,494] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:387, R:6] s1=145.5732, s2=145.8336, ratio=0.9982 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:45,536] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:328, R:4] s1=192.0944, s2=191.5197, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,563] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:28, R:7] s1=78.0043, s2=77.7651, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,566] D ThreadPoolExecutor-0_0 - get_fixed_pseudo_random_rings:200 - [P:29] Candidates: [6, 3, 7, 5]
[2025-04-16 22:41:45,588] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:447, R:6] s1=145.8922, s2=145.6019, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,590] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:448] Candidates: [7, 1, 4, 3]
[2025-04-16 22:41:45,597] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:178, R:6] s1=145.2504, s2=145.0239, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,663] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:117, R:7] s1=77.6111, s2=77.7905, ratio=0.9977 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:45,667] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:118] Candidates: [4, 0, 3, 1]
[2025-04-16 22:41:45,722] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:298, R:5] s1=192.3158, s2=191.8057, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,751] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:58] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:45,787] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:207, R:2] s1=141.9450, s2=141.3791, ratio=1.0040 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,793] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:208] Candidates: [2, 0, 1, 3]
[2025-04-16 22:41:45,804] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:268, R:4] s1=184.1404, s2=184.9793, ratio=0.9955 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:45,835] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:238, R:6] s1=145.4602, s2=145.1557, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:45,837] D ThreadPoolExecutor-0_7 - get_fixed_pseudo_random_rings:200 - [P:239] Candidates: [0, 3, 1, 2]
[2025-04-16 22:41:45,985] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:88] Selected rings for extraction: [7, 3]
[2025-04-16 22:41:46,187] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:418, R:4] s1=192.0145, s2=191.4326, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:46,215] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:448] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:46,239] I ThreadPoolExecutor-0_0 - _extract_single_pair_task:476 - [P:29] Selected rings for extraction: [5, 7]
[2025-04-16 22:41:46,249] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:118] Selected rings for extraction: [4, 3]
[2025-04-16 22:41:46,609] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:357, R:6] s1=145.9351, s2=145.6602, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:46,625] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:358] Candidates: [3, 0, 2, 6]
[2025-04-16 22:41:46,636] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:148, R:4] s1=192.1178, s2=192.7356, ratio=0.9968 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:46,780] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:387, R:7] s1=78.2303, s2=78.0386, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:46,792] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:388] Candidates: [4, 6, 7, 1]
[2025-04-16 22:41:46,850] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:208] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:46,854] I ThreadPoolExecutor-0_7 - _extract_single_pair_task:476 - [P:239] Selected rings for extraction: [3, 2]
[2025-04-16 22:41:46,866] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:328, R:7] s1=78.2986, s2=78.0670, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:46,872] D ThreadPoolExecutor-0_10 - get_fixed_pseudo_random_rings:200 - [P:329] Candidates: [2, 6, 1, 5]
[2025-04-16 22:41:46,898] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:178, R:5] s1=195.4427, s2=194.8608, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:46,901] D ThreadPoolExecutor-0_5 - get_fixed_pseudo_random_rings:200 - [P:179] Candidates: [4, 6, 2, 7]
[2025-04-16 22:41:47,108] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:58, R:7] s1=77.8587, s2=77.6518, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,161] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:298, R:3] s1=166.4202, s2=165.7214, ratio=1.0042 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,163] D ThreadPoolExecutor-0_9 - get_fixed_pseudo_random_rings:200 - [P:299] Candidates: [7, 2, 4, 6]
[2025-04-16 22:41:47,276] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:268, R:3] s1=162.6707, s2=163.2570, ratio=0.9964 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:47,279] D ThreadPoolExecutor-0_8 - get_fixed_pseudo_random_rings:200 - [P:269] Candidates: [5, 4, 7, 1]
[2025-04-16 22:41:47,492] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:88, R:7] s1=78.0462, s2=77.8821, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,526] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:358] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:47,539] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:418, R:6] s1=145.9492, s2=145.7146, ratio=1.0016 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,542] D ThreadPoolExecutor-0_13 - get_fixed_pseudo_random_rings:200 - [P:419] Candidates: [6, 5, 7, 2]
[2025-04-16 22:41:47,547] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:448, R:4] s1=192.1798, s2=191.6069, ratio=1.0030 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,557] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:388] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:47,596] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:118, R:4] s1=191.9753, s2=191.2363, ratio=1.0039 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,602] I ThreadPoolExecutor-0_10 - _extract_single_pair_task:476 - [P:329] Selected rings for extraction: [6, 5]
[2025-04-16 22:41:47,627] I ThreadPoolExecutor-0_5 - _extract_single_pair_task:476 - [P:179] Selected rings for extraction: [6, 4]
[2025-04-16 22:41:47,663] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:29, R:5] s1=191.6793, s2=192.2773, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:47,813] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:148, R:7] s1=77.5525, s2=77.4197, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,816] D ThreadPoolExecutor-0_4 - get_fixed_pseudo_random_rings:200 - [P:149] Candidates: [2, 6, 3, 4]
[2025-04-16 22:41:47,884] I ThreadPoolExecutor-0_9 - _extract_single_pair_task:476 - [P:299] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:47,927] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:239, R:3] s1=166.0927, s2=165.6388, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,939] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:208, R:3] s1=167.6027, s2=167.0759, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:47,971] I ThreadPoolExecutor-0_8 - _extract_single_pair_task:476 - [P:269] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:48,191] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:58, R:5] s1=193.0997, s2=192.5566, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,193] D ThreadPoolExecutor-0_1 - get_fixed_pseudo_random_rings:200 - [P:59] Candidates: [7, 2, 0, 4]
[2025-04-16 22:41:48,211] I ThreadPoolExecutor-0_13 - _extract_single_pair_task:476 - [P:419] Selected rings for extraction: [6, 7]
[2025-04-16 22:41:48,363] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:88, R:3] s1=165.9366, s2=165.4026, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,365] D ThreadPoolExecutor-0_2 - get_fixed_pseudo_random_rings:200 - [P:89] Candidates: [5, 6, 7, 3]
[2025-04-16 22:41:48,403] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:388, R:4] s1=192.0974, s2=191.5494, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,418] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:329, R:6] s1=145.8268, s2=145.4838, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,487] I ThreadPoolExecutor-0_4 - _extract_single_pair_task:476 - [P:149] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:48,513] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:448, R:7] s1=78.2378, s2=78.0894, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,517] D ThreadPoolExecutor-0_14 - get_fixed_pseudo_random_rings:200 - [P:449] Candidates: [4, 5, 3, 0]
[2025-04-16 22:41:48,548] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:179, R:6] s1=145.2227, s2=145.0609, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,566] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:118, R:3] s1=165.8191, s2=166.3351, ratio=0.9969 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:48,568] D ThreadPoolExecutor-0_3 - get_fixed_pseudo_random_rings:200 - [P:119] Candidates: [2, 3, 6, 4]
[2025-04-16 22:41:48,576] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:358, R:6] s1=145.9059, s2=145.6212, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,597] I ThreadPoolExecutor-0_1 - _extract_single_pair_task:476 - [P:59] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:48,608] D ThreadPoolExecutor-0_0 - extract_single_bit:349 - [P:29, R:7] s1=78.2742, s2=78.0884, ratio=1.0024 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,709] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:299, R:4] s1=189.1771, s2=190.3818, ratio=0.9937 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:48,733] D ThreadPoolExecutor-0_7 - extract_single_bit:349 - [P:239, R:2] s1=142.2643, s2=141.5448, ratio=1.0051 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,796] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:269, R:4] s1=188.3831, s2=188.2515, ratio=1.0007 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,879] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:208, R:2] s1=141.9439, s2=141.3686, ratio=1.0041 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:48,881] D ThreadPoolExecutor-0_6 - get_fixed_pseudo_random_rings:200 - [P:209] Candidates: [2, 7, 3, 5]
[2025-04-16 22:41:48,921] I ThreadPoolExecutor-0_2 - _extract_single_pair_task:476 - [P:89] Selected rings for extraction: [5, 6]
[2025-04-16 22:41:49,000] I ThreadPoolExecutor-0_14 - _extract_single_pair_task:476 - [P:449] Selected rings for extraction: [4, 5]
[2025-04-16 22:41:49,005] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:419, R:6] s1=145.9253, s2=145.6493, ratio=1.0019 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,034] I ThreadPoolExecutor-0_3 - _extract_single_pair_task:476 - [P:119] Selected rings for extraction: [4, 6]
[2025-04-16 22:41:49,156] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:388, R:7] s1=78.1871, s2=78.0439, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,158] D ThreadPoolExecutor-0_12 - get_fixed_pseudo_random_rings:200 - [P:389] Candidates: [1, 6, 3, 2]
[2025-04-16 22:41:49,184] D ThreadPoolExecutor-0_10 - extract_single_bit:349 - [P:329, R:5] s1=193.0597, s2=192.5005, ratio=1.0029 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,243] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:358, R:3] s1=167.8052, s2=167.0875, ratio=1.0043 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,244] D ThreadPoolExecutor-0_11 - get_fixed_pseudo_random_rings:200 - [P:359] Candidates: [5, 0, 4, 7]
[2025-04-16 22:41:49,302] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:149, R:4] s1=192.7356, s2=192.1211, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,306] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:59, R:4] s1=191.1127, s2=190.7359, ratio=1.0020 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,334] D ThreadPoolExecutor-0_5 - extract_single_bit:349 - [P:179, R:4] s1=193.1203, s2=192.4961, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,360] I ThreadPoolExecutor-0_6 - _extract_single_pair_task:476 - [P:209] Selected rings for extraction: [7, 5]
[2025-04-16 22:41:49,423] D ThreadPoolExecutor-0_9 - extract_single_bit:349 - [P:299, R:7] s1=78.3546, s2=77.9315, ratio=1.0054 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,457] D ThreadPoolExecutor-0_8 - extract_single_bit:349 - [P:269, R:5] s1=188.8794, s2=190.2332, ratio=0.9929 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:49,609] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:449, R:4] s1=192.1967, s2=191.5877, ratio=1.0032 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,640] D ThreadPoolExecutor-0_13 - extract_single_bit:349 - [P:419, R:7] s1=78.3553, s2=78.1635, ratio=1.0025 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,654] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:119, R:4] s1=191.2339, s2=191.8587, ratio=0.9967 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:49,661] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:89, R:5] s1=193.6662, s2=193.1368, ratio=1.0027 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,686] I ThreadPoolExecutor-0_12 - _extract_single_pair_task:476 - [P:389] Selected rings for extraction: [6, 3]
[2025-04-16 22:41:49,715] I ThreadPoolExecutor-0_11 - _extract_single_pair_task:476 - [P:359] Selected rings for extraction: [4, 7]
[2025-04-16 22:41:49,722] D ThreadPoolExecutor-0_4 - extract_single_bit:349 - [P:149, R:6] s1=145.1751, s2=145.3364, ratio=0.9989 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:49,727] D ThreadPoolExecutor-0_1 - extract_single_bit:349 - [P:59, R:7] s1=77.9147, s2=77.8283, ratio=1.0011 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,848] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:209, R:7] s1=77.6082, s2=77.4654, ratio=1.0018 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:49,921] D ThreadPoolExecutor-0_3 - extract_single_bit:349 - [P:119, R:6] s1=145.1814, s2=145.3989, ratio=0.9985 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:49,927] D ThreadPoolExecutor-0_2 - extract_single_bit:349 - [P:89, R:6] s1=145.7232, s2=145.4690, ratio=1.0017 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,026] D ThreadPoolExecutor-0_14 - extract_single_bit:349 - [P:449, R:5] s1=193.2561, s2=192.6230, ratio=1.0033 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,030] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:389, R:6] s1=145.6541, s2=145.9457, ratio=0.9980 vs thr=1.0 -> Bit=1
[2025-04-16 22:41:50,035] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:359, R:4] s1=192.1125, s2=191.5171, ratio=1.0031 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,076] D ThreadPoolExecutor-0_6 - extract_single_bit:349 - [P:209, R:5] s1=195.4806, s2=194.9399, ratio=1.0028 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,162] D ThreadPoolExecutor-0_11 - extract_single_bit:349 - [P:359, R:7] s1=78.2586, s2=78.0984, ratio=1.0021 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,169] D ThreadPoolExecutor-0_12 - extract_single_bit:349 - [P:389, R:3] s1=167.6151, s2=167.0338, ratio=1.0035 vs thr=1.0 -> Bit=0
[2025-04-16 22:41:50,172] I MainThread - extract_watermark_from_video:649 - Extraction finished. Processed pairs:453. Pairs with errors/Nones:0.
[2025-04-16 22:41:50,173] I MainThread - extract_watermark_from_video:665 - Total bits collected: 906. Valid bits: 906 (100.0% success rate).
[2025-04-16 22:41:50,173] I MainThread - extract_watermark_from_video:672 - Attempting to decode 3 potential packets (255 bits each) from valid bits...
[2025-04-16 22:41:50,174] I MainThread - decode_ecc:253 - Galois ECC: Декодировано, исправлено 2 ошибок.
[2025-04-16 22:41:50,174] D MainThread - decode_ecc:267 - Decode ECC: Extracted 64 payload bits.
[2025-04-16 22:41:50,175] I MainThread - decode_ecc:253 - Galois ECC: Декодировано, исправлено -1 ошибок.
[2025-04-16 22:41:50,175] D MainThread - decode_ecc:267 - Decode ECC: Extracted 64 payload bits.
[2025-04-16 22:41:50,176] I MainThread - decode_ecc:253 - Galois ECC: Декодировано, исправлено -1 ошибок.
[2025-04-16 22:41:50,176] D MainThread - decode_ecc:267 - Decode ECC: Extracted 64 payload bits.
[2025-04-16 22:41:50,176] I MainThread - extract_watermark_from_video:696 - Decode summary: Success=3, Failed=0. Total ECC symbol corrections: 2.
[2025-04-16 22:41:50,177] I MainThread - extract_watermark_from_video:700 - Voting results:
[2025-04-16 22:41:50,177] I MainThread - extract_watermark_from_video:701 -   ID 48a49976264bdf27: 2 votes
[2025-04-16 22:41:50,177] I MainThread - extract_watermark_from_video:701 -   ID 48e89076278bdf27: 1 votes
[2025-04-16 22:41:50,177] I MainThread - extract_watermark_from_video:704 - Winner selected: 48a49976264bdf27 with 2/3 votes (66.7%).
[2025-04-16 22:41:50,177] I MainThread - extract_watermark_from_video:707 - Extraction done. Total time: 70.71 sec.
[2025-04-16 22:41:50,177] I MainThread - main:747 - Decoded ID: 48a49976264bdf27
[2025-04-16 22:41:50,178] I MainThread - main:754 - ID MATCH.
[2025-04-16 22:41:50,178] I MainThread - main:758 - --- Extraction Main Process Finished ---
[2025-04-16 22:41:50,178] I MainThread - main:759 - --- Total Extractor Time: 72.26 sec ---
[2025-04-16 22:41:50,326] I MainThread - <module>:840 - Profiling stats saved: profile_extract_opencl_batched_galois_t4.txt
[2025-04-16 22:41:50,326] I MainThread - <module>:848 - Attempting to restore original dtcwt backend: numpy
[2025-04-16 22:41:50,326] I MainThread - <module>:860 - DTCWT backend restored to: numpy
