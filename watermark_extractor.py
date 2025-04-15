# Файл: extractor.py (Версия: Galois BCH, m=8, t=4, Syntax Fix V2)
import cv2
import numpy as np
import random
import logging
import time
import json # Импорт json нужен для load_saved_rings (хотя он больше не используется в слепом режиме)
import os
import imagehash
import hashlib
from PIL import Image
from scipy.fftpack import dct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import concurrent.futures
import uuid
from math import ceil

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    print("galois: импортирован.")
    _test_bch_ok = False
    _test_decode_ok = False
    BCH_CODE_OBJECT = None # Инициализируем заранее
    try:
        # Используем параметры, которые точно должны работать (m=8, t=4)
        _test_m = 8; _test_t = 4; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        if _test_bch_galois.t == _test_t:
             print(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) initialized OK.")
             _test_bch_ok = True
             BCH_CODE_OBJECT = _test_bch_galois # Сохраняем объект
        else: print(f"galois BCH init mismatch: t={_test_bch_galois.t}")

        # Проверяем decode с data и recv_ecc, если инициализация ОК
        if _test_bch_ok:
            _n_bits = _test_bch_galois.n
            _dummy_cw_bits = np.zeros(_n_bits, dtype=np.uint8)

            # --- ИСПРАВЛЕНО: Используем класс поля GF2 ---
            GF2 = galois.GF(2) # Получаем класс поля GF(2)
            _dummy_cw_vec = GF2(_dummy_cw_bits) # Создаем FieldArray через класс
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            # Вызываем decode
            _msg, _flips = _test_bch_galois.decode(_dummy_cw_vec, errors=True)
            if _flips is not None:
                 print(f"galois: decode(codeword) test OK (returned flips={_flips}).")
                 _test_decode_ok = True
            else:
                 print("galois: decode(codeword) returned None during test?") # Странно

    except AttributeError as ae: print(f"galois: ОШИБКА атрибута/метода: {ae}")
    except ValueError as ve: print(f"galois: ОШИБКА ValueError: {ve}")
    except TypeError as te: print(f"galois: ОШИБКА TypeError (возможно, 'field'): {te}")
    except Exception as test_err: print(f"galois: ОШИБКА теста: {test_err}")

    GALOIS_AVAILABLE = _test_bch_ok and _test_decode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None # Сброс если тесты не прошли
    if GALOIS_AVAILABLE: print("galois: Тесты инициализации и decode(codeword) пройдены.")
    else: print("galois: Не прошел базовые тесты. ECC будет отключен.")

except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print(f"galois: Ошибка импорта: {import_err}")

# --- Остальные импорты ---
import cProfile
import pstats
from collections import Counter

# --- Основные Параметры ---
LAMBDA_PARAM: float = 0.1; ALPHA_MIN: float = 1.005; ALPHA_MAX: float = 1.12; N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0; EMBED_COMPONENT: int = 2; CANDIDATE_POOL_SIZE: int = 4
BITS_PER_PAIR: int = 2; NUM_RINGS_TO_USE: int = BITS_PER_PAIR; RING_SELECTION_METHOD: str = 'pool_entropy_selection'
PAYLOAD_LEN_BYTES: int = 8; USE_ECC: bool = True; BCH_M: int = 8; BCH_T: int = 4; MAX_PACKET_REPEATS: int = 5
FPS: int = 30; LOG_FILENAME: str = 'watermarking_extract.log'; INPUT_EXTENSION: str = '.avi'
ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'; MAX_WORKERS_EXTRACT: Optional[int] = None

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO, format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Извлечения (Метод: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}) ---")
logging.info(f"Ожид. Payload: {PAYLOAD_LEN_BYTES * 8}bit, Ожид. ECC: {USE_ECC} (Galois BCH m={BCH_M}, t={BCH_T}), Доступно/Работает: {GALOIS_AVAILABLE}, Max Repeats: {MAX_PACKET_REPEATS}")
logging.info(f"Ожид. Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC ожидается, но galois недоступна/не работает! Декодирование ECC невозможно.")
elif not USE_ECC: logging.info("ECC не ожидается.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error(f"NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE!")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning(f"NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции ---
def dct_1d(signal_1d: np.ndarray) -> np.ndarray: return dct(signal_1d, type=2, norm='ortho')

# --- ИСПРАВЛЕННЫЙ СИНТАКСИС dtcwt_transform ---
def dtcwt_transform(y_plane: np.ndarray, frame_number: int = -1) -> Optional[Pyramid]:
    if np.any(np.isnan(y_plane)): pass
    try:
        t = Transform2d(); rows, cols = y_plane.shape; pad_rows = rows % 2 != 0; pad_cols = cols % 2 != 0
        if pad_rows or pad_cols: yp = np.pad(y_plane, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        else: yp = y_plane
        py = t.forward(yp.astype(np.float32), nlevels=1)
        # --- ИСПРАВЛЕНО: Убрано присваивание внутри if ---
        if hasattr(py, 'lowpass') and py.lowpass is not None:
            return py
        else:
            # logging.error(f"[F:{frame_number}] DTCWT no lowpass.") # Можно раскомментировать лог
            return None
    # --- ИСПРАВЛЕНО: Добавлено 'as e' ---
    except Exception as e:
        logging.error(f"[F:{frame_number}] DTCWT transform error: {e}")
        return None

# --- Функции Работы с Кольцами ---
@functools.lru_cache(maxsize=8)
def _ring_division_internal(subband_shape: Tuple[int, int], n_rings: int) -> List[Optional[np.ndarray]]:
    H, W = subband_shape
    if H < 2 or W < 2:
        return [None] * n_rings
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    rr, cc = np.indices((H, W), dtype=np.float32) # Используем cc
    distances = np.sqrt((rr - center_r)**2 + (cc - center_c)**2)
    min_dist, max_dist = np.min(distances), np.max(distances)

    if max_dist < 1e-6:
        ring_bins = np.array([0.0, 1.0])
        n_rings_eff = 1
    else:
        ring_bins = np.linspace(0.0, max_dist + 1e-6, n_rings + 1)
        n_rings_eff = n_rings

    if len(ring_bins) < 2:
        return [None] * n_rings

    ring_indices = np.digitize(distances, ring_bins) - 1
    ring_indices[distances < ring_bins[1]] = 0
    ring_indices = np.clip(ring_indices, 0, n_rings_eff - 1)

    # --- ИСПРАВЛЕНО: Возвращаем аннотацию и форматируем цикл ---
    rc: List[Optional[np.ndarray]] = [None] * n_rings # Возвращаем аннотацию
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0:
            rc[rdx] = coords
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    return rc
@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(subband_shape:Tuple[int,int], n_rings:int)->List[Optional[np.ndarray]]: return _ring_division_internal(subband_shape,n_rings)
def ring_division(lp:np.ndarray, nr:int=N_RINGS, fn:int=-1)->List[Optional[np.ndarray]]:
    if not isinstance(lp,np.ndarray) or lp.ndim!=2: return [None]*nr
    sh=lp.shape;
    try:
        cl=get_ring_coords_cached(sh,nr);
        if not isinstance(cl,list) or not all(isinstance(i,(np.ndarray,type(None))) for i in cl):
             get_ring_coords_cached.cache_clear(); cl=_ring_division_internal(sh,nr)
        return [a.copy() if a is not None else None for a in cl]
    except Exception as e: # Добавлено 'as e'
        logging.error(f"[F:{fn}] Ring division error: {e}")
        return [None]*nr

# --- Остальные функции (calculate_entropies, compute_adaptive_alpha_entropy, get_fixed_pseudo_random_rings, bits_to_bytes, decode_ecc, read_video, extract_frame_pair, extract_frame_pair_multi_ring, _extract_frame_pair_worker, extract_watermark_from_video - без изменений синтаксиса) ---
def calculate_entropies(rv:np.ndarray, fn:int=-1, ri:int=-1)->Tuple[float,float]:
    eps=1e-12; ve=0.; ee=0.;
    if rv.size>0:
        mn,mx=np.min(rv),np.max(rv);
        if mn<0. or mx>1.: rvc=np.clip(rv,0.,1.)
        else: rvc=rv
        h,_=np.histogram(rvc,bins=256,range=(0.,1.),density=False); tc=rvc.size;
        if tc>0: p=h/tc; p=p[p>eps];
        if p.size>0: ve=-np.sum(p*np.log2(p)); ee=-np.sum(p*np.exp(1.-p))
    return ve,ee
def compute_adaptive_alpha_entropy(rv:np.ndarray, ri:int, fn:int)->float:
    if rv.size<10: return ALPHA_MIN
    ve,_=calculate_entropies(rv,fn,ri); lv=np.var(rv);
    en=np.clip(ve/MAX_THEORETICAL_ENTROPY,0.,1.); vmp=0.005; vsc=500;
    tn=1./(1.+np.exp(-vsc*(lv-vmp))); we=.6; wt=.4;
    mf=np.clip((we*en+wt*tn),0.,1.); fa=ALPHA_MIN+(ALPHA_MAX-ALPHA_MIN)*mf;
    return np.clip(fa,ALPHA_MIN,ALPHA_MAX)
def get_fixed_pseudo_random_rings(pi:int, nr:int, ps:int)->List[int]:
    if ps<=0: return []
    if ps>nr: ps=nr
    sd=str(pi).encode('utf-8'); hd=hashlib.sha256(sd).digest(); sv=int.from_bytes(hd,'big'); prng=random.Random(sv)
    try: ci=prng.sample(range(nr),ps)
    except ValueError: ci=list(range(nr))
    logging.debug(f"[P:{pi}] Candidates: {ci}"); return ci
def bits_to_bytes(b:List[Optional[int]])->Optional[bytearray]:
    vb=[x for x in b if x is not None]; # Отфильтровали None (здесь их быть не должно)
    if not vb: return bytearray()       # Если список пуст
    nb=len(vb);                         # nb = 64
    nB=ceil(nb/8.);                     # nB = ceil(64/8) = 8
    pl=nB*8-nb;                         # pl = 8*8 - 64 = 0
    if pl>0: vb.extend([0]*int(pl))     # Паддинг не добавляется
    ba=bytearray();
    for i in range(0,len(vb),8):        # Цикл от 0 до 64 с шагом 8 (0, 8, 16, ... 56) - правильно
        bc=vb[i:i+8];                   # Берем 8 бит
        try:
             bv=int("".join(map(str,bc)),2); # Конвертируем '1010...' в число
             ba.append(bv)                  # Добавляем байт
        except Exception: return None       # Ошибка конвертации?
    return ba                           # Должен вернуть 8 байт
def decode_ecc(packet_bytes_in: bytearray, bch_code: galois.BCH, expected_data_len_bytes: int) -> Tuple[Optional[bytes], int]:
    """
    Декодирует пакет с использованием Galois. Добавлено логирование для bits_to_bytes.
    """
    # ... (начало функции без изменений) ...
    if not GALOIS_AVAILABLE or bch_code is None:
        # ... (обработка как раньше) ...
        if len(packet_bytes_in) >= expected_data_len_bytes: return bytes(packet_bytes_in[:expected_data_len_bytes]), 0
        else: return None, -1
    n = bch_code.n; k = bch_code.k
    expected_payload_len_bits = expected_data_len_bytes * 8
    if expected_payload_len_bits > k: return None, -1
    expected_packet_len_bytes = ceil(n / 8.0); current_packet_len_bytes = len(packet_bytes_in)
    packet_bytes = bytearray(packet_bytes_in)
    if current_packet_len_bytes < expected_packet_len_bytes:
        padding = expected_packet_len_bytes - current_packet_len_bytes; packet_bytes.extend([0] * padding)
    elif current_packet_len_bytes > expected_packet_len_bytes:
         packet_bytes = packet_bytes[:expected_packet_len_bytes]
    try:
        packet_bits = np.unpackbits(np.frombuffer(packet_bytes, dtype=np.uint8)); packet_bits = packet_bits[:n];
        if packet_bits.size != n: raise ValueError(f"Bit conv error: {packet_bits.size}!=n {n}")
        GF = bch_code.field; received_vector = GF(packet_bits);
        try:
            corrected_message_vector, N_corrected_symbols = bch_code.decode(received_vector, errors=True)
        except galois.errors.UncorrectableError: return None, -1

        if N_corrected_symbols == -1: return None, -1
        else:
             logging.info(f"Galois ECC: Corrected {N_corrected_symbols} symbol errors.")
             corrected_k_bits = corrected_message_vector.view(np.ndarray).astype(np.uint8)

             if corrected_k_bits.size >= expected_payload_len_bits:
                 corrected_payload_bits = corrected_k_bits[:expected_payload_len_bits]
                 # --- ДОБАВЛЕНО ЛОГИРОВАНИЕ ---
                 logging.debug(f"Decode ECC: Extracted {len(corrected_payload_bits)} payload bits: {''.join(map(str, corrected_payload_bits.tolist()))}")
                 # --- КОНЕЦ ЛОГИРОВАНИЯ ---
             else:
                 logging.error(f"Corrected k bits len {corrected_k_bits.size} < payload bits {expected_payload_len_bits}")
                 return None, -1

             corrected_payload_bytes = bits_to_bytes(corrected_payload_bits.tolist())

             # --- ДОБАВЛЕНО ЛОГИРОВАНИЕ ---
             if corrected_payload_bytes is None:
                 logging.error("bits_to_bytes returned None!")
                 return None, -1
             else:
                 logging.debug(f"Decode ECC: bits_to_bytes returned {len(corrected_payload_bytes)} bytes: {corrected_payload_bytes.hex()}")
             # --- КОНЕЦ ЛОГИРОВАНИЯ ---


             if len(corrected_payload_bytes) == expected_data_len_bytes:
                 return bytes(corrected_payload_bytes), N_corrected_symbols
             else:
                 # --- УБРАН ПАДДИНГ ЗДЕСЬ, ОШИБКА УЖЕ ПРОИЗОШЛА ---
                 logging.error(f"Final payload byte length {len(corrected_payload_bytes)} != expected {expected_data_len_bytes}")
                 return None, -1 # Возвращаем ошибку

    except Exception as e:
        logging.error(f"Exception during Galois ECC decoding: {e}", exc_info=True)
        return None, -1
# --- КОНЕЦ ИСПРАВЛЕННОЙ ФУНКЦИИ ---
def read_video(vp:str)->Tuple[List[np.ndarray],float]:
    logging.info(f"Reading: {vp}"); fr=[]; fps=float(FPS); cap=None; h,w=-1,-1
    try:
        cap=cv2.VideoCapture(vp);
        if not cap.isOpened(): return fr,fps
        fps=float(cap.get(cv2.CAP_PROP_FPS) or FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Props:{w}x{h}@{fps:.2f}~{fc}f"); h,w=h,w; rc,nc,ic=0,0,0
        while True:
            ret,f=cap.read();
            if not ret: break
            if f is None: nc+=1; continue
            if f.ndim==3 and f.shape[:2]==(h,w) and f.dtype==np.uint8: fr.append(f); rc+=1
            else: ic+=1
        logging.info(f"Read done. V:{rc},N:{nc},I:{ic}")
    except Exception as e: logging.error(f"Read error: {e}")
    finally:
        if cap: cap.release()
    if not fr: logging.error(f"No valid frames read.")
    return fr,fps
def extract_frame_pair(f1:np.ndarray, f2:np.ndarray, ri:int, nr:int=N_RINGS, fn:int=0, ec:int=EMBED_COMPONENT) -> Optional[int]:
    pn = fn // 2 # Индекс пары для логов
    try:
        if f1 is None or f2 is None or f1.shape!=f2.shape: return None
        if f1.dtype!=np.uint8: f1=np.clip(f1,0,255).astype(np.uint8)
        if f2.dtype!=np.uint8: f2=np.clip(f2,0,255).astype(np.uint8)
        try: y1=cv2.cvtColor(f1,cv2.COLOR_BGR2YCrCb); y2=cv2.cvtColor(f2,cv2.COLOR_BGR2YCrCb)
        except cv2.error: return None
        try: c1=y1[:,:,ec].astype(np.float32)/255.; c2=y2[:,:,ec].astype(np.float32)/255.
        except IndexError: return None
        p1=dtcwt_transform(c1,fn); p2=dtcwt_transform(c2,fn+1);
        if p1 is None or p2 is None or p1.lowpass is None or p2.lowpass is None: return None
        L1=p1.lowpass; L2=p2.lowpass;
        r1c=ring_division(L1,nr,fn); r2c=ring_division(L2,nr,fn+1);
        if not(0<=ri<nr and ri<len(r1c) and ri<len(r2c)): return None
        cd1=r1c[ri]; cd2=r2c[ri];
        if cd1 is None or cd2 is None or cd1.ndim!=2 or cd2.ndim!=2: return None
        try: rs1,cs1=cd1[:,0],cd1[:,1]; rv1=L1[rs1,cs1].astype(np.float32); rs2,cs2=cd2[:,0],cd2[:,1]; rv2=L2[rs2,cs2].astype(np.float32)
        except IndexError: return None
        if rv1.size==0 or rv2.size==0: return None
        if rv1.size!=rv2.size: sz=min(rv1.size,rv2.size); rv1=rv1[:sz]; rv2=rv2[:sz]
        d1=dct_1d(rv1); d2=dct_1d(rv2);
        try: S1=svd(d1.reshape(-1,1),compute_uv=False); S2=svd(d2.reshape(-1,1),compute_uv=False)
        except np.linalg.LinAlgError: return None
        s1=S1[0] if S1.size>0 else 0.; s2=S2[0] if S2.size>0 else 0.;
        a=compute_adaptive_alpha_entropy(rv1,ri,fn); eps=1e-12; thr=1.0; # Порог 1.0
        r=s1/(s2+eps); bit=0 if r>=thr else 1;

        # --- ВОЗВРАЩАЕМ ЛОГИРОВАНИЕ ---
        logging.info(f"[P:{pn}, R:{ri}] s1={s1:.4f}, s2={s2:.4f}, ratio={r:.4f} vs thr={thr:.4f} (alpha={a:.4f}) -> Bit={bit}")
        # --- КОНЕЦ ВОЗВРАЩЕНИЯ ---

        return bit
    except Exception as e:
        pn_err = fn // 2 if fn >=0 else -1
        logging.error(f"extract_frame_pair failed (P:{pn_err}, R:{ri}): {e}", exc_info=False) # Убрал exc_info чтобы не засорять лог
        return None
def extract_frame_pair_multi_ring(f1:np.ndarray, f2:np.ndarray, ris:List[int], nr:int=N_RINGS, fn:int=0, ec:int=EMBED_COMPONENT)->List[Optional[int]]:
    if not ris: return []
    return [extract_frame_pair(f1,f2,ri,nr,fn,ec) for ri in ris]
def _extract_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, List[Optional[int]]]:
    pi=args['pair_idx']; f1=args['frame1']; f2=args['frame2']; nr=args['n_rings']; nrtu=args['num_rings_to_use']; cps=args['candidate_pool_size']; ec=args['embed_component']; fn=2*pi; final_rings=[]
    try:
        cand_rings=get_fixed_pseudo_random_rings(pi,nr,cps);
        if len(cand_rings)<nrtu: return pi,[None]*nrtu
        try:
            comp1=f1[:,:,ec].astype(np.float32)/255.; p1=dtcwt_transform(comp1,fn);
            if p1 is None or p1.lowpass is None: raise RuntimeError("DTCWT L1 failed")
            L1=p1.lowpass; all_coords=ring_division(L1,nr,fn); c_entropies=[]; min_pix=10
            for r_idx in cand_rings:
                e=-float('inf');
                if 0<=r_idx<nr: coords=all_coords[r_idx];
                if coords is not None and coords.shape[0]>=min_pix:
                    try: r,c=coords[:,0],coords[:,1]; rv=L1[r,c]; ve,_=calculate_entropies(rv,fn,r_idx)
                    except Exception: pass
                    if np.isfinite(ve): e=ve
                c_entropies.append((e,r_idx))
            c_entropies.sort(key=lambda x:x[0], reverse=True); final_rings=[i for e,i in c_entropies if e>-float('inf')][:nrtu]
            if len(final_rings)<nrtu: return pi,[None]*nrtu
        except Exception as sel_err: return pi,[None]*nrtu
        bits=extract_frame_pair_multi_ring(f1,f2,final_rings,nr,fn,ec)
        if len(bits)!=nrtu: bits=[None]*nrtu; [bits.__setitem__(i, bits[i]) for i in range(min(len(bits),nrtu))]
        return pi,bits
    except Exception: return pi,[None]*nrtu
def extract_watermark_from_video(
        frames:List[np.ndarray], nr:int=N_RINGS, nrtu:int=NUM_RINGS_TO_USE, bp:int=BITS_PER_PAIR,
        cps:int=CANDIDATE_POOL_SIZE, ec:int=EMBED_COMPONENT, mpr:int=MAX_PACKET_REPEATS,
        ue:bool=USE_ECC, bm:int=BCH_M, bt:int=BCH_T, plb:int=PAYLOAD_LEN_BYTES,
        mw:Optional[int]=MAX_WORKERS_EXTRACT) -> Optional[bytes]:
    logging.info(f"Starting extraction (Pool+Select, Bits/Pair:{bp})")
    start=time.time(); nf=len(frames); tpa=nf//2; ppc=0; fpe=0;
    if tpa==0: logging.error("No frame pairs to process."); return None # Добавлена проверка

    p_len=plb*8; eff_ecc=False; n=-1; k=-1; ecc_l=-1;
    eff_ecc = ue and GALOIS_AVAILABLE # Флаг Galois
    bch_code_to_use = BCH_CODE_OBJECT if eff_ecc else None # Объект Galois

    # --- ИСПРАВЛЕННЫЙ БЛОК TRY...EXCEPT ---
    if eff_ecc and bch_code_to_use:
        try:
            n = bch_code_to_use.n
            k = bch_code_to_use.k
            ecc_l = n - k # Рассчитываем ecc_len
            logging.info(f"Galois BCH OK: n={n}, k={k}, t={bch_code_to_use.t}")
            if plb*8 <= k:
                p_len = n # Используем полную длину пакета
                logging.info(f"Expecting ECC packets ({p_len}b).")
            else:
                # Если payload не влезает в k, ECC не используется
                logging.warning(f"Payload size ({plb*8}) > Galois k ({k}). Disabling ECC.")
                p_len = plb*8 # Ожидаем только payload
                eff_ecc = False
                bch_code_to_use = None
        except Exception as e: # Ловим любую ошибку при доступе к атрибутам
            logging.error(f"Error getting Galois BCH params: {e}. Disabling ECC.")
            p_len = plb*8 # Возвращаемся к длине payload
            eff_ecc = False
            bch_code_to_use = None
    else:
        # Этот блок выполняется, если изначально eff_ecc был False
        p_len = plb*8
        logging.info(f"ECC disabled or unavailable. Expecting raw payload ({p_len}b).")
    # --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---

    if p_len<=0: logging.error("Calculated packet length is zero or negative."); return None

    pte=min(tpa, ceil(mpr*p_len/bp)); logging.info(f"Extracting from {pte} pairs (max repeats:{mpr}).");
    if pte==0: logging.warning("Zero pairs to extract."); return None

    # ... (остальная часть функции extract_watermark_from_video без изменений) ...
    ebpp:Dict[int,List[Optional[int]]]={}; tasks=[]; skipped=[]
    for i in range(pte):
        i1=2*i; i2=i1+1;
        if i2>=nf or frames[i1] is None or frames[i2] is None: skipped.append(i); continue
        args={'pair_idx':i,'frame1':frames[i1],'frame2':frames[i2],'n_rings':nr,'num_rings_to_use':nrtu,'candidate_pool_size':cps,'embed_component':ec}
        tasks.append(args)
    if not tasks: logging.error("No valid tasks created."); return None

    exec_cls=concurrent.futures.ThreadPoolExecutor; logging.info(f"Submitting {len(tasks)} tasks (mw={mw})...")
    try:
        with exec_cls(max_workers=mw) as executor:
            f2pi={executor.submit(_extract_frame_pair_worker,a):a['pair_idx'] for a in tasks}
            for f in concurrent.futures.as_completed(f2pi):
                pi=f2pi[f];
                try: _,brl=f.result(); ebpp[pi]=brl; ppc+=1
                except Exception as exc: logging.error(f"Pair {pi} worker exception: {exc}"); ebpp[pi]=[None]*bp; ppc+=1; fpe+=1
                if ebpp.get(pi) is None or None in ebpp.get(pi,[]): fpe+=1
    except Exception as e: logging.critical(f"Executor error: {e}")
    logging.info(f"Extraction finished. Processed:{ppc}. Pairs w/ errors:{fpe}."); eba:List[Optional[int]]=[]; tebc=0
    for i in range(pte): bl=ebpp.get(i,[None]*bp); eba.extend(bl); tebc+=len(bl)
    logging.info(f"Total bits collected: {tebc}")

    npp=tebc//p_len if p_len>0 else 0;
    if npp==0 and tebc>=p_len: npp=1
    logging.info(f"Decoding {npp} potential packets ({p_len} bits)..."); dp:List[bytes]=[]; dsc=0; dfc=0; dect=0
    for i in range(npp):
        si=i*p_len; ei=si+p_len;
        if ei>tebc: break
        pbl=eba[si:ei];
        if None in pbl: dfc+=1; continue
        pb=bits_to_bytes(pbl);
        if pb is None: dfc+=1; continue
        payload:Optional[bytes]=None; errors:int=-1
        if eff_ecc and bch_code_to_use is not None:
             # Передаем рассчитанные n, k, ecc_l в decode_ecc
            payload,errors = decode_ecc(pb, bch_code_to_use, plb) # Вызываем новую decode_ecc
        else:
            if len(pb)>=plb: payload=bytes(pb[:plb]); errors=0
            else: payload=None; errors=-1
        if payload is not None:
            if len(payload)==plb: dp.append(payload); dsc+=1;
            if errors>0: dect+=errors # Считаем символьные ошибки
            # else: dfc+=1 # Не считаем ошибкой, если 0 ошибок и payload есть
        else: dfc+=1
    logging.info(f"Decode summary: Success={dsc}, Failed={dfc}. ECC fixes(symbols):{dect}.");
    if not dp: logging.error("No valid payloads decoded."); return None # Добавлено сообщение
    pc=Counter(dp); logging.info("Voting results:");
    for pld,c in pc.most_common(): logging.info(f"  ID {pld.hex()}: {c} votes")
    mcp,wc=pc.most_common(1)[0]; conf=wc/dsc if dsc>0 else 0.
    logging.info(f"Winner selected with {wc}/{dsc} votes ({conf:.1%})."); fpb=mcp;
    end=time.time(); logging.info(f"Extraction done. Time: {end-start:.2f} sec.")
    return fpb
# --- Основная Функция ---
def main():
    start=time.time(); input_base="watermarked_galois_t4"; input_video=input_base+INPUT_EXTENSION; orig_id=None # Имя файла от embedder с t=4
    if os.path.exists(ORIGINAL_WATERMARK_FILE):
        try:
            with open(ORIGINAL_WATERMARK_FILE,"r") as f: orig_id=f.read().strip();
            if orig_id and len(orig_id)==PAYLOAD_LEN_BYTES*2: int(orig_id,16); logging.info(f"Read original ID.")
            else: logging.error("Invalid original ID file."); orig_id=None
        except Exception as e: logging.error(f"Read original ID failed: {e}"); orig_id=None
    else: logging.warning(f"{ORIGINAL_WATERMARK_FILE} not found.")

    logging.info("--- Starting Extraction Main Process (Galois) ---")
    if not os.path.exists(input_video): logging.critical(f"Input missing: '{input_video}'."); print(f"ERROR: Input missing."); return
    frames,fps = read_video(input_video);
    if not frames: return
    logging.info(f"Read {len(frames)} frames.")

    extracted_bytes = extract_watermark_from_video(
        frames=frames, nr=N_RINGS, nrtu=NUM_RINGS_TO_USE, bp=BITS_PER_PAIR, cps=CANDIDATE_POOL_SIZE,
        ec=EMBED_COMPONENT, mpr=MAX_PACKET_REPEATS, ue=USE_ECC, bm=BCH_M, bt=BCH_T, # bt=4
        plb=PAYLOAD_LEN_BYTES, mw=MAX_WORKERS_EXTRACT)

    print(f"\n--- Extraction Results ---"); ext_hex=None
    if extracted_bytes:
        if len(extracted_bytes)==PAYLOAD_LEN_BYTES: ext_hex=extracted_bytes.hex(); print("  Payload OK."); print(f"  Decoded ID (Hex): {ext_hex}"); logging.info(f"Decoded ID: {ext_hex}")
        else: print(f"  ERROR: Payload length mismatch! Got {len(extracted_bytes)}B."); logging.error(f"Payload length mismatch!")
    else: print(f"  Extraction FAILED."); logging.error("Extraction failed.")

    if orig_id:
        print(f"  Original ID (Hex): {orig_id}")
        if ext_hex and ext_hex==orig_id: print("\n  >>> ID MATCH <<<"); logging.info("ID MATCH.")
        else: print("\n  >>> !!! ID MISMATCH or FAILED !!! <<<"); logging.warning("ID MISMATCH.")
    else: print("\n  Original ID unavailable.")
    logging.info("--- Extraction Main Process Finished ---")
    total_time = time.time()-start; logging.info(f"--- Total Extractor Time: {total_time:.2f} sec ---")
    print(f"\nExtraction finished. Log: {LOG_FILENAME}")

# --- Точка Входа ---
if __name__ == "__main__":
    effective_use_ecc = USE_ECC and GALOIS_AVAILABLE # Пересчет флага
    if USE_ECC and not effective_use_ecc: print("\nWARNING: USE_ECC=True, but galois unavailable/failed. ECC disabled.")
    profiler = cProfile.Profile(); profiler.enable()
    try: main()
    except FileNotFoundError as e: print(f"\nERROR: File not found: {e}")
    except Exception as e: logging.critical(f"Unhandled exception (Extractor): {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. See log.")
    finally:
        profiler.disable(); stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        profile_file = f"profile_extract_galois_t{BCH_T}.txt" # Имя соответствует galois t=4
        try:
            with open(profile_file, "w") as f: stats_file = pstats.Stats(profiler, stream=f); stats_file.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling saved: {profile_file}"); print(f"Profiling saved: {profile_file}")
        except IOError as e: logging.error(f"Could not save profiling stats: {e}")