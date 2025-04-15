# Файл: embedder.py (Версия: Galois BCH, m=8, t=4, Syntax Fix V4)
import cv2
import numpy as np
import random
import logging
import time
import concurrent.futures
import json
import os
# import imagehash # Закомментирован, так как не используется в этой версии
import hashlib
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.linalg import svd
from dtcwt import Transform2d, Pyramid
from typing import List, Tuple, Optional, Dict, Any
import functools
import uuid
from math import ceil

# --- Попытка импорта и инициализации Galois ---
try:
    import galois
    print("galois: импортирован.")
    _test_bch_ok = False
    _test_encode_ok = False
    BCH_CODE_OBJECT = None # Инициализируем заранее
    try:
        _test_m = 8; _test_t = 4; _test_n = (1 << _test_m) - 1; _test_d = 2 * _test_t + 1
        _test_bch_galois = galois.BCH(_test_n, d=_test_d)
        if _test_bch_galois.t == _test_t:
             print(f"galois BCH(n={_test_bch_galois.n}, k={_test_bch_galois.k}, t={_test_bch_galois.t}) initialized OK.")
             _test_bch_ok = True
             BCH_CODE_OBJECT = _test_bch_galois
        else: print(f"galois BCH init mismatch: t={_test_bch_galois.t}")

        if _test_bch_ok:
            _k_bits = _test_bch_galois.k
            _dummy_msg_bits = np.zeros(_k_bits, dtype=np.uint8)
            GF2 = galois.GF(2)
            _dummy_msg_vec = GF2(_dummy_msg_bits)
            _codeword = _test_bch_galois.encode(_dummy_msg_vec)
            print("galois: encode() test OK.")
            _test_encode_ok = True
    except AttributeError as ae: print(f"galois: ОШИБКА атрибута/метода: {ae}")
    except ValueError as ve: print(f"galois: ОШИБКА ValueError: {ve}")
    except TypeError as te: print(f"galois: ОШИБКА TypeError: {te}")
    except Exception as test_err: print(f"galois: ОШИБКА теста: {test_err}")

    GALOIS_AVAILABLE = _test_bch_ok and _test_encode_ok
    if not GALOIS_AVAILABLE: BCH_CODE_OBJECT = None # Сброс если тесты не прошли
    if GALOIS_AVAILABLE: print("galois: Тесты инициализации и encode пройдены.")
    else: print("galois: Не прошел базовые тесты. ECC будет отключен.")

except ImportError: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print("galois library not found.")
except Exception as import_err: GALOIS_AVAILABLE = False; BCH_CODE_OBJECT = None; print(f"galois: Ошибка импорта: {import_err}")

# --- Остальные импорты ---
import cProfile
import pstats

# --- Основные Параметры ---
LAMBDA_PARAM: float = 0.04; ALPHA_MIN: float = 1.005; ALPHA_MAX: float = 1.1; N_RINGS: int = 8
MAX_THEORETICAL_ENTROPY = 8.0; EMBED_COMPONENT: int = 1; USE_PERCEPTUAL_MASKING: bool = True
CANDIDATE_POOL_SIZE: int = 4; BITS_PER_PAIR: int = 2; NUM_RINGS_TO_USE: int = BITS_PER_PAIR
RING_SELECTION_METHOD: str = 'pool_entropy_selection'; PAYLOAD_LEN_BYTES: int = 8; USE_ECC: bool = True
BCH_M: int = 8; BCH_T: int = 4; MAX_PACKET_REPEATS: int = 5; FPS: int = 30
LOG_FILENAME: str = 'watermarking_embed.log'; OUTPUT_CODEC: str = 'XVID'; OUTPUT_EXTENSION: str = '.avi'
SELECTED_RINGS_FILE: str = 'selected_rings_embed.json'; ORIGINAL_WATERMARK_FILE: str = 'original_watermark_id.txt'
MAX_WORKERS: Optional[int] = None

# --- Настройка Логирования ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO, format='[%(asctime)s] %(levelname).1s %(threadName)s - %(funcName)s:%(lineno)d - %(message)s')
# logging.getLogger().setLevel(logging.DEBUG)

# --- Логирование Конфигурации ---
effective_use_ecc = USE_ECC and GALOIS_AVAILABLE
logging.info(f"--- Запуск Скрипта Встраивания (Метод: {RING_SELECTION_METHOD}, Pool: {CANDIDATE_POOL_SIZE}, Select: {NUM_RINGS_TO_USE}) ---")
logging.info(f"Payload: {PAYLOAD_LEN_BYTES * 8}bit, ECC: {effective_use_ecc} (Galois BCH m={BCH_M}, t={BCH_T}), Max Repeats: {MAX_PACKET_REPEATS}")
logging.info(f"Альфа: MIN={ALPHA_MIN}, MAX={ALPHA_MAX}, N_RINGS_Total={N_RINGS}")
logging.info(f"Маскировка: {USE_PERCEPTUAL_MASKING} (Lambda={LAMBDA_PARAM}), Компонент: {['Y', 'Cr', 'Cb'][EMBED_COMPONENT]}")
logging.info(f"Выход: Кодек={OUTPUT_CODEC}, Расширение={OUTPUT_EXTENSION}")
if USE_ECC and not GALOIS_AVAILABLE: logging.warning("ECC вкл, но galois недоступна/не работает! Без ECC.")
elif not USE_ECC: logging.info("ECC выкл.")
if OUTPUT_CODEC in ['XVID', 'mp4v', 'MJPG']: logging.warning(f"Lossy кодек '{OUTPUT_CODEC}'.")
if NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE: logging.error("NUM_RINGS_TO_USE > CANDIDATE_POOL_SIZE!"); NUM_RINGS_TO_USE=CANDIDATE_POOL_SIZE; logging.warning("Adjusted NUM_RINGS_TO_USE.")
if NUM_RINGS_TO_USE != BITS_PER_PAIR: logging.warning("NUM_RINGS_TO_USE != BITS_PER_PAIR.")

# --- Базовые Функции ---
def dct_1d(s: np.ndarray) -> np.ndarray: return dct(s, type=2, norm='ortho')
def idct_1d(c: np.ndarray) -> np.ndarray: return idct(c, type=2, norm='ortho')
def dtcwt_transform(yp: np.ndarray, fn: int = -1) -> Optional[Pyramid]:
    if np.any(np.isnan(yp)): logging.warning(f"[F:{fn}] NaNs!")
    try:
        t=Transform2d(); r,c=yp.shape; pr=r%2!=0; pc=c%2!=0;
        if pr or pc: ypp=np.pad(yp,((0,pr),(0,pc)),mode='reflect')
        else: ypp=yp
        py=t.forward(ypp.astype(np.float32), nlevels=1);
        if hasattr(py,'lowpass') and py.lowpass is not None: py.padding_info=(pr,pc); return py
        else: logging.error(f"[F:{fn}] DTCWT no lowpass."); return None
    except Exception as e: logging.error(f"[F:{fn}] DTCWT fail: {e}", exc_info=True); return None
def dtcwt_inverse(py: Pyramid, fn: int = -1) -> Optional[np.ndarray]:
    if not isinstance(py, Pyramid) or not hasattr(py, 'lowpass'): return None
    try:
        t=Transform2d(); rp=t.inverse(py).astype(np.float32); pr,pc=getattr(py,'padding_info',(False,False));
        if pr or pc: r,c=rp.shape; er=r-pr if pr else r; ec=c-pc if pc else c; ry=rp[:er,:ec]
        else: ry=rp
        if np.any(np.isnan(ry)): logging.warning(f"[F:{fn}] NaNs after inverse!")
        return ry
    except Exception as e: logging.error(f"[F:{fn}] DTCWT inv fail: {e}", exc_info=True); return None
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

    rc: List[Optional[np.ndarray]]
    rc = [None] * n_rings

    # --- ИСПРАВЛЕНО: Цикл разбит на несколько строк ---
    for rdx in range(n_rings_eff):
        coords = np.argwhere(ring_indices == rdx)
        if coords.shape[0] > 0:
            rc[rdx] = coords
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    return rc

@functools.lru_cache(maxsize=8)
def get_ring_coords_cached(ss: Tuple[int, int], nr: int) -> List[Optional[np.ndarray]]: return _ring_division_internal(ss, nr)
def ring_division(lp: np.ndarray, nr: int = N_RINGS, fn: int = -1) -> List[Optional[np.ndarray]]:
    if not isinstance(lp,np.ndarray) or lp.ndim!=2: return [None]*nr
    sh=lp.shape;
    try:
        cl=get_ring_coords_cached(sh,nr);
        if not isinstance(cl,list) or not all(isinstance(i,(np.ndarray,type(None))) for i in cl): get_ring_coords_cached.cache_clear(); cl=_ring_division_internal(sh,nr)
        return [a.copy() if a is not None else None for a in cl]
    except Exception as e: logging.error(f"[F:{fn}] Ring div error: {e}"); return [None]*nr
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
def calculate_perceptual_mask(ip:np.ndarray, fn:int=-1)->Optional[np.ndarray]:
    if not isinstance(ip, np.ndarray) or ip.ndim!=2: return None
    try:
        p32=ip.astype(np.float32); gx=cv2.Sobel(p32,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(p32,cv2.CV_32F,0,1,ksize=3)
        gm=np.sqrt(gx**2+gy**2); k=(11,11); s=5; lm=cv2.GaussianBlur(p32,k,s);
        ms=cv2.GaussianBlur(p32**2,k,s); sm=lm**2; ls=np.sqrt(np.maximum(ms-sm,0))
        cm=np.maximum(gm,ls); eps=1e-9; mc=np.max(cm);
        mn=cm/(mc+eps) if mc>eps else np.zeros_like(cm); mn=np.clip(mn,0.,1.)
        return mn.astype(np.float32)
    except Exception as e: logging.error(f"[F:{fn}] Mask error: {e}"); return np.ones_like(ip,dtype=np.float32)
def add_ecc(data_bits: np.ndarray, bch_code: galois.BCH) -> Optional[np.ndarray]:
    """Добавляет ECC с использованием библиотеки galois."""
    if not GALOIS_AVAILABLE or bch_code is None:
        logging.warning("Galois не доступна или не инициализирована, ECC не добавляется.")
        return data_bits # Возвращаем исходные биты

    k = bch_code.k
    n = bch_code.n

    if data_bits.size > k:
        logging.error(f"Размер данных ({data_bits.size} бит) > Galois k ({k} бит).")
        return None
    elif data_bits.size < k:
        # Дополняем нулями СПРАВА до длины k
        padding_len = k - data_bits.size
        message_bits = np.pad(data_bits, (0, padding_len), 'constant').astype(np.uint8)
        # logging.debug(f"Данные ({data_bits.size}b) дополнены {padding_len} нулями до {k}b.")
    else:
        message_bits = data_bits.astype(np.uint8)

    try:
        # Преобразуем биты сообщения в FieldArray
        GF = bch_code.field # Получаем поле из объекта кода
        message_vector = GF(message_bits) # Создаем FieldArray

        # Кодируем
        codeword_vector = bch_code.encode(message_vector) # Возвращает FieldArray длины n

        # --- ИСПРАВЛЕНО: Преобразуем FieldArray в NumPy массив ---
        # Используем .view(np.ndarray) или .elements
        packet_bits = codeword_vector.view(np.ndarray).astype(np.uint8)
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---


        if packet_bits.size != n:
            logging.error(f"Galois encode: длина кодового слова {packet_bits.size} != ожидаемой n {n}.")
            return None
        else:
            logging.info(f"Galois ECC: Data({data_bits.size}b->{k}b) -> Packet({packet_bits.size}b, n={n}).")

        return packet_bits

    except Exception as e:
        logging.error(f"Ошибка при кодировании с Galois: {e}", exc_info=True)
        return None
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
def write_video(fr:List[np.ndarray], op:str, fps:float, cd:str=OUTPUT_CODEC): # cd вместо codec
    if not fr: logging.error("No frames to write."); return
    logging.info(f"Writing: {op}(FPS:{fps:.2f},Codec:{cd})") # Используем cd
    wr=None
    try:
        f=next((x for x in fr if x is not None and x.ndim==3),None);
        if f is None: logging.error("No valid frames to get size."); return
        h,w=f.shape[:2]; logging.info(f"Out res:{w}x{h}")
        fcc=cv2.VideoWriter_fourcc(*cd) # Используем cd
        base,_=os.path.splitext(op); op=base+OUTPUT_EXTENSION
        wr=cv2.VideoWriter(op,fcc,fps,(w,h)); wcd=cd
        if not wr.isOpened():
            logging.error(f"Fail codec {cd}.");
            if OUTPUT_EXTENSION.lower()=='.avi' and cd!='MJPG':
                fb='MJPG'; logging.warning(f"Fallback {fb}."); fcc=cv2.VideoWriter_fourcc(*fb); wr=cv2.VideoWriter(op,fcc,fps,(w,h)); wcd=fb
            if not wr.isOpened(): logging.critical("Writer fail."); return
        wc,sc=0,0; bf=np.zeros((h,w,3),dtype=np.uint8)
        for frame in fr: # Используем frame
            if frame is not None and frame.shape[:2]==(h,w) and frame.dtype==np.uint8: wr.write(frame); wc+=1
            else: wr.write(bf); sc+=1
        logging.info(f"Write done({wcd}). W:{wc},S:{sc}")
    except Exception as e: logging.error(f"Video write error: {e}", exc_info=True) # Используем 'e'
    finally:
        if wr: wr.release()
def embed_frame_pair(frame1_bgr:np.ndarray,frame2_bgr:np.ndarray,bits:List[int],selected_ring_indices:List[int],n_rings:int=N_RINGS,frame_number:int=0,use_perceptual_masking:bool=USE_PERCEPTUAL_MASKING,embed_component:int=EMBED_COMPONENT)->Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
    pn=frame_number//2;
    if len(bits)!=len(selected_ring_indices): return None,None
    if not bits: return frame1_bgr,frame2_bgr
    try:
        if frame1_bgr is None or frame2_bgr is None or frame1_bgr.shape!=frame2_bgr.shape: return None,None
        try: f1y=cv2.cvtColor(frame1_bgr,cv2.COLOR_BGR2YCrCb); f2y=cv2.cvtColor(frame2_bgr,cv2.COLOR_BGR2YCrCb)
        except cv2.error: return None,None
        try: c1=f1y[:,:,embed_component].astype(np.float32)/255.; c2=f2y[:,:,embed_component].astype(np.float32)/255.; Y1=f1y[:,:,0];Cb1=f1y[:,:,2];Cr1=f1y[:,:,1]; Y2=f2y[:,:,0];Cb2=f2y[:,:,2];Cr2=f2y[:,:,1]
        except IndexError: return None,None
        p1=dtcwt_transform(c1,frame_number); p2=dtcwt_transform(c2,frame_number+1);
        if p1 is None or p2 is None or p1.lowpass is None or p2.lowpass is None: return None,None
        L1=p1.lowpass.copy(); L2=p2.lowpass.copy(); r1c=ring_division(L1,n_rings,frame_number); r2c=ring_division(L2,n_rings,frame_number+1); pm=None
        if use_perceptual_masking: mf=calculate_perceptual_mask(c1,frame_number);
        if mf is not None:
             if mf.shape!=L1.shape: pm=cv2.resize(mf,(L1.shape[1],L1.shape[0]))
             else: pm=mf
        else: pm=np.ones_like(L1)
        mods=0
        for ri,bit in zip(selected_ring_indices,bits):
            if not (0<=ri<n_rings): continue
            try: cd1=r1c[ri]; cd2=r2c[ri];
            except IndexError: continue
            if cd1 is None or cd2 is None: continue
            rs1,cs1=cd1[:,0],cd1[:,1]; rs2,cs2=cd2[:,0],cd2[:,1]; v1=L1[rs1,cs1].astype(np.float32); v2=L2[rs2,cs2].astype(np.float32)
            if v1.size==0 or v2.size==0: continue
            if v1.size!=v2.size:
                sz=min(v1.size,v2.size);
                if sz==0: continue;
                v1=v1[:sz];v2=v2[:sz]; rs1=rs1[:sz];cs1=cs1[:sz]; rs2=rs2[:sz];cs2=cs2[:sz]
            a=compute_adaptive_alpha_entropy(v1,ri,frame_number); d1=dct_1d(v1); d2=dct_1d(v2)
            try: U1,S1,Vt1=svd(d1.reshape(-1,1),False); U2,S2,Vt2=svd(d2.reshape(-1,1),False)
            except np.linalg.LinAlgError: continue
            s1=S1[0] if S1.size>0 else 0.; s2=S2[0] if S2.size>0 else 0.; eps=1e-12; ratio=s1/(s2+eps); ns1,ns2=s1,s2; mod=False; a2=a*a; inv_a=1/(a+eps)
            if bit==0:
                if ratio<a: ns1=(s1*a2+a*s2)/(a2+1); ns2=(a*s1+s2)/(a2+1); mod=True
            else:
                if ratio>=inv_a: ns1=(s1+a*s2)/(1+a2); ns2=(a*s1+a2*s2)/(1+a2); mod=True
            if mod:
                mods+=1; nS1=np.array([ns1]); nS2=np.array([ns2]);
                if len(S1)>1: nS1=np.concatenate(([ns1],S1[1:]))
                if len(S2)>1: nS2=np.concatenate(([ns2],S2[1:]))
                try: d1m=(U1@np.diag(nS1)@Vt1).flat; d2m=(U2@np.diag(nS2)@Vt2).flat; v1m=idct_1d(d1m); v2m=idct_1d(d2m)
                except Exception: continue
                if v1m.size!=v1.size or v2m.size!=v2.size: continue
                del1=v1m-v1; del2=v2m-v2; mf1=np.ones_like(del1); mf2=np.ones_like(del2)
                if use_perceptual_masking and pm is not None:
                    try: mv1=pm[rs1,cs1]; mv2=pm[rs2,cs2]; mf1*=(LAMBDA_PARAM+(1-LAMBDA_PARAM)*mv1); mf2*=(LAMBDA_PARAM+(1-LAMBDA_PARAM)*mv2)
                    except Exception: pass
                try: L1[rs1,cs1]+=del1*mf1; L2[rs2,cs2]+=del2*mf2
                except IndexError: continue
        p1.lowpass=L1; p2.lowpass=L2;
        # --- ИСПРАВЛЕНО: Используем frame_number ---
        c1m=dtcwt_inverse(p1,frame_number); c2m=dtcwt_inverse(p2,frame_number+1);
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
        if c1m is None or c2m is None: return None,None
        tsh=(Y1.shape[0],Y1.shape[1]);
        if c1m.shape!=tsh: c1m=cv2.resize(c1m,(tsh[1],tsh[0]));
        if c2m.shape!=tsh: c2m=cv2.resize(c2m,(tsh[1],tsh[0]));
        c1s=np.clip(c1m*255,0,255).astype(np.uint8); c2s=np.clip(c2m*255,0,255).astype(np.uint8)
        ny1=np.stack((Y1,Cr1,Cb1),axis=-1); ny2=np.stack((Y2,Cr2,Cb2),axis=-1)
        if embed_component==0: ny1[:,:,0]=c1s; ny2[:,:,0]=c2s
        elif embed_component==1: ny1[:,:,1]=c1s; ny2[:,:,1]=c2s
        else: ny1[:,:,2]=c1s; ny2[:,:,2]=c2s
        f1m=cv2.cvtColor(ny1,cv2.COLOR_YCrCb2BGR); f2m=cv2.cvtColor(ny2,cv2.COLOR_YCrCb2BGR)
        return f1m,f2m
    except Exception as e: logging.error(f"Embed pair failed (P:{pn}): {e}"); return None,None

def _embed_frame_pair_worker(args: Dict[str, Any]) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    pi=args['pair_idx']; fn=2*pi; bits_arg=args['bits']; f1=args['frame1']; f2=args['frame2']; nr_arg=args['n_rings']
    nrtu=args['num_rings_to_use']; cps=args['candidate_pool_size']; ec_arg=args['embed_component']; upm_arg=args['use_perceptual_masking']
    final_rings=[]
    try:
        if len(bits_arg)!=nrtu: raise ValueError("Bit/Ring mismatch")
        cand_rings = get_fixed_pseudo_random_rings(pi, nr_arg, cps)
        if len(cand_rings)<nrtu: return fn,None,None,[]
        try:
            comp1=f1[:,:,ec_arg].astype(np.float32)/255.; pyr1=dtcwt_transform(comp1,fn);
            if pyr1 is None or pyr1.lowpass is None: raise RuntimeError("DTCWT L1 failed")
            L1=pyr1.lowpass; all_coords=ring_division(L1,nr_arg,fn); c_entropies=[]; min_pix=10
            for r_idx in cand_rings:
                e=-float('inf');
                if 0<=r_idx<nr_arg: coords=all_coords[r_idx];
                if coords is not None and coords.shape[0]>=min_pix:
                    try: r,c=coords[:,0],coords[:,1]; rv=L1[r,c]; ve,_=calculate_entropies(rv,fn,r_idx)
                    except Exception: pass
                    if np.isfinite(ve): e=ve
                c_entropies.append((e,r_idx))
            c_entropies.sort(key=lambda x:x[0], reverse=True); final_rings=[i for e,i in c_entropies if e>-float('inf')][:nrtu]
            if len(final_rings)<nrtu: return fn,None,None,final_rings
        except Exception as sel_err: logging.error(f"[P:{pi}] Select error: {sel_err}."); return fn,None,None,[]
        f1m,f2m=embed_frame_pair(frame1_bgr=f1,frame2_bgr=f2,bits=bits_arg,selected_ring_indices=final_rings,n_rings=nr_arg,frame_number=fn,use_perceptual_masking=upm_arg,embed_component=ec_arg)
        return fn,f1m,f2m,final_rings
    except Exception as e: logging.error(f"Worker {pi} error: {e}"); return fn,None,None,[]
def embed_watermark_in_video(
        frames: List[np.ndarray], packet_bits: np.ndarray, n_rings: int = N_RINGS, num_rings_to_use: int = NUM_RINGS_TO_USE,
        bits_per_pair: int = BITS_PER_PAIR, candidate_pool_size: int = CANDIDATE_POOL_SIZE, max_packet_repeats: int = MAX_PACKET_REPEATS,
        fps: float = FPS, max_workers: Optional[int] = MAX_WORKERS, use_perceptual_masking: bool = USE_PERCEPTUAL_MASKING, embed_component: int = EMBED_COMPONENT):
    nf=len(frames); tp=nf//2; p_len=packet_bits.size;
    if tp==0 or p_len==0: return frames[:]
    pairs_needed=ceil(max_packet_repeats*p_len/bits_per_pair); pairs_proc=min(tp,pairs_needed); total_bits=pairs_proc*bits_per_pair
    repeats=ceil(total_bits/p_len) if p_len>0 else 1; bits_flat=np.tile(packet_bits,repeats)[:total_bits]
    logging.info(f"Embed: {total_bits}b ({bits_per_pair}/pair) in {pairs_proc} pairs. Packet:{p_len}b. Repeats:{max_packet_repeats} target, {total_bits/p_len:.2f} actual.")
    start=time.time(); watermarked_frames=frames[:]; tasks=[]
    if pairs_proc==0: return watermarked_frames
    for pi in range(pairs_proc):
        i1=2*pi; i2=i1+1;
        if i2>=nf or frames[i1] is None or frames[i2] is None: continue
        sbi=pi*bits_per_pair; ebi=sbi+bits_per_pair; cb=bits_flat[sbi:ebi].tolist()
        if len(cb)!=bits_per_pair: continue
        args={'pair_idx':pi,'frame1':frames[i1],'frame2':frames[i2],'bits':cb,'n_rings':n_rings,'num_rings_to_use':num_rings_to_use,'candidate_pool_size':candidate_pool_size,'frame_number':i1,'use_perceptual_masking':use_perceptual_masking,'embed_component':embed_component}
        tasks.append(args)
    if not tasks: return watermarked_frames
    results:Dict[int,Tuple[Optional[np.ndarray],Optional[np.ndarray]]]={}; rings_log:Dict[int,List[int]]={}; pc=0; ec=0; tc=len(tasks)
    try:
        logging.info(f"Submitting {tc} tasks (mw={max_workers})...");
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            f2f={executor.submit(_embed_frame_pair_worker,a):a['frame_number'] for a in tasks}
            for i,f in enumerate(concurrent.futures.as_completed(f2f)):
                fn=f2f[f]; pi=fn//2
                try: fnr,f1m,f2m,sr=f.result()
                except Exception as exc: logging.error(f'Pair {pi} thread exception: {exc}'); ec+=1; continue
                if sr: rings_log[pi]=sr
                if f1m is not None and f2m is not None: results[fnr]=(f1m,f2m); pc+=1
                else: ec+=1; logging.error(f"Pair {pi} failed embedding.")
    except Exception as e: logging.critical(f"Executor error: {e}"); return watermarked_frames
    logging.info(f"Executor done. Success:{pc}, Failed:{ec}."); uc=0
    for i1,(f1m,f2m) in results.items():
        i2=i1+1;
        if i1<len(watermarked_frames) and f1m is not None: watermarked_frames[i1]=f1m; uc+=1
        if i2<len(watermarked_frames) and f2m is not None: watermarked_frames[i2]=f2m; uc+=1
    logging.info(f"Applied results to {uc} frames.");
    # --- ИСПРАВЛЕННЫЙ БЛОК TRY...EXCEPT ---
    if rings_log:
        try:
            ser_log = {str(k): v for k, v in rings_log.items()}
            with open(SELECTED_RINGS_FILE, 'w') as f:
                 json.dump(ser_log, f, indent=4)
            logging.info(f"Saved rings log: {SELECTED_RINGS_FILE}")
        except Exception as e: # Используем 'e'
             logging.error(f"Could not save rings log: {e}", exc_info=True) # Логируем ошибку
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    else: logging.warning("No rings recorded.")
    end=time.time(); logging.info(f"Embedding done. Time: {end-start:.2f}s.")
    return watermarked_frames
# --- Основная Функция (Galois) ---
def main():
    start=time.time(); input_video="input2.mp4"; base_output="watermarked_galois_t4"; output_video=base_output+OUTPUT_EXTENSION
    logging.info("--- Starting Embedding Main Process (Galois) ---")
    frames,fps=read_video(input_video);
    if not frames: return
    fps_use=float(FPS) if fps<=0 else fps;
    if len(frames)//2==0: return

    orig_id_bytes=os.urandom(PAYLOAD_LEN_BYTES); orig_id_hex=orig_id_bytes.hex()
    logging.info(f"Generated Payload ID ({PAYLOAD_LEN_BYTES*8}b, Hex): {orig_id_hex}")
    packet_bits:Optional[np.ndarray]=None; eff_ecc=USE_ECC and GALOIS_AVAILABLE
    bch_code=BCH_CODE_OBJECT if eff_ecc else None

    if eff_ecc and bch_code:
        try:
            n=bch_code.n; k=bch_code.k; ecc=n-k; logging.info(f"Using Galois BCH: n={n}, k={k}, t={bch_code.t}")
            if PAYLOAD_LEN_BYTES*8>k: logging.error(f"Payload>Galois k!"); eff_ecc=False; packet_bits=None; bch_code=None
            else: payload_bits=np.unpackbits(np.frombuffer(orig_id_bytes,dtype=np.uint8)); packet_bits=add_ecc(payload_bits,bch_code);
            if packet_bits is None: raise RuntimeError("add_ecc failed.")
        except Exception as e: logging.error(f"Galois ECC failed: {e}. No ECC."); eff_ecc=False; packet_bits=None; bch_code=None
    else: logging.info("ECC disabled or unavailable.")

    if packet_bits is None: packet_bits=np.unpackbits(np.frombuffer(orig_id_bytes,dtype=np.uint8)); logging.info(f"Using raw payload({packet_bits.size}b).")
    else: logging.info(f"Using ECC packet({packet_bits.size}b).")

    try:
        with open(ORIGINAL_WATERMARK_FILE,"w") as f: f.write(orig_id_hex)
        logging.info(f"Original ID saved: {ORIGINAL_WATERMARK_FILE}")
    except IOError as e: logging.error(f"Save ID failed: {e}")

    watermarked_frames = embed_watermark_in_video(
        frames=frames, packet_bits=packet_bits, n_rings=N_RINGS, num_rings_to_use=NUM_RINGS_TO_USE,
        bits_per_pair=BITS_PER_PAIR, candidate_pool_size=CANDIDATE_POOL_SIZE, max_packet_repeats=MAX_PACKET_REPEATS,
        fps=fps_use, max_workers=MAX_WORKERS, use_perceptual_masking=USE_PERCEPTUAL_MASKING, embed_component=EMBED_COMPONENT)

    # --- ИСПРАВЛЕНО: Проверка и использование watermarked_frames ---
    if watermarked_frames and len(watermarked_frames)==len(frames):
        write_video(watermarked_frames, output_video, fps=fps_use, cd=OUTPUT_CODEC) # Используем cd
        logging.info(f"Video saved: {output_video}")
        try:
            if os.path.exists(output_video): logging.info(f"Output size: {os.path.getsize(output_video)/(1024*1024):.2f}MB")
            else: logging.error(f"Output missing.")
        except OSError as e: logging.error(f"Get size failed: {e}")
    else: logging.error("Embedding failed. No output.")
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    logging.info("--- Embedding Finished ---")
    total_time=time.time()-start; logging.info(f"--- Total Time: {total_time:.2f}s ---")
    print(f"\nEmbed finished."); print(f"Out: {output_video}"); print(f"Log: {LOG_FILENAME}"); print(f"ID: {ORIGINAL_WATERMARK_FILE}"); print(f"Rings: {SELECTED_RINGS_FILE}"); print("\nRun extractor.")

# --- Точка Входа ---
if __name__ == "__main__":
    if USE_ECC and not GALOIS_AVAILABLE: print("\nERROR: ECC required but galois failed/unavailable.")
    prof=cProfile.Profile(); prof.enable()
    try: main()
    except FileNotFoundError as e: print(f"\nERROR: {e}")
    except ValueError as e: print(f"\nERROR: {e}.")
    except RuntimeError as e: print(f"\nERROR: {e}.")
    except Exception as e: logging.critical(f"Unhandled Embedder: {e}", exc_info=True); print(f"\nCRITICAL ERROR: {e}. See log.")
    finally:
        prof.disable(); stats=pstats.Stats(prof); stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        pfile=f"profile_embed_galois_t{BCH_T}.txt";
        try:
            with open(pfile,"w") as f: sf=pstats.Stats(prof,stream=f); sf.strip_dirs().sort_stats("cumulative").print_stats()
            logging.info(f"Profiling saved: {pfile}"); print(f"Profiling saved: {pfile}")
        except IOError as e: logging.error(f"Save profile failed: {e}")