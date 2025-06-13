import os
import logging
from pathlib import Path
from typing import List, NamedTuple
from collections import defaultdict
import gc
import psutil
from dataclasses import dataclass

import cv2
import numpy as np
import streamlit as st
import torch

import threading
import queue
import time
from threading import Lock, Event, Condition

# Deep learning framework
from ultralytics import YOLO

st.set_page_config(
    page_title="Hybrid Optimal Video Detection",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
tracker_yaml_path = ROOT / "./models/bytetrack.yaml"

# BALANCED OPTIMAL SETTINGS - RAM-GPU dengeleme
class OptimalSettings:
    """Balanced settings for RAM-GPU optimization"""
    
    def __init__(self):
        self.gpu_memory_gb = 0
        self.gpu_name = "CPU"
        
        if torch.cuda.is_available():
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_name = torch.cuda.get_device_name(0)
            
        self.cpu_cores = os.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calculate BALANCED settings
        self.settings = self._calculate_balanced_settings()
        
        print(f"⚡ Balanced Optimal Settings calculated:")
        print(f"   Hardware: {self.gpu_name} ({self.gpu_memory_gb:.1f}GB), {self.cpu_cores} cores")
        print(f"   Settings: {self.settings}")
        
    def _calculate_balanced_settings(self):
        """RAM-GPU dengeleme ile optimal ayarlar"""
        
        # KÜÇÜK batch size - RAM koruma
        batch_size = min(16, max(8, int(self.gpu_memory_gb * 2)))  # Daha konservatif
        
        # Daha fazla predictor - GPU'yu meşgul et
        num_predictors = min(4, max(2, int(self.gpu_memory_gb // 1.5)))
        
        # Orta düzey preprocessor
        num_preprocessors = min(4, max(2, self.cpu_cores // 2))
        
        # KÜÇÜK queue'lar - RAM tasarrufu
        base_queue = 48  # 96'dan 48'e düştü
        batch_queue_size = 8  # Çok küçük batch queue
        
        return {
            'BATCH_SIZE': batch_size,
            'NUM_PREDICTOR_THREADS': num_predictors,
            'NUM_PREPROCESSOR_THREADS': num_preprocessors,
            'READ_QUEUE_SIZE': base_queue,
            'PREPROCESS_QUEUE_SIZE': base_queue,
            'BATCH_QUEUE_SIZE': batch_queue_size,
            'RESULT_QUEUE_SIZE': base_queue,
            'MEMORY_CLEANUP_INTERVAL': 10,  # Daha sık temizlik
            'EXPECTED_GPU_USAGE': min(90, 60 + (self.gpu_memory_gb * 5)),
            'FRAME_SKIP_INTERVAL': 2  # Her 2 frame'de bir
        }

# Initialize optimal settings
if 'optimal_settings' not in st.session_state:
    st.session_state.optimal_settings = OptimalSettings()

SETTINGS = st.session_state.optimal_settings.settings
BATCH_SIZE = SETTINGS['BATCH_SIZE']
NUM_PREDICTOR_THREADS = SETTINGS['NUM_PREDICTOR_THREADS']
NUM_PREPROCESSOR_THREADS = SETTINGS['NUM_PREPROCESSOR_THREADS']
FRAME_SKIP_INTERVAL = SETTINGS['FRAME_SKIP_INTERVAL']

device = "cuda" if torch.cuda.is_available() else "cpu"

# Memory-efficient model havuzu
class MemoryEfficientModelPool:
    def __init__(self, model_path, pool_size):
        self.model_path = model_path
        self.pool_size = pool_size
        self.models = []
        self.lock = Lock()

        self._initialize_pool()

    def _initialize_pool(self):
        """Memory-efficient model pool"""
        print(f"🚀 {self.pool_size} memory-optimized model yükleniyor...")

        for i in range(self.pool_size):
            model = self._create_optimized_model()
            if model:
                self.models.append(model)
                print(f"✅ Model {i + 1}/{self.pool_size} hazır")
            else:
                print(f"❌ Model {i + 1} yüklenemedi")

        if not self.models:
            print("⚠️ Fallback model oluşturuluyor...")
            fallback_model = self._create_optimized_model()
            if fallback_model:
                self.models.append(fallback_model)

    def _create_optimized_model(self):
        """Memory-optimized model oluştur"""
        try:
            model = YOLO(self.model_path)
            model.tracker = str(tracker_yaml_path)

            if torch.cuda.is_available():
                # Model optimizasyonu
                model.model.half().to(device)
                model.model.eval()  # Inference modu
                
                # Küçük warmup - memory tasarrufu
                dummy = torch.randn(1, 3, 640, 640).half().to(device)
                with torch.no_grad():
                    _ = model.model(dummy)
                
                # Immediate cleanup
                del dummy
                torch.cuda.empty_cache()

            return model
        except Exception as e:
            logger.error(f"Model oluşturulamadı: {e}")
            return None

    def get_model(self):
        with self.lock:
            if self.models:
                return self.models.pop()
        time.sleep(0.05)  # Daha kısa bekleme
        return self._create_optimized_model()

    def return_model(self, model):
        if not model:
            return

        with self.lock:
            if len(self.models) < self.pool_size:
                self.models.append(model)
            else:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Initialize optimized model pool
if 'optimized_model_pool' not in st.session_state:
    st.session_state.optimized_model_pool = MemoryEfficientModelPool(
        MODEL_LOCAL_PATH, 
        NUM_PREDICTOR_THREADS + 1
    )

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack", 
    "Alligator Crack",
    "Potholes"
]

class PerformanceTracker:
    """Memory-efficient performance tracking"""
    def __init__(self):
        self.lock = Lock()
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.unique_tracks = set()
        self.class_counts = defaultdict(int)
        self.start_time = time.time()
        
    def add_total(self, count=1):
        with self.lock:
            self.total_frames += count
            
    def add_processed(self, count=1):
        with self.lock:
            self.processed_frames += count
            
    def add_skipped(self, count=1):
        with self.lock:
            self.skipped_frames += count
            
    def add_track(self, class_id, track_id):
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            if track_key not in self.unique_tracks:
                self.unique_tracks.add(track_key)
                if 0 <= class_id < len(CLASSES):
                    self.class_counts[CLASSES[class_id]] += 1
    
    def get_progress_info(self):
        with self.lock:
            actual_processed = self.processed_frames + self.skipped_frames
            progress = (actual_processed / max(1, self.total_frames)) * 100
            elapsed = time.time() - self.start_time
            
            if actual_processed > 0:
                remaining = (elapsed / actual_processed) * (self.total_frames - actual_processed)
            else:
                remaining = 0
            
            skip_rate = (self.skipped_frames / max(1, actual_processed)) * 100
                
            return {
                'progress_percent': min(100, progress),
                'processed': self.processed_frames,
                'skipped': self.skipped_frames,
                'total': self.total_frames,
                'elapsed_time': elapsed,
                'estimated_remaining': remaining,
                'unique_tracks': len(self.unique_tracks),
                'class_counts': dict(self.class_counts),
                'skip_rate': skip_rate
            }

class BalancedOptimalProcessor:
    """Memory-balanced video processor"""
    
    def __init__(self, model_pool):
        self.model_pool = model_pool
        self.tracker = PerformanceTracker()
        self.shutdown_event = Event()
        self.completion_event = Event()
        self.termination_lock = Lock()
        self.preprocessors_finished = 0
        self.predictors_finished = 0
        self.last_cleanup = time.time()

    def aggressive_memory_cleanup(self):
        """Agresif bellek temizliği"""
        current_time = time.time()
        if current_time - self.last_cleanup > SETTINGS['MEMORY_CLEANUP_INTERVAL']:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # GPU senkronizasyonu
                self.last_cleanup = current_time
                print(f"🧹 Aggressive memory cleanup - RAM freed")
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")

    def reader_thread(self, video_capture, read_queue, total_frames):
        """Memory-efficient frame reading with skip"""
        frames_read = 0
        frames_sent = 0
        read_errors = 0
        
        print(f"📖 Memory-efficient reader başladı: {total_frames} frame (her {FRAME_SKIP_INTERVAL} frame'de bir)")
        
        try:
            while frames_read < total_frames and not self.shutdown_event.is_set():
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # Frame skipping - MEMORY SAVE
                    if frames_read % FRAME_SKIP_INTERVAL == 0:
                        # Memory-efficient frame handling
                        frame_copy = frame.copy()  # Shallow copy
                        
                        timeout_counter = 0
                        while not self.shutdown_event.is_set() and timeout_counter < 3:  # Daha kısa timeout
                            try:
                                read_queue.put((frames_read, frame_copy), timeout=0.3)
                                frames_sent += 1
                                break
                            except queue.Full:
                                timeout_counter += 1
                                continue
                        
                        if timeout_counter >= 3:
                            print(f"⚠️ Queue dolu, frame {frames_read} atlandı")
                            del frame_copy  # Explicit cleanup
                            self.tracker.add_skipped()
                        else:
                            self.tracker.add_total()
                    else:
                        self.tracker.add_skipped()
                        
                    frames_read += 1
                    
                    # Daha sık progress report
                    if frames_read % 1000 == 0:
                        print(f"📖 Okunan: {frames_read}/{total_frames}, Gönderilen: {frames_sent}")
                        
                    # Memory cleanup during reading
                    if frames_read % 2000 == 0:
                        gc.collect()
                        
                except Exception as e:
                    read_errors += 1
                    if read_errors > 5:  # Daha düşük tolerance
                        print(f"❌ Çok fazla okuma hatası")
                        break
                    logger.error(f"Frame okuma hatası: {e}")
                    continue
            
            # Sonlandırma sinyalleri
            for _ in range(NUM_PREPROCESSOR_THREADS):
                try:
                    read_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                    
            print(f"📖 Okuma tamamlandı: {frames_read} toplam, {frames_sent} gönderildi")
            
        except Exception as e:
            logger.error(f"Reader fatal error: {e}")

    def preprocessor_worker(self, worker_id, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor):
        """Memory-efficient preprocessor"""
        frames_processed = 0
        
        print(f"🔧 Memory-efficient Preprocessor {worker_id} başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = read_queue.get(timeout=0.8)  # Daha kısa timeout
                    
                    if item is None:
                        try:
                            preprocess_queue.put(None, timeout=0.8)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.preprocessors_finished += 1
                            print(f"🔧 Preprocessor {worker_id} bitti ({self.preprocessors_finished}/{NUM_PREPROCESSOR_THREADS})")
                        break
                        
                    frame_idx, frame = item
                    
                    # Memory-efficient frame processing
                    processed_frame = self._memory_efficient_process_frame(frame, roi_vertical, roi_horizontal, resize_factor)
                    
                    # Explicit cleanup
                    del frame
                    
                    if processed_frame is not None:
                        try:
                            preprocess_queue.put((frame_idx, processed_frame), timeout=0.8)
                            frames_processed += 1
                        except queue.Full:
                            print(f"⚠️ Preprocess queue dolu, worker {worker_id}")
                            del processed_frame  # Cleanup on failure
                            continue
                    
                    # Local memory cleanup
                    if frames_processed % 50 == 0:
                        gc.collect()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Preprocessor {worker_id} hatası: {e}")
                    
            print(f"🔧 Preprocessor {worker_id} tamamlandı: {frames_processed} frame")
            
        except Exception as e:
            logger.error(f"Preprocessor {worker_id} fatal error: {e}")

    def _memory_efficient_process_frame(self, frame, roi_vertical, roi_horizontal, resize_factor):
        """Memory-optimized frame processing"""
        try:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # In-place resize (memory efficient)
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # In-place ROI masking
            roi_top = int(new_height * (1 - roi_vertical))
            roi_side = int(new_width * roi_horizontal)
            
            if roi_top > 0:
                frame_resized[:roi_top] = 0
            if roi_side > 0:
                frame_resized[:, :roi_side] = 0
                frame_resized[:, new_width-roi_side:] = 0
            
            # Ensure contiguous memory layout
            if not frame_resized.flags['C_CONTIGUOUS']:
                frame_resized = np.ascontiguousarray(frame_resized)
                    
            return frame_resized
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
            return None

    def batch_collector(self, preprocess_queue, batch_queue):
        """Memory-efficient batch collection"""
        current_batch = []
        current_indices = []
        none_count = 0
        batch_id = 0
        
        print(f"📦 Memory-efficient batch collector başladı (batch size: {BATCH_SIZE})")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = preprocess_queue.get(timeout=0.8)
                    
                    if item is None:
                        none_count += 1
                        print(f"📦 Sonlandırma sinyali {none_count}/{NUM_PREPROCESSOR_THREADS}")
                        
                        if none_count >= NUM_PREPROCESSOR_THREADS:
                            if current_batch:
                                batch_id += 1
                                print(f"📦 SON batch gönderildi: {len(current_batch)} frame")
                                try:
                                    batch_queue.put((current_batch, current_indices, batch_id), timeout=1.5)
                                except queue.Full:
                                    print("⚠️ Son batch gönderilemedi")
                                    # Cleanup on failure
                                    del current_batch, current_indices
                            
                            for _ in range(NUM_PREDICTOR_THREADS):
                                try:
                                    batch_queue.put(None, timeout=0.8)
                                except queue.Full:
                                    pass
                            break
                        continue
                        
                    frame_idx, frame = item
                    current_batch.append(frame)
                    current_indices.append(frame_idx)
                    
                    # Smaller batch size for memory efficiency
                    if len(current_batch) >= BATCH_SIZE:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, batch_id), timeout=0.8)
                            print(f"📦 Batch {batch_id} gönderildi: {len(current_batch)} frame")
                        except queue.Full:
                            print(f"⚠️ Batch queue dolu: batch {batch_id}")
                            # Cleanup on failure
                            del current_batch, current_indices
                            current_batch = []
                            current_indices = []
                            continue
                            
                        current_batch = []
                        current_indices = []
                        
                        # Memory cleanup after each batch
                        if batch_id % 10 == 0:
                            gc.collect()
                        
                except queue.Empty:
                    # Smaller timeout batch size
                    if current_batch and len(current_batch) >= BATCH_SIZE // 3:  # Daha küçük threshold
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, batch_id), timeout=0.5)
                            print(f"📦 Timeout batch {batch_id}: {len(current_batch)} frame")
                            current_batch = []
                            current_indices = []
                        except queue.Full:
                            pass
                    continue
                except Exception as e:
                    logger.error(f"Batch collector hatası: {e}")
                    
            print(f"📦 Batch collector tamamlandı: {batch_id} batch oluşturuldu")
            
        except Exception as e:
            logger.error(f"Batch collector fatal error: {e}")

    def predictor_worker(self, worker_id, batch_queue, result_queue, score_threshold):
        """GPU-optimized predictor worker"""
        model = None
        batches_processed = 0
        
        print(f"🤖 GPU-optimized Predictor {worker_id} başladı")
        
        try:
            model = self.model_pool.get_model()
            
            while not self.shutdown_event.is_set():
                try:
                    batch_data = batch_queue.get(timeout=1.5)
                    
                    if batch_data is None:
                        try:
                            result_queue.put(None, timeout=0.8)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.predictors_finished += 1
                            print(f"🤖 Predictor {worker_id} bitti ({self.predictors_finished}/{NUM_PREDICTOR_THREADS})")
                        break
                        
                    frames, indices, batch_id = batch_data
                    
                    # Frequent memory cleanup
                    if batches_processed % 5 == 0:
                        self.aggressive_memory_cleanup()
                    
                    # Optimized YOLO inference
                    processed_results = self._optimized_yolo_inference(model, frames, indices, batch_id, score_threshold)
                    
                    # Explicit cleanup
                    del frames
                    
                    if processed_results:
                        self.tracker.add_processed(len(processed_results))
                        try:
                            result_queue.put(processed_results, timeout=0.8)
                        except queue.Full:
                            print(f"⚠️ Result queue dolu: batch {batch_id}")
                            del processed_results  # Cleanup on failure
                        
                        print(f"🤖 Worker {worker_id} batch {batch_id} tamamlandı: {len(processed_results)} frame")
                    
                    batches_processed += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Predictor {worker_id} hatası: {e}")
                    # Emergency cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            print(f"🤖 Predictor {worker_id} tamamlandı: {batches_processed} batch işlendi")
                    
        except Exception as e:
            logger.error(f"Predictor {worker_id} fatal error: {e}")
        finally:
            if model:
                self.model_pool.return_model(model)

    def _optimized_yolo_inference(self, model, frames, indices, batch_id, score_threshold):
        """Memory & GPU optimized YOLO inference"""
        try:
            # YOLO inference with optimizations
            results = model.track(
                frames,
                device=device,
                conf=score_threshold,
                imgsz=640,
                verbose=False,
                persist=True,
                tracker=str(tracker_yaml_path),
                half=True,
                augment=False,  # Disable augmentation for speed
                agnostic_nms=True  # Faster NMS
            )
            
            processed_results = []
            for i, result in enumerate(results):
                if self.shutdown_event.is_set():
                    break
                    
                # Extract tracking info
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        class_id = int(box.cls)
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id)
                            self.tracker.add_track(class_id, track_id)
                
                # Create annotated frame
                annotated = result.plot()
                processed_results.append({
                    'frame_idx': indices[i],
                    'annotated_frame': annotated,
                    'batch_id': batch_id
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"YOLO inference hatası: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def writer_thread(self, result_queue, cv2writer):
        """Memory-efficient video writer"""
        frame_buffer = {}
        expected_frame_idx = 0
        none_count = 0
        frames_written = 0
        
        print(f"🎬 Memory-efficient video yazma başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    results = result_queue.get(timeout=1.5)
                    
                    if results is None:
                        none_count += 1
                        print(f"🎬 Writer sonlandırma sinyali {none_count}/{NUM_PREDICTOR_THREADS}")
                        
                        if none_count >= NUM_PREDICTOR_THREADS:
                            break
                        continue
                        
                    # Add to buffer
                    for result in results:
                        frame_idx = result['frame_idx']
                        frame_buffer[frame_idx] = result['annotated_frame']
                    
                    # Write sequential frames
                    while expected_frame_idx in frame_buffer:
                        frame = frame_buffer.pop(expected_frame_idx)
                        cv2writer.write(frame)
                        frames_written += 1
                        expected_frame_idx += 1
                        
                        if frames_written % 100 == 0:  # Daha sık rapor
                            print(f"🎬 Yazılan: {frames_written} frame")
                    
                    # Smaller buffer control - memory efficiency
                    if len(frame_buffer) > 100:  # 200'den 100'e düştü
                        print(f"⚠️ Frame buffer: {len(frame_buffer)} frame")
                        if frame_buffer:
                            min_available = min(frame_buffer.keys())
                            if min_available > expected_frame_idx + 50:  # 100'den 50'ye
                                print(f"⚠️ Gap atlama: {expected_frame_idx} → {min_available}")
                                expected_frame_idx = min_available
                    
                    # Memory cleanup during writing
                    if frames_written % 500 == 0:
                        gc.collect()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer hatası: {e}")
            
            # Write remaining frames
            if frame_buffer:
                print(f"🎬 Kalan {len(frame_buffer)} frame yazılıyor...")
                for idx in sorted(frame_buffer.keys()):
                    cv2writer.write(frame_buffer[idx])
                    frames_written += 1
            
            print(f"🎬 Video yazma tamamlandı: {frames_written} frame")
            self.completion_event.set()
            
        except Exception as e:
            logger.error(f"Writer fatal error: {e}")
            self.completion_event.set()

def create_tracking_counters():
    st.write("### 🎯 Detection Results")
    counter_cols = st.columns(4)
    counter_placeholders = {}
    colors = ["🟡", "🔴", "🟠", "🔵"]
    
    for i, class_name in enumerate(CLASSES):
        with counter_cols[i]:
            counter_placeholders[class_name] = st.empty()
            counter_placeholders[class_name].metric(f"{colors[i]} {class_name}", "0")
    
    return counter_placeholders

def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

# Setup directories
temp_dir = "./temp"
os.makedirs(temp_dir, exist_ok=True)

temp_file_input = f"{temp_dir}/video_input.mp4"
temp_file_infer = f"{temp_dir}/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

def processVideo(video_file, score_threshold, roi_vertical=0.6, roi_horizontal=0.2, resize_factor=0.5):
    """Balanced RAM-GPU optimized video processing"""
    
    write_bytesio_to_file(temp_file_input, video_file)
    
    video_capture = cv2.VideoCapture(temp_file_input)
    if not video_capture.isOpened():
        st.error('Video dosyası açılamadı')
        return
        
    # Video özellikleri
    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    processed_width = int(_width * resize_factor)
    processed_height = int(_height * resize_factor)

    st.write("**Video Analizi:**")
    st.write(f"Giriş: {_width}x{_height} @ {_fps:.1f} FPS | {_total_frames} frame")
    st.write(f"Çıkış: {processed_width}x{processed_height}")
    
    st.write("**Balanced System Settings:**")
    settings_info = (f"Batch: {BATCH_SIZE} (memory-optimized) | "
                    f"Predictors: {NUM_PREDICTOR_THREADS} (GPU-optimized) | "
                    f"Preprocessors: {NUM_PREPROCESSOR_THREADS} | "
                    f"Frame Skip: {FRAME_SKIP_INTERVAL}")
    st.write(settings_info)

    # UI elementleri
    progress_bar = st.progress(0, text="Balanced processing başlatılıyor...")
    counter_placeholders = create_tracking_counters()
    performance_dashboard = st.empty()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2writer = cv2.VideoWriter(temp_file_infer, fourcc, _fps, (processed_width, processed_height))

    # Balanced Processor
    processor = BalancedOptimalProcessor(st.session_state.optimized_model_pool)

    # Memory-efficient queue'lar
    read_queue = queue.Queue(maxsize=SETTINGS['READ_QUEUE_SIZE'])
    preprocess_queue = queue.Queue(maxsize=SETTINGS['PREPROCESS_QUEUE_SIZE'])
    batch_queue = queue.Queue(maxsize=SETTINGS['BATCH_QUEUE_SIZE'])
    result_queue = queue.Queue(maxsize=SETTINGS['RESULT_QUEUE_SIZE'])

    threads = []
    start_time = time.time()

    try:
        print(f"⚡ Balanced RAM-GPU video işleme başlıyor:")
        print(f"   Total frames: {_total_frames}")
        print(f"   Expected processed: ~{_total_frames//FRAME_SKIP_INTERVAL}")
        print(f"   Memory-optimized batch size: {BATCH_SIZE}")
        print(f"   GPU-optimized predictors: {NUM_PREDICTOR_THREADS}")
        
        # Thread'leri oluştur
        reader = threading.Thread(
            target=processor.reader_thread,
            args=(video_capture, read_queue, _total_frames),
            name="MemoryReader", daemon=True
        )
        threads.append(reader)

        # Memory-efficient preprocessor threads
        for i in range(NUM_PREPROCESSOR_THREADS):
            preprocessor = threading.Thread(
                target=processor.preprocessor_worker,
                args=(i, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor),
                name=f"MemoryPreprocessor-{i}", daemon=True
            )
            threads.append(preprocessor)

        # Batch collector
        collector = threading.Thread(
            target=processor.batch_collector,
            args=(preprocess_queue, batch_queue),
            name="MemoryCollector", daemon=True
        )
        threads.append(collector)

        # GPU-optimized predictor threads
        for i in range(NUM_PREDICTOR_THREADS):
            predictor = threading.Thread(
                target=processor.predictor_worker,
                args=(i, batch_queue, result_queue, score_threshold),
                name=f"GPUPredictor-{i}", daemon=True
            )
            threads.append(predictor)

        # Memory-efficient writer thread
        writer = threading.Thread(
            target=processor.writer_thread,
            args=(result_queue, cv2writer),
            name="MemoryWriter", daemon=True
        )
        threads.append(writer)

        # Tüm thread'leri başlat
        for t in threads:
            t.start()

        # Enhanced progress tracking
        last_update = time.time()
        
        while not processor.completion_event.is_set() and not processor.shutdown_event.is_set():
            current_time = time.time()
            
            if current_time - last_update >= 1.0:
                progress_info = processor.tracker.get_progress_info()
                
                # Progress bar güncelle
                if progress_info['total'] > 0:
                    total_handled = progress_info['processed'] + progress_info['skipped']
                    progress = min(total_handled / _total_frames, 1.0)
                    elapsed_time = progress_info['elapsed_time']
                    remaining_time = progress_info['estimated_remaining']
                    
                    progress_bar.progress(
                        progress,
                        text=f"Balanced Processing: {progress_info['processed']} işlendi, {progress_info['skipped']} atlandı ({progress*100:.1f}%) | Kalan: {remaining_time:.0f}s"
                    )
                
                # Enhanced performance dashboard
                with performance_dashboard.container():
                    perf_cols = st.columns(5)
                    
                    with perf_cols[0]:
                        elapsed_min = elapsed_time / 60
                        st.metric("Süre", f"{elapsed_min:.1f} dk")
                        
                    with perf_cols[1]:
                        if progress_info['processed'] > 0:
                            fps = progress_info['processed'] / elapsed_time
                            st.metric("İşleme FPS", f"{fps:.1f}")
                        else:
                            st.metric("İşleme FPS", "0.0")
                        
                    with perf_cols[2]:
                        remaining_min = remaining_time / 60
                        st.metric("Kalan", f"{remaining_min:.1f} dk")
                        
                    with perf_cols[3]:
                        st.metric("Skip Rate", f"{progress_info['skip_rate']:.1f}%")
                        
                    with perf_cols[4]:
                        st.metric("Tracks", f"🎯 {progress_info['unique_tracks']}")
                
                # Counter'ları güncelle
                for class_name in CLASSES:
                    count = progress_info['class_counts'].get(class_name, 0)
                    counter_placeholders[class_name].metric(f"{class_name}", f"{count}")
                
                last_update = current_time
            
            time.sleep(0.5)

        # Completion waiting
        print("⏳ Balanced processing tamamlanması bekleniyor...")
        if not processor.completion_event.wait(timeout=300):
            st.warning("⚠️ İşlem zaman aşımı - zorla tamamlanıyor")
            processor.shutdown_event.set()

    except KeyboardInterrupt:
        st.warning("🛑 İşlem kullanıcı tarafından durduruldu")
        processor.shutdown_event.set()
        
    except Exception as e:
        st.error(f"❌ İşlem hatası: {e}")
        processor.shutdown_event.set()
        
    finally:
        # Cleanup
        processor.shutdown_event.set()
        video_capture.release()
        cv2writer.release()
        
        # Final aggressive cleanup
        processor.aggressive_memory_cleanup()
        
        # Results
        end_time = time.time()
        final_progress = processor.tracker.get_progress_info()
        processing_time = end_time - start_time
        
        # Performance analysis
        theoretical_time = _total_frames / _fps
        speedup_factor = theoretical_time / processing_time if processing_time > 0 else 0
        skip_speedup = FRAME_SKIP_INTERVAL if FRAME_SKIP_INTERVAL > 1 else 1
        total_speedup = speedup_factor * skip_speedup
        
        st.success(f"⚡ Balanced RAM-GPU Processing Tamamlandı!")
        
        # Enhanced results summary
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("İşlem Süresi", f"{processing_time:.1f}s")
            st.metric("Base Speedup", f"{speedup_factor:.1f}x")
            
        with summary_cols[1]:
            if final_progress['processed'] > 0:
                avg_fps = final_progress['processed'] / processing_time
                st.metric("Avg FPS", f"{avg_fps:.1f}")
            else:
                st.metric("Avg FPS", "0.0")
            total_handled = final_progress['processed'] + final_progress['skipped']
            completion_rate = (total_handled / max(1, final_progress['total'])) * 100
            st.metric("Completion", f"{completion_rate:.1f}%")
            
        with summary_cols[2]:
            st.metric("Skip Speedup", f"{skip_speedup:.1f}x")
            st.metric("Total Speedup", f"{total_speedup:.1f}x")
            
        with summary_cols[3]:
            st.metric("Total Tracks", f"{final_progress['unique_tracks']}")
            if processing_time > 0:
                tracks_per_min = (final_progress['unique_tracks'] / processing_time) * 60
                st.metric("Tracks/Min", f"{tracks_per_min:.1f}")
        
        # Frame processing stats
        st.write("**📊 Processing Statistics:**")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("İşlenen Frame", f"{final_progress['processed']}")
        with stats_cols[1]:
            st.metric("Atlanan Frame", f"{final_progress['skipped']}")
        with stats_cols[2]:
            st.metric("Skip Efficiency", f"{final_progress['skip_rate']:.1f}%")
        
        # Performance evaluation
        if completion_rate >= 95 and total_speedup >= 1.8:
            st.success("🎯 MÜKEMMEL: Balanced RAM-GPU optimization başarılı!")
        elif completion_rate >= 85 and total_speedup >= 1.4:
            st.info("👍 İYİ: Solid balanced performance")
        else:
            st.warning("📊 GELİŞTİRİLEBİLİR: Daha fazla optimization gerekli")
        
        print(f"✅ Balanced processing tamamlandı:")
        print(f"   Processed: {final_progress['processed']} frames")
        print(f"   Skipped: {final_progress['skipped']} frames") 
        print(f"   Total speedup: {total_speedup:.1f}x")
        print(f"   Memory efficiency: OPTIMIZED")
        print(f"   GPU efficiency: ENHANCED")

# UI Layout
col1, col2 = st.columns(2)
with col1:
    if os.path.exists(temp_file_infer):
        try:
            with open(temp_file_infer, "rb") as f:
                st.download_button(
                    label="📥 Download Balanced-Optimized Video",
                    data=f,
                    file_name="balanced_optimal_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        except Exception:
            st.info("Download available after processing")
    else:
        st.info("Balanced-optimized video will appear here")
        
with col2:
    if st.button('🔄 Reset Balanced System', use_container_width=True, type="primary"):
        # Complete reset with aggressive cleanup
        for key in ['optimal_settings', 'optimized_model_pool']:
            if key in st.session_state:
                del st.session_state[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        st.rerun()

st.title("⚖️ Balanced RAM-GPU Video Processor")
st.write("**✨ Optimizations:** Memory Efficient + GPU Balanced + Frame Skipping")

# System info
sys_info = st.expander("🔧 System Configuration")
with sys_info:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Memory Optimizations:**")
        st.write(f"- Small batch size: {BATCH_SIZE}")
        st.write(f"- Reduced queue sizes: {SETTINGS['READ_QUEUE_SIZE']}")
        st.write(f"- Aggressive cleanup: {SETTINGS['MEMORY_CLEANUP_INTERVAL']}s")
        st.write(f"- Frame skipping: {FRAME_SKIP_INTERVAL}x")
    with col2:
        st.write("**GPU Optimizations:**")
        st.write(f"- Multiple predictors: {NUM_PREDICTOR_THREADS}")
        st.write(f"- Half precision: Enabled")
        st.write(f"- Fast NMS: Enabled") 
        st.write(f"- Model pooling: Optimized")

# File upload
video_file = st.file_uploader(
    "📹 Upload Video File", 
    type=".mp4", 
    disabled=st.session_state.runningInference
)

# Settings
score_threshold = st.slider(
    "Detection Confidence", 
    0.0, 1.0, 0.5, 0.05, 
    disabled=st.session_state.runningInference
)

# Preprocessing
st.write("---")
st.write("### 🔧 Preprocessing Configuration")

prep_cols = st.columns(3)
with prep_cols[0]:
    resize_factor = st.slider(
        "Size Factor", 
        0.1, 1.0, 0.5, 0.1,
        disabled=st.session_state.runningInference,
        help="Lower = faster processing, less memory"
    )
with prep_cols[1]:
    roi_vertical = st.slider(
        "Vertical ROI", 
        0.1, 1.0, 0.6, 0.1,
        disabled=st.session_state.runningInference,
        help="Keep bottom portion"
    )
with prep_cols[2]:
    roi_horizontal = st.slider(
        "Horizontal ROI", 
        0.0, 0.5, 0.2, 0.1,
        disabled=st.session_state.runningInference,
        help="Remove side margins"
    )

# Process button
if video_file is not None:
    if st.button(
        '⚖️ Start Balanced Processing',
        use_container_width=True,
        disabled=st.session_state.runningInference,
        type="secondary",
        key="processing_button"
    ):
        st.info(f"⚖️ Processing {video_file.name} with balanced RAM-GPU optimization...")
        processVideo(video_file, score_threshold, roi_vertical, roi_horizontal, resize_factor)