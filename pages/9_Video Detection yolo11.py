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
    page_title="Road Defect Detection & Tracking",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Updated model path for YOLO11n
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"  # Updated to YOLO11n
tracker_yaml_path = ROOT / "./models/bytetrack.yaml"

# HYBRID OPTIMAL SETTINGS - Best balance of speed + stability
class OptimalSettings:
    """Smart settings calculator based on hardware analysis"""
    
    def __init__(self):
        self.gpu_memory_gb = 0
        self.gpu_name = "CPU"
        
        if torch.cuda.is_available():
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_name = torch.cuda.get_device_name(0)
            
        self.cpu_cores = os.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Calculate OPTIMAL settings (not too conservative, not too aggressive)
        self.settings = self._calculate_optimal_settings()
        
        print(f"⚡ Hybrid Optimal Settings calculated:")
        print(f"   Hardware: {self.gpu_name} ({self.gpu_memory_gb:.1f}GB), {self.cpu_cores} cores")
        print(f"   Settings: {self.settings}")
        
    def _calculate_optimal_settings(self):
        """Calculate the sweet spot for stable high performance"""
        
        batch_size=min(32,int(self.gpu_memory_gb *4))
        num_predictors= 2
        num_preprocessors=2
        base_queue=96
        batch_queue_size=base_queue //4
        
        
        return {
            'BATCH_SIZE': batch_size,
            'NUM_PREDICTOR_THREADS': num_predictors,
            'NUM_PREPROCESSOR_THREADS': num_preprocessors,
            'READ_QUEUE_SIZE': base_queue,
            'PREPROCESS_QUEUE_SIZE': base_queue,
            'BATCH_QUEUE_SIZE': batch_queue_size,  # Smaller for batches
            'RESULT_QUEUE_SIZE': base_queue,
            'MEMORY_CLEANUP_INTERVAL': 20,  # seconds
            'EXPECTED_GPU_USAGE': min(85, 50 + (self.gpu_memory_gb * 5))  # Realistic target
        }

# Initialize optimal settings
if 'optimal_settings' not in st.session_state:
    st.session_state.optimal_settings = OptimalSettings()

SETTINGS = st.session_state.optimal_settings.settings
BATCH_SIZE = SETTINGS['BATCH_SIZE']
NUM_PREDICTOR_THREADS = SETTINGS['NUM_PREDICTOR_THREADS']
NUM_PREPROCESSOR_THREADS = SETTINGS['NUM_PREPROCESSOR_THREADS']

device = "cuda" if torch.cuda.is_available() else "cpu"


# Basit ve güvenli model havuzu yönetimi
class OptimizedModelPool:
    def __init__(self, model_path, pool_size):
        self.model_path = model_path
        self.pool_size = pool_size
        self.models = []
        self.lock = Lock()

        self._initialize_pool()

    def _initialize_pool(self):
        """Havuzdaki modelleri baştan oluşturur."""
        print(f"🚀 {self.pool_size} model yükleniyor...")

        for i in range(self.pool_size):
            model = self._create_model()
            if model:
                self.models.append(model)
                print(f"✅ Model {i + 1}/{self.pool_size} hazır")
            else:
                print(f"❌ Model {i + 1} yüklenemedi")

        if not self.models:
            print("⚠️ Hiç model yüklenemedi, yedek olarak bir model oluşturuluyor...")
            fallback_model = self._create_model()
            if fallback_model:
                self.models.append(fallback_model)
            print(f"📦 Yedek model oluşturuldu: {len(self.models)} model")

    def _create_model(self):
        """Yeni bir model oluşturur, optimize eder."""
        try:
            # Load YOLO11n model
            model = YOLO(self.model_path)
            model.tracker = str(tracker_yaml_path)

            if torch.cuda.is_available():
                model.model.half().to(device)
              
                dummy = torch.randn(1, 3, 640, 640).half().to(device)
                with torch.no_grad():
                    _ = model.model(dummy)

            return model
        except Exception as e:
            logger.error(f"Model oluşturulamadı: {e}")
            return None

    def get_model(self):
        """Kullanım için model alır, yoksa yeni bir tane oluşturur."""
        with self.lock:
            if self.models:
                return self.models.pop()

        # Eğer havuz boşsa, kısa bekleme sonrası yeni model oluştur
        time.sleep(0.1)
        return self._create_model()

    def return_model(self, model):
        """Kullanım sonrası modeli geri havuza koyar."""
        if not model:
            return

        with self.lock:
            if len(self.models) < self.pool_size:
                self.models.append(model)
            else:
                # Havuz doluysa, bu modeli serbest bırak
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# Initialize optimized model pool
if 'optimized_model_pool' not in st.session_state:
    st.session_state.optimized_model_pool = OptimizedModelPool(
        MODEL_LOCAL_PATH, 
        NUM_PREDICTOR_THREADS + 1  # One extra for safety
    )

# Updated class names for road defect detection (6 classes: 0-5)
CLASSES = [
    "Longitudinal Crack",    # 0: D00 - Boyuna çatlak
    "Transverse Crack",      # 1: D10 - Enine çatlak
    "Alligator Crack",       # 2: D20 - Timsah derisi çatlağı
    "Pothole",               # 3: D40 - Çukur
    
]

class PerformanceTracker:
    """Sade ve hızlı performans takibi"""
    def __init__(self):
        self.lock = Lock()
        self.total_frames = 0
        self.processed_frames = 0
        self.unique_tracks = set()
        self.class_counts = defaultdict(int)
        self.start_time = time.time()
        
    def add_total(self, count=1):
        with self.lock:
            self.total_frames += count
            
    def add_processed(self, count=1):
        with self.lock:
            self.processed_frames += count
            
    def add_track(self, class_id, track_id):
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            if track_key not in self.unique_tracks:
                self.unique_tracks.add(track_key)
                if 0 <= class_id < len(CLASSES):
                    self.class_counts[CLASSES[class_id]] += 1
    
    def get_progress_info(self):
        with self.lock:
            progress = (self.processed_frames / max(1, self.total_frames)) * 100
            elapsed = time.time() - self.start_time
            
            if self.processed_frames > 0:
                remaining = (elapsed / self.processed_frames) * (self.total_frames - self.processed_frames)
            else:
                remaining = 0
                
            return {
                'progress_percent': min(100, progress),
                'processed': self.processed_frames,
                'total': self.total_frames,
                'elapsed_time': elapsed,
                'estimated_remaining': remaining,
                'unique_tracks': len(self.unique_tracks),
                'class_counts': dict(self.class_counts)
            }


class HybridOptimalProcessor:
    """Optimize edilmiş video işleme sistemi - VIDEO OUTPUT REMOVED"""
    
    def __init__(self, model_pool):
        self.model_pool = model_pool
        self.tracker = PerformanceTracker()
        self.shutdown_event = Event()
        self.completion_event = Event()
        self.termination_lock = Lock()
        self.preprocessors_finished = 0
        self.predictors_finished = 0
        self.last_cleanup = time.time()

    def memory_cleanup(self):
        """Akıllı bellek temizliği"""
        current_time = time.time()
        if current_time - self.last_cleanup > SETTINGS['MEMORY_CLEANUP_INTERVAL']:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.last_cleanup = current_time
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")

    def reader_thread(self, video_capture, read_queue, total_frames):
        """Video okuma thread'i"""
        frames_read = 0
        read_errors = 0
        
        print(f"📖 Video okuma başladı: {total_frames} frame")
        
        try:
            while frames_read < total_frames and not self.shutdown_event.is_set():
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                        
                    # Frame'i kuyruğa ekle
                    timeout_counter = 0
                    while not self.shutdown_event.is_set() and timeout_counter < 5:
                        try:
                            read_queue.put((frames_read, frame), timeout=0.5)
                            break
                        except queue.Full:
                            timeout_counter += 1
                            continue
                    
                    if timeout_counter >= 5:
                        print(f"⚠️ Queue dolu: frame {frames_read}")
                        
                    frames_read += 1
                    self.tracker.add_total()
                    
                    # İlerleme raporu
                    if frames_read % 500 == 0:
                        print(f"📖 Okunan: {frames_read}/{total_frames}")
                        
                except Exception as e:
                    read_errors += 1
                    if read_errors > 10:
                        print(f"❌ Çok fazla okuma hatası")
                        break
                    logger.error(f"Frame okuma hatası: {e}")
                    continue
            
            # Sonlandırma sinyalleri gönder
            for _ in range(NUM_PREPROCESSOR_THREADS):
                try:
                    read_queue.put(None, timeout=2.0)
                except queue.Full:
                    pass
                    
            print(f"📖 Okuma tamamlandı: {frames_read} frame, {read_errors} hata")
            
        except Exception as e:
            logger.error(f"Reader fatal error: {e}")

    def preprocessor_worker(self, worker_id, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor):
        """Frame ön işleme worker'ı"""
        frames_processed = 0
        
        print(f"🔧 Preprocessor {worker_id} başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = read_queue.get(timeout=1.0)
                    
                    if item is None:
                        # Sonlandırma sinyali ilet
                        try:
                            preprocess_queue.put(None, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.preprocessors_finished += 1
                            print(f"🔧 Preprocessor {worker_id} bitti ({self.preprocessors_finished}/{NUM_PREPROCESSOR_THREADS})")
                        break
                        
                    frame_idx, frame = item
                    
                    # Frame işle
                    processed_frame = self._process_frame(frame, roi_vertical, roi_horizontal, resize_factor)
                    
                    if processed_frame is not None:
                        try:
                            preprocess_queue.put((frame_idx, processed_frame), timeout=1.0)
                            frames_processed += 1
                        except queue.Full:
                            print(f"⚠️ Preprocess queue dolu, worker {worker_id}")
                            continue
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Preprocessor {worker_id} hatası: {e}")
                    
            print(f"🔧 Preprocessor {worker_id} tamamlandı: {frames_processed} frame")
            
        except Exception as e:
            logger.error(f"Preprocessor {worker_id} fatal error: {e}")

    def _process_frame(self, frame, roi_vertical, roi_horizontal, resize_factor):
        """Frame işleme - temiz ve hızlı"""
        try:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # Yeniden boyutlandır
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # ROI maskeleme
            roi_top = int(new_height * (1 - roi_vertical))
            roi_side = int(new_width * roi_horizontal)
            
            if roi_top > 0:
                frame_resized[:roi_top] = 0
            if roi_side > 0:
                frame_resized[:, :roi_side] = 0
                frame_resized[:, new_width-roi_side:] = 0
                    
            return frame_resized
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
            return None

    def batch_collector(self, preprocess_queue, batch_queue):
        """Batch toplama - akıllı ve verimli"""
        current_batch = []
        current_indices = []
        none_count = 0
        batch_id = 0
        
        print(f"📦 Batch collector başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = preprocess_queue.get(timeout=1.0)
                    
                    if item is None:
                        none_count += 1
                        print(f"📦 Sonlandırma sinyali {none_count}/{NUM_PREPROCESSOR_THREADS}")
                        
                        if none_count >= NUM_PREPROCESSOR_THREADS:
                            # Son batch'i gönder
                            if current_batch:
                                batch_id += 1
                                print(f"📦 SON batch gönderildi: {len(current_batch)} frame")
                                try:
                                    batch_queue.put((current_batch, current_indices, batch_id), timeout=2.0)
                                except queue.Full:
                                    print("⚠️ Son batch gönderilemedi")
                            
                            # Predictor'lara sonlandırma sinyali
                            for _ in range(NUM_PREDICTOR_THREADS):
                                try:
                                    batch_queue.put(None, timeout=1.0)
                                except queue.Full:
                                    pass
                            break
                        continue
                        
                    frame_idx, frame = item
                    current_batch.append(frame)
                    current_indices.append(frame_idx)
                    
                    # Batch hazır mı?
                    if len(current_batch) >= BATCH_SIZE:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, batch_id), timeout=1.0)
                            print(f"📦 Batch {batch_id} gönderildi: {len(current_batch)} frame")
                        except queue.Full:
                            print(f"⚠️ Batch queue dolu: batch {batch_id}")
                            continue
                            
                        current_batch = []
                        current_indices = []
                        
                except queue.Empty:
                    # Timeout batch kontrolü
                    if current_batch and len(current_batch) >= BATCH_SIZE // 2:
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
        """YOLO tahmin worker'ı - TRACKING DATA ONLY"""
        model = None
        batches_processed = 0
        
        print(f"🤖 Predictor {worker_id} başladı")
        
        try:
            model = self.model_pool.get_model()
            
            while not self.shutdown_event.is_set():
                try:
                    batch_data = batch_queue.get(timeout=2.0)
                    
                    if batch_data is None:
                        # Sonlandırma sinyali ilet
                        try:
                            result_queue.put(None, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.predictors_finished += 1
                            print(f"🤖 Predictor {worker_id} bitti ({self.predictors_finished}/{NUM_PREDICTOR_THREADS})")
                        break
                        
                    frames, indices, batch_id = batch_data
                    
                    # Bellek temizliği
                    if batches_processed % 10 == 0:
                        self.memory_cleanup()
                    
                    # YOLO tahmin yap - ONLY TRACKING DATA
                    processed_results = self._run_yolo_inference(model, frames, indices, batch_id, score_threshold)
                    
                    if processed_results:
                        self.tracker.add_processed(len(frames))
                        try:
                            result_queue.put(processed_results, timeout=1.0)
                        except queue.Full:
                            print(f"⚠️ Result queue dolu: batch {batch_id}")
                        
                        print(f"🤖 Worker {worker_id} batch {batch_id} tamamlandı: {len(frames)} frame")
                    
                    batches_processed += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Predictor {worker_id} hatası: {e}")
                    
            print(f"🤖 Predictor {worker_id} tamamlandı: {batches_processed} batch işlendi")
                    
        except Exception as e:
            logger.error(f"Predictor {worker_id} fatal error: {e}")
        finally:
            if model:
                self.model_pool.return_model(model)

    def _run_yolo_inference(self, model, frames, indices, batch_id, score_threshold):
        """YOLO inference - TRACKING DATA ONLY (NO VIDEO OUTPUT)"""
        try:
            results = model.track(
                frames,
                device=device,
                conf=score_threshold,
                imgsz=640,
                verbose=False,
                persist=True,
                tracker=str(tracker_yaml_path),
                half=True
            )
            
            processed_results = []
            for i, result in enumerate(results):
                if self.shutdown_event.is_set():
                    break
                    
                # Tracking bilgilerini çıkar - VIDEO OUTPUT REMOVED
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        class_id = int(box.cls)
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id)
                            self.tracker.add_track(class_id, track_id)
                
                # ONLY TRACKING DATA - NO ANNOTATED FRAME
                processed_results.append({
                    'frame_idx': indices[i],
                    'batch_id': batch_id,
                    'tracking_data': True  # Just a flag to indicate processing
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"YOLO inference hatası: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def data_processor_thread(self, result_queue):
        """Data processing thread - NO VIDEO WRITING"""
        frames_processed = 0
        none_count = 0
        
        print(f"📊 Data processor başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    results = result_queue.get(timeout=2.0)
                    
                    if results is None:
                        none_count += 1
                        print(f"📊 Data processor sonlandırma sinyali {none_count}/{NUM_PREDICTOR_THREADS}")
                        
                        if none_count >= NUM_PREDICTOR_THREADS:
                            break
                        continue
                        
                    # Process tracking data only - NO VIDEO OPERATIONS
                    for result in results:
                        frames_processed += 1
                        
                        if frames_processed % 200 == 0:
                            print(f"📊 İşlenen: {frames_processed} frame")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Data processor hatası: {e}")
            
            print(f"📊 Data processing tamamlandı: {frames_processed} frame")
            self.completion_event.set()
            
        except Exception as e:
            logger.error(f"Data processor fatal error: {e}")
            self.completion_event.set()


def create_tracking_counters():
    st.write("### 🛣️ Road Defect Detection Results")
    counter_cols = st.columns(3)
    counter_placeholders = {}
    
    # Updated colors and layout for 6 road defect classes
    defect_data = [
        ("🟡", "Longitudinal Crack"),
        ("🔴", "Transverse Crack"), 
        ("🟠", "Alligator Crack"),
        ("🔵", "Pothole"),
      
    ]
    
    for i, (color, class_name) in enumerate(defect_data):
        col_idx = i % 4
        with counter_cols[col_idx]:
            counter_placeholders[class_name] = st.empty()
            counter_placeholders[class_name].metric(f"{color} {class_name}", "0")
    
    return counter_placeholders


def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


# Setup directories
temp_dir = "./temp"
os.makedirs(temp_dir, exist_ok=True)

temp_file_input = f"{temp_dir}/video_input.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False


def processVideo(video_file, score_threshold, roi_vertical=0.6, roi_horizontal=0.2, resize_factor=0.5):
    """Optimize edilmiş video işleme - NO VIDEO OUTPUT"""
    
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
    st.write(f"İşlem: {processed_width}x{processed_height} (Video output disabled for performance)")
    
    st.write("**Sistem Ayarları:**")
    settings_info = (f"Batch Boyutu: {BATCH_SIZE} | "
                    f"Predictor: {NUM_PREDICTOR_THREADS} | "
                    f"Preprocessor: {NUM_PREPROCESSOR_THREADS}")
    st.write(settings_info)

    # UI elementleri
    progress_bar = st.progress(0, text="İşlem başlatılıyor...")
    counter_placeholders = create_tracking_counters()
    performance_dashboard = st.empty()

    # Processor oluştur
    processor = HybridOptimalProcessor(st.session_state.optimized_model_pool)

    # Queue'ları oluştur - NO VIDEO WRITER QUEUE
    read_queue = queue.Queue(maxsize=SETTINGS['READ_QUEUE_SIZE'])
    preprocess_queue = queue.Queue(maxsize=SETTINGS['PREPROCESS_QUEUE_SIZE'])
    batch_queue = queue.Queue(maxsize=SETTINGS['BATCH_QUEUE_SIZE'])
    result_queue = queue.Queue(maxsize=SETTINGS['RESULT_QUEUE_SIZE'])

    threads = []
    start_time = time.time()

    try:
        print(f"⚡ Road defect detection başlıyor: {_total_frames} frame")
        
        # Thread'leri oluştur
        reader = threading.Thread(
            target=processor.reader_thread,
            args=(video_capture, read_queue, _total_frames),
            name="Reader", daemon=True
        )
        threads.append(reader)

        # Preprocessor thread'leri
        for i in range(NUM_PREPROCESSOR_THREADS):
            preprocessor = threading.Thread(
                target=processor.preprocessor_worker,
                args=(i, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor),
                name=f"Preprocessor-{i}", daemon=True
            )
            threads.append(preprocessor)

        # Batch collector
        collector = threading.Thread(
            target=processor.batch_collector,
            args=(preprocess_queue, batch_queue),
            name="Collector", daemon=True
        )
        threads.append(collector)

        # Predictor thread'leri
        for i in range(NUM_PREDICTOR_THREADS):
            predictor = threading.Thread(
                target=processor.predictor_worker,
                args=(i, batch_queue, result_queue, score_threshold),
                name=f"Predictor-{i}", daemon=True
            )
            threads.append(predictor)

        # Data processor thread (replaces video writer)
        data_processor = threading.Thread(
            target=processor.data_processor_thread,
            args=(result_queue,),
            name="DataProcessor", daemon=True
        )
        threads.append(data_processor)

        # Tüm thread'leri başlat
        for t in threads:
            t.start()

        # İlerleme takibi - basitleştirilmiş
        last_update = time.time()
        
        while not processor.completion_event.is_set() and not processor.shutdown_event.is_set():
            current_time = time.time()
            
            if current_time - last_update >= 1.0:
                # Progress info al
                progress_info = processor.tracker.get_progress_info()
                
                # Progress bar güncelle
                if progress_info['total'] > 0:
                    progress = min(progress_info['processed'] / _total_frames, 1.0)
                    elapsed_time = progress_info['elapsed_time']
                    remaining_time = progress_info['estimated_remaining']
                    
                    progress_bar.progress(
                        progress,
                        text=f"İşleniyor: {progress_info['processed']}/{progress_info['total']} frame ({progress*100:.1f}%) | Kalan: {remaining_time:.0f}s"
                    )
                
                # Performans dashboard'u
                with performance_dashboard.container():
                    perf_cols = st.columns(4)
                    
                    with perf_cols[0]:
                        elapsed_min = elapsed_time / 60
                        st.metric("Geçen Süre", f"{elapsed_min:.1f} dk")
                        
                    with perf_cols[1]:
                        if progress_info['processed'] > 0:
                            fps = progress_info['processed'] / elapsed_time
                            st.metric("Ortalama FPS", f"{fps:.1f}")
                        else:
                            st.metric("Ortalama FPS", "0.0")
                        
                    with perf_cols[2]:
                        remaining_min = remaining_time / 60
                        st.metric("Tahmini Kalan", f"{remaining_min:.1f} dk")
                        
                    with perf_cols[3]:
                        st.metric("Bulunan Track", f"🎯 {progress_info['unique_tracks']}")
                
                # Counter'ları güncelle
                for class_name in CLASSES:
                    count = progress_info['class_counts'].get(class_name, 0)
                    counter_placeholders[class_name].metric(f"{class_name}", f"{count}")
                
                last_update = current_time
            
            time.sleep(0.5)

        # Tamamlanma bekleme
        print("⏳ İşlem tamamlanması bekleniyor...")
        if not processor.completion_event.wait(timeout=300):  # 5 dakika max
            st.warning("⚠️ İşlem zaman aşımı - zorla tamamlanıyor")
            processor.shutdown_event.set()

    except KeyboardInterrupt:
        st.warning("🛑 İşlem kullanıcı tarafından durduruldu")
        processor.shutdown_event.set()
        
    except Exception as e:
        st.error(f"❌ İşlem hatası: {e}")
        processor.shutdown_event.set()
        
    finally:
        # Temizlik
        processor.shutdown_event.set()
        video_capture.release()
        
        # Son bellek temizliği
        processor.memory_cleanup()
        
        # Son sonuçlar
        end_time = time.time()
        final_progress = processor.tracker.get_progress_info()
        processing_time = end_time - start_time
        
        # Performans analizi
        theoretical_time = _total_frames / _fps
        speedup_factor = theoretical_time / processing_time if processing_time > 0 else 0
        
        st.success(f"🛣️ Road Defect Detection Tamamlandı!")
        
        # Sonuç özeti
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("İşlem Süresi", f"{processing_time:.1f}s")
            st.metric("Hızlanma Faktörü", f"{speedup_factor:.1f}x")
            
        with summary_cols[1]:
            if final_progress['processed'] > 0:
                avg_fps = final_progress['processed'] / processing_time
                st.metric("Ortalama FPS", f"{avg_fps:.1f}")
            else:
                st.metric("Ortalama FPS", "0.0")
            completion_rate = (final_progress['processed'] / max(1, final_progress['total'])) * 100
            st.metric("Tamamlanma", f"{completion_rate:.1f}%")
            
        with summary_cols[2]:
            st.metric("Toplam Defect Track", f"{final_progress['unique_tracks']}")
            if processing_time > 0:
                tracks_per_min = (final_progress['unique_tracks'] / processing_time) * 60
                st.metric("Track/Dakika", f"{tracks_per_min:.1f}")
        
        # Defect sınıfı özeti
        st.write("### 📊 Tespit Edilen Yol Defektleri:")
        defect_summary_cols = st.columns(3)
        
        class_counts = final_progress['class_counts']
        for i, class_name in enumerate(CLASSES):
            col_idx = i % 3
            with defect_summary_cols[col_idx]:
                count = class_counts.get(class_name, 0)
                if count > 0:
                    st.write(f"**{class_name}**: {count} adet")
                else:
                    st.write(f"{class_name}: {count} adet")
        
        # Performans değerlendirmesi - updated for road defect detection
        if completion_rate >= 95 and speedup_factor >= 1.0:
            st.success("🎯 MÜKEMMEL: Optimal performans elde edildi! Yol defektleri başarıyla tespit edildi.")
        elif completion_rate >= 85 and speedup_factor >= 0.8:
            st.info("👍 İYİ: Solid performans elde edildi. Yol defekt tespiti başarılı.")
        else:
            st.warning("📊 GELİŞTİRİLEBİLİR: Video çözünürlüğü veya batch boyutu azaltılabilir")
        
        print(f"✅ Road defect detection tamamlandı: {final_progress['processed']}/{final_progress['total']} frame")
        
        
# UI Layout - UPDATED FOR NO VIDEO OUTPUT
col1, col2 = st.columns(2)
with col1:
    st.info("🚀 Performance optimized: Video output disabled")
    st.write("Only tracking data and statistics are processed")
        
with col2:
    if st.button('🔄 Reset System', use_container_width=True, type="primary"):
        # Complete reset
        for key in ['optimal_settings', 'optimized_model_pool']:
            if key in st.session_state:
                del st.session_state[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

st.title("🛣️ YOLO11n Road Defect Detection & Tracking")
st.write("**Optimized for Performance**: Video output disabled, only tracking data processed")

# File upload
video_file = st.file_uploader(
    "📹 Upload Road Video File", 
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
        help="Lower = faster processing"
    )
with prep_cols[1]:
    roi_vertical = st.slider(
        "Vertical ROI", 
        0.1, 1.0, 0.6, 0.1,
        disabled=st.session_state.runningInference,
        help="Keep bottom portion (road area)"
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
        '🛣️ Start Road Defect Detection',
        use_container_width=True,
        disabled=st.session_state.runningInference,
        type="secondary",
        key="processing_button"
    ):
        st.info(f"🛣️ Processing {video_file.name} for road defect detection...")
        processVideo(video_file, score_threshold, roi_vertical, roi_horizontal, resize_factor)