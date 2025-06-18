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
    page_title="YOLO11 Road Defect Frame Extractor",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Updated model path for YOLO11
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"  # YOLO11 model
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
        
        print(f"🛣️ YOLO11 Road Defect Settings calculated:")
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
        print(f"🚀 {self.pool_size} YOLO11 model yükleniyor...")

        for i in range(self.pool_size):
            model = self._create_model()
            if model:
                self.models.append(model)
                print(f"✅ YOLO11 Model {i + 1}/{self.pool_size} hazır")
            else:
                print(f"❌ Model {i + 1} yüklenemedi")

        if not self.models:
            print("⚠️ Hiç model yüklenemedi, yedek olarak bir model oluşturuluyor...")
            fallback_model = self._create_model()
            if fallback_model:
                self.models.append(fallback_model)
            print(f"📦 Yedek model oluşturuldu: {len(self.models)} model")

    def _create_model(self):
        """Yeni bir YOLO11 model oluşturur, optimize eder."""
        try:
            # Load YOLO11 model
            model = YOLO(self.model_path)
            model.tracker = str(tracker_yaml_path)

            if torch.cuda.is_available():
                model.model.half().to(device)
              
                dummy = torch.randn(1, 3, 640, 640).half().to(device)
                with torch.no_grad():
                    _ = model.model(dummy)

            return model
        except Exception as e:
            logger.error(f"YOLO11 Model oluşturulamadı: {e}")
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

# Updated YOLO11 Road Defect Classes (7 classes: 0-6)
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
        self.saved_tracks = set()  # Kaydedilen ID'leri takip etmek için
        self.class_counts = defaultdict(int)
        self.start_time = time.time()
        self.detected_frames = 0  # Tespit içeren frame sayısı
        
    def add_total(self, count=1):
        with self.lock:
            self.total_frames += count
            
    def add_processed(self, count=1):
        with self.lock:
            self.processed_frames += count
            
    def add_detected_frame(self):
        with self.lock:
            self.detected_frames += 1
            
    def add_track(self, class_id, track_id):
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            if track_key not in self.unique_tracks:
                self.unique_tracks.add(track_key)
                if 0 <= class_id < len(CLASSES):
                    self.class_counts[CLASSES[class_id]] += 1
    
    def is_track_saved(self, class_id, track_id):
        """Bu ID daha önce kaydedildi mi kontrol et"""
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            return track_key in self.saved_tracks
    
    def mark_track_saved(self, class_id, track_id):
        """Bu ID'yi kaydedildi olarak işaretle"""
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            self.saved_tracks.add(track_key)
    
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
                'class_counts': dict(self.class_counts),
                'detected_frames': self.detected_frames,
                'saved_tracks': len(self.saved_tracks)
            }


class HybridOptimalProcessor:
    """Optimize edilmiş video işleme sistemi - Frame extractor versiyonu"""
    
    def __init__(self, model_pool, output_dir):
        self.model_pool = model_pool
        self.tracker = PerformanceTracker()
        self.shutdown_event = Event()
        self.completion_event = Event()
        self.termination_lock = Lock()
        self.preprocessors_finished = 0
        self.predictors_finished = 0
        self.last_cleanup = time.time()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

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
                            preprocess_queue.put((frame_idx, processed_frame, frame), timeout=1.0)  # Orijinal frame'i de gönder
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
        """Frame işleme - SOL ALTA odaklanacak şekilde güncellendi"""
        try:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # Yeniden boyutlandır
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # ROI maskeleme - SOL ALT bölgeye odaklanma
            roi_top = int(new_height * (1 - roi_vertical))
            roi_right = int(new_width * (1 - roi_horizontal))  # Sağdan kesilecek alan
            
            # Güvenlik kontrolü - ROI sınırları
            roi_top = max(0, min(roi_top, new_height - 1))
            roi_right = max(1, min(roi_right, new_width))  # En az 1 pixel bırak
            
            # Üst kısmı siyahlama (gökyüzü vb.)
            if roi_top > 0 and roi_top < new_height:
                frame_resized[:roi_top] = 0
            
            # Sağ tarafı siyahlama (sol alt bölgeye odaklanma için)
            if roi_right > 0 and roi_right < new_width:
                frame_resized[:, roi_right:] = 0
                    
            return frame_resized
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
            return None

    def batch_collector(self, preprocess_queue, batch_queue):
        """Batch toplama - akıllı ve verimli"""
        current_batch = []
        current_indices = []
        current_originals = []  # Orijinal frame'leri saklama
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
                                    batch_queue.put((current_batch, current_indices, current_originals, batch_id), timeout=2.0)
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
                        
                    frame_idx, frame, original_frame = item
                    current_batch.append(frame)
                    current_indices.append(frame_idx)
                    current_originals.append(original_frame)
                    
                    # Batch hazır mı?
                    if len(current_batch) >= BATCH_SIZE:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, current_originals, batch_id), timeout=1.0)
                            print(f"📦 Batch {batch_id} gönderildi: {len(current_batch)} frame")
                        except queue.Full:
                            print(f"⚠️ Batch queue dolu: batch {batch_id}")
                            continue
                            
                        current_batch = []
                        current_indices = []
                        current_originals = []
                        
                except queue.Empty:
                    # Timeout batch kontrolü
                    if current_batch and len(current_batch) >= BATCH_SIZE // 2:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, current_originals, batch_id), timeout=0.5)
                            print(f"📦 Timeout batch {batch_id}: {len(current_batch)} frame")
                            current_batch = []
                            current_indices = []
                            current_originals = []
                        except queue.Full:
                            pass
                    continue
                except Exception as e:
                    logger.error(f"Batch collector hatası: {e}")
                    
            print(f"📦 Batch collector tamamlandı: {batch_id} batch oluşturuldu")
            
        except Exception as e:
            logger.error(f"Batch collector fatal error: {e}")

    def predictor_worker(self, worker_id, batch_queue, result_queue, score_threshold):
        """YOLO11 tahmin worker'ı"""
        model = None
        batches_processed = 0
        
        print(f"🤖 YOLO11 Predictor {worker_id} başladı")
        
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
                            print(f"🤖 YOLO11 Predictor {worker_id} bitti ({self.predictors_finished}/{NUM_PREDICTOR_THREADS})")
                        break
                        
                    frames, indices, originals, batch_id = batch_data
                    
                    # Bellek temizliği
                    if batches_processed % 10 == 0:
                        self.memory_cleanup()
                    
                    # YOLO11 tahmin yap
                    processed_results = self._run_yolo11_inference(model, frames, indices, originals, batch_id, score_threshold)
                    
                    if processed_results:
                        self.tracker.add_processed(len(frames))
                        try:
                            result_queue.put(processed_results, timeout=1.0)
                        except queue.Full:
                            print(f"⚠️ Result queue dolu: batch {batch_id}")
                        
                        print(f"🤖 YOLO11 Worker {worker_id} batch {batch_id} tamamlandı: {len(frames)} frame")
                    
                    batches_processed += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"YOLO11 Predictor {worker_id} hatası: {e}")
                    
            print(f"🤖 YOLO11 Predictor {worker_id} tamamlandı: {batches_processed} batch işlendi")
                    
        except Exception as e:
            logger.error(f"YOLO11 Predictor {worker_id} fatal error: {e}")
        finally:
            if model:
                self.model_pool.return_model(model)

    def _run_yolo11_inference(self, model, frames, indices, originals, batch_id, score_threshold):
        """YOLO11 inference - temiz ve güvenli"""
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
                    
                # Tespit var mı kontrol et ve ID'li tespitleri filtrele
                valid_detections = []
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        try:
                            class_id = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
                            # ID kontrolü - sadece ID'li tespitleri kabul et
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0]) if hasattr(box.id, '__iter__') else int(box.id)
                                # Ensure class_id is within our road defect classes (0-6)
                                if 0 <= class_id < len(CLASSES):
                                    self.tracker.add_track(class_id, track_id)
                                    valid_detections.append((class_id, track_id))
                        except (ValueError, TypeError, AttributeError) as e:
                            # ID parse hatası - bu tespiti atla
                            logger.warning(f"ID parse hatası: {e}")
                            continue
                
                # Sadece ID'li tespit içeren frame'leri kaydet
                if valid_detections:
                    self.tracker.add_detected_frame()
                    
                    # Annotated frame oluştur
                    try:
                        annotated = result.plot()
                    except Exception as e:
                        logger.error(f"Annotation hatası: {e}")
                        annotated = originals[i].copy()  # Fallback to original
                    
                    processed_results.append({
                        'frame_idx': indices[i],
                        'annotated_frame': annotated,
                        'original_frame': originals[i],
                        'batch_id': batch_id,
                        'valid_detections': valid_detections,
                        'has_detection': True
                    })
                else:
                    # Tespit yoksa sadece counter'ı güncelle
                    processed_results.append({
                        'frame_idx': indices[i],
                        'annotated_frame': None,
                        'original_frame': None,
                        'batch_id': batch_id,
                        'valid_detections': [],
                        'has_detection': False
                    })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"YOLO11 inference hatası: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def frame_saver_thread(self, result_queue):
        """Frame kaydetme thread'i - Sadece ID'li ve daha önce kaydedilmemiş frame'leri kaydeder"""
        none_count = 0
        frames_saved = 0
        
        print(f"💾 Frame saver başladı")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    results = result_queue.get(timeout=2.0)
                    
                    if results is None:
                        none_count += 1
                        print(f"💾 Saver sonlandırma sinyali {none_count}/{NUM_PREDICTOR_THREADS}")
                        
                        if none_count >= NUM_PREDICTOR_THREADS:
                            break
                        continue
                        
                    # Sadece ID'li ve yeni tespitleri kaydet
                    for result in results:
                        if result['has_detection'] and result['valid_detections'] and result['annotated_frame'] is not None:
                            # Bu frame'de kaydedilecek yeni ID var mı kontrol et
                            should_save = False
                            for class_id, track_id in result['valid_detections']:
                                if not self.tracker.is_track_saved(class_id, track_id):
                                    should_save = True
                                    break
                            
                            if should_save:
                                frame_idx = result['frame_idx']
                                annotated_frame = result['annotated_frame']
                                
                                # Sadece annotated frame'i kaydet (orijinal frame kaydetme)
                                try:
                                    # Annotated frame kaydet
                                    annotated_filename = self.output_dir / f"frame_{frame_idx:06d}_detected.jpg"
                                    cv2.imwrite(str(annotated_filename), annotated_frame)
                                    
                                    # Bu frame'deki tüm ID'leri kaydedildi olarak işaretle
                                    for class_id, track_id in result['valid_detections']:
                                        self.tracker.mark_track_saved(class_id, track_id)
                                    
                                    frames_saved += 1
                                    
                                    if frames_saved % 25 == 0:
                                        print(f"💾 Kaydedilen: {frames_saved} benzersiz ID frame'i")
                                        
                                except Exception as e:
                                    logger.error(f"Frame kaydetme hatası: {e}")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Frame saver hatası: {e}")
            
            print(f"💾 Frame kaydetme tamamlandı: {frames_saved} benzersiz frame kaydedildi")
            self.completion_event.set()
            
        except Exception as e:
            logger.error(f"Frame saver fatal error: {e}")
            self.completion_event.set()


def create_tracking_counters():
    st.write("### 🛣️ Road Defect Detection Results")
    
    # Create a single row with 4 columns for 4 classes
    counter_cols = st.columns(4)
    
    counter_placeholders = {}
    
    # Colors and emojis for 4 road defect classes
    road_defect_info = [
        ("🟡", "Longitudinal Crack"),  # 0
        ("🔴", "Transverse Crack"),    # 1  
        ("🟠", "Alligator Crack"),     # 2
        ("🔵", "Pothole"),             # 3
    ]
    
    for i in range(4):
        color, class_name = road_defect_info[i]
        with counter_cols[i]:
            counter_placeholders[class_name] = st.empty()
            counter_placeholders[class_name].metric(f"{color} {class_name}", "0")
    
    return counter_placeholders



def write_bytesio_to_file(filename, bytesio):
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())


# Setup directories
temp_dir = "./temp"
output_frames_dir = "./output_frames"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

temp_file_input = f"{temp_dir}/video_input.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False


def processVideoFrameExtraction(video_file, score_threshold, roi_vertical=0.6, roi_horizontal=0.3, resize_factor=0.5):
    """Optimize edilmiş YOLO11 road defect frame extraction - Sol alt odaklı"""
    
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
    st.write(f"İşleme: {processed_width}x{processed_height} (Sol alt odaklı ROI)")
    
    st.write("**YOLO11 Road Defect System:**")
    settings_info = (f"Model: YOLO11 (last.pt) | "
                    f"Classes: 7 Road Defects | "
                    f"Batch: {BATCH_SIZE} | "
                    f"Predictors: {NUM_PREDICTOR_THREADS} | "
                    f"Preprocessors: {NUM_PREPROCESSOR_THREADS} | "
                    f"Mode: Sadece ID'li tespitler")
    st.write(settings_info)

    # UI elementleri
    progress_bar = st.progress(0, text="YOLO11 Road Defect Frame Extraction başlatılıyor...")
    counter_placeholders = create_tracking_counters()
    performance_dashboard = st.empty()

    # Output klasörünü temizle ve hazırla
    import shutil
    if os.path.exists(output_frames_dir):
        shutil.rmtree(output_frames_dir)
    os.makedirs(output_frames_dir, exist_ok=True)

    # Processor oluştur
    processor = HybridOptimalProcessor(st.session_state.optimized_model_pool, output_frames_dir)

    # Queue'ları oluştur
    read_queue = queue.Queue(maxsize=SETTINGS['READ_QUEUE_SIZE'])
    preprocess_queue = queue.Queue(maxsize=SETTINGS['PREPROCESS_QUEUE_SIZE'])
    batch_queue = queue.Queue(maxsize=SETTINGS['BATCH_QUEUE_SIZE'])
    result_queue = queue.Queue(maxsize=SETTINGS['RESULT_QUEUE_SIZE'])

    threads = []
    start_time = time.time()

    try:
        print(f"🛣️ YOLO11 Road Defect Frame Extraction başlıyor: {_total_frames} frame (Sadece ID'li)")
        
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

        # YOLO11 Predictor thread'leri
        for i in range(NUM_PREDICTOR_THREADS):
            predictor = threading.Thread(
                target=processor.predictor_worker,
                args=(i, batch_queue, result_queue, score_threshold),
                name=f"YOLO11-Predictor-{i}", daemon=True
            )
            threads.append(predictor)

        # Frame Saver thread
        frame_saver = threading.Thread(
            target=processor.frame_saver_thread,
            args=(result_queue,),
            name="FrameSaver", daemon=True
        )
        threads.append(frame_saver)

        # Tüm thread'leri başlat
        for t in threads:
            t.start()

        # İlerleme takibi - road defect odaklı
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
                        text=f"🛣️ Frame Extraction: {progress_info['processed']}/{progress_info['total']} frame ({progress*100:.1f}%) | ID'li: {progress_info['saved_tracks']} | Kalan: {remaining_time:.0f}s"
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
                            st.metric("YOLO11 FPS", f"{fps:.1f}")
                        else:
                            st.metric("YOLO11 FPS", "0.0")
                        
                    with perf_cols[2]:
                        st.metric("Tespit Frame'leri", f"📸 {progress_info['detected_frames']}")
                        
                    with perf_cols[3]:
                        st.metric("Kaydedilen ID'ler", f"🆔 {progress_info['saved_tracks']}")
                
                # Road defect counter'ları güncelle
                for class_name in CLASSES:
                    count = progress_info['class_counts'].get(class_name, 0)
                    counter_placeholders[class_name].metric(f"{class_name}", f"{count}")
                
                last_update = current_time
            
            time.sleep(0.5)

        # Tamamlanma bekleme
        print("⏳ YOLO11 frame extraction tamamlanması bekleniyor...")
        if not processor.completion_event.wait(timeout=300):  # 5 dakika max
            st.warning("⚠️ İşlem zaman aşımı - zorla tamamlanıyor")
            processor.shutdown_event.set()

    except KeyboardInterrupt:
        st.warning("🛑 İşlem kullanıcı tarafından durduruldu")
        processor.shutdown_event.set()
        
    except Exception as e:
        st.error(f"❌ YOLO11 işlem hatası: {e}")
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
        
        st.success(f"🛣️ YOLO11 Frame Extraction Tamamlandı!")
        
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
            st.metric("Tespit Edilen Frame", f"📸 {final_progress['detected_frames']}")
            st.metric("Benzersiz ID'ler", f"🆔 {final_progress['saved_tracks']}")
        
        # Road defect detection summary
        st.write("### 📊 Tespit Edilen Yol Defektleri:")
        defect_summary_cols = st.columns(4)
        
        class_counts = final_progress['class_counts']
        defect_colors = ["🟡", "🔴", "🟠", "🔵"]
        
        for i, class_name in enumerate(CLASSES):
            col_idx = i % 4
            with defect_summary_cols[col_idx]:
                count = class_counts.get(class_name, 0)
                color = defect_colors[i] if i < len(defect_colors) else "🔶"
                if count > 0:
                    st.write(f"**{color} {class_name}**: {count} adet")
                else:
                    st.write(f"{color} {class_name}: {count} adet")
        
        # Output klasörü bilgisi
        st.write("### 📁 Çıktı Klasörü:")
        frame_files = list(Path(output_frames_dir).glob("*.jpg"))
        st.write(f"**Klasör**: `{output_frames_dir}`")
        st.write(f"**Kaydedilen dosya sayısı**: {len(frame_files)} adet")
        
        if len(frame_files) > 0:
            st.write("**Dosya formatı**: `frame_XXXXXX_detected.jpg` (Sadece tespit edilen frame'ler)")
            
            # Örnek dosyalar göster
            sample_files = sorted(frame_files)[:6]  # İlk 6 dosyayı göster
            st.write("**Örnek dosyalar**:")
            for file_path in sample_files:
                st.write(f"- {file_path.name}")
                
        # Performans değerlendirmesi
        if final_progress['detected_frames'] > 0:
            detection_rate = (final_progress['detected_frames'] / final_progress['processed']) * 100
            unique_rate = (final_progress['saved_tracks'] / final_progress['detected_frames']) * 100
            st.write(f"### 📈 İstatistikler:")
            st.write(f"- **Tespit Oranı**: {detection_rate:.1f}% (tespit içeren frame oranı)")
            st.write(f"- **Benzersizlik Oranı**: {unique_rate:.1f}% (kaydedilen benzersiz ID oranı)")
            
            if final_progress['saved_tracks'] >= 10:
                st.success("🎯 Yüksek ID çeşitliliği - çok sayıda benzersiz defekt tespit edildi")
            elif final_progress['saved_tracks'] >= 5:
                st.info("👍 Orta ID çeşitliliği - bazı benzersiz defektler tespit edildi")
            else:
                st.info("📊 Düşük ID çeşitliliği - az sayıda benzersiz defekt")
        else:
            st.info("ℹ️ Hiç ID'li defekt tespit edilmedi - eşik değerini düşürmeyi deneyin")
        
        # Zip dosyası oluşturma seçeneği
        if len(frame_files) > 0:
            if st.button("📦 Tüm Frame'leri ZIP olarak Hazırla", use_container_width=True):
                import zipfile
                zip_path = f"{temp_dir}/detected_frames.zip"
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in frame_files:
                        zipf.write(file_path, file_path.name)
                
                st.success(f"✅ ZIP dosyası hazır: {len(frame_files)} dosya")
                
                # Download button
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="📥 Tespit Edilen Frame'leri İndir",
                        data=f,
                        file_name="yolo11_detected_frames.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
        
        print(f"✅ YOLO11 Frame Extraction tamamlandı: {final_progress['saved_tracks']} benzersiz ID frame'i kaydedildi")


# UI Layout
col1, col2 = st.columns(2)
with col1:
    frames_count = 0
    if os.path.exists(output_frames_dir):
        frames_count = len(list(Path(output_frames_dir).glob("*.jpg")))
    
    if frames_count > 0:
        st.metric("💾 Kaydedilen Frame", f"{frames_count} adet")
        st.info(f"Frame'ler: `{output_frames_dir}` klasöründe")
    else:
        st.info("Benzersiz ID'li frame'ler burada görünecek")
        
with col2:
    if st.button('🔄 Reset YOLO11 System', use_container_width=True, type="primary"):
        # Complete reset
        for key in ['optimal_settings', 'optimized_model_pool']:
            if key in st.session_state:
                del st.session_state[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Output klasörünü temizle
        if os.path.exists(output_frames_dir):
            import shutil
            shutil.rmtree(output_frames_dir)
        st.rerun()

st.title("🛣️ YOLO11 Road Defect Frame Extractor")
st.write("**✨ Extract unique ID-tracked frames with road defects using YOLO11**")

# Model info panel
model_info = st.expander("🔧 YOLO11 Model Information")
with model_info:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Configuration:**")
        st.write(f"- Model: YOLO11 (last.pt)")
        st.write(f"- Classes: {len(CLASSES)} Road Defect Types")
        st.write(f"- Input Resolution: 640x640")
        st.write(f"- Tracking: ByteTrack")
        st.write(f"- Output: Sadece ID'li frame'ler")
        st.write(f"- ROI: Sol alt odaklı")
    with col2:
        st.write("**Road Defect Classes:**")
        for i, class_name in enumerate(CLASSES):
            defect_codes = ["D00", "D10", "D20", "D40", "D43", "D44", "D50"]
            st.write(f"- {i}: {class_name} ({defect_codes[i]})")

# Important notice
st.info("🆔 **Önemli**: Bu sistem sadece track ID'si olan tespitleri kaydeder ve her ID'yi yalnızca bir kez kaydeder. Bu sayede benzersiz defekt örnekleri elde edilir.")

# File upload
video_file = st.file_uploader(
    "📹 Upload Road Video File", 
    type=".mp4", 
    disabled=st.session_state.runningInference,
    help="Upload a video file containing road footage for defect detection and frame extraction"
)

# Settings
score_threshold = st.slider(
    "Detection Confidence Threshold", 
    0.0, 1.0, 0.5, 0.05, 
    disabled=st.session_state.runningInference,
    help="Higher values = fewer but more confident detections"
)

# Preprocessing
st.write("---")
st.write("### 🔧 Video Preprocessing Configuration")

prep_cols = st.columns(3)
with prep_cols[0]:
    resize_factor = st.slider(
        "Size Factor", 
        0.1, 1.0, 0.5, 0.1,
        disabled=st.session_state.runningInference,
        help="Lower = faster processing, less detail"
    )
with prep_cols[1]:
    roi_vertical = st.slider(
        "Vertical ROI", 
        0.1, 1.0, 0.7, 0.1,  # Daha fazla alt kısım
        disabled=st.session_state.runningInference,
        help="Keep bottom portion (road area)"
    )
with prep_cols[2]:
    roi_horizontal = st.slider(
        "Left Focus ROI", 
        0.0, 0.7, 0.3, 0.1,  # Sol odak için daha fazla alan
        disabled=st.session_state.runningInference,
        help="Remove right side (focus on left-bottom)"
    )

# ROI visualization
st.write("**🎯 ROI Preview**: Sol alt bölgeye odaklanılacak")
roi_cols = st.columns(3)
with roi_cols[0]:
    st.metric("Alt Bölge", f"{roi_vertical*100:.0f}%", "Yolun alt kısmı")
with roi_cols[1]:
    st.metric("Sol Odak", f"{(1-roi_horizontal)*100:.0f}%", "Sol taraf korunacak")
with roi_cols[2]:
    st.metric("İşlem Boyutu", f"{resize_factor*100:.0f}%", "Hız optimizasyonu")

# Process button
if video_file is not None:
    if st.button(
        '🛣️ Start YOLO11 Frame Extraction (ID-Only)',
        use_container_width=True,
        disabled=st.session_state.runningInference,
        type="secondary",
        key="processing_button"
    ):
        st.info(f"🛣️ Processing {video_file.name} for unique ID-tracked road defect frames...")
        processVideoFrameExtraction(video_file, score_threshold, roi_vertical, roi_horizontal, resize_factor)
else:
    st.warning("📹 Lütfen işlem için bir video dosyası yükleyin")

# Footer
st.write("---")
st.write("### 📋 System Information")
info_cols = st.columns(3)
with info_cols[0]:
    st.write("**Processing Mode:**")
    st.write("- ID-tracked detections only")
    st.write("- Unique frames per ID")
    st.write("- Left-bottom ROI focus")
with info_cols[1]:
    st.write("**Output Format:**")
    st.write("- Annotated frames only")
    st.write("- JPG format")
    st.write("- frame_XXXXXX_detected.jpg")
with info_cols[2]:
    st.write("**Performance:**")
    if torch.cuda.is_available():
        st.write(f"- GPU: {st.session_state.optimal_settings.gpu_name}")
        st.write(f"- VRAM: {st.session_state.optimal_settings.gpu_memory_gb:.1f}GB")
    else:
        st.write("- CPU Mode")
    st.write(f"- Threads: {NUM_PREDICTOR_THREADS}+{NUM_PREPROCESSOR_THREADS}")

# Performance tips
with st.expander("💡 Performance Tips"):
    st.write("""
    **Hızlandırma için:**
    - Size Factor'ı düşürün (0.3-0.5)
    - Confidence Threshold'u yükseltin (0.6-0.8)
    - ROI alanını daraltın
    
    **Kalite için:**
    - Size Factor'ı yükseltin (0.7-1.0)
    - Confidence Threshold'u düşürün (0.3-0.5)
    - ROI alanını genişletin
    
    **ID Tracking için:**
    - Video kalitesi önemli
    - Sabit kamera açısı tercih edilir
    - Yeterli ışık koşulları gerekli
    """)

# Debug information (only show if processing)
if st.session_state.runningInference:
    with st.expander("🔍 Debug Information"):
        st.write("**Current Settings:**")
        st.json(SETTINGS)
        
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
                st.write(f"**GPU Memory:** {gpu_used:.1f}GB / {gpu_memory:.1f}GB")
            except:
                st.write("**GPU Memory:** Unable to read")
        
        st.write(f"**Model Pool:** {len(st.session_state.optimized_model_pool.models)} models available")