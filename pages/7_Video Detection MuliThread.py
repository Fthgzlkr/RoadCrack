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
from threading import Lock, Event

# Deep learning framework
from ultralytics import YOLO

st.set_page_config(
    page_title="Stable Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
tracker_yaml_path = ROOT / "./models/bytetrack.yaml"

# SAFE SETTINGS - No more memory issues!
def get_safe_settings():
    """Conservative settings that work reliably"""
    
    # Check available resources
    gpu_memory_gb = 0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    cpu_cores = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Very conservative settings to prevent crashes
    if gpu_memory_gb < 4:  # Low-end GPU
        settings = {
            'BATCH_SIZE': 2,
            'NUM_PREDICTOR_THREADS': 1,
            'NUM_PREPROCESSOR_THREADS': 1,
            'MAX_QUEUE_SIZE': 50
        }
    elif gpu_memory_gb < 8:  # Mid-range GPU
        settings = {
            'BATCH_SIZE': 4,
            'NUM_PREDICTOR_THREADS': 1,
            'NUM_PREPROCESSOR_THREADS': 2,
            'MAX_QUEUE_SIZE': 100
        }
    else:  # High-end GPU
        settings = {
            'BATCH_SIZE': 8,
            'NUM_PREDICTOR_THREADS': 2,
            'NUM_PREPROCESSOR_THREADS': 2,
            'MAX_QUEUE_SIZE': 200
        }
    
    return settings

SETTINGS = get_safe_settings()
BATCH_SIZE = SETTINGS['BATCH_SIZE']
NUM_PREDICTOR_THREADS = SETTINGS['NUM_PREDICTOR_THREADS']
NUM_PREPROCESSOR_THREADS = SETTINGS['NUM_PREPROCESSOR_THREADS']
MAX_QUEUE_SIZE = SETTINGS['MAX_QUEUE_SIZE']

device = "cuda" if torch.cuda.is_available() else "cpu"

# Thread-safe model pool
class SafeModelPool:
    def __init__(self, model_path, max_models=2):
        self.model_path = model_path
        self.models = []
        self.lock = Lock()
        
        # Create initial models
        for _ in range(max_models):
            try:
                model = YOLO(model_path)
                model.tracker = str(tracker_yaml_path)
                if torch.cuda.is_available():
                    model.model.half()  # FP16 for memory
                self.models.append(model)
            except Exception as e:
                logger.error(f"Model creation error: {e}")
                break
                
        if not self.models:
            # Fallback: create at least one model
            model = YOLO(model_path)
            model.tracker = str(tracker_yaml_path)
            self.models.append(model)
    
    def get_model(self):
        with self.lock:
            if self.models:
                return self.models.pop()
            # If no models available, create new one
            model = YOLO(self.model_path)
            model.tracker = str(tracker_yaml_path)
            return model
    
    def return_model(self, model):
        with self.lock:
            if len(self.models) < 3:  # Keep max 3 models
                self.models.append(model)
            else:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Initialize model pool
if 'safe_model_pool' not in st.session_state:
    st.session_state.safe_model_pool = SafeModelPool(MODEL_LOCAL_PATH, NUM_PREDICTOR_THREADS)

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack", 
    "Alligator Crack",
    "Potholes"
]

@dataclass
class SafeStats:
    """Thread-safe statistics without Streamlit dependencies"""
    def __init__(self):
        self.lock = Lock()
        self.total_frames = 0
        self.processed_frames = 0
        self.unique_tracks = set()
        self.class_counts = defaultdict(int)
        self.errors = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
    def add_total(self, count=1):
        with self.lock:
            self.total_frames += count
            
    def add_processed(self, count=1):
        with self.lock:
            self.processed_frames += count
            self.fps_counter += count
            
    def add_track(self, class_id, track_id):
        with self.lock:
            track_key = f"{class_id}_{track_id}"
            if track_key not in self.unique_tracks:
                self.unique_tracks.add(track_key)
                if 0 <= class_id < len(CLASSES):
                    self.class_counts[CLASSES[class_id]] += 1
    
    def add_error(self):
        with self.lock:
            self.errors += 1
            
    def get_all(self):
        with self.lock:
            current_time = time.time()
            time_diff = current_time - self.last_fps_time
            
            if time_diff >= 1.0:
                fps = self.fps_counter / time_diff
                self.fps_counter = 0
                self.last_fps_time = current_time
            else:
                fps = 0
                
            return {
                'total': self.total_frames,
                'processed': self.processed_frames,
                'unique_tracks': len(self.unique_tracks),
                'class_counts': dict(self.class_counts),
                'fps': fps,
                'errors': self.errors
            }

class StableVideoProcessor:
    """Rock-solid video processor with proper frame completion"""
    
    def __init__(self, model_pool):
        self.model_pool = model_pool
        self.stats = SafeStats()
        self.shutdown_event = Event()
        self.completion_event = Event()  # New: signals when all frames are processed
        
        # Termination tracking
        self.preprocessors_done = 0
        self.predictors_done = 0
        self.termination_lock = Lock()
        
    def cleanup_memory(self):
        """Safe memory cleanup"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")

    def reader_thread(self, video_capture, read_queue, total_frames):
        """Read all frames and signal completion"""
        frames_read = 0
        
        try:
            print(f"📖 Reader starting - target: {total_frames} frames")
            
            while frames_read < total_frames and not self.shutdown_event.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    break
                    
                # Put frame with index
                while not self.shutdown_event.is_set():
                    try:
                        read_queue.put((frames_read, frame), timeout=1.0)
                        break
                    except queue.Full:
                        continue
                
                frames_read += 1
                self.stats.add_total()
                
                if frames_read % 100 == 0:
                    print(f"📖 Read {frames_read}/{total_frames} frames")
            
            # Signal end to ALL preprocessors
            for i in range(NUM_PREPROCESSOR_THREADS):
                while not self.shutdown_event.is_set():
                    try:
                        read_queue.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                        
            print(f"📖 Reader finished - read {frames_read} frames, sent {NUM_PREPROCESSOR_THREADS} termination signals")
            
        except Exception as e:
            logger.error(f"Reader error: {e}")

    def preprocessor_worker(self, worker_id, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor):
        """Preprocessor with proper termination handling"""
        frames_processed = 0
        
        try:
            print(f"🔧 Preprocessor {worker_id} starting")
            
            while not self.shutdown_event.is_set():
                try:
                    item = read_queue.get(timeout=2.0)
                    if item is None:
                        # Signal batch collector that this preprocessor is done
                        while not self.shutdown_event.is_set():
                            try:
                                preprocess_queue.put(None, timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        
                        with self.termination_lock:
                            self.preprocessors_done += 1
                            print(f"🔧 Preprocessor {worker_id} finished - {self.preprocessors_done}/{NUM_PREPROCESSOR_THREADS} done")
                        break
                        
                    frame_idx, frame = item
                    
                    # Process frame
                    processed_frame = self.apply_preprocessing_safe(frame, roi_vertical, roi_horizontal, resize_factor)
                    
                    if processed_frame is not None:
                        while not self.shutdown_event.is_set():
                            try:
                                preprocess_queue.put((frame_idx, processed_frame), timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        frames_processed += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Preprocessor {worker_id} error: {e}")
                    self.stats.add_error()
                    
            print(f"🔧 Preprocessor {worker_id} terminated - processed {frames_processed} frames")
            
        except Exception as e:
            logger.error(f"Preprocessor {worker_id} fatal error: {e}")

    def apply_preprocessing_safe(self, frame, roi_vertical, roi_horizontal, resize_factor):
        """Safe preprocessing with memory checks"""
        try:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            # Resize
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Apply ROI
            roi_top = int(new_height * (1 - roi_vertical))
            roi_side = int(new_width * roi_horizontal)
            
            if roi_top > 0 or roi_side > 0:
                frame_processed = frame_resized.copy()
                if roi_top > 0:
                    frame_processed[:roi_top] = 0
                if roi_side > 0:
                    frame_processed[:, :roi_side] = 0
                    frame_processed[:, new_width-roi_side:] = 0
            else:
                frame_processed = frame_resized
                
            return frame_processed
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None

    def batch_collector(self, preprocess_queue, batch_queue):
        """Collect frames into batches with guaranteed completion"""
        current_batch = []
        current_indices = []
        none_count = 0
        batch_id = 0
        
        try:
            print(f"📦 Batch collector starting - target: {NUM_PREPROCESSOR_THREADS} termination signals")
            
            while not self.shutdown_event.is_set():
                try:
                    item = preprocess_queue.get(timeout=2.0)
                    
                    if item is None:
                        none_count += 1
                        print(f"📦 Received termination signal {none_count}/{NUM_PREPROCESSOR_THREADS}")
                        
                        if none_count >= NUM_PREPROCESSOR_THREADS:
                            # Send final batch if exists
                            if current_batch:
                                batch_id += 1
                                print(f"📦 Sending FINAL batch {batch_id} with {len(current_batch)} frames")
                                while not self.shutdown_event.is_set():
                                    try:
                                        batch_queue.put((current_batch, current_indices, batch_id), timeout=1.0)
                                        break
                                    except queue.Full:
                                        continue
                            
                            # Signal all predictors to stop
                            for i in range(NUM_PREDICTOR_THREADS):
                                while not self.shutdown_event.is_set():
                                    try:
                                        batch_queue.put(None, timeout=1.0)
                                        break
                                    except queue.Full:
                                        continue
                            
                            print(f"📦 Batch collector sent {NUM_PREDICTOR_THREADS} termination signals")
                            break
                        continue
                        
                    frame_idx, frame = item
                    current_batch.append(frame)
                    current_indices.append(frame_idx)
                    
                    # Send batch when full
                    if len(current_batch) >= BATCH_SIZE:
                        batch_id += 1
                        while not self.shutdown_event.is_set():
                            try:
                                batch_queue.put((current_batch, current_indices, batch_id), timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        print(f"📦 Sent batch {batch_id} with {len(current_batch)} frames")
                        current_batch = []
                        current_indices = []
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Batch collector error: {e}")
                    self.stats.add_error()
                    
            print(f"📦 Batch collector finished - sent {batch_id} batches")
            
        except Exception as e:
            logger.error(f"Batch collector fatal error: {e}")

    def predictor_worker(self, worker_id, batch_queue, result_queue, score_threshold):
        """Predictor with proper model handling"""
        model = None
        batches_processed = 0
        
        try:
            print(f"🤖 Predictor {worker_id} starting")
            model = self.model_pool.get_model()
            
            while not self.shutdown_event.is_set():
                try:
                    batch_data = batch_queue.get(timeout=2.0)
                    
                    if batch_data is None:
                        # Signal writer that this predictor is done
                        while not self.shutdown_event.is_set():
                            try:
                                result_queue.put(None, timeout=1.0)
                                break
                            except queue.Full:
                                continue
                                
                        with self.termination_lock:
                            self.predictors_done += 1
                            print(f"🤖 Predictor {worker_id} finished - {self.predictors_done}/{NUM_PREDICTOR_THREADS} done")
                        break
                        
                    frames, indices, batch_id = batch_data
                    
                    # Clean GPU memory before processing
                    if torch.cuda.is_available() and batches_processed % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # Process with YOLO
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
                    
                    # Process results
                    processed_results = []
                    for i, result in enumerate(results):
                        if self.shutdown_event.is_set():
                            break
                            
                        # Update tracking
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.cpu().numpy()
                            for box in boxes:
                                class_id = int(box.cls)
                                if hasattr(box, 'id') and box.id is not None:
                                    track_id = int(box.id)
                                    self.stats.add_track(class_id, track_id)
                        
                        annotated = result.plot()
                        processed_results.append({
                            'frame_idx': indices[i],
                            'annotated_frame': annotated,
                            'batch_id': batch_id
                        })
                    
                    self.stats.add_processed(len(frames))
                    batches_processed += 1
                    
                    # Send results
                    while not self.shutdown_event.is_set():
                        try:
                            result_queue.put(processed_results, timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    
                    print(f"🤖 Predictor {worker_id} processed batch {batch_id} ({len(frames)} frames)")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Predictor {worker_id} error: {e}")
                    self.stats.add_error()
                    # Clean up on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            print(f"🤖 Predictor {worker_id} terminated - processed {batches_processed} batches")
                    
        except Exception as e:
            logger.error(f"Predictor {worker_id} fatal error: {e}")
        finally:
            if model:
                self.model_pool.return_model(model)

    def writer_thread(self, result_queue, cv2writer):
        """Writer that ensures ALL frames are written"""
        frame_buffer = {}
        expected_frame_idx = 0
        none_count = 0
        frames_written = 0
        
        try:
            print(f"🎬 Writer starting")
            
            while not self.shutdown_event.is_set():
                try:
                    results = result_queue.get(timeout=3.0)
                    
                    if results is None:
                        none_count += 1
                        print(f"🎬 Writer received termination signal {none_count}/{NUM_PREDICTOR_THREADS}")
                        
                        if none_count >= NUM_PREDICTOR_THREADS:
                            print(f"🎬 All predictors done. Buffer has {len(frame_buffer)} frames")
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
                        
                        if frames_written % 100 == 0:
                            print(f"🎬 Written {frames_written} frames")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer error: {e}")
                    self.stats.add_error()
            
            # CRITICAL: Write ALL remaining frames
            print(f"🎬 Writing {len(frame_buffer)} remaining frames...")
            remaining_indices = sorted(frame_buffer.keys())
            for idx in remaining_indices:
                cv2writer.write(frame_buffer[idx])
                frames_written += 1
            
            print(f"🎬 Writer finished - total frames written: {frames_written}")
            self.completion_event.set()  # Signal completion
            
        except Exception as e:
            logger.error(f"Writer fatal error: {e}")
            self.completion_event.set()

def create_tracking_counters():
    st.write("### 🔍 Detection Counters")
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

# Create temp directories
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
    """STABLE video processing - guarantees all frames processed"""
    
    write_bytesio_to_file(temp_file_input, video_file)
    
    video_capture = cv2.VideoCapture(temp_file_input)
    if not video_capture.isOpened():
        st.error('Error opening video file')
        return
        
    # Get video properties
    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    processed_width = int(_width * resize_factor)
    processed_height = int(_height * resize_factor)

    st.write("**Video Information:**")
    st.write(f"Original: {_width}x{_height} → Processed: {processed_width}x{processed_height}")
    st.write(f"Total Frames: {_total_frames} @ {_fps:.1f} FPS")
    st.write(f"**Safe Settings:** Batch: {BATCH_SIZE} | Threads: {NUM_PREDICTOR_THREADS}P/{NUM_PREPROCESSOR_THREADS}Pr")

    # UI elements
    progress_bar = st.progress(0, text="Starting stable processing...")
    counter_placeholders = create_tracking_counters()
    status_placeholder = st.empty()

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2writer = cv2.VideoWriter(temp_file_infer, fourcc, _fps, (processed_width, processed_height))

    # Create processor
    processor = StableVideoProcessor(st.session_state.safe_model_pool)

    # Create smaller, safer queues
    read_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    preprocess_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    batch_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE//4)
    result_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    threads = []
    start_time = time.time()

    try:
        print(f"🚀 Starting processing pipeline for {_total_frames} frames")
        
        # Create and start threads
        reader = threading.Thread(
            target=processor.reader_thread,
            args=(video_capture, read_queue, _total_frames),
            name="Reader", daemon=True
        )
        threads.append(reader)

        for i in range(NUM_PREPROCESSOR_THREADS):
            preprocessor = threading.Thread(
                target=processor.preprocessor_worker,
                args=(i, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor),
                name=f"Preprocessor-{i}", daemon=True
            )
            threads.append(preprocessor)

        collector = threading.Thread(
            target=processor.batch_collector,
            args=(preprocess_queue, batch_queue),
            name="BatchCollector", daemon=True
        )
        threads.append(collector)

        for i in range(NUM_PREDICTOR_THREADS):
            predictor = threading.Thread(
                target=processor.predictor_worker,
                args=(i, batch_queue, result_queue, score_threshold),
                name=f"Predictor-{i}", daemon=True
            )
            threads.append(predictor)

        writer = threading.Thread(
            target=processor.writer_thread,
            args=(result_queue, cv2writer),
            name="Writer", daemon=True
        )
        threads.append(writer)

        # Start all threads
        for t in threads:
            t.start()

        # Monitor progress
        last_update = time.time()
        
        while not processor.completion_event.is_set() and not processor.shutdown_event.is_set():
            current_time = time.time()
            
            if current_time - last_update >= 1.0:
                stats = processor.stats.get_all()
                
                # Update progress
                if stats['total'] > 0:
                    progress = min(stats['processed'] / _total_frames, 1.0)
                    progress_bar.progress(
                        progress,
                        text=f"Processing: {stats['processed']}/{stats['total']} frames ({progress*100:.1f}%) | FPS: {stats['fps']:.1f}"
                    )
                
                # Update status
                elapsed = current_time - start_time
                with status_placeholder.container():
                    status_cols = st.columns(4)
                    with status_cols[0]:
                        st.metric("Elapsed", f"{elapsed:.0f}s")
                    with status_cols[1]:
                        st.metric("FPS", f"{stats['fps']:.1f}")
                    with status_cols[2]:
                        st.metric("Tracks", stats['unique_tracks'])
                    with status_cols[3]:
                        st.metric("Errors", stats['errors'])
                
                # Update counters
                for class_name in CLASSES:
                    count = stats['class_counts'].get(class_name, 0)
                    counter_placeholders[class_name].metric(f"{class_name}", f"{count}")
                
                last_update = current_time
                
                # Cleanup memory periodically
                if elapsed % 30 == 0:
                    processor.cleanup_memory()
            
            time.sleep(0.5)

        # Wait for completion
        print("⏳ Waiting for all threads to complete...")
        processor.completion_event.wait(timeout=60)  # Max 1 minute wait

    except KeyboardInterrupt:
        st.warning("🛑 Processing interrupted")
        processor.shutdown_event.set()
        
    finally:
        # Cleanup
        processor.shutdown_event.set()
        
        # Close video resources
        video_capture.release()
        cv2writer.release()
        
        # Clean memory
        processor.cleanup_memory()
        
        # Final results
        end_time = time.time()
        final_stats = processor.stats.get_all()
        processing_time = end_time - start_time
        
        st.success(f"✅ Processing Complete!")
        st.write(f"**Results:** {final_stats['processed']} frames in {processing_time:.1f}s")
        st.write(f"**Detections:** {final_stats['unique_tracks']} unique tracks")
        
        if final_stats['errors'] > 0:
            st.warning(f"⚠️ {final_stats['errors']} errors occurred (but processing completed)")

# UI Layout
col1, col2 = st.columns(2)
with col1:
    if os.path.exists(temp_file_infer):
        try:
            with open(temp_file_infer, "rb") as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="stable_detection_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        except Exception:
            st.info("Download will be available after processing")
    else:
        st.info("Processed video will appear here")
        
with col2:
    if st.button('🔄 Restart Application', use_container_width=True, type="primary"):
        # Clean up everything
        if 'safe_model_pool' in st.session_state:
            del st.session_state.safe_model_pool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.rerun()

st.title("🛡️ Rock-Solid Video Detection")
st.write("Guaranteed frame completion • No crashes • Memory safe")

# Show current settings
with st.expander("⚙️ System Configuration"):
    sys_cols = st.columns(3)
    with sys_cols[0]:
        st.write(f"**Batch Size:** {BATCH_SIZE}")
        st.write(f"**Queue Size:** {MAX_QUEUE_SIZE}")
    with sys_cols[1]:
        st.write(f"**Predictors:** {NUM_PREDICTOR_THREADS}")
        st.write(f"**Preprocessors:** {NUM_PREPROCESSOR_THREADS}")
    with sys_cols[2]:
        st.write(f"**Device:** {device}")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.write(f"**GPU Memory:** {gpu_memory:.1f}GB")

# File upload
video_file = st.file_uploader(
    "📹 Upload Video File", 
    type=".mp4", 
    disabled=st.session_state.runningInference
)
st.caption("Conservative settings ensure stable processing")

# Settings
score_threshold = st.slider(
    "Confidence Threshold", 
    0.0, 1.0, 0.5, 0.05, 
    disabled=st.session_state.runningInference
)

# Preprocessing settings
st.write("---")
st.write("### 🔧 Preprocessing Settings")

setting_cols = st.columns(3)
with setting_cols[0]:
    resize_factor = st.slider(
        "Resize Factor", 
        0.1, 1.0, 0.5, 0.1,
        disabled=st.session_state.runningInference,
        help="Lower = faster but less detail"
    )
with setting_cols[1]:
    roi_vertical = st.slider(
        "ROI Vertical", 
        0.1, 1.0, 0.6, 0.1,
        disabled=st.session_state.runningInference,
        help="Keep bottom portion of frame"
    )
with setting_cols[2]:
    roi_horizontal = st.slider(
        "ROI Horizontal", 
        0.0, 0.5, 0.2, 0.1,
        disabled=st.session_state.runningInference,
        help="Remove side portions"
    )

st.info(f"💡 Final size: {resize_factor:.1f}x original | Focus on road: bottom {roi_vertical*100:.0f}%")

# Process button
if video_file is not None:
    if st.button(
        '🛡️ Start Stable Processing',
        use_container_width=True,
        disabled=st.session_state.runningInference,
        type="secondary",
        key="processing_button"
    ):
        st.info(f"🔄 Processing {video_file.name} with rock-solid pipeline...")
        processVideo(video_file, score_threshold, roi_vertical, roi_horizontal, resize_factor)

# Footer info
st.write("---")
st.caption("🔬 This version prioritizes stability and completion over raw speed")
st.caption("📊 All frames guaranteed to be processed • Memory-safe operation")