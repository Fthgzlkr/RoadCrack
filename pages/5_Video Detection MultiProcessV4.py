# Fixed RTX 3060 Laptop Video Processing Pipeline
# ULTRA MINIMAL FIX: Sadece writer değiştirildi + Detection Counter Eklendi

import cv2
import numpy as np
import streamlit as st
import torch
import threading
import queue
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

# Deep learning framework
from ultralytics import YOLO

# Configuration
st.set_page_config(
    page_title="RTX 3060 Laptop - Fixed Version with Counters",
    page_icon="🏎️",
    layout="centered"
)

# Constants
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"

class ProductionRTX3060Optimizer:
    """Production-ready RTX 3060 Laptop optimizer"""
    
    def __init__(self):
        self.setup_production_environment()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = self.calculate_optimal_batch_size()
        self.apply_all_optimizations()
        
    def setup_production_environment(self):
        """Production environment setup"""
        env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256,garbage_collection_threshold:0.8',
            'CUBLAS_WORKSPACE_CONFIG': ':16:8',
            'CUDA_LAUNCH_BLOCKING': '0',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
    
    def calculate_optimal_batch_size(self):
        """RTX 3060 Laptop için optimal batch size hesapla"""
        if not torch.cuda.is_available():
            return 4
        
        try:
            torch.cuda.empty_cache()
            test_batches = [  16, 12, 8]
            
            for batch_size in test_batches:
                try:
                    test_tensor = torch.randn(batch_size, 3, 640, 640, dtype=torch.half).cuda()
                    memory_used = torch.cuda.memory_allocated(0) / 1024**3
                    
                    if memory_used < 5.2:
                        del test_tensor
                        torch.cuda.empty_cache()
                        return batch_size
                    
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
            
            return 8
            
        except Exception:
            return 16
    
    def apply_all_optimizations(self):
        """Tüm optimizasyonları uygula"""
        if not torch.cuda.is_available():
            return
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.cuda.empty_cache()
        gc.collect()

# Initialize optimizer
optimizer = ProductionRTX3060Optimizer()
DEVICE = optimizer.device
BATCH_SIZE = optimizer.batch_size

# Model loading with all optimizations
@st.cache_resource
def load_production_model():
    """Production model loading with full optimization"""
    model = YOLO(MODEL_LOCAL_PATH)
    
    if DEVICE == "cuda":
        model.to("cuda")
        model.model.half()
        
        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.half).cuda()
        with torch.no_grad():
            for _ in range(5):
                _ = model.model(dummy_input)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            model.model = torch.compile(model.model, mode='max-autotune')
            st.success(f"✅ Model compiled with PyTorch 2.0 (Batch: {BATCH_SIZE})")
        except Exception as e:
            st.info(f"ℹ️ Model loaded without compilation (Batch: {BATCH_SIZE})")
    
    return model

# Load model
net = load_production_model()

CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

# 🔥 NEW: Detection Counter Display Class
class DetectionCounterDisplay:
    """Real-time detection counter display class"""
    
    def __init__(self):
        # Initialize Streamlit placeholders for counters
        st.write("### 📊 Real-Time Detection Counters")
        
        # Create 4 columns for each class
        self.counter_cols = st.columns(4)
        
        # Create metric placeholders for each class
        self.counter_placeholders = {}
        self.class_colors = {
            "Longitudinal Crack": "🟡",
            "Transverse Crack": "🔴", 
            "Alligator Crack": "🟠",
            "Potholes": "🔵"
        }
        
        for i, class_name in enumerate(CLASSES):
            with self.counter_cols[i]:
                color_emoji = self.class_colors[class_name]
                self.counter_placeholders[class_name] = st.empty()
                # Initialize with 0
                self.counter_placeholders[class_name].metric(
                    f"{color_emoji} {class_name}", 
                    "0",
                    delta=None
                )
    
    def update_counters(self, detection_counts):
        """Update the real-time counters"""
        for class_name in CLASSES:
            if class_name in detection_counts:
                color_emoji = self.class_colors[class_name]
                count = detection_counts[class_name]
                
                # Update the counter display
                self.counter_placeholders[class_name].metric(
                    f"{color_emoji} {class_name}",
                    f"{count:,}",  # Add comma formatting for large numbers
                    delta=None
                )

class FixedVideoProcessor:
    """Fixed video processor - reader error çözümü + Detection Counter Integration"""
    
    def __init__(self, input_path: str, output_path: str, counter_display=None):
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = BATCH_SIZE
        
        # 🔥 NEW: Counter display reference
        self.counter_display = counter_display
        
        # Performance monitoring
        self.start_time = time.time()
        self.processed_frames = 0
        self.total_frames = 0
        self.detection_counts = {cls: 0 for cls in CLASSES}
        
        # Thread-safe counters
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Fixed queue sizes - daha büyük ve güvenli
        self.queue_size = min(256, BATCH_SIZE * 8)  # Increased
        self.frame_queue = queue.Queue(maxsize=self.queue_size)
        self.result_queue = queue.Queue(maxsize=self.queue_size)
        
        # Control flags
        self.stop_flag = threading.Event()
        self.reader_finished = threading.Event()
        self.processor_finished = threading.Event()
        
    def fixed_frame_reader(self, cap):
        """FIXED frame reader with debugging"""
        frame_count = 0
        consecutive_failures = 0
        max_failures = 5
        
        print("Frame reader started...")
        
        try:
            while not self.stop_flag.is_set():
                try:
                    # Try to read frame with error handling
                    ret, frame = cap.read()
                    
                    if not ret:
                        print(f"Video completed. Total frames read: {frame_count}")
                        break
                    
                    # Reset failure counter on successful read
                    consecutive_failures = 0
                    frame_count += 1
                    
                    # DEBUG: Log first frame
                    if frame_count == 1:
                        print(f"First frame read: {frame.shape}, dtype: {frame.dtype}")
                    
                    # Put frame in queue with debug
                    try:
                        self.frame_queue.put(frame, timeout=1.0)
                        with self.frame_lock:
                            self.total_frames += 1
                        
                        # DEBUG: Log progress
                        if frame_count % 100 == 0:
                            print(f"Reader: {frame_count} frames read, queue size: {self.frame_queue.qsize()}")
                            
                    except queue.Full:
                        if self.stop_flag.is_set():
                            break
                        print(f"Queue full at frame {frame_count}, skipping frame")
                        continue
                        
                except Exception as e:
                    consecutive_failures += 1
                    print(f"Frame read error {consecutive_failures}/{max_failures}: {e}")
                    
                    if consecutive_failures >= max_failures:
                        print(f"Too many consecutive failures, stopping reader")
                        break
                    
                    time.sleep(0.01)
                    continue
                    
        except Exception as e:
            print(f"Critical reader error: {e}")
        finally:
            # Always signal end regardless of how we exit
            try:
                print(f"Reader sending END signal after {frame_count} frames")
                self.frame_queue.put(None, timeout=1.0)
            except Exception as e:
                print(f"Failed to send END signal: {e}")
            self.reader_finished.set()
            print(f"Frame reader finished. Total frames: {frame_count}")
            print(f"Final frame queue size: {self.frame_queue.qsize()}")
    
    def fixed_batch_processor(self, confidence: float):
        """FIXED batch processor with debugging + Real-time counter updates"""
        batch_buffer = []
        processed_count = 0
        
        print("Batch processor started...")
        
        try:
            while not self.stop_flag.is_set():
                try:
                    # Get frame with longer timeout and debug
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    if frame is None:
                        print(f"Processor received END signal. Processing final batch of {len(batch_buffer)} frames")
                        # End signal received, process remaining batch
                        if batch_buffer:
                            self._process_fixed_batch(batch_buffer, confidence)
                            processed_count += 1
                        break
                    
                    # DEBUG: Log frame receipt
                    if len(batch_buffer) == 0:
                        print(f"Processor received first frame: {frame.shape}")
                    
                    batch_buffer.append(frame)
                    
                    # Process when batch is full
                    if len(batch_buffer) >= self.batch_size:
                        print(f"Processing batch of {len(batch_buffer)} frames")
                        self._process_fixed_batch(batch_buffer, confidence)
                        processed_count += 1
                        batch_buffer = []
                        print(f"Completed batch {processed_count}")
                        
                        # 🔥 NEW: Update counters after each batch
                        if self.counter_display and processed_count % 2 == 0:  # Update every 2 batches to avoid too frequent updates
                            with self.detection_lock:
                                current_counts = self.detection_counts.copy()
                            self.counter_display.update_counters(current_counts)
                        
                except queue.Empty:
                    # Timeout occurred, check if we should process partial batch
                    if self.reader_finished.is_set() and batch_buffer:
                        print(f"Reader finished, processing final partial batch of {len(batch_buffer)} frames")
                        self._process_fixed_batch(batch_buffer, confidence)
                        processed_count += 1
                        batch_buffer = []
                        
                        # 🔥 NEW: Final counter update
                        if self.counter_display:
                            with self.detection_lock:
                                current_counts = self.detection_counts.copy()
                            self.counter_display.update_counters(current_counts)
                    continue
                except Exception as e:
                    print(f"Processor error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Critical processor error: {e}")
        finally:
            # Signal end to writer
            try:
                print("Processor sending END signal to writer")
                self.result_queue.put(None, timeout=0.5)
            except:
                print("Failed to send END signal to writer")
            self.processor_finished.set()
            print(f"Batch processor finished. Processed batches: {processed_count}")
            
            # DEBUG: Check queue sizes
            print(f"Final queue sizes - Frame queue: {self.frame_queue.qsize()}, Result queue: {self.result_queue.qsize()}")
    
    def _process_fixed_batch(self, frames, confidence):
        """FAST batch processing - like original code, no preprocessing resize + Enhanced detection counting"""
        if not frames or self.stop_flag.is_set():
            return
        
        try:
            # NO PREPROCESSING - Direct inference like original code
            # GPU inference on original frames
            try:
                with torch.no_grad():
                    results = net.track(
                        frames,  # Direct original frames - NO RESIZE!
                        device=DEVICE,
                        conf=confidence,
                        verbose=False,
                        half=True,
                        batch=len(frames),
                        stream=True,
                        workers=1
                    )
                    
                    results_list = list(results)
                    
            except Exception as e:
                print(f"Inference error: {e}")
                results_list = [None] * len(frames)
            
            # 🔥 NEW: Enhanced detection counting with batch processing
            batch_detections = {cls: 0 for cls in CLASSES}
            
            # Process results - FAST
            for i, (frame, result) in enumerate(zip(frames, results_list)):
                try:
                    if result is not None:
                        # Count detections - ENHANCED for batch counting
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.cpu().numpy()
                            for box in boxes:
                                class_id = int(box.cls)
                                if 0 <= class_id < len(CLASSES):
                                    class_name = CLASSES[class_id]
                                    batch_detections[class_name] += 1
                        
                        # Get annotated frame - YOLO plot() returns original size!
                        annotated = result.plot() if hasattr(result, 'plot') else frame
                    else:
                        annotated = frame
                    
                    # Send to writer queue
                    try:
                        self.result_queue.put(annotated, timeout=1.0)
                        with self.frame_lock:
                            self.processed_frames += 1
                    except queue.Full:
                        if self.stop_flag.is_set():
                            break
                        
                except Exception as e:
                    # Fast fallback
                    try:
                        self.result_queue.put(frame, timeout=0.5)
                        with self.frame_lock:
                            self.processed_frames += 1
                    except:
                        pass
            
            # 🔥 NEW: Update global detection counts (thread-safe)
            if any(batch_detections.values()):  # Only update if there were detections
                with self.detection_lock:
                    for class_name, count in batch_detections.items():
                        self.detection_counts[class_name] += count
                        
                # Print batch detection summary
                total_batch_detections = sum(batch_detections.values())
                if total_batch_detections > 0:
                    print(f"Batch detections: {batch_detections} (Total: {total_batch_detections})")
                        
        except Exception as e:
            print(f"Batch processing error: {e}")
    
    def fixed_writer(self, writer):
        """FIXED video writer - proper codec handling"""
        written_frames = 0
        
        try:
            while not self.stop_flag.is_set():
                try:
                    # Get frame with timeout
                    frame = self.result_queue.get(timeout=2.0)
                    
                    if frame is None:
                        # End signal received
                        break
                    
                    # Write frame with proper validation
                    if isinstance(frame, np.ndarray) and frame.size > 0:
                        try:
                            # Ensure frame is the right size and format
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                # Convert RGB to BGR for OpenCV (YOLO plot returns RGB)
                                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                
                                # Ensure frame is uint8
                                if bgr_frame.dtype != np.uint8:
                                    bgr_frame = bgr_frame.astype(np.uint8)
                                
                                # Write frame
                                success = writer.write(bgr_frame)
                                if success:
                                    written_frames += 1
                                else:
                                    print(f"Failed to write frame {written_frames}")
                            else:
                                print(f"Invalid frame shape: {frame.shape}")
                                
                        except Exception as e:
                            print(f"Frame conversion error: {e}")
                            # Try writing original frame as fallback
                            try:
                                if frame.dtype != np.uint8:
                                    frame = frame.astype(np.uint8)
                                success = writer.write(frame)
                                if success:
                                    written_frames += 1
                            except Exception as e2:
                                print(f"Fallback write failed: {e2}")
                                continue
                    
                except queue.Empty:
                    # Check if processor finished
                    if self.processor_finished.is_set():
                        break
                    continue
                except Exception as e:
                    print(f"Writer error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Critical writer error: {e}")
        finally:
            print(f"Video writer finished. Written frames: {written_frames}")
            # Ensure writer is properly closed
            try:
                if hasattr(writer, 'release'):
                    writer.release()
            except:
                pass
    
    def process_video_fixed(self, confidence: float):
        """EXACTLY SAME AS ORIGINAL - NO CHANGES + Final counter update"""
        # Open video with error handling
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.input_path}")
        
        try:
            # Video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {width}x{height}, {fps}fps, {total_video_frames} frames")
            
            # Video writer with multiple codec attempts
            writer = None
            output_path = self.output_path
            
            # Try multiple codec/format combinations
            codec_attempts = [
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
                ('MJPG', '.avi'),
                ('X264', '.mp4')
            ]
            
            for codec, ext in codec_attempts:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_path = output_path.replace('.mp4', ext) if ext != '.mp4' else output_path
                    writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                    
                    if writer.isOpened():
                        self.output_path = test_path
                        print(f"Video writer initialized with {codec} codec, output: {test_path}")
                        break
                    else:
                        writer.release()
                        writer = None
                except Exception as e:
                    print(f"Failed to initialize writer with {codec}: {e}")
                    if writer:
                        writer.release()
                        writer = None
            
            if not writer or not writer.isOpened():
                raise ValueError(f"Cannot create output video with any codec")
            
            # Progress tracking
            progress_bar = st.progress(0, text="🔧 Fixed processing başlıyor...")
            stats_placeholder = st.empty()
            
            # Start threads
            threads = [
                threading.Thread(target=self.fixed_frame_reader, args=(cap,), name="FixedReader"),
                threading.Thread(target=self.fixed_batch_processor, args=(confidence,), name="FixedProcessor"),
                threading.Thread(target=self.fixed_writer, args=(writer,), name="FixedWriter")
            ]
            
            start_time = time.time()
            
            # Start all threads
            for thread in threads:
                thread.daemon = True  # Allow clean exit
                thread.start()
            
            # Monitor progress
            last_update = time.time()
            counter_update_interval = 0.5  # Update counters every 0.5 seconds
            
            try:
                while any(t.is_alive() for t in threads):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Update progress every second
                    if current_time - last_update >= 1.0:
                        with self.frame_lock:
                            processed = self.processed_frames
                            total = max(self.total_frames, 1)
                        
                        # Calculations
                        progress = min(processed / total_video_frames, 1.0) if total_video_frames > 0 else 0
                        current_fps = processed / elapsed if elapsed > 0 else 0
                        
                        # Queue status
                        queue_status = f"Frame Q: {self.frame_queue.qsize()}, Result Q: {self.result_queue.qsize()}"
                        
                        # Update UI
                        progress_bar.progress(
                            progress,
                            text=f"🔧 Fixed Processing: {processed}/{total_video_frames} | "
                                 f"FPS: {current_fps:.1f} | {queue_status}"
                        )
                        
                        # 🔥 NEW: Show detection progress in stats
                        with self.detection_lock:
                            total_detections = sum(self.detection_counts.values())
                        
                        stats_placeholder.info(
                            f"📊 Status: {queue_status} | Batch: {self.batch_size} | "
                            f"🎯 Total Detections: {total_detections:,}"
                        )
                        
                        # 🔥 NEW: Update counters more frequently
                        if self.counter_display and current_time - last_update >= counter_update_interval:
                            with self.detection_lock:
                                current_counts = self.detection_counts.copy()
                            self.counter_display.update_counters(current_counts)
                        
                        last_update = current_time
                    
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                st.warning("⚠️ Processing interrupted")
                self.stop_flag.set()
            
            finally:
                # CRITICAL: Proper shutdown sequence
                self.stop_flag.set()
                print("Stop flag set, waiting for threads to finish...")
                
                # Wait for threads with longer timeout
                for thread in threads:
                    thread.join(timeout=10.0)  # Increased timeout
                    if thread.is_alive():
                        print(f"Thread {thread.name} still alive after timeout")
                
                print("All threads finished, closing video resources...")
                
                # CRITICAL: Close writer FIRST with proper sequence
                if writer and writer.isOpened():
                    print("Closing video writer...")
                    writer.release()
                    print("Video writer closed")
                
                # Then close video capture
                if cap and cap.isOpened():
                    print("Closing video capture...")
                    cap.release()
                    print("Video capture closed")
                
                # 🔥 NEW: Final counter update
                if self.counter_display:
                    with self.detection_lock:
                        final_counts = self.detection_counts.copy()
                    self.counter_display.update_counters(final_counts)
                
                # Check if output file was created properly
                if os.path.exists(self.output_path):
                    file_size = os.path.getsize(self.output_path)
                    print(f"Output video file size: {file_size} bytes")
                    if file_size < 1000:  # Less than 1KB indicates problem
                        print("WARNING: Output file is too small, video may be corrupt")
                else:
                    print("ERROR: Output video file was not created")
                
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Final results
            end_time = time.time()
            total_time = end_time - start_time
            final_fps = self.processed_frames / total_time if total_time > 0 else 0
            
            return {
                'processed_frames': self.processed_frames,
                'total_time': total_time,
                'fps': final_fps,
                'detections': self.detection_counts.copy(),
                'efficiency': (self.processed_frames / total_video_frames * 100) if total_video_frames > 0 else 0
            }
            
        except Exception as e:
            cap.release()
            raise e

# Helper functions - SAME AS ORIGINAL
def write_uploaded_file(filename, bytesio):
    """Write uploaded file to disk"""
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def process_video_fixed_main(video_file, confidence, counter_display=None):
    """Main fixed processing function - DIRECT DOWNLOAD LIKE ORIGINAL + Counter Integration"""
    os.makedirs('./temp', exist_ok=True)
    temp_input = "./temp/fixed_input.mp4"
    temp_output = "./temp/fixed_output.mp4"
    
    write_uploaded_file(temp_input, video_file)
    
    # 🔥 NEW: Pass counter display to processor
    processor = FixedVideoProcessor(temp_input, temp_output, counter_display)
    
    try:
        results = processor.process_video_fixed(confidence)
        
        st.success("🔥 Fixed Processing Completed with Real-time Counters!")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Total Time", f"{int(results['total_time']//60):02d}:{int(results['total_time']%60):02d}")
        with col2:
            st.metric("🚀 Average FPS", f"{results['fps']:.1f}")
        with col3:
            st.metric("📊 Frames Processed", f"{results['processed_frames']:,}")
        with col4:
            st.metric("📈 Efficiency", f"{results['efficiency']:.1f}%")
        
        # 🔥 NEW: Enhanced Detection Results Summary
        st.write("### 🎯 Final Detection Summary")
        
        # Calculate total detections
        total_detections = sum(results['detections'].values())
        
        # Create a summary row
        summary_col1, summary_col2, summary_col3 = st.columns([1, 1, 2])
        with summary_col1:
            st.metric("🔥 Total Detections", f"{total_detections:,}")
        with summary_col2:
            avg_per_frame = total_detections / results['processed_frames'] if results['processed_frames'] > 0 else 0
            st.metric("📈 Avg per Frame", f"{avg_per_frame:.2f}")
        with summary_col3:
            # Show detection distribution
            if total_detections > 0:
                detection_percentages = []
                for class_name in CLASSES:
                    count = results['detections'][class_name]
                    percentage = (count / total_detections) * 100
                    detection_percentages.append(f"{class_name}: {percentage:.1f}%")
                st.info("📊 Distribution: " + " | ".join(detection_percentages))
        
        # Detailed class breakdown
        st.write("### 📈 Detailed Class Breakdown")
        det_col1, det_col2, det_col3, det_col4 = st.columns(4)
        
        detections = results['detections']
        class_colors = ["🟡", "🔴", "🟠", "🔵"]
        
        for i, (class_name, color) in enumerate(zip(CLASSES, class_colors)):
            col = [det_col1, det_col2, det_col3, det_col4][i]
            with col:
                count = detections[class_name]
                percentage = (count / total_detections * 100) if total_detections > 0 else 0
                col.metric(
                    f"{color} {class_name}", 
                    f"{count:,}",
                    delta=f"{percentage:.1f}% of total"
                )
        
        # Return the temp output path for direct access
        return temp_output
        
    except Exception as e:
        st.error(f"❌ Fixed processing error: {e}")
        return None

# System monitoring class - SAME AS ORIGINAL
class RTX3060SystemMonitor:
    @staticmethod
    def get_system_status():
        status = {}
        
        if torch.cuda.is_available():
            status['gpu_name'] = torch.cuda.get_device_name(0)
            status['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            status['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3
        else:
            status['gpu_available'] = False