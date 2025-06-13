import os
import logging
from pathlib import Path
from collections import defaultdict
import gc
import psutil
import time
import sys

import cv2
import numpy as np
import torch
import threading
import queue
from threading import Lock, Event

from ultralytics import YOLO

# Console-based video detector with TensorRT support
class ConsoleVideoDetector:
    def __init__(self, model_path, tracker_yaml_path, use_tensorrt=True):
        self.model_path = model_path
        self.tracker_yaml_path = tracker_yaml_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_tensorrt = use_tensorrt and torch.cuda.is_available()
        
        # Performance tracking
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'unique_tracks': set(),
            'class_counts': defaultdict(int),
            'start_time': 0,
            'phase_times': {},
            'current_phase': 'Initializing'
        }
        
        # Hardware analysis
        self._analyze_hardware()
        self._calculate_settings()
        
        # Thread control
        self.shutdown_event = Event()
        self.completion_event = Event()
        self.termination_lock = Lock()
        self.preprocessors_finished = 0
        self.predictors_finished = 0
        self.last_cleanup = time.time()
        
        # Initialize model with TensorRT
        self._setup_model()
        
    def _analyze_hardware(self):
        """Analyze hardware capabilities"""
        self.gpu_memory_gb = 0
        self.gpu_name = "CPU"
        
        if torch.cuda.is_available():
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_name = torch.cuda.get_device_name(0)
            
        self.cpu_cores = os.cpu_count()
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"🔍 Hardware Analysis:")
        print(f"   GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f}GB)")
        print(f"   CPU: {self.cpu_cores} cores")
        print(f"   RAM: {self.ram_gb:.1f}GB")
        
    def _calculate_settings(self):
        """Calculate optimal settings - same as original Streamlit code"""
        batch_size = min(32, int(self.gpu_memory_gb * 4))
        num_predictors = 2
        num_preprocessors = 2
        base_queue = 96
        batch_queue_size = base_queue // 4
        
        self.settings = {
            'BATCH_SIZE': batch_size,
            'NUM_PREDICTOR_THREADS': num_predictors,
            'NUM_PREPROCESSOR_THREADS': num_preprocessors,
            'READ_QUEUE_SIZE': base_queue,
            'PREPROCESS_QUEUE_SIZE': base_queue,
            'BATCH_QUEUE_SIZE': batch_queue_size,
            'RESULT_QUEUE_SIZE': base_queue,
            'MEMORY_CLEANUP_INTERVAL': 20,
            'EXPECTED_GPU_USAGE': min(85, 50 + (self.gpu_memory_gb * 5))
        }
        
        print(f"⚙️  Settings: Batch={batch_size}, Predictors={num_predictors}, Preprocessors={num_preprocessors}")
        
    def _setup_model(self):
        """Setup YOLO model with optional TensorRT optimization"""
        phase_start = time.time()
        self.stats['current_phase'] = 'Loading Model'
        
        print(f"🚀 Loading model: {Path(self.model_path).name}")
        print(f"🎯 Loading tracker: {Path(self.tracker_yaml_path).name}")
        
        self.model = YOLO(self.model_path)
        self.model.tracker = str(self.tracker_yaml_path)
        
        if torch.cuda.is_available():
            self.model.model.half().to(self.device)
            self.model.model.eval()
            
            # Warmup
            print("🔥 Warming up GPU...")
            dummy = torch.randn(1, 3, 640, 640).half().to(self.device)
            with torch.no_grad():
                _ = self.model.model(dummy)
            del dummy
            torch.cuda.empty_cache()
            
            # TensorRT optimization
            if self.use_tensorrt:
                try:
                    print("⚡ Attempting TensorRT optimization...")
                    self.stats['current_phase'] = 'TensorRT Optimization'
                    
                    # Export to TensorRT engine with dynamic batch
                    tensorrt_path = str(Path(self.model_path).with_suffix('.engine'))
                    
                    if not os.path.exists(tensorrt_path):
                        print(f"🔧 Creating TensorRT engine with dynamic batch: {Path(tensorrt_path).name}")
                        success = self.model.export(
                            format='engine',
                            imgsz=640,
                            half=True,
                            dynamic=True,  # Enable dynamic batch size
                            simplify=True,
                            workspace=4,  # 4GB workspace
                            verbose=False,
                            batch=self.settings['BATCH_SIZE']  # Set max batch size
                        )
                        
                        if success:
                            print("✅ TensorRT engine created successfully")
                            # Load TensorRT model
                            self.model = YOLO(tensorrt_path)
                            self.model.tracker = str(self.tracker_yaml_path)
                        else:
                            print("⚠️ TensorRT export failed, using original model")
                    else:
                        print(f"✅ Using existing TensorRT engine: {Path(tensorrt_path).name}")
                        self.model = YOLO(tensorrt_path)
                        self.model.tracker = str(self.tracker_yaml_path)
                        
                except Exception as e:
                    print(f"⚠️ TensorRT optimization failed: {e}")
                    print("📝 Continuing with standard CUDA model")
                    self.use_tensorrt = False
            
        self.stats['phase_times']['Model Setup'] = time.time() - phase_start
        self.stats['current_phase'] = 'Ready'
        
        optimization = "TensorRT" if self.use_tensorrt else "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"✅ Model ready with {optimization} optimization")
        
    def _create_model_pool(self):
        """Create optimized model pool - same as Streamlit version"""
        pool_size = self.settings['NUM_PREDICTOR_THREADS'] + 1
        self.model_pool = []
        
        print(f"📦 Creating model pool with {pool_size} instances...")
        
        for i in range(pool_size):
            try:
                if self.use_tensorrt:
                    # Use same TensorRT engine
                    model = YOLO(str(Path(self.model_path).with_suffix('.engine')))
                else:
                    model = YOLO(self.model_path)
                    
                model.tracker = str(self.tracker_yaml_path)
                
                if torch.cuda.is_available():
                    model.model.half().to(self.device)
                    model.model.eval()
                    
                self.model_pool.append(model)
                print(f"✅ Model {i+1}/{pool_size} ready")
                
            except Exception as e:
                print(f"❌ Model {i+1} failed: {e}")
                
        if not self.model_pool:
            print("⚠️ Creating fallback model...")
            self.model_pool.append(self.model)
            
    def _get_model(self):
        """Get model from pool"""
        if self.model_pool:
            return self.model_pool.pop()
        return self.model
        
    def _return_model(self, model):
        """Return model to pool"""
        if len(self.model_pool) < self.settings['NUM_PREDICTOR_THREADS'] + 1:
            self.model_pool.append(model)
        else:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def memory_cleanup(self):
        """Memory cleanup - same as Streamlit version"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.settings['MEMORY_CLEANUP_INTERVAL']:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.last_cleanup = current_time
            except Exception as e:
                print(f"❌ Memory cleanup error: {e}")
                
    def reader_thread(self, video_capture, read_queue, total_frames):
        """Video reader thread - same logic as Streamlit"""
        phase_start = time.time()
        self.stats['current_phase'] = 'Reading Video'
        
        frames_read = 0
        read_errors = 0
        
        print(f"📖 Video reading started: {total_frames} frames")
        
        try:
            while frames_read < total_frames and not self.shutdown_event.is_set():
                try:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                        
                    timeout_counter = 0
                    while not self.shutdown_event.is_set() and timeout_counter < 5:
                        try:
                            read_queue.put((frames_read, frame), timeout=0.5)
                            break
                        except queue.Full:
                            timeout_counter += 1
                            continue
                    
                    if timeout_counter >= 5:
                        print(f"⚠️ Queue full: frame {frames_read}")
                        
                    frames_read += 1
                    self.stats['total_frames'] = frames_read
                    
                    if frames_read % 1000 == 0:
                        print(f"📖 Read: {frames_read}/{total_frames}")
                        
                except Exception as e:
                    read_errors += 1
                    if read_errors > 10:
                        print(f"❌ Too many read errors")
                        break
                    continue
            
            # Send termination signals
            for _ in range(self.settings['NUM_PREPROCESSOR_THREADS']):
                try:
                    read_queue.put(None, timeout=2.0)
                except queue.Full:
                    pass
                    
            self.stats['phase_times']['Reading'] = time.time() - phase_start
            print(f"📖 Reading completed: {frames_read} frames, {read_errors} errors")
            
        except Exception as e:
            print(f"❌ Reader fatal error: {e}")
            
    def preprocessor_worker(self, worker_id, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor):
        """Preprocessor worker - same as Streamlit"""
        frames_processed = 0
        
        print(f"🔧 Preprocessor {worker_id} started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = read_queue.get(timeout=1.0)
                    
                    if item is None:
                        try:
                            preprocess_queue.put(None, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.preprocessors_finished += 1
                            print(f"🔧 Preprocessor {worker_id} finished ({self.preprocessors_finished}/{self.settings['NUM_PREPROCESSOR_THREADS']})")
                        break
                        
                    frame_idx, frame = item
                    
                    # Process frame - same logic
                    processed_frame = self._process_frame(frame, roi_vertical, roi_horizontal, resize_factor)
                    
                    if processed_frame is not None:
                        try:
                            preprocess_queue.put((frame_idx, processed_frame), timeout=1.0)
                            frames_processed += 1
                        except queue.Full:
                            continue
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Preprocessor {worker_id} error: {e}")
                    
            print(f"🔧 Preprocessor {worker_id} completed: {frames_processed} frames")
            
        except Exception as e:
            print(f"❌ Preprocessor {worker_id} fatal error: {e}")
            
    def _process_frame(self, frame, roi_vertical, roi_horizontal, resize_factor):
        """Frame processing - exact same as Streamlit"""
        try:
            height, width = frame.shape[:2]
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            roi_top = int(new_height * (1 - roi_vertical))
            roi_side = int(new_width * roi_horizontal)
            
            if roi_top > 0:
                frame_resized[:roi_top] = 0
            if roi_side > 0:
                frame_resized[:, :roi_side] = 0
                frame_resized[:, new_width-roi_side:] = 0
                    
            return frame_resized
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return None
            
    def batch_collector(self, preprocess_queue, batch_queue):
        """Batch collector - same as Streamlit"""
        phase_start = time.time()
        self.stats['current_phase'] = 'Batch Collection'
        
        current_batch = []
        current_indices = []
        none_count = 0
        batch_id = 0
        
        print(f"📦 Batch collector started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    item = preprocess_queue.get(timeout=1.0)
                    
                    if item is None:
                        none_count += 1
                        print(f"📦 Termination signal {none_count}/{self.settings['NUM_PREPROCESSOR_THREADS']}")
                        
                        if none_count >= self.settings['NUM_PREPROCESSOR_THREADS']:
                            if current_batch:
                                batch_id += 1
                                print(f"📦 FINAL batch sent: {len(current_batch)} frames")
                                try:
                                    batch_queue.put((current_batch, current_indices, batch_id), timeout=2.0)
                                except queue.Full:
                                    pass
                            
                            for _ in range(self.settings['NUM_PREDICTOR_THREADS']):
                                try:
                                    batch_queue.put(None, timeout=1.0)
                                except queue.Full:
                                    pass
                            break
                        continue
                        
                    frame_idx, frame = item
                    current_batch.append(frame)
                    current_indices.append(frame_idx)
                    
                    if len(current_batch) >= self.settings['BATCH_SIZE']:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, batch_id), timeout=1.0)
                            print(f"📦 Batch {batch_id} sent: {len(current_batch)} frames")
                        except queue.Full:
                            continue
                            
                        current_batch = []
                        current_indices = []
                        
                except queue.Empty:
                    if current_batch and len(current_batch) >= self.settings['BATCH_SIZE'] // 2:
                        batch_id += 1
                        try:
                            batch_queue.put((current_batch, current_indices, batch_id), timeout=0.5)
                            print(f"📦 Timeout batch {batch_id}: {len(current_batch)} frames")
                            current_batch = []
                            current_indices = []
                        except queue.Full:
                            pass
                    continue
                    
            self.stats['phase_times']['Batch Collection'] = time.time() - phase_start
            print(f"📦 Batch collector completed: {batch_id} batches created")
            
        except Exception as e:
            print(f"❌ Batch collector fatal error: {e}")
            
    def predictor_worker(self, worker_id, batch_queue, result_queue, score_threshold):
        """YOLO predictor worker - enhanced with TensorRT"""
        model = None
        batches_processed = 0
        
        optimization = "TensorRT" if self.use_tensorrt else "CUDA"
        print(f"🤖 Predictor {worker_id} started ({optimization})")
        
        try:
            model = self._get_model()
            
            while not self.shutdown_event.is_set():
                try:
                    batch_data = batch_queue.get(timeout=2.0)
                    
                    if batch_data is None:
                        try:
                            result_queue.put(None, timeout=1.0)
                        except queue.Full:
                            pass
                            
                        with self.termination_lock:
                            self.predictors_finished += 1
                            print(f"🤖 Predictor {worker_id} finished ({self.predictors_finished}/{self.settings['NUM_PREDICTOR_THREADS']})")
                        break
                        
                    frames, indices, batch_id = batch_data
                    
                    if batches_processed % 10 == 0:
                        self.memory_cleanup()
                    
                    # YOLO inference with TensorRT
                    processed_results = self._run_yolo_inference(model, frames, indices, batch_id, score_threshold)
                    
                    if processed_results:
                        self.stats['processed_frames'] += len(frames)
                        try:
                            result_queue.put(processed_results, timeout=1.0)
                        except queue.Full:
                            pass
                        
                        print(f"🤖 Worker {worker_id} batch {batch_id} completed: {len(frames)} frames")
                    
                    batches_processed += 1
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Predictor {worker_id} error: {e}")
                    
            print(f"🤖 Predictor {worker_id} completed: {batches_processed} batches processed")
                    
        except Exception as e:
            print(f"❌ Predictor {worker_id} fatal error: {e}")
        finally:
            if model:
                self._return_model(model)
                
    def _run_yolo_inference(self, model, frames, indices, batch_id, score_threshold):
        """YOLO inference - same as Streamlit with tracking"""
        try:
            results = model.track(
                frames,
                device=self.device,
                conf=score_threshold,
                imgsz=640,
                verbose=False,
                persist=True,
                tracker=str(self.tracker_yaml_path),
                half=True
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
                            track_key = f"{class_id}_{track_id}"
                            if track_key not in self.stats['unique_tracks']:
                                self.stats['unique_tracks'].add(track_key)
                                self.stats['class_counts'][class_id] += 1
                
                # Create annotated frame
                annotated = result.plot()
                processed_results.append({
                    'frame_idx': indices[i],
                    'annotated_frame': annotated,
                    'batch_id': batch_id
                })
            
            return processed_results
            
        except Exception as e:
            print(f"❌ YOLO inference error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
            
    def writer_thread(self, result_queue, cv2writer):
        """Video writer thread - same as Streamlit"""
        phase_start = time.time()
        self.stats['current_phase'] = 'Writing Video'
        
        frame_buffer = {}
        expected_frame_idx = 0
        none_count = 0
        frames_written = 0
        
        print(f"🎬 Video writing started")
        
        try:
            while not self.shutdown_event.is_set():
                try:
                    results = result_queue.get(timeout=2.0)
                    
                    if results is None:
                        none_count += 1
                        print(f"🎬 Writer termination signal {none_count}/{self.settings['NUM_PREDICTOR_THREADS']}")
                        
                        if none_count >= self.settings['NUM_PREDICTOR_THREADS']:
                            break
                        continue
                        
                    for result in results:
                        frame_idx = result['frame_idx']
                        frame_buffer[frame_idx] = result['annotated_frame']
                    
                    while expected_frame_idx in frame_buffer:
                        frame = frame_buffer.pop(expected_frame_idx)
                        cv2writer.write(frame)
                        frames_written += 1
                        expected_frame_idx += 1
                        
                        if frames_written % 500 == 0:
                            print(f"🎬 Written: {frames_written} frames")
                    
                    if len(frame_buffer) > 200:
                        print(f"⚠️ Large frame buffer: {len(frame_buffer)} frames")
                        if frame_buffer:
                            min_available = min(frame_buffer.keys())
                            if min_available > expected_frame_idx + 100:
                                print(f"⚠️ Skipping gap: {expected_frame_idx} → {min_available}")
                                expected_frame_idx = min_available
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Writer error: {e}")
            
            # Write remaining frames
            if frame_buffer:
                print(f"🎬 Writing remaining {len(frame_buffer)} frames...")
                for idx in sorted(frame_buffer.keys()):
                    cv2writer.write(frame_buffer[idx])
                    frames_written += 1
            
            self.stats['phase_times']['Writing'] = time.time() - phase_start
            print(f"🎬 Video writing completed: {frames_written} frames")
            self.completion_event.set()
            
        except Exception as e:
            print(f"❌ Writer fatal error: {e}")
            self.completion_event.set()
            
    def _progress_monitor(self):
        """Enhanced progress monitoring with detailed timing"""
        last_update = time.time()
        
        while not self.completion_event.is_set() and not self.shutdown_event.is_set():
            current_time = time.time()
            
            if current_time - last_update >= 2.0:
                elapsed = current_time - self.stats['start_time']
                processed = self.stats['processed_frames']
                total = self.stats['total_frames']
                
                if total > 0:
                    progress = (processed / total) * 100
                    fps = processed / max(1, elapsed)
                    remaining = (elapsed / max(1, processed)) * (total - processed) if processed > 0 else 0
                    
                    print(f"\n⏱️  Time: {elapsed:.1f}s | Phase: {self.stats['current_phase']}")
                    print(f"📊 Progress: {progress:.1f}% | Processed: {processed}/{total} frames")
                    print(f"🚀 FPS: {fps:.1f} | Remaining: {remaining:.0f}s")
                    
                    if self.stats['unique_tracks']:
                        unique_tracks = len(self.stats['unique_tracks'])
                        print(f"🎯 Unique tracks: {unique_tracks}")
                        
                        track_summary = []
                        for class_id, count in self.stats['class_counts'].items():
                            track_summary.append(f"Class {class_id}: {count}")
                        if track_summary:
                            print(f"📋 Tracks: {' | '.join(track_summary)}")
                
                last_update = current_time
                
            time.sleep(1.0)
            
    def process_video(self, input_path, output_path, score_threshold=0.5, roi_vertical=0.6, roi_horizontal=0.2, resize_factor=0.5):
        """Main video processing function with robust video handling"""
        print(f"\n⚡ Starting optimized video detection")
        print(f"📁 Input: {Path(input_path).name}")
        print(f"📁 Output: {Path(output_path).name}")
        print(f"🎯 Settings: conf={score_threshold}, roi_v={roi_vertical}, roi_h={roi_horizontal}, resize={resize_factor}")
        
        # Create model pool
        self._create_model_pool()
        
        # Get video info with robust handling
        video_capture = cv2.VideoCapture(input_path)
        if not video_capture.isOpened():
            print(f"❌ Cannot open video: {input_path}")
            return
        
        # Try different backends if needed
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
        
        _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = video_capture.get(cv2.CAP_PROP_FPS)
        _total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate video properties
        if _width <= 0 or _height <= 0 or _fps <= 0:
            print(f"❌ Invalid video properties: {_width}x{_height} @ {_fps} FPS")
            video_capture.release()
            return
            
        if _total_frames <= 0:
            print("⚠️ Cannot get total frame count, using frame-by-frame reading")
            _total_frames = 999999  # Large number for unknown length
        
        processed_width = int(_width * resize_factor)
        processed_height = int(_height * resize_factor)
        
        print(f"🎬 Video: {_width}x{_height} @ {_fps:.1f} FPS | {_total_frames} frames")
        print(f"📐 Output: {processed_width}x{processed_height}")
        
        self.stats['start_time'] = time.time()
        
        # Setup queues - same sizes as Streamlit
        read_queue = queue.Queue(maxsize=self.settings['READ_QUEUE_SIZE'])
        preprocess_queue = queue.Queue(maxsize=self.settings['PREPROCESS_QUEUE_SIZE'])
        batch_queue = queue.Queue(maxsize=self.settings['BATCH_QUEUE_SIZE'])
        result_queue = queue.Queue(maxsize=self.settings['RESULT_QUEUE_SIZE'])
        
        # Robust video writer setup
        try:
            # Try H264 codec first
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            cv2writer = cv2.VideoWriter(output_path, fourcc, _fps, (processed_width, processed_height))
            
            # Test if writer is working
            if not cv2writer.isOpened():
                print("⚠️ H264 codec failed, trying mp4v...")
                cv2writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                cv2writer = cv2.VideoWriter(output_path, fourcc, _fps, (processed_width, processed_height))
                
            if not cv2writer.isOpened():
                print("⚠️ mp4v codec failed, trying XVID...")
                cv2writer.release()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = output_path.replace('.mp4', '.avi')  # Change extension
                cv2writer = cv2.VideoWriter(output_path, fourcc, _fps, (processed_width, processed_height))
                
            if not cv2writer.isOpened():
                print(f"❌ Cannot create video writer for: {output_path}")
                video_capture.release()
                return
                
            print(f"✅ Video writer ready: {Path(output_path).suffix} format")
            
        except Exception as e:
            print(f"❌ Video writer setup failed: {e}")
            video_capture.release()
            return
        
        threads = []
        
        try:
            print(f"\n🚀 Starting {len(self.model_pool)} threads pipeline...")
            
            # Create threads - same structure as Streamlit
            reader = threading.Thread(
                target=self.reader_thread,
                args=(video_capture, read_queue, _total_frames),
                name="Reader", daemon=True
            )
            threads.append(reader)
            
            for i in range(self.settings['NUM_PREPROCESSOR_THREADS']):
                preprocessor = threading.Thread(
                    target=self.preprocessor_worker,
                    args=(i, read_queue, preprocess_queue, roi_vertical, roi_horizontal, resize_factor),
                    name=f"Preprocessor-{i}", daemon=True
                )
                threads.append(preprocessor)
            
            collector = threading.Thread(
                target=self.batch_collector,
                args=(preprocess_queue, batch_queue),
                name="Collector", daemon=True
            )
            threads.append(collector)
            
            for i in range(self.settings['NUM_PREDICTOR_THREADS']):
                predictor = threading.Thread(
                    target=self.predictor_worker,
                    args=(i, batch_queue, result_queue, score_threshold),
                    name=f"Predictor-{i}", daemon=True
                )
                threads.append(predictor)
            
            writer = threading.Thread(
                target=self.writer_thread,
                args=(result_queue, cv2writer),
                name="Writer", daemon=True
            )
            threads.append(writer)
            
            monitor = threading.Thread(
                target=self._progress_monitor,
                name="Monitor", daemon=True
            )
            threads.append(monitor)
            
            # Start all threads
            print(f"🎬 Starting {len(threads)} threads...")
            for t in threads:
                t.start()
                time.sleep(0.1)  # Small delay between thread starts
            
            # Wait for completion with timeout
            print("⏳ Processing started, waiting for completion...")
            completed = self.completion_event.wait(timeout=3600)  # 1 hour max
            
            if not completed:
                print("⚠️ Processing timeout reached")
                self.shutdown_event.set()
            
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.shutdown_event.set()
        except Exception as e:
            print(f"\n❌ Processing error: {e}")
            self.shutdown_event.set()
        
        finally:
            print("\n🔄 Cleaning up...")
            self.shutdown_event.set()
            
            # Wait for threads to finish
            for t in threads:
                t.join(timeout=5.0)
            
            video_capture.release()
            cv2writer.release()
            
            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Final results
            total_time = time.time() - self.stats['start_time']
            processed = self.stats['processed_frames']
            unique_tracks = len(self.stats['unique_tracks'])
            
            print(f"\n✅ Processing completed!")
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"📊 Processed frames: {processed}")
            
            if processed > 0:
                print(f"🚀 Average FPS: {processed/total_time:.1f}")
                print(f"🎯 Unique tracks: {unique_tracks}")
                
                if self.stats['class_counts']:
                    print(f"📋 Track breakdown:")
                    for class_id, count in self.stats['class_counts'].items():
                        print(f"   Class {class_id}: {count} unique tracks")
            else:
                print("⚠️ No frames were processed - check video file and codec compatibility")
            
            if self.stats['phase_times']:
                print(f"\n📋 Phase timing breakdown:")
                for phase, duration in self.stats['phase_times'].items():
                    print(f"   {phase}: {duration:.1f}s")
            
            optimization = "TensorRT" if self.use_tensorrt else "CUDA" if torch.cuda.is_available() else "CPU"
            print(f"🔥 Optimization used: {optimization}")
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)  # MB
                print(f"📁 Output file: {Path(output_path).name} ({file_size:.1f} MB)")
            else:
                print(f"⚠️ Output file not created: {output_path}")

def install_tensorrt_info():
    """TensorRT kurulum bilgisi"""
    print("\n🔧 TensorRT Installation Guide:")
    print("1. Download TensorRT from NVIDIA Developer:")
    print("   https://developer.nvidia.com/tensorrt")
    print("2. Install with pip:")
    print("   pip install tensorrt")
    print("3. Or conda:")
    print("   conda install tensorrt -c conda-forge")
    print("\n⚡ TensorRT can provide 20-40% speedup for inference!")

def main():
    # Default paths
    model_path = r"C:\Users\fthgz\OneDrive\Belgeler\RoadDamageDetection-main\models\YOLOv8_Small_RDD.pt"
    tracker_path = r"C:\Users\fthgz\OneDrive\Belgeler\RoadDamageDetection-main\models\bytetrack.yaml"
    input_video = r"C:\Users\fthgz\OneDrive\Belgeler\RoadDamageDetection-main\42 AJF 307_Uygun 30dk.mp4"
    output_video = r"C:\Users\fthgz\OneDrive\Belgeler\RoadDamageDetection-main\cuda_detected_output.mp4"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    if not os.path.exists(tracker_path):
        print(f"❌ Tracker file not found: {tracker_path}")
        return
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    try:
        # Create detector WITHOUT TensorRT for now
        detector = ConsoleVideoDetector(
            model_path=model_path,
            tracker_yaml_path=tracker_path,
            use_tensorrt=False  # TensorRT devre dışı - sadece CUDA kullan
        )
        
        print("🔥 Running with CUDA optimization (TensorRT disabled)")
        
        # Process video with same parameters as Streamlit
        detector.process_video(
            input_path=input_video,
            output_path=output_video,
            score_threshold=0.5,    # Detection confidence
            roi_vertical=0.6,       # Keep bottom 60%
            roi_horizontal=0.2,     # Remove 20% from sides
            resize_factor=0.5       # Resize to 50%
        )
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n💡 TensorRT kurulumu için:")
    print("   conda install tensorrt -c conda-forge")
    print("   veya NVIDIA Developer'dan indirin")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Use default paths
        main()
    elif len(sys.argv) == 6:
        # Custom arguments: model tracker input output tensorrt_flag
        model_path = sys.argv[1]
        tracker_path = sys.argv[2]
        input_video = sys.argv[3]
        output_video = sys.argv[4]
        use_tensorrt = sys.argv[5].lower() in ['true', '1', 'yes']
        
        detector = ConsoleVideoDetector(
            model_path=model_path,
            tracker_yaml_path=tracker_path,
            use_tensorrt=use_tensorrt
        )
        
        detector.process_video(
            input_path=input_video,
            output_path=output_video
        )
    else:
        print("Usage:")
        print("  python detector.py  # Use default paths")
        print("  python detector.py <model.pt> <tracker.yaml> <input.mp4> <output.mp4> <use_tensorrt>")
        print("Example:")
        print("  python detector.py model.pt bytetrack.yaml input.mp4 output.mp4 true")
        sys.exit(1)