import os
import logging
from pathlib import Path
from typing import List, NamedTuple
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
import torch

import threading
import queue
import time

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection with Tracking + Preprocessing",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
tracker_yaml_path = ROOT / "./models/bytetrack.yaml"

BATCH_SIZE = 24  # GPU verimini artırmak için

device = "cuda" if torch.cuda.is_available() else "cpu"

# Session-specific caching
cache_key = "yolov8smallrdd_tracker"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH) 
    net.tracker = str(tracker_yaml_path)
    
    
        
    st.session_state[cache_key] = net



CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack", 
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray
    track_id: int = None

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

_total_frames_count = 0
_processed_frames_count = 0
_total_lock = threading.Lock()
_processed_lock = threading.Lock()

# TRACKING: Global variables
_unique_track_ids = set()
_track_lock = threading.Lock()
_class_counts = defaultdict(int)
_class_lock = threading.Lock()

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

def write_bytesio_to_file(filename, bytesio):
    """Write the contents of the given BytesIO to a file."""
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def apply_preprocessing(frame, roi_vertical, roi_horizontal, resize_factor):
  
    # 1. Boyut küçült
    height, width = frame.shape[:2]
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # 2. ROI maskesi oluştur
    roi_top = int(new_height * (1 - roi_vertical))  # Üstten kesilecek
    roi_side = int(new_width * roi_horizontal)      # Kenarlardan kesilecek
    
    # 3. Maske uygula - sadece orta-alt kısım kalacak yol kısımı
    mask = np.zeros_like(frame_resized)
    mask[roi_top:, roi_side:new_width-roi_side] = 255
    frame_processed = cv2.bitwise_and(frame_resized, mask)
    
    return frame_processed

def create_tracking_counters():
    st.write("### 🔍 Nesne Takip Sayaçları")
    counter_cols = st.columns(4)
    counter_placeholders = {}
    colors = ["🟡", "🔴", "🟠", "🔵"]
    
    for i, class_name in enumerate(CLASSES):
        with counter_cols[i]:
            counter_placeholders[class_name] = st.empty()
            counter_placeholders[class_name].metric(
                f"{colors[i]} {class_name}", 
                "0"
            )
    
    return counter_placeholders

def processVideo(video_file, score_threshold, roi_vertical=0.6, roi_horizontal=0.2, resize_factor=0.5):
  
    
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _total_frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _total_frames/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds)

        # Preprocessing sonrası boyutları hesapla
        processed_width = int(_width * resize_factor)
        processed_height = int(_height * resize_factor)

        st.write("**Orijinal Video:**")
        st.write(f"Duration: {_duration_strings} | Size: {_width}x{_height} | FPS: {_fps:.1f} | Total Frames: {_total_frames}")
        
        st.write("**Preprocessing Ayarları:**")
        st.write(f"Resize Factor: {resize_factor:.2f} | ROI Vertical: {roi_vertical:.2f} | ROI Horizontal: {roi_horizontal:.2f}")
        st.write(f"İşlenmiş Boyut: {processed_width}x{processed_height}")

        inferenceBarText = "Performing inference with integrated preprocessing..."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()
        
        # TRACKING: Sayaç placeholder'larını oluştur
        counter_placeholders = create_tracking_counters()
        track_stats = st.empty()

        stop_event = threading.Event()
        shutdown_flag = threading.Event()

        # Video writer - preprocessing sonrası boyutları kullan
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (processed_width, processed_height))

        # Frame okuma yazma kuyrukları
        read_queue = queue.Queue(maxsize=512)
        write_queue = queue.Queue(maxsize=512)

        # Reset counters thread-safe
        global _total_frames_count, _processed_frames_count, _unique_track_ids, _class_counts
        with _total_lock:
            _total_frames_count = 0
        with _processed_lock:
            _processed_frames_count = 0
        with _track_lock:
            _unique_track_ids = set()
        with _class_lock:
            _class_counts = defaultdict(int)

        

        # Yazıcı thread e pre processing entegre edildi 
        def reader():
            global _total_frames_count
            while not stop_event.is_set() and not shutdown_flag.is_set():
                ret, frame = videoCapture.read()
                if not ret or shutdown_flag.is_set():
                    try:
                        read_queue.put(None, timeout=1)
                    except queue.Full:
                        pass
                    break
                
                #  PREPROCESSING BURADA YAPILIYOR 
                try:
                    processed_frame = apply_preprocessing(
                        frame, 
                        roi_vertical=roi_vertical,
                        roi_horizontal=roi_horizontal, 
                        resize_factor=resize_factor
                    )
                    
                    # Preprocessed frame'i queue'ya koy
                    while not stop_event.is_set() and not shutdown_flag.is_set():
                        try:
                            read_queue.put(processed_frame, timeout=1)
                            with _total_lock:
                                _total_frames_count += 1
                            break
                        except queue.Full:
                            continue
                            
                except Exception as e:
                    print(f"Preprocessing error: {e}")
                    continue  # Skip this frame
 
        # predictor thread direkt olarak pre process edilmiş frame i alır
        def predictor():
            global _processed_frames_count
            buffer = []
            while not stop_event.is_set() and not shutdown_flag.is_set():
                try:
                    frame = read_queue.get(timeout=0.5)
                    
                    if frame is None or shutdown_flag.is_set():
                        #
                        if buffer and not shutdown_flag.is_set():
                            process_batch(buffer)
                            with _processed_lock:
                                _processed_frames_count += len(buffer)
                        try:
                            write_queue.put(None, timeout=1)
                        except queue.Full:
                            pass
                        break

                    buffer.append(frame)
                    if len(buffer) == BATCH_SIZE:
                        if not shutdown_flag.is_set():
                            process_batch(buffer)
                            with _processed_lock:
                                _processed_frames_count += len(buffer)
                        buffer = []
                
                except queue.Empty:
                    if shutdown_flag.is_set():
                        break
                    continue

        def process_batch(frames):
            if shutdown_flag.is_set():
                return
                
            try:
                processed_frames = []

                for frame in frames:
                    if shutdown_flag.is_set():
                        return
                    processed_frames.append(frame)
        
                if not shutdown_flag.is_set():
                    # tracker kullanılıyor detection değil bytetrack kullanılıyor
                    results = net.track(
                        processed_frames, 
                        device=device, 
                        conf=score_threshold, 
                        imgsz=640, 
                        verbose=True,
                        persist=True,
                        tracker=str(tracker_yaml_path)
                    )

                    for i, result in enumerate(results):
                        if shutdown_flag.is_set():
                            return
                        
                        # track id ler için olan kısım
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.cpu().numpy()
                            for box in boxes:
                                class_id = int(box.cls)
                                
                                if hasattr(box, 'id') and box.id is not None:
                                    track_id = int(box.id)
                                    track_key = f"{class_id}_{track_id}"
                                    
                                    with _track_lock:
                                        if track_key not in _unique_track_ids:
                                            _unique_track_ids.add(track_key)
                                            
                                            with _class_lock:
                                                if 0 <= class_id < len(CLASSES):
                                                    class_name = CLASSES[class_id]
                                                    _class_counts[class_name] += 1
                        
                        annotated = result.plot()
                        try:
                            write_queue.put(annotated, timeout=1)
                        except queue.Full:
                            if shutdown_flag.is_set():
                                return
                            pass
        
            except Exception as e:
                print(f"Batch işleme hatası: {e}")
        
        # yazıcı thread kısmı
        def writer_thread():
            while not stop_event.is_set() and not shutdown_flag.is_set():
                try:
                    frame = write_queue.get(timeout=0.5)
                    if frame is None or shutdown_flag.is_set():
                        break
                    cv2writer.write(frame) 
                except queue.Empty:
                    if shutdown_flag.is_set():
                        break
                    continue

        # threadleri başlat
        threads = [
            threading.Thread(target=reader, name="Reader+Preprocessor"),
            threading.Thread(target=predictor, name="Predictor"), 
            threading.Thread(target=writer_thread, name="Writer")
        ]

        start_time = time.time()

        for t in threads:
            t.start()

        # trackin takip kısmı önemsiz kısım
        try:
            last_counter_update = time.time()
            
            while any(t.is_alive() for t in threads) and not shutdown_flag.is_set():
                with _total_lock:
                    total = _total_frames_count
                with _processed_lock:
                    done = _processed_frames_count
                if total > 0:
                    inferenceBar.progress(min(done / _total_frames, 1.0), text=f"Processed: {done}/{total} ({(done/_total_frames*100):.1f}%) | Queues: R{read_queue.qsize()} W{write_queue.qsize()} | {(time.time()-start_time):.1f}s")
                
                
                current_time = time.time()
                if current_time - last_counter_update >= 0.5:
                    with _track_lock:
                        track_count = len(_unique_track_ids)
                    
                    track_stats.info(f"🔍 Benzersiz Takip Edilen Nesneler: {track_count}")
                    
                    with _class_lock:
                        for class_name in CLASSES:
                            counter_placeholders[class_name].metric(
                                f"{class_name}", 
                                f"{_class_counts[class_name]}"
                            )
                    
                    last_counter_update = current_time
                
               
                
                time.sleep(0.2)
            
           
            with _total_lock:
                total = _total_frames_count
            with _processed_lock:
                done = _processed_frames_count
            if total > 0:
                inferenceBar.progress(min(done / _total_frames, 1.0), text=f"Completed: {done}/{total} ({(done/_total_frames*100):.1f}%)")

        except KeyboardInterrupt:
            print("🛑 Graceful shutdown başlatılıyor...")
            shutdown_flag.set()
            stop_event.set()
            
            try:
                read_queue.put(None, timeout=1)
                write_queue.put(None, timeout=1)
            except:
                pass
                
        finally:
            print("Thread'ların kapanması bekleniyor...")
            for t in threads:
                t.join(timeout=3.0)
                if t.is_alive():
                    print(f"⚠️ {t.name} hala çalışıyor")
            
            videoCapture.release()
            cv2writer.release()

            end_time = time.time()
            
            with _track_lock:
                final_track_count = len(_unique_track_ids)
            
            with _class_lock:
                final_class_counts = dict(_class_counts)
            
            if shutdown_flag.is_set():
                print(f"⚠️ Video işleme durduruldu. Süre: {end_time - start_time:.2f} saniye")
                st.warning("Video işleme durduruldu!")
                inferenceBar.progress(0, text="İşlem durduruldu")
            else:
                print(f"✅ Video işleme tamamlandı. Süre: {end_time - start_time:.2f} saniye")
                st.success(f"Video Processed! 🎯 Benzersiz tespit: {final_track_count}")
                
                st.write("### 🔍 Tespit Sonuçları")
                st.markdown(f"""
                * **Preprocessing:** Resize {resize_factor:.2f}x, ROI V{roi_vertical:.2f}/H{roi_horizontal:.2f}
                * **Tracking:** ByteTrack
                * **Benzersiz Nesneler:** {final_track_count}
                * **İşlem Süresi:** {end_time - start_time:.2f} saniye
                
                **Sınıf Dağılımı:**
                """)
                
                class_cols = st.columns(4)
                for i, class_name in enumerate(CLASSES):
                    with class_cols[i]:
                        count = final_class_counts.get(class_name, 0)
                        total = sum(final_class_counts.values())
                        percentage = (count / total * 100) if total > 0 else 0
                        st.metric(
                            f"{class_name}", 
                            f"{count}",
                            delta=f"{percentage:.1f}%"
                        )

# UI Layout
col1, col2 = st.columns(2)
with col1:
    with open(temp_file_infer, "rb") as f:
        st.download_button(
            label="Download Prediction Video",
            data=f,
            file_name="RDD_Tracking_Prediction.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
with col2:
    if st.button('Restart Apps', use_container_width=True, type="primary"):
        st.rerun()

st.title("Road Damage Detection with Integrated Preprocessing + Tracking")
st.write("Video preprocessing (resize + ROI) ve tracking tek pipeline'da yapılır - zaman kaybı yok!")

# Video upload
video_file = st.file_uploader("Upload Video", type=".mp4", disabled=st.session_state.runningInference,)
st.caption("Max 1GB .mp4 video files supported.")

# YOLO settings
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)

# PREPROCESSING CONTROLS - Yeni UI!
st.write("---")
st.write("### ⚙️ Preprocessing Settings")

col_pre1, col_pre2, col_pre3 = st.columns(3)

with col_pre1:
    resize_factor = st.slider(
        "Resize Factor", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        disabled=st.session_state.runningInference,
        help="Video boyutunu küçültme oranı. 0.5 = yarıya indir"
    )

with col_pre2:
    roi_vertical = st.slider(
        "ROI Vertical", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        disabled=st.session_state.runningInference,
        help="Alt kısmın ne kadarının kalacağı. 0.6 = alt %60'ı al"
    )

with col_pre3:
    roi_horizontal = st.slider(
        "ROI Horizontal", 
        min_value=0.0, 
        max_value=0.5, 
        value=0.2, 
        step=0.1,
        disabled=st.session_state.runningInference,
        help="Kenarlardan ne kadar kesileceği. 0.2 = her kenardan %20 kes"
    )

st.info(f"💡 Final boyut: Original x {resize_factor:.1f} | ROI: Alt %{roi_vertical*100:.0f}, Kenar -%{roi_horizontal*100:.0f}")

if video_file is not None:
    if st.button('🚀 Process Video (Preprocessing + Tracking)', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = f"Processing {video_file.name} with integrated preprocessing + tracking"
        st.warning(_warning)
        processVideo(
            video_file, 
            score_threshold,
            roi_vertical=roi_vertical,
            roi_horizontal=roi_horizontal, 
            resize_factor=resize_factor
        )