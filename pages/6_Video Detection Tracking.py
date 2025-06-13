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
    page_title="Video Detection with Tracking",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
tracker_yaml_path = ROOT / "./models/bytetrack.yaml"

BATCH_SIZE = 12  # GPU verimini artırmak için Eğer GPU'n güçlü ise BATCH_SIZE = 8 veya 16 yapabilirsin.

device = "cuda" if torch.cuda.is_available() else "cpu" # Sistemde Cuda Destekliyor mu Kontrol edilir.

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd_tracker"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH) 
    
    # TRACKING MOD: Tracker'ı başlat
    net.tracker = str(tracker_yaml_path)  # Alternatif: "botsort" byte track daha iyi... oluşturduğumuz yaml dosyasından al
    
    # CUDA GPU ya taşı
    if device == "cuda":
        net.to("cuda")
        st.write(f"Model GPU (CUDA) ile yüklendi. Tracker: Bytetracks")
    else:
        st.write(f"CUDA desteklenmiyor, model CPU ile çalışıyor. Tracker: Bytetrack")
        
    st.session_state[cache_key] = net

print(f"Model loaded to {device.upper()} with tracker: {net.tracker}")

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
    track_id: int = None  # TRACKING: Track ID ekledik benzersiz trackler için

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

_total_frames_count = 0
_processed_frames_count = 0
_total_lock = threading.Lock()
_processed_lock = threading.Lock()

# TRACKING: Benzersiz track ID'leri için global değişkenler
_unique_track_ids = set()
_track_lock = threading.Lock()
_class_counts = defaultdict(int)
_class_lock = threading.Lock()

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

# TRACKING: Sayaçları gösterecek fonksiyon
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

def processVideo(video_file, score_threshold):
    
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

        st.write("Video Duration :", _duration_strings)
        st.write("Width, Height, FPS, Total FPS :", _width, _height, _fps, _total_frames)

        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()
        
        # TRACKING: Sayaç placeholder'larını oluştur
        counter_placeholders = create_tracking_counters()
        track_stats = st.empty()

        stop_event = threading.Event()
        # GRACEFUL SHUTDOWN EKLENTİSİ: Global shutdown koordinasyonu için
        shutdown_flag = threading.Event()

        # Issue with opencv-python with pip doesn't support h264 codec due to license, so we cant show the mp4 video on the streamlit in the cloud
        # If you can install the opencv through conda using this command, maybe you can render the video for the streamlit
        # $ conda install -c conda-forge opencv
        # fourcc_mp4 = cv2.VideoWriter_fourcc(*'h264')
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # 2. Frame okuma kuyruğu ve yazma kuyruğu
        read_queue = queue.Queue(maxsize=128)
        write_queue = queue.Queue(maxsize=128)

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

        # 2000 frame milestone flag
        milestone_2000_logged = False

        # 4. Okuyucu Thread tüm frameleri oku
        def reader():
            global _total_frames_count
            # GRACEFUL: Çift kontrol ekledik + 2000 frame limiti kaldırıldı
            while not stop_event.is_set() and not shutdown_flag.is_set():
                ret, frame = videoCapture.read()
                # GRACEFUL: Shutdown kontrolü eklendi
                if not ret or shutdown_flag.is_set():
                    try:
                        read_queue.put(None, timeout=1) # Kuyruk tıkalı kalmasın diye 1 saniye timeout süresi atanır.
                    except queue.Full:
                        pass
                    break
                
                # GRACEFUL: Shutdown kontrolü eklenmiş döngü
                while not stop_event.is_set() and not shutdown_flag.is_set():
                    try:
                        read_queue.put(frame, timeout=1) # frame bilgisi 1 saniye içinde yazılamazsa stop_event kontrol edilerek tekrar denenir, proje durdumu bilmek adına.
                        with _total_lock:
                            _total_frames_count += 1
                        break  # frame kuyruğa başarıyla konduysa döngüyü terk et
                    except queue.Full:
                        continue  # Kuyruk dolu, stop_event kontrolüyle tekrar dene
 
        # 5. Tahminci Thread
        def predictor():
            global _processed_frames_count
            buffer = []
            # GRACEFUL: Çift kontrol ekledik
            while not stop_event.is_set() and not shutdown_flag.is_set():
                try:
                    # GRACEFUL: Timeout ekledik blocking'i önlemek için
                    frame = read_queue.get(timeout=0.5)  # get_nowait yerine timeout
                    
                    # GRACEFUL: Shutdown kontrolü eklendi
                    if frame is None or shutdown_flag.is_set():
                        # Kalan buffer'ı işle (graceful cleanup)
                        if buffer and not shutdown_flag.is_set():
                            process_batch(buffer) #Kuyruk kapanmadan önce son birikmiş kare varsa, onları da GPU'da işleyip işlemeyi tamamla.
                            with _processed_lock:
                                _processed_frames_count += len(buffer)
                        try:
                            write_queue.put(None, timeout=1)
                        except queue.Full:
                            pass
                        break # Bu da "tahminler bitti" sinyalidir. Bu iş parçacığı işi bitirmiştir ve çıkış yapar. write_queue'ya None atılarak yazıcı (writer) thread bilgilendirilir.

                    buffer.append(frame)
                    if len(buffer) == BATCH_SIZE: #Eğer buffer içindeki kare sayısı BATCH_SIZE kadar olduysa:
                        # GRACEFUL: Shutdown kontrolü eklendi
                        if not shutdown_flag.is_set():
                            process_batch(buffer) # process_batch() fonksiyonu çağrılır. Bu GPU ile toplu tahmin yapan fonksiyondur.
                            with _processed_lock:
                                _processed_frames_count += len(buffer)
                        buffer = []  # Ardından buffer sıfırlanır ve yeni kareleri beklemeye devam eder.
                
                except queue.Empty:
                    # GRACEFUL: Timeout durumunda shutdown kontrolü
                    if shutdown_flag.is_set():
                        break
                    continue  # Normal devam et

        def process_batch(frames):
            # GRACEFUL: İşlem öncesi shutdown kontrolü
            if shutdown_flag.is_set():
                return
                
            try:
                processed_frames = []

                for frame in frames:
                    # GRACEFUL: Her frame'de shutdown kontrolü
                    if shutdown_flag.is_set():
                        return
                    processed_frames.append(frame)
        
                # GRACEFUL: GPU işlemi öncesi kontrol
                if not shutdown_flag.is_set():
                    # TRACKING: predict yerine track kullan
                    results = net.track(
                        processed_frames, 
                        device=device, 
                        conf=score_threshold, 
                        imgsz=640, 
                        verbose=True,
                        persist=True,  # Track ID'leri kareler arasında koru
                        tracker=str(tracker_yaml_path)  # ByteTrack algoritması
                    )

                    for i, result in enumerate(results):
                        # GRACEFUL: Her sonuç için shutdown kontrolü
                        if shutdown_flag.is_set():
                            return
                        
                        # TRACKING: Track ID'leri işle
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.cpu().numpy()
                            for box in boxes:
                                class_id = int(box.cls)
                                
                                # Track ID varsa kontrol et
                                if hasattr(box, 'id') and box.id is not None:
                                    track_id = int(box.id)
                                    track_key = f"{class_id}_{track_id}"
                                    
                                    # Daha önce görülmemiş nesne
                                    with _track_lock:
                                        if track_key not in _unique_track_ids:
                                            _unique_track_ids.add(track_key)
                                            
                                            # Sınıf bazlı sayaç güncelle
                                            with _class_lock:
                                                if 0 <= class_id < len(CLASSES):
                                                    class_name = CLASSES[class_id]
                                                    _class_counts[class_name] += 1
                                                    
                                                    # Counter'ı güncelle (ana thread'e bırakıyoruz)
                                                    # Bu satır işlevsizdir, sadece gösterim amacıyla koydum
                                                    # counter_placeholders[class_name].metric(
                                                    #    f"{class_name}", 
                                                    #    f"{_class_counts[class_name]}"
                                                    # )
                        
                        annotated = result.plot()
                        try:
                            write_queue.put(annotated, timeout=1)
                        except queue.Full:
                            if shutdown_flag.is_set():
                                return
                            pass  # Queue doluysa frame'i atla
        
            except Exception as e:
                print(f"Batch işleme hatası: {e}")
        
        # 6. Yazıcı Thread
        def writer_thread():
            # GRACEFUL: Çift kontrol ekledik
            while not stop_event.is_set() and not shutdown_flag.is_set():
                try:
                    # GRACEFUL: Timeout ekledik
                    frame = write_queue.get(timeout=0.5)  # get_nowait yerine timeout
                    # GRACEFUL: Shutdown kontrolü eklendi
                    if frame is None or shutdown_flag.is_set():
                        break
                    cv2writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except queue.Empty:
                    # GRACEFUL: Timeout durumunda shutdown kontrolü
                    if shutdown_flag.is_set():
                        break
                    continue  # Normal devam et

        # 7. Thread'leri başlat
        threads = [
            threading.Thread(target=reader, name="Reader"),
            threading.Thread(target=predictor, name="Predictor"),
            threading.Thread(target=writer_thread, name="Writer")
        ]

        start_time = time.time()

        for t in threads:
            # GRACEFUL: Daemon kaldırıldı, kontrollü kapanma için
            t.start()

        # Ana thread, ilerlemeyi takip edip progress bar'ı günceller
        try:
            # GRACEFUL: Shutdown kontrolü eklendi + 2000 frame limiti kaldırıldı
            last_counter_update = time.time()
            
            while any(t.is_alive() for t in threads) and not shutdown_flag.is_set():
                with _total_lock:
                    total = _total_frames_count
                with _processed_lock:
                    done = _processed_frames_count
                if total > 0:
                    inferenceBar.progress(min(done / _total_frames, 1.0), text=f"Processed frames: {done} / {total}  %{(done / _total_frames*100):.2f}  Total:{_total_frames} read_queue:{read_queue.qsize()}  write_queue:{write_queue.qsize()} {(time.time() - start_time):.1f}")
                
                # TRACKING: Track istatistiklerini göster (her 0.5 saniyede bir)
                current_time = time.time()
                if current_time - last_counter_update >= 0.5:
                    with _track_lock:
                        track_count = len(_unique_track_ids)
                    
                    track_stats.info(f"🔍 Benzersiz Takip Edilen Nesneler: {track_count}")
                    
                    # Sınıf sayaçlarını güncelle
                    with _class_lock:
                        for class_name in CLASSES:
                            counter_placeholders[class_name].metric(
                                f"{class_name}", 
                                f"{_class_counts[class_name]}"
                            )
                    
                    last_counter_update = current_time
                
                # 2000 frame'de batch size ve süre bilgisi yazdır (sadece bir kez)
                if done >= 2000 and not milestone_2000_logged:
                    elapsed_time = time.time() - start_time
                    print(f"📊 2000 Frame Milestone - BATCH_SIZE: {BATCH_SIZE}, Süre: {elapsed_time:.2f} saniye")
                    milestone_2000_logged = True
                
                time.sleep(0.2)
            
            # Son güncelleme %100 için
            with _total_lock:
                total = _total_frames_count
            with _processed_lock:
                done = _processed_frames_count
            if total > 0:
                inferenceBar.progress(min(done / _total_frames, 1.0), text=f"Processed frames: {done} / {total}  %{(done / _total_frames*100):.2f}  Total:{_total_frames} read_queue:{read_queue.qsize()}  write_queue:{write_queue.qsize()} {(time.time() - start_time):.1f}")

        except KeyboardInterrupt:
            # GRACEFUL: Koordineli shutdown başlat
            print("🛑 Graceful shutdown başlatılıyor...")
            shutdown_flag.set()  # Tüm thread'lere dur sinyali gönder
            stop_event.set()
            
            # GRACEFUL: Queue'lara son sinyaller gönder
            try:
                read_queue.put(None, timeout=1)
                write_queue.put(None, timeout=1)
            except:
                pass  # Queue dolu olabilir, sorun değil
                
        finally:
            # GRACEFUL: Thread'lerin temiz kapanmasını bekle
            print("Thread'ların kapanması bekleniyor...")
            for t in threads:
                t.join(timeout=3.0)  # Her thread için 3 saniye bekle
                if t.is_alive():
                    print(f"⚠️ {t.name} hala çalışıyor")
            
            # Kaynakları temizle
            videoCapture.release()
            cv2writer.release()

            end_time = time.time()
            
            # TRACKING: Final stats
            with _track_lock:
                final_track_count = len(_unique_track_ids)
            
            with _class_lock:
                final_class_counts = dict(_class_counts)
            
            # GRACEFUL: Durum mesajları
            if shutdown_flag.is_set():
                print(f"⚠️ Video işleme kullanıcı tarafından durduruldu. BATCH_SIZE: {BATCH_SIZE} Süre: {end_time - start_time:.2f} saniye")
                st.warning("Video işleme durduruldu!")
                inferenceBar.progress(0, text="İşlem durduruldu")
            else:
                print(f"✅ Video işleme tamamlandı. BATCH_SIZE: {BATCH_SIZE} Süre: {end_time - start_time:.2f} saniye")
                st.success(f"Video Processed! 🎯 Benzersiz tespit edilen nesneler: {final_track_count}")
                
                # TRACKING: Final sonuçları göster
                st.write("### 🔍 Tespit Sonuçları")
                st.markdown(f"""
                * **Takip Algoritması:** ByteTrack
                * **Benzersiz Nesneler:** {final_track_count}
                * **İşlem Süresi:** {end_time - start_time:.2f} saniye
                
                **Sınıf Dağılımı:**
                """)
                
                # Sınıf bazlı sonuçları göster
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

col1, col2 = st.columns(2)
with col1:
    # Also rerun the appplication after download
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
        # Rerun the application
        st.rerun()

st.title("Road Damage Detection with Tracking")
st.write("Video içindeki yol hasarlarını tespit edip takip eden sistem. Her hasar benzersiz olarak sayılır.")

video_file = st.file_uploader("Upload Video", type=".mp4", disabled=st.session_state.runningInference,)
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction. You can change the threshold before running the inference.")

if video_file is not None:
    if st.button('Process Video with Tracking', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = "Processing Video " + video_file.name + " with object tracking"
        st.warning(_warning)
        processVideo(video_file, score_threshold)