import cv2
import numpy as np
import os
import time
from pathlib import Path
import argparse

class FastROIProcessor:
    def __init__(self, roi_top=0.5, roi_bottom=1.0, roi_left=0.1, roi_right=0.7):
        """
        Maksimum hızda ROI işlemci
        Sadece gerekli işlemleri yapar - önizleme yok
        """
        self.roi_top = roi_top
        self.roi_bottom = roi_bottom
        self.roi_left = roi_left
        self.roi_right = roi_right
        
        print(f"⚡ Hızlı ROI İşlemci")
        print(f"   ROI: %{roi_top*100:.0f}-%{roi_bottom*100:.0f} x %{roi_left*100:.0f}-%{roi_right*100:.0f}")
    
    def get_roi_coordinates(self, height, width):
        """ROI koordinatlarını hesapla - optimize edilmiş"""
        y1 = int(height * self.roi_top)
        y2 = int(height * self.roi_bottom)
        x1 = int(width * self.roi_left)
        x2 = int(width * self.roi_right)
        return x1, y1, x2, y2
    
    def process_video_fast(self, input_path, output_path, mode='roi_only'):
        """
        Maksimum hızda video işleme
        mode: 'roi_only' (sadece ROI) veya 'full_frame' (tam frame)
        """
        
        # Video açma
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {input_path}")
            return None
        
        # Video özellikleri - tek seferde al
        props = [
            cv2.CAP_PROP_FPS,
            cv2.CAP_PROP_FRAME_WIDTH, 
            cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FRAME_COUNT
        ]
        fps, width, height, total_frames = [int(cap.get(prop)) for prop in props]
        
        print(f"\n📹 Video: {width}x{height}, {fps} FPS, {total_frames:,} frame")
        print(f"⏱️  Orijinal süre: {total_frames/fps:.1f} saniye")
        
        # ROI koordinatları - bir kez hesapla
        x1, y1, x2, y2 = self.get_roi_coordinates(height, width)
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        print(f"📏 ROI: {roi_width}x{roi_height} piksel")
        
        # Çıktı boyutunu belirle
        if mode == 'roi_only':
            output_width, output_height = roi_width, roi_height
            print(f"💾 Çıktı modu: Sadece ROI ({output_width}x{output_height})")
        else:
            output_width, output_height = width, height
            print(f"💾 Çıktı modu: Tam frame ({output_width}x{output_height})")
        
        # Video yazıcı
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            print(f"❌ Çıktı video oluşturulamadı: {output_path}")
            cap.release()
            return None
        
        # Performans değişkenleri
        frame_count = 0
        start_time = time.time()
        last_update = start_time
        
        print(f"\n🚀 İşlem başlıyor...")
        print(f"⏰ Başlangıç: {time.strftime('%H:%M:%S')}")
        print("=" * 80)
        
        # Ana işlem döngüsü - maksimum hız için optimize edilmiş
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # ROI işlemi
            if mode == 'roi_only':
                # Sadece ROI alanını kırp - en hızlı
                roi_frame = frame[y1:y2, x1:x2]
                out.write(roi_frame)
            else:
                # Tam frame ile ROI mask - biraz daha yavaş
                frame[0:y1, :] = 0          # Üst kısmı siyah
                frame[y2:height, :] = 0     # Alt kısmı siyah  
                frame[:, 0:x1] = 0          # Sol kısmı siyah
                frame[:, x2:width] = 0      # Sağ kısmı siyah
                out.write(frame)
            
            # Performans raporu (her 5 saniyede bir)
            current_time = time.time()
            if current_time - last_update >= 5.0:
                elapsed = current_time - start_time
                progress = (frame_count / total_frames) * 100
                fps_actual = frame_count / elapsed
                
                # ETA hesaplama
                remaining_frames = total_frames - frame_count
                eta_seconds = remaining_frames / fps_actual if fps_actual > 0 else 0
                eta_minutes = eta_seconds / 60
                
                # Hız oranı
                speed_ratio = fps_actual / fps
                
                print(f"📊 İlerleme: {progress:6.2f}% | "
                      f"Frame: {frame_count:7,}/{total_frames:,} | "
                      f"Hız: {fps_actual:6.1f} FPS ({speed_ratio:.2f}x) | "
                      f"Kalan: {eta_minutes:5.1f}dk")
                
                last_update = current_time
        
        # Temizlik
        cap.release()
        out.release()
        
        # Final raporu
        total_time = time.time() - start_time
        final_fps = frame_count / total_time
        speed_ratio = final_fps / fps
        
        print("=" * 80)
        print(f"✅ İşlem Tamamlandı!")
        print(f"📊 Final Performans:")
        print(f"   Toplam süre: {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
        print(f"   İşlenen frame: {frame_count:,}")
        print(f"   Ortalama hız: {final_fps:.1f} FPS")
        print(f"   Hız oranı: {speed_ratio:.2f}x ({'hızlı' if speed_ratio > 1 else 'yavaş'})")
        print(f"   Frame/saniye: {frame_count/total_time:.1f}")
        
        # Dosya bilgileri
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024*1024)
            original_size_mb = file_size_mb / speed_ratio  # Tahmini
            
            print(f"💾 Dosya Bilgileri:")
            print(f"   Çıktı: {output_path}")
            print(f"   Boyut: {file_size_mb:.1f} MB")
            print(f"   Boyut azalma: {((output_width*output_height)/(width*height))*100:.1f}%")
        
        return {
            'total_time': total_time,
            'frames_processed': frame_count,
            'fps': final_fps,
            'speed_ratio': speed_ratio,
            'file_size_mb': file_size_mb if 'file_size_mb' in locals() else 0
        }

def main():
    parser = argparse.ArgumentParser(description='Hızlı ROI Video İşlemci')
    parser.add_argument('--input', '-i', required=True, help='Giriş video dosyası')
    parser.add_argument('--output', '-o', help='Çıkış video dosyası')
    parser.add_argument('--mode', '-m', choices=['roi_only', 'full_frame'], 
                       default='roi_only', help='İşlem modu')
    
    # ROI parametreleri
    parser.add_argument('--roi-top', type=float, default=0.5, help='Üst sınır (0.5)')
    parser.add_argument('--roi-bottom', type=float, default=1.0, help='Alt sınır (1.0)')
    parser.add_argument('--roi-left', type=float, default=0.1, help='Sol sınır (0.1)')
    parser.add_argument('--roi-right', type=float, default=0.7, help='Sağ sınır (0.7)')
    
    # Performans seçenekleri
    parser.add_argument('--benchmark', action='store_true', help='Benchmark modu - her iki modu test et')
    
    args = parser.parse_args()
    
    # Dosya kontrolü
    if not os.path.exists(args.input):
        print(f"❌ Dosya bulunamadı: {args.input}")
        return
    
    # Çıktı dosyası
    input_path = Path(args.input)
    if not args.output:
        suffix = "_roi" if args.mode == 'roi_only' else "_masked"
        args.output = str(input_path.parent / f"{input_path.stem}{suffix}.mp4")
    
    print(f"🎬 Giriş: {args.input}")
    print(f"💾 Çıktı: {args.output}")
    print(f"🎯 Mod: {args.mode}")
    
    # İşlemci oluştur
    processor = FastROIProcessor(
        roi_top=args.roi_top,
        roi_bottom=args.roi_bottom,
        roi_left=args.roi_left,
        roi_right=args.roi_right
    )
    
    if args.benchmark:
        # Benchmark modu - her iki modu test et
        print("\n🏃‍♂️ BENCHMARK MODU")
        
        # ROI only test
        output_roi = str(input_path.parent / f"{input_path.stem}_benchmark_roi.mp4")
        print(f"\n1️⃣ ROI Only modu test ediliyor...")
        result_roi = processor.process_video_fast(args.input, output_roi, 'roi_only')
        
        # Full frame test  
        output_full = str(input_path.parent / f"{input_path.stem}_benchmark_full.mp4")
        print(f"\n2️⃣ Full Frame modu test ediliyor...")
        result_full = processor.process_video_fast(args.input, output_full, 'full_frame')
        
        # Karşılaştırma
        print(f"\n📈 BENCHMARK SONUÇLARI:")
        print(f"{'Mod':<12} {'Süre':<8} {'FPS':<8} {'Hız':<8} {'Dosya':<10}")
        print("-" * 50)
        print(f"{'ROI Only':<12} {result_roi['total_time']:<8.1f} {result_roi['fps']:<8.1f} "
              f"{result_roi['speed_ratio']:<8.2f}x {result_roi['file_size_mb']:<10.1f}MB")
        print(f"{'Full Frame':<12} {result_full['total_time']:<8.1f} {result_full['fps']:<8.1f} "
              f"{result_full['speed_ratio']:<8.2f}x {result_full['file_size_mb']:<10.1f}MB")
        
        speed_diff = result_roi['speed_ratio'] / result_full['speed_ratio']
        print(f"\n🚀 ROI Only, Full Frame'den {speed_diff:.1f}x daha hızlı!")
        
    else:
        # Normal işlem
        result = processor.process_video_fast(args.input, args.output, args.mode)
        
        if result:
            print(f"\n🎉 Başarıyla tamamlandı!")
            if result['speed_ratio'] > 1:
                print(f"⚡ Gerçek zamandan {result['speed_ratio']:.1f}x daha hızlı işlendi!")
            else:
                print(f"🐌 Gerçek zamandan {1/result['speed_ratio']:.1f}x daha yavaş işlendi")

if __name__ == "__main__":
    # Kullanım örnekleri:
    
    # En hızlı (sadece ROI):
    # python fast_roi.py --input "video.mp4"
    
    # Tam frame (ROI dışı siyah):
    # python fast_roi.py --input "video.mp4" --mode full_frame
    
    # Benchmark (her iki modu test et):
    # python fast_roi.py --input "video.mp4" --benchmark
    
    # Özel ROI:
    # python fast_roi.py --input "video.mp4" --roi-top 0.4 --roi-right 0.8
    
    main()