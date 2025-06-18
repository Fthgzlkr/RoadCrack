import os
import glob
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def count_yolo_labels(dataset_path):
    """
    YOLO formatındaki label dosyalarını analiz eder
    """
    # Sınıf isimleri (YAML'daki sıraya göre)
    class_names = {
        0: "Longitudinal Crack (D00)",
        1: "Transverse Crack (D10)", 
        2: "Alligator Crack (D20)",
        3: "Pothole (D40)",
        4: "Crosswalk Blur (D43)",
        5: "White Line (D44)",
        6: "Utility Hole (D50)",
        7: "Repair"
    }
    
    # Sayaçlar
    train_counter = Counter()
    val_counter = Counter()
    total_counter = Counter()
    
    # Dosya sayıları
    train_files = 0
    val_files = 0
    train_empty = 0
    val_empty = 0
    
    # Train labellarını say
    train_labels_path = os.path.join(dataset_path, "labels", "train")
    if os.path.exists(train_labels_path):
        label_files = glob.glob(os.path.join(train_labels_path, "*.txt"))
        train_files = len(label_files)
        
        print(f"🔍 Train klasöründe {train_files} label dosyası bulundu...")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    
                if not lines or lines == ['']:
                    train_empty += 1
                    continue
                    
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        train_counter[class_id] += 1
                        total_counter[class_id] += 1
                        
            except Exception as e:
                print(f"⚠️ Hata: {label_file} - {e}")
    
    # Val labellarını say
    val_labels_path = os.path.join(dataset_path, "labels", "val")
    if os.path.exists(val_labels_path):
        label_files = glob.glob(os.path.join(val_labels_path, "*.txt"))
        val_files = len(label_files)
        
        print(f"🔍 Val klasöründe {val_files} label dosyası bulundu...")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    
                if not lines or lines == ['']:
                    val_empty += 1
                    continue
                    
                for line in lines:
                    if line.strip():
                        class_id = int(line.split()[0])
                        val_counter[class_id] += 1
                        total_counter[class_id] += 1
                        
            except Exception as e:
                print(f"⚠️ Hata: {label_file} - {e}")
    
    return {
        'train_counter': train_counter,
        'val_counter': val_counter,
        'total_counter': total_counter,
        'class_names': class_names,
        'stats': {
            'train_files': train_files,
            'val_files': val_files,
            'train_empty': train_empty,
            'val_empty': val_empty
        }
    }

def print_detailed_report(results):
    """
    Detaylı rapor yazdırır
    """
    train_counter = results['train_counter']
    val_counter = results['val_counter']
    total_counter = results['total_counter']
    class_names = results['class_names']
    stats = results['stats']
    
    print("\n" + "="*80)
    print("📊 YOLO DATASET LABEL ANALİZİ")
    print("="*80)
    
    # Genel istatistikler
    print(f"\n📁 DOSYA İSTATİSTİKLERİ:")
    print(f"   🟢 Train dosyaları: {stats['train_files']:,}")
    print(f"   🔵 Val dosyaları: {stats['val_files']:,}")
    print(f"   📄 Toplam dosya: {stats['train_files'] + stats['val_files']:,}")
    print(f"   📋 Boş train dosyaları: {stats['train_empty']:,}")
    print(f"   📋 Boş val dosyaları: {stats['val_empty']:,}")
    
    # Nesne sayıları
    total_train_objects = sum(train_counter.values())
    total_val_objects = sum(val_counter.values())
    total_objects = sum(total_counter.values())
    
    print(f"\n🎯 NESNE İSTATİSTİKLERİ:")
    print(f"   🟢 Train nesneleri: {total_train_objects:,}")
    print(f"   🔵 Val nesneleri: {total_val_objects:,}")
    print(f"   📊 Toplam nesne: {total_objects:,}")
    
    # Sınıf bazında detay
    print(f"\n🏷️ SINIF BAZINDA DETAY:")
    print(f"{'ID':<3} {'Sınıf Adı':<25} {'Train':<8} {'Val':<8} {'Toplam':<8} {'Oran':<6}")
    print("-" * 65)
    
    for class_id in sorted(class_names.keys()):
        name = class_names[class_id]
        train_count = train_counter.get(class_id, 0)
        val_count = val_counter.get(class_id, 0)
        total_count = total_counter.get(class_id, 0)
        percentage = (total_count / total_objects * 100) if total_objects > 0 else 0
        
        print(f"{class_id:<3} {name:<25} {train_count:<8,} {val_count:<8,} {total_count:<8,} {percentage:<5.1f}%")
    
    # Eksik sınıflar
    print(f"\n⚠️ EKSİK SINIFLAR:")
    missing_classes = []
    for class_id in class_names.keys():
        if total_counter.get(class_id, 0) == 0:
            missing_classes.append(f"{class_id}: {class_names[class_id]}")
    
    if missing_classes:
        for missing in missing_classes:
            print(f"   ❌ {missing}")
    else:
        print("   ✅ Tüm sınıflar mevcut!")
    
    # Dataset dengesi
    print(f"\n⚖️ DATASET DENGESİ ANALİZİ:")
    if total_objects > 0:
        max_count = max(total_counter.values()) if total_counter else 0
        min_count = min(total_counter.values()) if total_counter else 0
        avg_count = total_objects / len(class_names)
        
        print(f"   📈 En çok: {max_count:,} nesne")
        print(f"   📉 En az: {min_count:,} nesne")
        print(f"   📊 Ortalama: {avg_count:.0f} nesne")
        print(f"   ⚡ Dengesizlik oranı: {max_count/min_count:.1f}x" if min_count > 0 else "   ⚡ Dengesizlik oranı: ∞")

def plot_class_distribution(results, save_path=None):
    """
    Sınıf dağılımını görselleştirir
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        total_counter = results['total_counter']
        class_names = results['class_names']
        
        # Veri hazırlama
        classes = []
        counts = []
        
        for class_id in sorted(class_names.keys()):
            classes.append(f"{class_id}: {class_names[class_id].split('(')[0].strip()}")
            counts.append(total_counter.get(class_id, 0))
        
        # Grafik oluşturma
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(2, 1, 1)
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Sınıf Bazında Nesne Sayıları', fontsize=16, fontweight='bold')
        plt.xlabel('Sınıflar', fontsize=12)
        plt.ylabel('Nesne Sayısı', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Değerleri çubukların üzerine yaz
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(2, 1, 2)
        non_zero_classes = [cls for cls, cnt in zip(classes, counts) if cnt > 0]
        non_zero_counts = [cnt for cnt in counts if cnt > 0]
        
        if non_zero_counts:
            plt.pie(non_zero_counts, labels=non_zero_classes, autopct='%1.1f%%', startangle=90)
            plt.title('Sınıf Dağılım Oranları', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Grafik kaydedildi: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠️ Matplotlib/Seaborn bulunamadı. Grafik oluşturulamadı.")
        print("Kurulum için: pip install matplotlib seaborn")

def main():
    """
    Ana fonksiyon
    """
    # Dataset yolu
    dataset_path = r"C:\Users\fthgz\OneDrive\Belgeler\RoadDamageDetection-main\RDD2022_SPLIT"
    
    print("🚀 YOLO Label Analizi Başlatılıyor...")
    
    # Analiz yap
    results = count_yolo_labels(dataset_path)
    
    # Rapor yazdır
    print_detailed_report(results)
    
    # Grafik oluştur (opsiyonel)
    try:
        plot_class_distribution(results, save_path="class_distribution.png")
    except Exception as e:
        print(f"⚠️ Grafik oluşturulamadı: {e}")
    
    print("\n🎉 Analiz tamamlandı!")

if __name__ == "__main__":
    main()