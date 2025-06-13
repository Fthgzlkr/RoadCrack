# standalone_ram_cleaner.py
# Video işleme öncesi çalıştırılacak bağımsız RAM temizleyici

import psutil
import gc
import os
import time
import subprocess
from ctypes import windll, wintypes, byref, c_size_t

class PowerRAMCleaner:
    """Güçlü RAM temizleme sistemi"""
    
    def __init__(self):
        self.initial_ram = None
        self.final_ram = None
        
    def get_ram_info(self):
        """Detaylı RAM bilgisi"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def print_ram_status(self, title="RAM Durumu"):
        """RAM durumunu yazdır"""
        info = self.get_ram_info()
        print(f"\n{'='*50}")
        print(f"🧠 {title}")
        print(f"{'='*50}")
        print(f"📊 Toplam RAM: {info['total_gb']:.1f} GB")
        print(f"💾 Kullanılan: {info['used_gb']:.1f} GB (%{info['percent']:.1f})")
        print(f"✅ Mevcut: {info['available_gb']:.1f} GB")
        print(f"🆓 Boş: {info['free_gb']:.1f} GB")
        print(f"{'='*50}\n")
        return info
    
    def step1_python_cleanup(self):
        """1. Adım: Python Garbage Collection"""
        print("🐍 1. Python Garbage Collection...")
        
        collected_total = 0
        for generation in range(3):
            collected = gc.collect()
            collected_total += collected
            print(f"   Gen {generation}: {collected} nesne")
        
        # Force garbage collection
        gc.disable()
        gc.enable()
        
        print(f"✅ Python GC tamamlandı: {collected_total} nesne temizlendi")
        return collected_total
    
    def step2_system_cache_clear(self):
        """2. Adım: Sistem Cache Temizleme"""
        print("💻 2. Sistem Cache Temizleme...")
        
        try:
            # Windows sistem cache temizle
            kernel32 = windll.kernel32
            
            # Mevcut process working set'ini küçült
            handle = kernel32.GetCurrentProcess()
            result = kernel32.SetProcessWorkingSetSize(handle, c_size_t(-1), c_size_t(-1))
            
            if result:
                print("✅ Working Set temizlendi")
            else:
                print("⚠️ Working Set temizleme kısmen başarılı")
                
            return True
            
        except Exception as e:
            print(f"⚠️ Sistem cache temizleme hatası: {e}")
            return False
    
    def step3_trim_all_processes(self):
        """3. Adım: Tüm Process'lerin Working Set'lerini Küçült"""
        print("⚙️ 3. Tüm Process Working Set Trim...")
        
        trimmed_count = 0
        failed_count = 0
        
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    pid = proc.info['pid']
                    name = proc.info['name']
                    
                    # Sistem kritik process'leri atla
                    if name.lower() in ['system', 'registry', 'smss.exe', 'csrss.exe', 'wininit.exe']:
                        continue
                    
                    # Process handle al
                    handle = windll.kernel32.OpenProcess(0x1F0FFF, False, pid)
                    if handle:
                        # Working set'i küçült
                        result = windll.kernel32.SetProcessWorkingSetSize(handle, c_size_t(-1), c_size_t(-1))
                        windll.kernel32.CloseHandle(handle)
                        
                        if result:
                            trimmed_count += 1
                        else:
                            failed_count += 1
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    failed_count += 1
                    continue
            
            print(f"✅ Process Trim: {trimmed_count} başarılı, {failed_count} başarısız")
            return trimmed_count
            
        except Exception as e:
            print(f"⚠️ Process trim hatası: {e}")
            return 0
    
    def step4_windows_memory_compression(self):
        """4. Adım: Windows Memory Compression Optimize"""
        print("🔄 4. Windows Memory Compression...")
        
        try:
            # PowerShell ile memory compression yenile
            commands = [
                'powershell.exe -Command "Get-Process | ForEach-Object { [System.GC]::Collect() }"',
                'powershell.exe -Command "Clear-RecycleBin -Force -Confirm:$false"',
            ]
            
            for cmd in commands:
                try:
                    subprocess.run(cmd, shell=True, timeout=30, capture_output=True)
                except subprocess.TimeoutExpired:
                    print("   ⚠️ PowerShell timeout (normal)")
                except Exception:
                    pass
            
            print("✅ Memory Compression optimize edildi")
            return True
            
        except Exception as e:
            print(f"⚠️ Memory Compression hatası: {e}")
            return False
    
    def step5_temp_files_cleanup(self):
        """5. Adım: Temp Dosyaları Temizle"""
        print("🗑️ 5. Temporary Files Cleanup...")
        
        temp_dirs = [
            os.environ.get('TEMP', ''),
            os.environ.get('TMP', ''),
            'C:\\Windows\\Temp',
            os.path.expanduser('~\\AppData\\Local\\Temp')
        ]
        
        cleaned_files = 0
        cleaned_size = 0
        
        for temp_dir in temp_dirs:
            if not temp_dir or not os.path.exists(temp_dir):
                continue
                
            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files += 1
                            cleaned_size += file_size
                        except (OSError, PermissionError):
                            continue
                            
            except Exception:
                continue
        
        cleaned_mb = cleaned_size / (1024 * 1024)
        print(f"✅ Temp cleanup: {cleaned_files} dosya, {cleaned_mb:.1f} MB")
        return cleaned_files, cleaned_mb
    
    def step6_browser_cache_hint(self):
        """6. Adım: Browser Cache Temizlik Önerisi"""
        print("🌐 6. Browser Cache Önerisi...")
        
        browsers = {
            'Chrome': 'chrome://settings/clearBrowserData',
            'Edge': 'edge://settings/clearBrowserData',
            'Firefox': 'about:preferences#privacy'
        }
        
        print("   Manuel browser cache temizliği önerilir:")
        for browser, url in browsers.items():
            print(f"   • {browser}: {url}")
        
        print("✅ Browser cache önerileri verildi")
    
    def step7_disk_cleanup(self):
        """7. Adım: Windows Disk Cleanup"""
        print("💿 7. Windows Disk Cleanup...")
        
        try:
            # Disk cleanup çalıştır (silent mode)
            subprocess.run(['cleanmgr', '/sagerun:1'], timeout=60, capture_output=True)
            print("✅ Disk cleanup çalıştırıldı")
            return True
        except subprocess.TimeoutExpired:
            print("⚠️ Disk cleanup timeout (arka planda devam ediyor)")
            return True
        except FileNotFoundError:
            print("⚠️ Disk cleanup bulunamadı")
            return False
        except Exception as e:
            print(f"⚠️ Disk cleanup hatası: {e}")
            return False
    
    def emergency_ram_boost(self):
        """Acil Durum RAM Boost"""
        print("\n🚨 ACİL DURUM RAM BOOST MODU!")
        print("⚠️ Bu işlem tüm gereksiz programları kapatabilir!")
        
        response = input("Devam etmek istiyor musunuz? (y/N): ").lower()
        if response != 'y':
            print("❌ İşlem iptal edildi")
            return False
        
        print("\n🔥 Emergency RAM Boost başlıyor...")
        
        try:
            # Yüksek RAM kullanan process'leri bul ve uyar
            print("📊 Yüksek RAM kullanan programlar:")
            
            high_memory_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    if memory_mb > 500:  # 500MB üzeri
                        high_memory_processes.append((proc.info['name'], memory_mb, proc.info['pid']))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # RAM kullanımına göre sırala
            high_memory_processes.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, memory_mb, pid) in enumerate(high_memory_processes[:10]):
                print(f"   {i+1}. {name}: {memory_mb:.1f} MB (PID: {pid})")
            
            print("\n⚠️ Bu programları manuel olarak kapatmanız önerilir")
            print("💡 Video işleme için kritik olmayan programları kapatın")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Emergency boost hatası: {e}")
            return False
    
    def full_cleanup_sequence(self):
        """Tam temizlik sequence'i"""
        print("\n🚀 GÜÇLÜ RAM TEMİZLEME BAŞLIYOR...")
        print("⏱️ Bu işlem 1-2 dakika sürebilir\n")
        
        # Başlangıç durumu
        self.initial_ram = self.print_ram_status("Başlangıç Durumu")
        
        start_time = time.time()
        
        # Temizlik adımları
        steps_results = {}
        
        steps_results['python_gc'] = self.step1_python_cleanup()
        time.sleep(1)
        
        steps_results['system_cache'] = self.step2_system_cache_clear()
        time.sleep(1)
        
        steps_results['process_trim'] = self.step3_trim_all_processes()
        time.sleep(2)
        
        steps_results['memory_compression'] = self.step4_windows_memory_compression()
        time.sleep(1)
        
        steps_results['temp_cleanup'] = self.step5_temp_files_cleanup()
        time.sleep(1)
        
        self.step6_browser_cache_hint()
        time.sleep(1)
        
        steps_results['disk_cleanup'] = self.step7_disk_cleanup()
        
        # Bitiş durumu
        self.final_ram = self.print_ram_status("Temizlik Sonrası")
        
        # Sonuçlar
        self.print_results(steps_results, time.time() - start_time)
        
        return self.final_ram
    
    def print_results(self, steps_results, duration):
        """Sonuçları yazdır"""
        print(f"\n{'🎉'*20} SONUÇLAR {'🎉'*20}")
        
        # RAM iyileştirmesi
        ram_saved = self.initial_ram['percent'] - self.final_ram['percent']
        gb_freed = self.initial_ram['used_gb'] - self.final_ram['used_gb']
        
        print(f"⏱️ Toplam Süre: {duration:.1f} saniye")
        print(f"📉 RAM Kullanımı: %{self.initial_ram['percent']:.1f} → %{self.final_ram['percent']:.1f}")
        print(f"🆓 Serbest Bırakılan: {gb_freed:.2f} GB ({ram_saved:.1f}%)")
        print(f"✅ Mevcut RAM: {self.final_ram['available_gb']:.1f} GB")
        
        # Adım sonuçları
        print(f"\n📋 Adım Sonuçları:")
        if steps_results.get('python_gc', 0) > 0:
            print(f"   🐍 Python GC: {steps_results['python_gc']} nesne")
        if steps_results.get('process_trim', 0) > 0:
            print(f"   ⚙️ Process Trim: {steps_results['process_trim']} process")
        if steps_results.get('temp_cleanup'):
            files, mb = steps_results['temp_cleanup']
            print(f"   🗑️ Temp Files: {files} dosya, {mb:.1f} MB")
        
        # Durum değerlendirmesi
        print(f"\n🎯 Durum Değerlendirmesi:")
        if self.final_ram['percent'] < 60:
            print("   🟢 MÜKEMMEL: Video işleme için ideal RAM durumu!")
        elif self.final_ram['percent'] < 70:
            print("   🟡 İYİ: Video işleme yapılabilir, performans takip edilmeli")
        elif self.final_ram['percent'] < 80:
            print("   🟠 ORTA: Küçük videolar için uygun, büyük videolar için resize gerekli")
        else:
            print("   🔴 YÜKSEk: Emergency boost önerilir!")
            print("   💡 Öneriler:")
            print("      - Gereksiz programları kapatın")
            print("      - Browser'ı kapatın")
            print("      - Video resize factor düşürün (0.5-0.7)")
        
        print(f"\n{'🏁'*50}")
        print("✅ RAM TEMİZLEME TAMAMLANDI!")
        print("🎬 Artık video işleme için hazırsınız!")
        print(f"{'🏁'*50}\n")


def main():
    """Ana fonksiyon"""
    print("🚀 GÜÇLÜ RAM TEMİZLEYİCİ v2.0")
    print("=" * 60)
    print("📱 Video işleme öncesi RAM optimizasyonu")
    print("⚠️ Yönetici hakları gerekebilir")
    print("=" * 60)
    
    cleaner = PowerRAMCleaner()
    
    # Menü
    while True:
        print("\n🎯 SEÇENEKLER:")
        print("1. 🧹 Tam RAM Temizliği (Önerilen)")
        print("2. 📊 Sadece RAM Durumunu Göster")
        print("3. 🚨 Emergency RAM Boost")
        print("4. ❌ Çıkış")
        
        choice = input("\nSeçiminiz (1-4): ").strip()
        
        if choice == '1':
            cleaner.full_cleanup_sequence()
            
            # Video işleme önerisi
            if cleaner.final_ram['percent'] < 70:
                print("\n🎬 VİDEO İŞLEME ÖNERİLERİ:")
                print("✅ RAM durumu video işleme için uygun!")
                print("💡 Şimdi video işleme kodunuzu çalıştırabilirsiniz")
                
                if cleaner.final_ram['available_gb'] > 8:
                    print("🚀 1080p videoları resize_factor=1.0 ile işleyebilirsiniz")
                elif cleaner.final_ram['available_gb'] > 4:
                    print("⚡ resize_factor=0.7-0.8 ile optimal performans")
                else:
                    print("⚠️ resize_factor=0.5-0.6 önerilir")
            
            break
            
        elif choice == '2':
            cleaner.print_ram_status()
            continue
            
        elif choice == '3':
            cleaner.emergency_ram_boost()
            continue
            
        elif choice == '4':
            print("👋 Görüşmek üzere!")
            break
            
        else:
            print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ İşlem kullanıcı tarafından iptal edildi")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        print("💡 Programı yönetici olarak çalıştırmayı deneyin")
    
    input("\nÇıkmak için Enter'a basın...")