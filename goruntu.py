import tkinter as tk #kullanıcı arayüzü
from tkinter import filedialog # dosya açma, kaydetme
from tkinter import simpledialog #veri girişleri almak için 
import cv2 # Görüntü analizi-yapay zeka işlemleri
import numpy as np #bilimsel hesaplamalar için
import matplotlib.pyplot as plt #veri görselleştirme 
from PIL import Image #görüntü dosyalarıyla çalışmayı sağlar

def gri_donustur():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    gri_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gri Donusumu", gri_resim)
    cv2.imshow("Orijinal Resim", resim)
    cv2.waitKey(0) #kullanıcı resim seçene kadar bekle 
    cv2.destroyAllWindows() # pencereyi kapat



def binary_donustur(): #gri tonlamalı hale dönüştürür ve ardından ikili görüntüye çevirir
    dosya_yolu = filedialog.askopenfilename()
    orijinal_resim = cv2.imread(dosya_yolu)
    orijinal_kopya = orijinal_resim.copy()
    gri_resim = cv2.cvtColor(orijinal_resim, cv2.COLOR_BGR2GRAY)
    _, binary_resim = cv2.threshold(gri_resim, 128, 255, cv2.THRESH_BINARY)
    #128in üzerindeki piksel değerleri 255 (beyaz), altındaki piksel değerleri ise 0 (siyah)
    cv2.imshow("Binary Donusumu", binary_resim)
    cv2.imshow("Orijinal Resim", orijinal_kopya)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def goruntu_dondur():
    dondurme_penceresi = tk.Toplevel(root)
    entry_aci = tk.Entry(dondurme_penceresi)
    entry_aci.pack()
    btn_onayla = tk.Button(dondurme_penceresi, text="Resim seç", command=lambda: donme_onayla(entry_aci.get(), dondurme_penceresi))
    btn_onayla.pack()
    dondurme_penceresi.title("aci giriniz")
    dondurme_penceresi.geometry("400x100")



def donme_onayla(aci, pencere):
    aci = int(aci) #Kullanıcıdan alınan açı değeri string olarak gelir. Bu değeri int (tam sayı) formatına dönüştürür.
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    yukseklik, genislik = resim.shape[:2] #Resmin yüksekliği ve genişliği resim.shape ile alındı
    döndürme_matrisi = cv2.getRotationMatrix2D((genislik / 2, yukseklik / 2), aci, 1)
    #Döndürme merkezi,Döndürme açısı, 1=Ölçek faktörü: işlemi sırasında resmin büyütülüp küçültülmeyeceğini belirler.
    dondurulmus_goruntu = cv2.warpAffine(resim, döndürme_matrisi, (genislik, yukseklik))
    cv2.imshow("Orijinal Resim", resim)
    cv2.imshow("Goruntu Dondurme", dondurulmus_goruntu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pencere.destroy()



def goruntu_kirpma():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    img = Image.fromarray(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))
    
    kirpma_orani = simpledialog.askfloat("Kirpma", "Kirpma orani (%):")
    
    genislik, yukseklik = img.size
    kirpma_genislik = int(genislik * kirpma_orani / 100)
    kirpma_yukseklik = int(yukseklik * kirpma_orani / 100)
    
    kismi_goruntu = img.crop((kirpma_genislik, kirpma_yukseklik, genislik - kirpma_genislik, yukseklik - kirpma_yukseklik))
    kismi_goruntu.show()
   




def goruntu_yaklastir(faktor):
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    yeniden_boyutlandirilmis_resim = cv2.resize(resim, None, fx=faktor, fy=faktor)
    cv2.imshow("Orijinal Resim", resim)
    cv2.imshow("Goruntu yaklastirma", yeniden_boyutlandirilmis_resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def goruntu_uzaklastir(faktor):
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    yeniden_boyutlandirilmis_resim = cv2.resize(resim, None, fx=faktor, fy=faktor)
    cv2.imshow("Orijinal resim", resim)
    cv2.imshow("Goruntu uzaklastirma", yeniden_boyutlandirilmis_resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def renk_uzayi_donustur():
    def sec():
        secilen_index = listbox.curselection()
        if secilen_index:
            secilen_uzay.set(renk_uzaylari[secilen_index[0]])
            renk_uzayi_penceresi.destroy()
            dosya_yolu = filedialog.askopenfilename()
            if dosya_yolu:
                resim = cv2.imread(dosya_yolu)
                hedef_renk_uzayi = secilen_uzay.get()
                if hedef_renk_uzayi == 'RGB':
                    donusturulmus_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
                elif hedef_renk_uzayi == 'HSV':
                    donusturulmus_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2HSV)
                elif hedef_renk_uzayi == 'YUV':
                    donusturulmus_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2YUV)
                elif hedef_renk_uzayi == 'LAB':
                    donusturulmus_resim = cv2.cvtColor(resim, cv2.COLOR_BGR2LAB)
                else:
                    donusturulmus_resim = resim
                cv2.imshow("Orijinal Resim", resim)
                cv2.imshow(f"{hedef_renk_uzayi} Renk Uzayi", donusturulmus_resim)
                cv2.waitKey(0)
                cv2.destroyAllWindows()        
    renk_uzayi_penceresi = tk.Tk()
    renk_uzayi_penceresi.title("Renk Uzayi Secimi")
    renk_uzaylari = ['RGB', 'HSV', 'YUV', 'LAB'] 
    secilen_uzay = tk.StringVar(renk_uzayi_penceresi)
    secilen_uzay.set(renk_uzaylari[0])
    listbox = tk.Listbox(renk_uzayi_penceresi)
    for uzay in renk_uzaylari:
        listbox.insert(tk.END, uzay)
    listbox.pack()
    sec_dugmesi = tk.Button(renk_uzayi_penceresi, text="Seç", command=sec)
    sec_dugmesi.pack()



def histogram_genisletme():
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
        resim = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        genisletilmis_resim = cv2.equalizeHist(resim)
        yeni_min = simpledialog.askinteger("Yeni Minimum Değer", "Lütfen yeni minimum değeri girin:")
        yeni_max = simpledialog.askinteger("Yeni Maksimum Değer", "Lütfen yeni maksimum değeri girin:")
        if yeni_min is None or yeni_max is None:
            print("Geçersiz giriş. İşlem iptal edildi.")
            return
        genisletilmis_resim = np.clip(genisletilmis_resim, yeni_min, yeni_max)
        orijinal_hist = cv2.calcHist([resim], [0], None, [256], [0,256])
        #None değeri, tüm görüntüyü kullanmak
        genisletilmis_hist = cv2.calcHist([genisletilmis_resim], [0], None, [256], [0,256])
        plt.plot(orijinal_hist, color='b', label='Orijinal Histogram')
        plt.plot(genisletilmis_hist, color='r', label='Genisletilmis Histogram')
        plt.xlim([0, 256])
        plt.legend()
        cv2.imshow("Histogramı Genişletilmiş Resim", genisletilmis_resim)
        plt.show()



def histogram_germe():
    dosya_yolu = filedialog.askopenfilename()
    if dosya_yolu:
        resim = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)
        min_deger = np.min(resim)
        max_deger = np.max(resim)
        yeni_min = simpledialog.askinteger("Yeni Minimum Değer", f"Lütfen yeni minimum değeri ({min_deger} ile {max_deger} arasında) girin:")
        yeni_max = simpledialog.askinteger("Yeni Maksimum Değer", f"Lütfen yeni maksimum değeri ({min_deger} ile {max_deger} arasında) girin:")
        if yeni_min is None or yeni_max is None:
            print("Geçersiz giriş. İşlem iptal edildi.")
            return
        gerilmis_resim = (((resim - min_deger) / (max_deger - min_deger)) * (yeni_max - yeni_min)) + yeni_min
        gerilmis_resim = np.round(gerilmis_resim).astype(np.uint8)
        orijinal_hist = cv2.calcHist([resim], [0], None, [256], [0,256])
        gerilmis_hist = cv2.calcHist([gerilmis_resim], [0], None, [256], [0,256])
        plt.plot(orijinal_hist, color='b', label='Orijinal Histogram')
        plt.plot(gerilmis_hist, color='r', label='Gerilmis Histogram')
        plt.xlim([0, 256])
        plt.legend()
        plt.show()
        



def resim_sec():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    return resim

def ekleme_islemi():
    resim1 = resim_sec()
    resim2 = resim_sec()
    if resim1.shape != resim2.shape:
        print("Resimlerin boyutları eşit değil.")
        return
    ekleme_sonucu = cv2.add(resim1, resim2)
    cv2.imshow("Ekleme İşlemi", ekleme_sonucu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def bolme_islemi():
    resim1 = resim_sec()
    resim2 = resim_sec()
    if resim1.shape != resim2.shape:
        print("Resimlerin boyutları eşit değil.")
        return
    resim1_float = np.float32(resim1)
    resim2_float = np.float32(resim2)    
    bolme_sonucu = cv2.divide(resim1_float, resim2_float + np.finfo(float).eps)
    bolme_sonucu_uint8 = np.uint8(bolme_sonucu)
    cv2.imshow("Bölme İşlemi", bolme_sonucu_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def kontrast_artirma():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    kanal_b, kanal_g, kanal_r = cv2.split(resim)#kanalları ayırır
    artirilmis_b = cv2.equalizeHist(kanal_b)
    artirilmis_g = cv2.equalizeHist(kanal_g)
    artirilmis_r = cv2.equalizeHist(kanal_r)
    artirilmis_resim = cv2.merge((artirilmis_b, artirilmis_g, artirilmis_r))#kanalları birleştirir
    cv2.imshow("Kontrast Artirma", artirilmis_resim)
    cv2.imshow("Orijinal Resim", resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def konvolusyon_islemi():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    kernel = np.ones((5,5), np.float32) / 25 #5x5 boyutunda ve her elemanı 1/25 
    konvolusyon_sonucu = cv2.filter2D(resim, -1, kernel)
    # -1: Dönüş görüntüsünün derinliği
    cv2.imshow("Konvolusyon islemi", konvolusyon_sonucu)
    cv2.imshow("orjinal Resim",resim )
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def esikleme_islemi():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    griresim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)    
    esik_degeri = 128
    _, esiklenmis_resim = cv2.threshold(griresim, esik_degeri, 255, cv2.THRESH_BINARY)
    cv2.imshow("Esikleme islemi", esiklenmis_resim)
    cv2.imshow("gri Resim",griresim )
    cv2.imshow("orjinal resim",resim )
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def kenar_bulma_prewitt():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)   
    # prewitt ile kenar tesbiti yapıldı
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    #Görüntünün her bir renk kanalı için x ve y yönlerinde Prewitt operatörleri uygulandı
    kenarlar_x_r = cv2.filter2D(resim[:,:,0], -1, kernel_x)  
    kenarlar_y_r = cv2.filter2D(resim[:,:,0], -1, kernel_y)
    kenarlar_x_g = cv2.filter2D(resim[:,:,1], -1, kernel_x)  
    kenarlar_y_g = cv2.filter2D(resim[:,:,1], -1, kernel_y)
    kenarlar_x_b = cv2.filter2D(resim[:,:,2], -1, kernel_x)  
    kenarlar_y_b = cv2.filter2D(resim[:,:,2], -1, kernel_y)
    kenarlar_r = cv2.add(kenarlar_x_r, kenarlar_y_r)
    kenarlar_g = cv2.add(kenarlar_x_g, kenarlar_y_g)
    kenarlar_b = cv2.add(kenarlar_x_b, kenarlar_y_b)
    kenarlar = cv2.add(kenarlar_r, kenarlar_g)
    kenarlar = cv2.add(kenarlar, kenarlar_b)
    cv2.imshow("Kenar Bulma (Prewitt)", kenarlar)
    cv2.imshow("Orijinal", resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def gurultu_ekleme_ve_temizleme():
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)    
    gurultulu_resim = resim.copy()
    gurultu_miktari = 0.05  #piksel sayısının %5'ine kadar gürültü eklemek için
    gurultu_miktarı = int(gurultu_miktari * resim.size)
    gurultu_konumlar = [np.random.randint(0, i-1, gurultu_miktarı) for i in resim.shape]
    gurultulu_resim[gurultu_konumlar[0], gurultu_konumlar[1], gurultu_konumlar[2]] = 255
    gurultulu_resim[gurultu_konumlar[0], gurultu_konumlar[1], gurultu_konumlar[2]] = 0
    temizlenmis_mean = cv2.blur(gurultulu_resim, (3, 3))#(3,3) boyutunda bir çekirdek (kernel) kullanılır
    temizlenmis_median = cv2.medianBlur(gurultulu_resim, 3)
    cv2.imshow("gurultulu Resim", gurultulu_resim)
    cv2.imshow("Mean Filtresi ile Temizlenmis Resim", temizlenmis_mean)
    cv2.imshow("Median Filtresi ile Temizlenmis Resim", temizlenmis_median)
    cv2.imshow("orjinal", resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def unsharp_masking(): #görüntünün keskinliğini artırır
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    unsharp_resim = cv2.filter2D(resim, -1, kernel)
    #-1 parametresi, çıkış görüntüsünün aynı derinlikte
    cv2.imshow("Orijinal Resim", resim)
    cv2.imshow("Unsharp Filtre Uygulanmıs Resim", unsharp_resim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def morfolojik_islemler():  
    root = tk.Tk()
    root.withdraw()  
    dosya_yolu = filedialog.askopenfilename()
    resim = cv2.imread(dosya_yolu)
    kernel_boyutu = 5
    kernel = np.ones((kernel_boyutu, kernel_boyutu), np.uint8)
    root = tk.Tk()
    root.title("Morfolojik İslem Secimi")

    def secilen_islem():
        secilen_index = listbox.curselection()
        if secilen_index:
            secilen = secilen_index[0] + 1
            root.destroy()
            if secilen == 1:
                sonuc = cv2.dilate(resim, kernel, iterations=1)
                islem_adi = "Genisleme"
            elif secilen == 2:
                sonuc = cv2.erode(resim, kernel, iterations=1)
                islem_adi = "Asinma"
            elif secilen == 3:
                sonuc = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel)
                islem_adi = "Acma"
            elif secilen == 4:
                sonuc = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel)
                islem_adi = "Kapama"                
            cv2.imshow(islem_adi, sonuc)
            cv2.imshow("Orijinal Resim", resim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    listbox = tk.Listbox(root, selectmode=tk.SINGLE)
    listbox.insert(1, "Genisleme")
    listbox.insert(2, "Asinma")
    listbox.insert(3, "Acma")
    listbox.insert(4, "Kapama")
    listbox.pack()
    button = tk.Button(root, text="Sec", command=secilen_islem)
    button.pack()


root = tk.Tk()
root.title("Görüntü İşleme Uygulaması")
root.geometry("550x300")


btn_gri = tk.Button(root, text="Gri Dönüşüm", command=gri_donustur, width=20)
btn_binary = tk.Button(root, text="Binary Dönüşüm", command=binary_donustur, width=20)
btn_dondur = tk.Button(root, text="Görüntü Döndürme", command=goruntu_dondur, width=20)
btn_kirp = tk.Button(root, text="Görüntü Kırpma", command=goruntu_kirpma, width=20)
btn_yaklastir = tk.Button(root, text="Yaklaştır", command=lambda: goruntu_yaklastir(1.5), width=20)
btn_uzaklastir = tk.Button(root, text="Uzaklaştır", command=lambda: goruntu_uzaklastir(0.5), width=20)
btn_renk_uzayi = tk.Button(root, text="Renk Uzayı Dönüştürme", command=renk_uzayi_donustur, width=20)
btn_histogram_genisletme = tk.Button(root, text="Histogram Genişletme", command=histogram_genisletme, width=20)
btn_histogram_germe = tk.Button(root, text="Histogram Germe", command=histogram_germe, width=20)
btn_ekleme = tk.Button(root, text="Ekleme İşlemi", command=ekleme_islemi, width=20)
btn_bolme = tk.Button(root, text="Bölme İşlemi", command=bolme_islemi, width=20)
btn_kontrast_artirma = tk.Button(root, text="Kontrast Artırma", command=kontrast_artirma, width=20)
btn_konvolusyon = tk.Button(root, text="Konvolüsyon İşlemi (mean)", command=konvolusyon_islemi, width=20)
btn_esikleme = tk.Button(root, text="Eşikleme İşlemi", command=esikleme_islemi, width=20)
btn_kenar_bulma_prewitt = tk.Button(root, text="Kenar Bulma (Prewitt)", command=kenar_bulma_prewitt, width=20)
btn_gurultu_ekleme_ve_temizleme = tk.Button(root, text="Gürültü Ekle/Temizle", command=gurultu_ekleme_ve_temizleme, width=20)
btn_unsharp_masking = tk.Button(root, text="Unsharp Masking", command=unsharp_masking, width=20)
btn_morfolojik_islemler = tk.Button(root, text="Morfolojik İşlemler", command=morfolojik_islemler, width=20)

for btn in (btn_gri, btn_binary, btn_dondur, btn_kirp, btn_yaklastir, btn_uzaklastir, btn_renk_uzayi, 
    btn_histogram_genisletme, btn_histogram_germe, btn_ekleme, btn_bolme, btn_kontrast_artirma, 
    btn_konvolusyon, btn_esikleme, btn_kenar_bulma_prewitt, btn_gurultu_ekleme_ve_temizleme, 
    btn_unsharp_masking, btn_morfolojik_islemler): 
    btn.configure(bg="lightgray", fg="black") 

buttons = [
    btn_gri, btn_binary, btn_dondur, btn_kirp, btn_yaklastir, btn_uzaklastir, btn_renk_uzayi, 
    btn_histogram_genisletme, btn_histogram_germe, btn_ekleme, btn_bolme, btn_kontrast_artirma, 
    btn_konvolusyon, btn_esikleme, btn_kenar_bulma_prewitt, btn_gurultu_ekleme_ve_temizleme, 
    btn_unsharp_masking, btn_morfolojik_islemler
]

row_counter = 0
col_counter = 0
for button in buttons:
    button.grid(row=row_counter, column=col_counter, padx=10, pady=5)
    row_counter += 1
    if row_counter > 5:
        row_counter = 0
        col_counter += 1

root.configure(bg="gray")
root.mainloop()


