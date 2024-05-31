import string

import fasttext
import numpy as np

# Önceden eğitilmiş FastText modelini yükleme
model_path = "cc.en.300.bin"  # Örnek bir model dosya yolu
model = fasttext.load_model(model_path)

from transformers import AutoTokenizer, AutoModel
import torch

# Modeli ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
modelS = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

import requests
import json
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer





def cosinesimilatiry_patch(userList,dataList):
    for user in userList:
        # String'i virgül ile ayırarak sayıları bir listeye dönüştür
        vector_list = user.get("fasttext").split(',')
        vector_listS = user.get("scibert").split(',')
        # Her elemanı uygun veri türüne dönüştür (float olması bekleniyor)
        vectorUser_floats = [float(num) for num in vector_list]
        vectorUserS_floats = [float(num) for num in vector_listS]
        for data in dataList:
            # String'i virgül ile ayırarak sayıları bir listeye dönüştür
            vector_list = data.get("fasttext").split(',')
            vector_listS = data.get("scibert").split(',')
            # Her elemanı uygun veri türüne dönüştür (float olması bekleniyor)
            vectorData_floats = [float(num) for num in vector_list]
            vectorDataS_floats = [float(num) for num in vector_listS]
            # Kosinüs benzerliğini hesapla
            similarity_score_fasttext = cosine_similarity(vectorUser_floats, vectorData_floats)
            similarity_score_scibert = cosine_similarity(vectorUserS_floats, vectorDataS_floats)
            print("Kosinüs benzerliği fasttext:", similarity_score_fasttext,"Kosinüs benzerliği scibert:", similarity_score_scibert, " dataid: ",data.get("id"),"  userid: ",user.get("id"))
            jsonPost = {
                "cosine_similarity_fasttext": similarity_score_fasttext,
                "cosine_similarity_scibert": similarity_score_scibert,
                "user_id": user.get("id"),
                "data_id": data.get("id")
            }
            patch_istegi("calc/"+ str(data.get("id")),jsonPost)



def secim_patch(patch):
    userList = get_istegi("user")
    for user in userList:
        # Kullanıcı verisini alın
        secim_vektor = user.get('secim_'+patch)

        # Kullanıcı verisinin None olup olmadığını kontrol edin
        if secim_vektor is not None:
            # String'i virgüllere göre ayırma ve boş stringleri filtreleyerek her bir karakteri integer'a çevirme
            secim_vektor = np.array([int(char) for char in secim_vektor if char])

            # Dizinin uzunluğunu al
            length_of_secim_vektor = len(secim_vektor)
            print(length_of_secim_vektor)
            # Aynı uzunlukta, tüm elemanları 1 olan bir dizi oluşturma
            true_array = np.ones(length_of_secim_vektor, dtype=int)

            print("Secim",patch," Vektor:", secim_vektor)  # Örnek çıktı: [0 1 0 1 0 1 0 0 0]
            print("True Array:", true_array)  # Örnek çıktı: [1 1 1 1 1 1 1 1 1]
        else:
            print("secim_",patch," değeri bulunamadı.")
        # precision_fasttext, recall_fasttext = precision_recall(true_labels, predicted_labels)

        # precision_recall fonksiyonunu çağırma ve döndürülen değerleri alma
        presicion, recall = presicion_recall(true_array, secim_vektor)
        presicion = float(presicion)
        recall = float(recall)

        path = "user/" + str(user.get("id"))
        name = "presicion_"+patch
        print(path)
        print(name)
        json_istek = {
            name: presicion
        }
        name = "recall_" + patch
        patch_istegi(path, json_istek)
        print(path)
        print(name)
        json_istek2 = {
            name: recall
        }
        patch_istegi(path, json_istek2)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def presicion_recall(true_labels, predicted_labels):
    # Gerçek etiketler (true labels) ve tahmin edilen etiketler (predicted labels)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # TP, FP, FN ve TN hesaplaması
    TP = np.sum((true_labels == 1) & (predicted_labels == 1))
    FP = np.sum((true_labels == 0) & (predicted_labels == 1))
    FN = np.sum((true_labels == 1) & (predicted_labels == 0))
    TN = np.sum((true_labels == 0) & (predicted_labels == 0))

    # Zero division check for precision and recall
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return precision, recall

def cosinesimilatiry(userList,dataList):
    for user in userList:
        # String'i virgül ile ayırarak sayıları bir listeye dönüştür
        vector_list = user.get("fasttext").split(',')
        vector_listS = user.get("scibert").split(',')
        # Her elemanı uygun veri türüne dönüştür (float olması bekleniyor)
        vectorUser_floats = [float(num) for num in vector_list]
        vectorUserS_floats = [float(num) for num in vector_listS]
        for data in dataList:
            # String'i virgül ile ayırarak sayıları bir listeye dönüştür
            vector_list = data.get("fasttext").split(',')
            vector_listS = data.get("scibert").split(',')
            # Her elemanı uygun veri türüne dönüştür (float olması bekleniyor)
            vectorData_floats = [float(num) for num in vector_list]
            vectorDataS_floats = [float(num) for num in vector_listS]
            # Kosinüs benzerliğini hesapla
            similarity_score_fasttext = cosine_similarity(vectorUser_floats, vectorData_floats)
            similarity_score_scibert = cosine_similarity(vectorUserS_floats, vectorDataS_floats)
            print("Kosinüs benzerliği fasttext:", similarity_score_fasttext,"Kosinüs benzerliği scibert:", similarity_score_scibert, " dataid: ",data.get("id"),"  userid: ",user.get("id"))
            jsonPost = {
                "cosine_similarity_fasttext": similarity_score_fasttext,
                "cosine_similarity_scibert": similarity_score_scibert,
                "user_id": user.get("id"),
                "data_id": data.get("id")
            }
            post_istegi("calc",jsonPost)


def user_fasttext(userlist):
    print("user fasttextteuiz")
    for user in userlist:
        print(user.get("id"))
        fasttext_sonuc = fasttext_vector(user.get("interests"))
        path = "user/" + str(user.get("id"))
        print(path)
        json_istek = {
            "fasttext": fasttext_sonuc
        }
        patch_istegi(path, json_istek)



def user_scibert(userlist):
    print("user scibert")
    for user in userlist:
        print(user.get("id"))
        scibert_sonuc = scibert_vector(user.get("interests"))
        path = "user/" + str(user.get("id"))
        print(path)
        json_istek = {
            "scibert": scibert_sonuc
        }
        patch_istegi(path, json_istek)

def post_istegi(path,jsonPost):
    url = "http://localhost:8080/" + path
    # POST isteği gönderme
    response = requests.post(url, json=jsonPost)

    # Sunucudan gelen yanıtı kontrol etme
    if response.status_code == 200:
        print('İstek başarılı! Sunucu yanıtı:', response.text)
    else:
        print('İstek başarısız! Sunucu yanıtı:', response.status_code)


def patch_istegi(path,json_istek):
    url = "http://localhost:8080/"+path

    # PATCH isteği gönderme
    response = requests.patch(url, json=json_istek)

    # Yanıtı kontrol etme
    if response.status_code == 200:
        print("PATCH isteği başarıyla tamamlandı.")
    else:
        print("PATCH isteği başarısız oldu. Hata kodu:", response.status_code)
def get_istegi(path):
    # GET isteği yapılacak URL
    url = 'http://localhost:8080/'+path

    # GET isteği gönder
    response = requests.get(url)

    # İstek sonucunu kontrol et
    if response.status_code == 200:
        # JSON yanıtını Python sözlüğüne dönüştür
        data = response.json()
        # Veri sayısını bul
        num_data = len(data)
        print("İstek başarılı! Toplam", num_data, "adet veri döndü.")
    else:
        print("İstek başarısız! Hata kodu:", response.status_code)

    return data

def scibert_vector(text):

    # Metni tokenize etme ve tensor'a dönüştürme
    inputs = tokenizer(text, return_tensors="pt")

    # Modelden vektör elde etme
    with torch.no_grad():
        outputs = modelS(**inputs)

    # Son katmanın çıktısını almak (metin vektörü)
    vector = outputs.last_hidden_state[:, 0, :].squeeze()

    # Vektörü bir stringe dönüştür
    vector_str = ','.join(map(str, vector.tolist()))

    print("Metin Gömülme Vektörü:", vector)
    print(vector_str)

    return vector_str



def fasttext_vector(text):
    # Metin vektörlerini elde etme
    vector = model.get_sentence_vector(text)
    # Vektörü bir stringe dönüştür
    vector_str = ','.join(map(str, vector))
    print("Fasttext Metin vektörü:", vector)
    print(vector_str)

    return vector_str


def preprocess_text(text):
    # Küçük harfe dönüştürme
    text = text.lower()

    # Noktalama işaretlerini çıkarma
    text = ''.join([char for char in text if char not in punctuation])

    # Metni token'lara ayırma
    words = word_tokenize(text, language='english')

    # Stopwords'leri ve boşlukları temizleme
    words = [word for word in words if word not in stop_words and word.isalnum()]

    # Kelime köklerini bulma
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Temizlenmiş cümleyi birleştir
    words = ' '.join(words)


    return words


def json_dosyasindan_verileri_oku(json_dosya_yolu):
    # Stop words listesini indir

    with open(json_dosya_yolu, 'r') as dosya:
        dosya_icerigi = dosya.read()

        # Her bir JSON verisini ayır
        json_verileri = dosya_icerigi.strip().split('\n')

        url = 'http://localhost:8080/data'

        # Her bir JSON verisini işle
        for veri in json_verileri:
            json_veri = json.loads(veri)

            temizlenmis_cumle = preprocess_text(json_veri.get('abstract'))

            fasttext_sonuc = fasttext_vector(temizlenmis_cumle)
            scibert_sonuc = scibert_vector(temizlenmis_cumle)

            json_istek = {
            "name": json_veri.get('name'),
            "title": json_veri.get('title'),
            "abstract_":temizlenmis_cumle,
            "fulltext":json_veri.get('fulltext'),
            "keywords":json_veri.get('keywords'),
            "fasttext":fasttext_sonuc,
            "scibert":scibert_sonuc
            }
            # veya başka bir işlem yapabilirsiniz
            response = requests.post(url, json=json_istek)

            # İstek sonucunu kontrol et
            if response.status_code == 200:
                print("İstek başarılı! Yanıt:", response.json())
            else:
                print("İstek başarısız! Hata kodu:", response.status_code)


data = get_istegi("data")
user_list = get_istegi("user")

# Veri sayısını bul
num_data = len(data)

b = 0

if num_data == 0:

    nltk.download('stopwords')
    nltk.download('punkt')

    # stopwords listesi
    stop_words = set(stopwords.words('english'))

    # Metin içindeki noktalama işaretleri
    punctuation = set(string.punctuation)

    # JSON dosyasının yolunu belirtin
    dosya_yolu = 'test.json'

    # JSON dosyasındaki verileri oku ve işle
    json_dosyasindan_verileri_oku(dosya_yolu)
    user_fasttext(user_list)
    user_scibert(user_list)
    dataList = get_istegi("data")
    userList = get_istegi("user")
    cosinesimilatiry(userList, dataList)
    b = 1

if (b == 0):
    data_list = get_istegi("data")
    user_list = get_istegi("user")
    secim_patch("scibert")
    secim_patch("fasttext")
    user_fasttext(user_list)
    user_scibert(user_list)
    cosinesimilatiry_patch(user_list, data_list)

#scibert ve fasttext verilerini user için yenileyelim

#calctaki cosine similarityleri patchle


