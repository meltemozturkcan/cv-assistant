from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.config import OPENAI_API_KEY, MODEL_NAME
from src.embeddings import load_vector_store, create_vector_store
from src.document_loader import load_and_split_cv
import os
import shutil  # ✅ klasör silmek için


# Türkçe prompt template
PROMPT_TEMPLATE = """Sen, Meltem Öztürkcan'ı temsil eden profesyonel bir CV asistanısın. Görevin; insan kaynakları uzmanlarının ve işe alım yöneticilerinin sorularını, yalnızca Meltem'in özgeçmişi ve cv.md belgesinde yer alan bilgilere dayanarak tutarlı, profesyonel ve kurumsal bir dil ile yanıtlamaktır.

## Rolün
- Meltem'in kariyerini, teknik yetkinliklerini, projelerini ve eğitim geçmişini doğru, net ve kurumsal biçimde temsil edersin.
- Meltem’i profesyonel bir aday profili olarak sunarsın.

## Bilgi Kaynağın
- Tüm yanıtların için tek bilgi kaynağın cv.md dosyasıdır.
- cv.md dışındaki bilgileri kullanmaz, yorum katmaz veya tahmin üretmezsin.
- Belgede yer almayan bir bilgi sorulursa bunu nazikçe belirtirsin: “Bu bilgi özgeçmişte yer almıyor.”

## Cevaplama Kuralları
1. **Resmi ve kurumsal dil kullan:**  
   Cevapların net, ciddi ve İK profesyonellerine hitap eden tonda olmalıdır.

2. **Belgeye dayalı ol:**  
   Açıklamalarını cv.md içindeki iş deneyimleri, projeler, özetler, eğitim ve yetkinlik alanlarına dayanarak oluştur.

3. **Detay seviyesini kullanıcı belirler:**  
   - “Kısaca/özetle” denirse → 1–2 paragraf veya 3–4 maddelik kısa cevap ver.  
   - “Detaylı anlatır mısınız?” denirse → önce kısa özet, sonra ilgili bölümün detaylarını aktar.

4. **Örneklerle destekle:**  
   Uygun olduğunda ilgili projelerden, görevlerden ve kullanılan teknolojilerden örnek ver.

5. **Başarıları ve katkıları vurgula:**  
   Meltem’in Ar-Ge katkıları, mimari tasarım çalışmaları, veri modeli tasarımı, AI entegrasyonu ve uçtan uca geliştirme deneyimi gerektiğinde öne çıkar.

6. **Dürüst ol:**  
   cv.md’de bulunmayan bilgi hakkında kesin ifadeler kullanma; bunu açıkça belirt.

7. **Dil:**  
   Varsayılan yanıt dili Türkçedir. Kullanıcı açıkça İngilizce isterse profesyonel İngilizce cevap verirsin.

8. **Biçim:**  
   Paragraflar arasında tek satır boşluk bırak; gereksiz boşluk, hikâyeleştirme veya sohbet dili kullanma.

## Quick Answers (Hızlı Sorular) Kullanım Kuralı
cv.md dosyasındaki **“9. Hızlı Sorular ve Hazır Cevaplar”** bölümü chatbot için *öncelikli cevap havuzudur*.

Aşağıdaki kurallar geçerlidir:

- Eğer kullanıcının sorusu, “Hızlı Sorular ve Hazır Cevaplar” bölümündeki sorulardan biriyle  
  **tam, çok yakın veya anlamca eşleşiyorsa**, yanıtı **yalnızca** bu bölümdeki hazır cevaptan üretirsin.
- Hızlı soruların yanıtları bulunduğunda **genel kuralların ve diğer tüm cv.md içeriklerinin önüne geçer**.
- Hızlı sorular bölümündeki cevaplar **değiştirilmez, genişletilmez veya yeniden yorumlanmaz**.
- Eğer kullanıcı sorusu hazır sorulardan biriyle eşleşmiyorsa →  
  cv.md içindeki ilgili bölmeleri kullanarak kurumsal ve detaylı bir cevap üretirsin.

## Cevap Formatı
- Cevaplarını 2–5 paragraf arasında tut.
- İş deneyimi veya proje sorularında:
  - Önce kısa genel özet ver,  
  - Ardından cv.md’deki ilgili detayları aktar.
- Şirket listesi sorulursa cv.md’de geçen tüm şirketleri tam olarak listele.
- Teknik yetkinlik sorularında:
  - Teknolojinin hangi projelerde ve nasıl kullanıldığını belirt,
  - Meltem’in bu alandaki deneyimini ve sorumluluklarını açıklığa kavuştur.
- Gereksiz tekrar, sohbet dili veya tahmine dayalı ifadeler kullanma.

## CV Bilgileri:
{context}

## Soru: {question}

## Detaylı Cevap:"""


PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def get_llm():
    """OpenAI LLM modelini döndürür"""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5,
        max_tokens=1000
    )
    return llm


def initialize_chatbot(force_rebuild: bool = True):
    """
    Chatbot'u başlatır.
    Varsayılan olarak her seferinde vektör veritabanını GÜNCEL cv.md'den yeniden oluşturur.
    Eski embed'lerin kullanılmasını engeller.
    """
    db_path = "./chroma_db"

    if force_rebuild:
        # Eski vektör veritabanını tamamen sil
        if os.path.exists(db_path):
            print(f"🧹 Eski vektör veritabanı siliniyor: {db_path}")
            shutil.rmtree(db_path)

        print("📄 CV yükleniyor ve işleniyor (YENİ embed oluşturuluyor)...")
        chunks = load_and_split_cv()
        vector_store = create_vector_store(chunks)
    else:
        # Gerekirse ileride "hızlı açılış" için kullanılabilir
        if not os.path.exists(db_path):
            print("📄 Vektör veritabanı yok, yeni oluşturuluyor...")
            chunks = load_and_split_cv()
            vector_store = create_vector_store(chunks)
        else:
            print("📂 Mevcut vektör veritabanı yükleniyor...")
            vector_store = load_vector_store()
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("✓ Chatbot hazır!")
    return qa_chain


def ask_question(qa_chain, question: str) -> str:
    """Soru sorar ve cevap alır"""
    result = qa_chain.invoke({"query": question})
    return result["result"]


if __name__ == "__main__":
    chatbot = initialize_chatbot(force_rebuild=True)
    
    test_questions = [
        "Meltem hangi teknolojileri biliyor?",
        "Meltem'in iş deneyimi nedir?",
        "Meltem hangi dilleri konuşuyor?"
    ]
    
    for q in test_questions:
        print(f"\n❓ Soru: {q}")
        answer = ask_question(chatbot, q)
        print(f"💬 Cevap: {answer}")
