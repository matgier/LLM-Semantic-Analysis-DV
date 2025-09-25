from pickle import TRUE
from time import sleep
from matplotlib.patches import Patch
import numpy as np
import json
import requests
from typing import List, Dict, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler


from helpers.all_keys import *


# Konfiguracja API
OPEN_AI_API = False

VOYAGE_AI_API = False

GEMINI_AI_API = True 

OLLAMA= False

MODEL_OLLAMA = 'mistral:latest'
SENSITIVITY = 0



def normalization_and_centralization(embedding_value: List[float], centred: bool = True, normalized: bool = True, normalization_type: str = 'l2'):
    # Centrowanie (jeśli wymagane)
    embedding = embedding_value
    if not isinstance(embedding_value, np.ndarray):
        embedding_array = np.array(embedding_value)
    # Normalizacja (jeśli wymagane)
    # if centred:
    #     mean_value = np.mean(embedding_array)
    #     embedding_array = embedding_array - mean_value
    if normalized:
        if normalization_type == "l2":
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

        norm_value = np.linalg.norm(embedding_array)
        mean_value = np.mean(embedding_array)
        # print(f"    - Wartość normy: {norm_value}")
        # print(f"    - Wartość średnia: {mean_value}")

        # Zapis jako lista
        embedding = embedding_array.tolist()
    return embedding


def generate_embedding_ollama(
    text: str, 
    model: str = "mistral:latest", 
    host: str = "localhost", 
    port: int = 11434,
    normalize: bool = True,
    normalization_type: str = 'l2',
    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając wybranego modelu w Ollamie.
    
    Parametry:
    ----------
    text : str
        Tekst, który ma zostać przekształcony w embedding.
    model : str, domyślnie "mistral:latest"
        Nazwa modelu w Ollamie do generowania embeddingu. 
        Sugerowane modele: "nomic-embed-text", "llama2", "llama3", "mistral", "gemma".
    host : str, domyślnie "localhost"
        Host, na którym działa Ollama.
    port : int, domyślnie 11434
        Port, na którym nasłuchuje API Ollamy.
    normalize : bool, domyślnie True
        Czy znormalizować wektor embeddingu do jednostkowej długości.
    timeout : int, domyślnie 30
        Limit czasu w sekundach na odpowiedź od API.
        
    Zwraca:
    -------
    List[float]
        Lista floatów reprezentująca embedding tekstu
    
    Zgłasza:
    --------
    ConnectionError: Gdy nie można połączyć się z serwerem Ollama.
    ValueError: Gdy zwrócony embedding jest pusty lub gdy model nie istnieje.
    """
    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # Przygotuj URL do API Ollamy
    url = f"http://{host}:{port}/api/embeddings"
    
    # Przygotuj dane do wysłania
    data = {
        "model": model,
        "prompt": text
    }
    
    try:
        # Wywołaj API z timeout
        response = requests.post(url, json=data, timeout=timeout)
        
        # Sprawdź, czy zapytanie się powiodło
        if response.status_code != 200:
            error_msg = f"Błąd API: {response.status_code}, {response.text}"
            raise ValueError(error_msg)
        
        # Pobierz embedding z odpowiedzi
        result = response.json()
        embedding = result.get("embedding")
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu. Odpowiedź: {result}")
        
        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)

        return embedding
        
        
        
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Timeout podczas łączenia z Ollamą na {host}:{port}. "
                             f"Serwer nie odpowiedział w ciągu {timeout} sekund.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Nie można połączyć się z Ollamą na {host}:{port}. "
                             f"Upewnij się, że Ollama jest uruchomiona i dostępna.")
    except json.JSONDecodeError:
        raise ValueError(f"Nieprawidłowa odpowiedź z API Ollamy: {response.text}")



import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import gc
import os

def generate_embedding(
    text: str, 
    model_name: str = "mistralai/Mistral-7B-v0.1",
    #model_name: str = "meta-llama/Llama-3.1-8B",
    layer_index: int = 16,
    normalize: bool = True,
    normalization_type: str = 'l2',
    max_length: int = 8192
) -> List[float]:
    """
    Generuje najwyższej jakości embeddingi z Mistral-7B.
    
    PRIORYTET: MAKSYMALNA JAKOŚĆ I DOKŁADNOŚĆ
    ==========================================
    - Float32 dla pełnej precyzji numerycznej
    - Pełna długość kontekstu (4096 tokenów)
    - Prawidłowe mean pooling z attention mask
    - Brak chunking - pełny kontekst zachowany
    - Oryginalny model Mistral-7B bez kompresji
    
    Args:
        text: Tekst do przetworzenia
        model_name: Model Mistral-7B (nie zmieniać dla jakości)
        layer_index: Warstwa 16 (optymalna dla semantyki)
        normalize: Normalizacja L2 embeddingu
        normalization_type: 'l2' lub 'robust'
        max_length: Maksymalna długość (4096 dla Mistral)
        
    Returns:
        Lista floatów - embedding najwyższej jakości
    """
    
    # ========== WALIDACJA ==========
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # ========== SETUP DLA MAKSYMALNEJ JAKOŚCI ==========
    # Zawsze float32 dla pełnej precyzji
    dtype = torch.float16
    
    # Wybór device - MPS dla M1 lub CPU
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     # Wyłączamy problematyczne optymalizacje
    #     # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    #     # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    #     print(f"🎯 Tryb MPS")
    # else:
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)
    #print(f"🎯 Tryb CPU")
    
    # ========== ŁADOWANIE MODELU ==========
   # print(f"📥 Ładowanie {model_name} w pełnej precyzji...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=max_length
    )
    
    # Mistral może nie mieć pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model w pełnej precyzji
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True  # Sekwencyjne ładowanie
    ).to(device)
    
    model.eval()  # Tryb ewaluacji
    
    # ========== TOKENIZACJA ==========
    # Pełna tokenizacja bez truncation jeśli mieści się w max_length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True  # Kluczowe dla jakości
    )
    
    # Przeniesienie na device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    #print(f"📏 Długość sekwencji: {inputs['input_ids'].shape[1]} tokenów")
    
    # ========== FORWARD PASS W PEŁNEJ PRECYZJI ==========
    with torch.inference_mode():
        # Bez autocast - chcemy pełną precyzję
        outputs = model(**inputs)
        
        # Ekstrakcja hidden states z wybranej warstwy
        hidden_states = outputs.hidden_states
        print(f"Model ma {len(hidden_states)} warstw, żądany indeks: {layer_index}")
        if layer_index >= len(hidden_states):
            raise ValueError(f"Model ma {len(hidden_states)} warstw, żądany indeks: {layer_index}")
        
        layer_output = hidden_states[layer_index]  # [batch=1, seq_len, hidden_dim]
        
        # ========== MEAN POOLING Z ATTENTION MASK ==========
        # Kluczowe dla jakości - prawidłowe uśrednianie tylko rzeczywistych tokenów
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [1, seq_len, 1]
        
        # Maskowanie padding tokens
        masked_output = layer_output * attention_mask
        
        # Suma tylko rzeczywistych tokenów
        sum_output = torch.sum(masked_output, dim=1)  # [1, hidden_dim]
        sum_mask = torch.sum(attention_mask, dim=1).clamp(min=1e-9)  # [1, 1]
        
        # Średnia ważona
        pooled_output = sum_output / sum_mask  # [1, hidden_dim]
        
        # Konwersja na numpy w pełnej precyzji
        embedding_tensor = pooled_output[0].detach()  # [hidden_dim]
        
        # Czyszczenie pamięci
        del outputs, hidden_states, layer_output, inputs
        if device.type == "mps":
            try:
                torch.mps.synchronize()
            except:
                pass
    
    # ========== KONWERSJA I NORMALIZACJA ==========
    # Konwersja na numpy zachowując float32
    if device.type == "mps":
        embedding_array = embedding_tensor.cpu().numpy().astype(np.float16)
    else:
        embedding_array = embedding_tensor.numpy()
    
    del embedding_tensor
    
    # Normalizacja dla lepszej porównywalności
    if normalize:
        if normalization_type == "l2":
            # Standardowa normalizacja L2
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
                
        elif normalization_type == "robust":
            # Robust normalization - odporna na outliers
            median = np.median(embedding_array)
            mad = np.median(np.abs(embedding_array - median))
            if mad > 0:
                embedding_array = (embedding_array - median) / (1.4826 * mad)
    
    # ========== STATYSTYKI DLA WERYFIKACJI ==========
    # print(f"✅ Embedding najwyższej jakości:")
    # print(f"   - Wymiar: {len(embedding_array)}")
    # print(f"   - Norma: {np.linalg.norm(embedding_array):.4f}")
    # print(f"   - Średnia: {np.mean(embedding_array):.6f}")
    # print(f"   - Std: {np.std(embedding_array):.6f}")
    # print(f"   - Min/Max: [{np.min(embedding_array):.6f}, {np.max(embedding_array):.6f}]")
    
    # ========== CZYSZCZENIE ==========
    del model, tokenizer
    
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except:
            pass
    
    gc.collect()
    
    # Zwracamy jako listę float32
    return embedding_array.tolist()




def generate_embedding_openai(
    text: str, 
    model: str = "text-embedding-3-large", 
    api_key: str = API_KEY_OPENAI,
    normalize: bool = True,
    normalization_type: str = 'l2',
    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając API OpenAI.
    """
    sleep(1)
    # print("----> generate_embedding_openai")
    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    try:
        #print("-> Using OPENAI to create embedding!")
        # Inicjalizacja klienta OpenAI
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Wywołanie API z timeout
        response = client.embeddings.create(
            input=text,
            model=model,
            timeout=timeout
        )
        
        # Pobierz embedding z odpowiedzi
        embedding = response.data[0].embedding
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu.")

        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding
    except Exception as e:
        # Generyczna obsługa wyjątków dla uproszczenia
        raise ValueError(f"Błąd przy generowaniu embeddingu OpenAI: {str(e)}")


def generate_embedding_voyage(
    text: str, 
    model: str = "voyage-3-large", 
    api_key: str = None,
    input_type: str = None,
    normalize: bool = True,
    normalization_type: str = 'l2',

    timeout: int = 30
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając API Voyage AI.
    """
    sleep(1)

    # Upewnij się, że tekst nie jest pusty
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    # Pobierz klucz API
    if api_key is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Nie znaleziono klucza API. Podaj api_key jako parametr lub ustaw zmienną środowiskową VOYAGE_API_KEY.")
    
    try:
        print(f"-> Using VOYAGE AI to create embedding with model {model}!")
        
        # Endpoint API dla embeddingów
        url = "https://api.voyageai.com/v1/embeddings"
        
        # Przygotowanie nagłówków
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Przygotowanie danych
        data = {
            "model": model,
            "input": text
        }
        
        # Dodaj input_type jeśli został określony
        if input_type:
            if input_type not in ["query", "document"]:
                raise ValueError("input_type musi być 'query' lub 'document' lub None")
            data["input_type"] = input_type
        
        # Wykonanie żądania HTTP
        response = requests.post(
            url, 
            headers=headers, 
            json=data,
            timeout=timeout
        )
        
        # Sprawdzenie statusu odpowiedzi
        response.raise_for_status()
        
        # Parsowanie odpowiedzi
        result = response.json()
        
        # Sprawdzenie czy odpowiedź zawiera embedding
        if "data" not in result or len(result["data"]) == 0 or "embedding" not in result["data"][0]:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu. Odpowiedź API: {result}")
        
        # Pobierz embedding z odpowiedzi
        embedding = result["data"][0]["embedding"]
        
        # Sprawdź, czy embedding nie jest pusty
        if not embedding:
            raise ValueError(f"Model '{model}' zwrócił pusty embedding.")
        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding

        
    except Exception as e:
        # Generyczna obsługa wyjątków dla uproszczenia
        raise ValueError(f"Błąd przy generowaniu embeddingu Voyage: {str(e)}")

from google import genai
from google.genai.types import EmbedContentConfig
import numpy as np
from time import sleep
from typing import List

def generate_embedding_gemini(
    text: str,
    model: str = "gemini-embedding-exp-03-07",  # Zaktualizowany domyślny model
    api_key: str = API_KEY_GEMINI,
    normalize: bool = True,
    normalization_type: str = 'l2',
    timeout: int = 30,
    task_type: str = "RETRIEVAL_DOCUMENT",  # Nowy parametr
    output_dimensionality: int = None,  # Opcjonalna redukcja wymiarów
    title: str = None  # Opcjonalny tytuł dla kontekstu
) -> List[float]:
    """
    Generuje embedding dla podanego tekstu używając nowego Gemini API.
    
    Args:
        text: Tekst do przetworzenia
        model: Nazwa modelu (domyślnie gemini-embedding-001)
        api_key: Klucz API
        normalize: Czy normalizować embedding
        normalization_type: Typ normalizacji ('l2' lub 'robust')
        timeout: Timeout dla żądania
        task_type: Typ zadania dla embeddingu
        output_dimensionality: Opcjonalna redukcja wymiarowości
        title: Opcjonalny tytuł dla kontekstu
    """
    sleep(1)
    print("----> generate_embedding_gemini")

    # Walidacja wejścia
    if not text or not text.strip():
        raise ValueError("Tekst nie może być pusty")
    
    try:
        #print("-> Using GEMINI (new API) to create embedding!")
        
        # Inicjalizacja klienta z nowym API
        client = genai.Client(api_key=api_key)
        
        # Przygotowanie konfiguracji
        config_params = {
            "task_type": task_type
        }
        
        # Dodanie opcjonalnych parametrów jeśli są podane
        if output_dimensionality:
            config_params["output_dimensionality"] = output_dimensionality
        if title:
            config_params["title"] = title
            
        config = EmbedContentConfig(**config_params)
        
        # Wywołanie API z nową strukturą
        response = client.models.embed_content(
            model=model,
            contents=text,
            config=config
        )
        
        # Wyciągnięcie embeddingu z nowej struktury odpowiedzi
        if not response.embeddings or len(response.embeddings) == 0:
            raise ValueError(f"Model '{model}' nie zwrócił embeddingu.")
            
        embedding= response.embeddings[0].values
        if not embedding:
            raise ValueError(f"Embedding jest pusty dla modelu '{model}'.")

        embedding = normalization_and_centralization(embedding_value=embedding, normalized=normalize)
        return embedding

    except Exception as e:
        raise ValueError(f"Błąd przy generowaniu embeddingu Gemini: {str(e)}")

import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from typing import List, Dict, Union, Optional, Tuple

def compute_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Tworzy macierz odległości kosinusowych między embedingami.
    
    Args:
        embeddings (numpy.ndarray): Macierz embedingów
        
    Returns:
        numpy.ndarray: Macierz odległości kosinusowych
    """
    if embeddings.shape[0] <= 1:
        # Zwróć pustą macierz dla 0 lub 1 embeddingu
        return np.zeros((embeddings.shape[0], embeddings.shape[0]))
    
    distances = pdist(embeddings, metric='cosine')
    return squareform(distances)
    

def identify_semantic_axes(semantic_features: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Identyfikuje pary lub trójki kategorii, które tworzą osie semantyczne na podstawie
    ich nazw (np. 'kategoria_positive', 'kategoria_negative' i opcjonalnie 'kategoria_balanced'
    tworzą oś 'kategoria').
    
    Args:
        semantic_features (Dict[str, List[str]]): Słownik kategorii semantycznych
        
    Returns:
        Dict[str, Dict[str, str]]: Słownik osi semantycznych
    """
    axes = {}
    categories = list(semantic_features.keys())
    processed_categories = set()
    
    # Definicja możliwych konwencji nazewnictwa
    opposite_suffixes = [
        ('_beneficial', '_harmful'),
        ('_effective', '_ineffective'), 
        ('_strong', '_weak'),   # Dostosowane do dostarczonych danych
        ('_positive', '_negative'), 
        ('_respected', '_restricted'),
        ('_progressive', '_traditional'),   # Dostosowane do dostarczonych danych
        ('_regulated', '_free_market'),

    ]
    
    # Dostosowanie do specyficznego przypadku "balanced" w dostarczonych danych
    balanced_suffix = '_balanced'
    
    # Iteracja po wszystkich kategoriach
    for category in categories:
        if category in processed_categories:
            continue
            
        found_axis = False
        
        # Sprawdzanie, czy kategoria kończy się na "balanced" i czy istnieje para
        if category.endswith(balanced_suffix):
            base_name = category[:-len(balanced_suffix)]
            
            # Sprawdź czy istnieją odpowiednie kategorie "pozytywne" i "negatywne"
            potential_pairs = []
            for pos_suffix, neg_suffix in opposite_suffixes:
                pos_category = base_name + pos_suffix
                neg_category = base_name + neg_suffix
                if pos_category in categories and neg_category in categories:
                    potential_pairs.append((pos_category, neg_category))
            
            if potential_pairs:
                # Bierzemy pierwszą znalezioną parę
                pos_category, neg_category = potential_pairs[0]
                
                # Znaleziono trójkę kategorii tworzących oś z kategorią zbalansowaną w środku
                axis_name = base_name
                axes[axis_name] = {
                    'positive': pos_category,
                    'negative': neg_category,
                    'balanced': category  # Dodaj informację o kategorii zbalansowanej
                }
                processed_categories.add(category)
                processed_categories.add(pos_category)
                processed_categories.add(neg_category)
                found_axis = True
                print(f"Zidentyfikowano oś semantyczną z trzema punktami: {axis_name}")
                print(f"  Biegun pozytywny: {pos_category}")
                print(f"  Punkt środkowy: {category}")
                print(f"  Biegun negatywny: {neg_category}")
                continue
        
        # Standardowe sprawdzanie par przeciwieństw
        for pos_suffix, neg_suffix in opposite_suffixes:
            # Sprawdzenie czy kategoria kończy się na jeden z sufiksów pozytywnych
            if category.endswith(pos_suffix):
                base_name = category[:-len(pos_suffix)]
                opposite_category = base_name + neg_suffix
                
                if opposite_category in categories:
                    # Znaleziono parę kategorii tworzących oś
                    axis_name = base_name
                    axes[axis_name] = {
                        'positive': category,
                        'negative': opposite_category
                    }
                    
                    # Sprawdź, czy istnieje również kategoria zbalansowana
                    balanced_category = base_name + balanced_suffix
                    if balanced_category in categories:
                        axes[axis_name]['balanced'] = balanced_category
                        processed_categories.add(balanced_category)
                    
                    processed_categories.add(category)
                    processed_categories.add(opposite_category)
                    found_axis = True
                    print(f"Zidentyfikowano oś semantyczną: {axis_name}")
                    print(f"  Biegun pozytywny: {category}")
                    print(f"  Biegun negatywny: {opposite_category}")
                    if balanced_category in categories:
                        print(f"  Punkt środkowy: {balanced_category}")
                    break
                    
            # Sprawdzenie czy kategoria kończy się na jeden z sufiksów negatywnych
            elif category.endswith(neg_suffix):
                base_name = category[:-len(neg_suffix)]
                opposite_category = base_name + pos_suffix
                
                if opposite_category in categories:
                    # Znaleziono parę kategorii tworzących oś
                    axis_name = base_name
                    axes[axis_name] = {
                        'positive': opposite_category,
                        'negative': category
                    }
                    
                    # Sprawdź, czy istnieje również kategoria zbalansowana
                    balanced_category = base_name + balanced_suffix
                    if balanced_category in categories:
                        axes[axis_name]['balanced'] = balanced_category
                        processed_categories.add(balanced_category)
                    
                    processed_categories.add(category)
                    processed_categories.add(opposite_category)
                    found_axis = True
                    print(f"Zidentyfikowano oś semantyczną: {axis_name}")
                    print(f"  Biegun pozytywny: {opposite_category}")
                    print(f"  Biegun negatywny: {category}")
                    if balanced_category in categories:
                        print(f"  Punkt środkowy: {balanced_category}")
                    break
        
        # Jeśli nie znaleziono osi, możemy utworzyć sztuczną kategorię zbalansowaną
        # jako centroid między pozytywną a negatywną kategorią
        if not found_axis and category not in processed_categories:
            # Ta zmiana jest opcjonalna - można automatycznie tworzyć zbalansowane kategorie
            # nawet gdy nie są zdefiniowane w danych wejściowych
            print(f"Ostrzeżenie: Nie znaleziono przeciwstawnej kategorii dla '{category}'")
    
    print(f"Zidentyfikowano {len(axes)} osi semantycznych")
    return axes

# Dodana funkcja do obsługi dłuższych tekstów w semantic_features
def preprocess_semantic_features(semantic_features: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Przetwarza słownik semantic_features, aby zawierał listy słów kluczowych zamiast długich tekstów.
    
    Args:
        semantic_features (Dict[str, List[str]]): Słownik z kategoriami semantycznymi i ich opisami
            
    Returns:
        Dict[str, List[str]]: Przetworzony słownik z kategoriami i listami słów kluczowych
    """
    processed_features = semantic_features
    
    for category, texts in semantic_features.items():
        if len(texts) == 1 and len(texts[0].split()) > 15:
            pass
            # # To jest długi tekst, podziel go na kluczowe frazy
            # text = texts[0]
            
            # # Podziel tekst na zdania
            # import re
            # sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # # Wybierz kluczowe frazy (możemy użyć prostej ekstrakcji)
            # key_phrases = []
            # for sentence in sentences:
            #     # Usuwamy zbyt krótkie frazy
            #     if len(sentence.split()) < 3:
            #         continue
                
            #     # Wybieramy pierwsze 5-8 słów z każdego zdania jako frazę
            #     words = sentence.split()
            #     phrase_length = min(8, len(words))
            #     phrase = " ".join(words[:phrase_length])
                
            #     # Dodajemy frazę, jeśli jest wystarczająco długa
            #     if len(phrase.split()) >= 3:
            #         key_phrases.append(phrase)
            
            # # Ogranicz liczbę fraz dla danej kategorii
            # max_phrases = 5
            # if len(key_phrases) > max_phrases:
            #     key_phrases = key_phrases[:max_phrases]
                
            # processed_features[category] = key_phrases
        else:
            # To już jest lista słów kluczowych, zachowujemy bez zmian
            processed_features[category] = texts
            
    return processed_features



def plot_semantic_axis_2d_mds(axis_name: str, result: Dict, figsize: Tuple[int, int] = (10, 8), 
                         title: Optional[str] = None, output_path: Optional[str] = None, 
                         show_centroids: bool = True):
    """
    Wizualizuje wybraną oś semantyczną w przestrzeni 2D z punktami słów kluczowych, centroidami kategorii
    i analizowanym tekstem. Uwzględnia zbalansowany centroid jeśli istnieje.
    """
    # Sprawdzenie, czy mamy wszystkie potrzebne dane
    if "mds_coords_2d" not in result:
        raise ValueError(f"Wynik dla osi '{axis_name}' nie zawiera danych MDS 2D.")
    
    # Funkcja do skracania długich tekstów
    def truncate_text(text, max_length=70):
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    # Tworzenie figury
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ustawienie tytułu
    if title is None:
        title = f"Oś semantyczna: {axis_name} (2D)"
    ax.set_title(title, fontsize=14)
    
    # Pobierz dane z wyniku
    mds_coords_2d = np.array(result["mds_coords_2d"])
    word_labels = result["word_labels"]
    all_words = result["all_words"]
    
    # Skróć wszystkie słowa, gdy są za długie
    truncated_words = [truncate_text(word, 50) for word in all_words]
    
    # Znajdź indeksy dla różnych typów słów
    neg_indices = [i for i, label in enumerate(word_labels) if label == "neg"]
    pos_indices = [i for i, label in enumerate(word_labels) if label == "pos"]
    balanced_indices = [i for i, label in enumerate(word_labels) if label == "balanced"]  # Dodane dla zbalansowanych kategorii
    
    # Pobierz współrzędne MDS dla każdego słowa
    neg_coords = mds_coords_2d[neg_indices] if neg_indices else np.array([])
    pos_coords = mds_coords_2d[pos_indices] if pos_indices else np.array([])
    balanced_coords = mds_coords_2d[balanced_indices] if balanced_indices else np.array([])  # Dodane dla zbalansowanych kategorii
    
    # Pobierz centroidy
    neg_centroid_mds = np.array(result["neg_centroid_mds_2d"])
    pos_centroid_mds = np.array(result["pos_centroid_mds_2d"])
    
    # Centroid zbalansowany, jeśli istnieje
    balanced_centroid_mds = np.array(result.get("balanced_centroid_mds_2d", [0, 0]))
    has_balanced_centroid = "balanced_centroid_mds_2d" in result
    
    # Pobierz pozycję analizowanego tekstu i wektor osi
    text_position = np.array(result["position_2d"])
    axis_unit_vector = np.array(result["axis_unit_vector"])
    orthogonal_vector = np.array(result["orthogonal_vector"])
    
    # Określ min, max i zakres osi X i Y
    all_x, all_y = [], []
    
    # Dodaj wszystkie punkty słów kluczowych
    for coords in [neg_coords, pos_coords, balanced_coords]:
        if len(coords) > 0:
            all_x.extend(coords[:, 0])
            all_y.extend(coords[:, 1])
    
    # Dodaj centroidy
    all_x.extend([neg_centroid_mds[0], pos_centroid_mds[0]])
    all_y.extend([neg_centroid_mds[1], pos_centroid_mds[1]])
    if has_balanced_centroid:
        all_x.append(balanced_centroid_mds[0])
        all_y.append(balanced_centroid_mds[1])
    
    # Dodaj analizowany tekst
    all_x.append(text_position[0])
    all_y.append(text_position[1])
    
    # Oblicz zakres z marginesem
    if all_x and all_y:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Dodaj margines 10%
        x_margin = 0.1 * x_range
        y_margin = 0.1 * y_range
        
        # Zapewnij, że zakres X i Y jest taki sam, aby uniknąć zniekształceń
        max_range = max(x_range + 2 * x_margin, y_range + 2 * y_margin)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_min = x_center - max_range / 2
        x_max = x_center + max_range / 2
        y_min = y_center - max_range / 2
        y_max = y_center + max_range / 2
    else:
        # Domyślny zakres, jeśli brak punktów
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
    
    # Skrócone nazwy kategorii dla etykiet
    neg_label = truncate_text(result["pole_labels"]["negative"])
    pos_label = truncate_text(result["pole_labels"]["positive"])
    balanced_label = truncate_text(result["pole_labels"].get("balanced", "zbalansowane"))
    
    # Rysowanie punktów dla słów negatywnego bieguna
    if len(neg_coords) > 0:
        ax.scatter(neg_coords[:, 0], neg_coords[:, 1], 
                c='red', s=80, alpha=0.5, label=f'Differentiating value {neg_label}')
        
        # Dodanie etykiet słów
        for i, idx in enumerate(neg_indices):
            word = truncated_words[idx]  # Użyj skróconej wersji
            ax.annotate(word, (neg_coords[i, 0], neg_coords[i, 1]), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc='aliceblue', ec="red", alpha=0.7))
    
    # Rysowanie punktów dla słów pozytywnego bieguna
    if len(pos_coords) > 0:
        ax.scatter(pos_coords[:, 0], pos_coords[:, 1], 
                c='green', s=80, alpha=0.5, label=f'Differentiating value {pos_label}')
        
        # Dodanie etykiet słów
        for i, idx in enumerate(pos_indices):
            word = truncated_words[idx]  # Użyj skróconej wersji
            ax.annotate(word, (pos_coords[i, 0], pos_coords[i, 1]), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc='mistyrose', ec="green", alpha=0.7))
    
    # Rysowanie punktów dla słów zbalansowanych
    if len(balanced_coords) > 0:
        ax.scatter(balanced_coords[:, 0], balanced_coords[:, 1], 
                c='blue', s=80, alpha=0.5, label=f'Qu {balanced_label}')
        
        # Dodanie etykiet słów
        for i, idx in enumerate(balanced_indices):
            word = truncated_words[idx]  # Użyj skróconej wersji
            ax.annotate(word, (balanced_coords[i, 0], balanced_coords[i, 1]), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc='lavender', ec="blue", alpha=0.7))
    
    # Rysowanie centroidów, jeśli zażądano
    if show_centroids:
        # Rysowanie centroidu negatywnego
        ax.scatter([neg_centroid_mds[0]], [neg_centroid_mds[1]], 
                c='red', s=150, alpha=0.8, marker='*', 
                edgecolor='black', linewidth=1.5, label=f'Negative quality {neg_label}')
        
        # Rysowanie centroidu pozytywnego
        ax.scatter([pos_centroid_mds[0]], [pos_centroid_mds[1]], 
                c='green', s=150, alpha=0.8, marker='*', 
                edgecolor='black', linewidth=1.5, label=f'Positive quality {pos_label}')
        
        # Rysowanie centroidu zbalansowanego, jeśli istnieje
        if has_balanced_centroid:
            ax.scatter([balanced_centroid_mds[0]], [balanced_centroid_mds[1]], 
                    c='blue', s=150, alpha=0.8, marker='*', 
                    edgecolor='black', linewidth=1.5, 
                    label=f'Centroid {balanced_label}')
        
        # Rysowanie głównej osi polaryzacji
        # Obliczamy zakres wykresu
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Wydłuż wektor, aby oś przechodziła przez cały wykres
        arrow_length = max(x_range, y_range) * 0.8
        
        # Rysuj linię osi głównej
        arrow_start = (neg_centroid_mds[0] * 0.5 + pos_centroid_mds[0] * 0.5) - axis_unit_vector[0] * arrow_length / 2
        arrow_end = (neg_centroid_mds[0] * 0.5 + pos_centroid_mds[0] * 0.5) + axis_unit_vector[0] * arrow_length / 2
        
        arrow_start_y = (neg_centroid_mds[1] * 0.5 + pos_centroid_mds[1] * 0.5) - axis_unit_vector[1] * arrow_length / 2
        arrow_end_y = (neg_centroid_mds[1] * 0.5 + pos_centroid_mds[1] * 0.5) + axis_unit_vector[1] * arrow_length / 2
        
        # Dodaj linię głównej osi
        ax.plot([arrow_start, arrow_end], [arrow_start_y, arrow_end_y], 
            'k-', alpha=0.7, linewidth=2, label="Main polarization axis")
        
        # # Dodaj strzałki wskazujące kierunek osi
        # ax.arrow(0, 0, axis_unit_vector[0] * arrow_length * 0.3, axis_unit_vector[1] * arrow_length * 0.3, 
        #         head_width=0.05, head_length=0.05, fc='k', ec='k', alpha=0.7)
        
        # Rysuj ortogonalną oś (prostopadłą)
        # Rysuj krótszą ortogonalną linię kreskowaną
        orth_arrow_length = max(x_range, y_range) * 0.5
        
        orth_arrow_start = -orthogonal_vector[0] * orth_arrow_length / 2
        orth_arrow_end = orthogonal_vector[0] * orth_arrow_length / 2
        
        orth_arrow_start_y = -orthogonal_vector[1] * orth_arrow_length / 2
        orth_arrow_end_y = orthogonal_vector[1] * orth_arrow_length / 2
        
        # Dodaj linię ortogonalną
        ax.plot([orth_arrow_start, orth_arrow_end], [orth_arrow_start_y, orth_arrow_end_y], 
            'k--', alpha=0.4, linewidth=1, label="Ortogonal axis")
        
        # Dodaj linię łączącą centroidy
        ax.plot([neg_centroid_mds[0], pos_centroid_mds[0]], [neg_centroid_mds[1], pos_centroid_mds[1]], 
            'k:', alpha=0.7, linewidth=1.5)
        
        # Dodaj linię do centroidu zbalansowanego, jeśli istnieje
        if has_balanced_centroid:
            ax.plot([neg_centroid_mds[0], balanced_centroid_mds[0]], [neg_centroid_mds[1], balanced_centroid_mds[1]], 
                'r:', alpha=0.5, linewidth=1)
            ax.plot([balanced_centroid_mds[0], pos_centroid_mds[0]], [balanced_centroid_mds[1], pos_centroid_mds[1]], 
                'g:', alpha=0.5, linewidth=1)
    
    # Dodanie analizowanego tekstu
    ax.scatter([text_position[0]], [text_position[1]], 
            c='green', s=150, alpha=0.9, marker='o', 
            edgecolor='black', linewidth=2, label='Analyzed text')
    
    # Dodanie projekcji na główną oś
    if "projection_on_main_axis" in result and "projection_point" in result:
        projection_point = np.array(result["projection_point"])
        
        # Narysuj projekcję jako punkt
        ax.scatter([projection_point[0]], [projection_point[1]], 
                c='green', s=100, alpha=0.5, marker='x')
        
        # Narysuj linię projekcji
        ax.plot([text_position[0], projection_point[0]], [text_position[1], projection_point[1]], 
            'b--', alpha=0.6, linewidth=1)
    
    # Dodaj linie do centroidów
    if show_centroids:
        ax.plot([text_position[0], neg_centroid_mds[0]], [text_position[1], neg_centroid_mds[1]], 
            'r--', alpha=0.3, linewidth=1)
        ax.plot([text_position[0], pos_centroid_mds[0]], [text_position[1], pos_centroid_mds[1]], 
            'g--', alpha=0.3, linewidth=1)
        if has_balanced_centroid:
            ax.plot([text_position[0], balanced_centroid_mds[0]], [text_position[1], balanced_centroid_mds[1]], 
                'purple', linestyle='--', alpha=0.3, linewidth=1)
        
    # Dodanie etykiety z informacjami o analizowanym tekście
    info_text = []
    
    if "direction_label" in result:
        direction_label = truncate_text(result['direction_label'])
        info_text.append(f"Direction: {direction_label}")


    # Informacje o odległościach
    if "neg_distance" in result and "pos_distance" in result:
        info_text.append(f"Cos. dis. neg: {result['neg_distance']:.4f}")
        info_text.append(f"Cos. dis. pos: {result['pos_distance']:.4f}")
        if has_balanced_centroid and "balanced_distance" in result:
            info_text.append(f"Cos. dis. bal: {result['balanced_distance']:.4f}")
    
    # # Informacje o euklidesowych odległościach
    # if "neg_euklidean_distance" in result and "pos_euklidean_distance" in result:
    #     info_text.append(f"Odl. eukl. neg: {result['neg_euklidean_distance']:.4f}")
    #     info_text.append(f"Odl. eukl. pos: {result['pos_euklidean_distance']:.4f}")
    #     if has_balanced_centroid and "balanced_euklidean_distance" in result:
    #         info_text.append(f"Odl. eukl. bal: {result['balanced_euklidean_distance']:.4f}")
    

    # Informacje o projekcji na głównej osi
    if "projection_on_main_axis" in result:
        info_text.append(f"Main. proj.: {result['projection_on_main_axis']:.4f}")
    
    # Informacje o wartości stress MDS
    if "stress_value" in result:
        info_text.append(f"Stress value: {result['stress_value']:.4f}")
        
    # Połącz wszystkie informacje
    info_str = "\n".join(info_text)
    

    ax.text(0.08,0.08, info_str, 
        fontsize=9, ha='left', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='green', alpha=0.8))
    
    # Dodanie legendy
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Ustawienie równych skal dla osi X i Y
    ax.set_aspect('equal')
    
    # Ustawienie zakresu osi
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Dodanie siatki
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Dodanie osi X i Y przechodzących przez punkt (0,0)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Dodanie opisów osi
    # Dodaj opisy głównej osi polaryzacji
    main_axis_label_x = axis_unit_vector[0] * (max(x_range, y_range) * 0.4)
    main_axis_label_y = axis_unit_vector[1] * (max(x_range, y_range) * 0.4)
    
    # # Skrócone etykiety dla osi polaryzacji
    # if has_balanced_centroid:
    #     axis_label = f"Główna oś polaryzacji\n{neg_label} ↔ {balanced_label} ↔ {pos_label}"
    # else:
    #     axis_label = f"Main polarization ax polaryzacji\n{neg_label} ↔ {pos_label}"
        
    # ax.text(main_axis_label_x, main_axis_label_y, 
    #     axis_label, 
    #     ha='center', va='center', fontsize=10, fontweight='bold', 
    #     bbox=dict(boxstyle="round,pad=0.2", fc='lightyellow', ec="orange", alpha=0.7))
    
    # Zapisanie lub wyświetlenie wykresu
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisany do: {output_path}")
    
    return fig

def plot_polarization_heatmap_mds(results: Dict[str, Dict], figsize: Tuple[int, int] = (10, 6), 
                             title: str = "", 
                             output_path: Optional[str] = None):
    """
    Tworzy wykres typu heatmap pokazujący polaryzację tekstu na wszystkich osiach semantycznych.
    Uwzględnia kategorię zbalansowaną w wizualizacji.
    """
    if not results:
        raise ValueError("Brak wyników do wizualizacji.")
    
    # Przygotowanie danych do heatmapy
    axes_names = []
    normalized_positions = []
    strengths = []
    directions = []
    orthogonal_projections = []
    
    for axis_name, result in results.items():
        axes_names.append(axis_name)
        
        # Upewnij się, że znormalizowane pozycje są w zakresie -1 do 1
        norm_pos = result['normalized_position']
        normalized_positions.append(norm_pos)
        
        # Siła polaryzacji
        strength = result['strength']
        strengths.append(strength)
        
        # Kierunek i projekcja ortogonalna
        # Zmodyfikujmy kod, aby obsługiwał trzy kategorie: -1 (negatywna), 0 (zbalansowana), 1 (pozytywna)
        if 'direction' in result:
            if result['direction'] == 'positive':
                directions.append(1)
            elif result['direction'] == 'balanced':
                directions.append(0)  # Wartość 0 dla zbalansowanej kategorii
            else:  # 'negative'
                directions.append(-1)
        else:
            directions.append(1 if result.get('direction_label') == 'positive' else -1)
            
        orthogonal_projections.append(result.get('orthogonal_projection', 0.0))
    
    # Sortowanie według siły polaryzacji
    sorted_indices = np.argsort(normalized_positions)[::-1]  # Od największej do najmniejszej siły
    
    sorted_axes = [axes_names[i] for i in sorted_indices]
    sorted_positions = [normalized_positions[i] for i in sorted_indices]
    # sorted_strengths = [strengths[i] for i in sorted_indices]
    sorted_directions = [directions[i] for i in sorted_indices]
    # sorted_orthogonal = [orthogonal_projections[i] for i in sorted_indices]
    
    # Tworzenie figury z trzema panelami
    fig, ax1 = plt.subplots(figsize=(figsize[0] * 1.5, figsize[1]))    
    # Panel 1: Heatmapa pozycji znormalizowanych
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green
    norm = plt.Normalize(-2, 2)  # Wymuszamy zakres -1 do 1 dla spójności
    
    # Tworzenie heatmapy
    im = ax1.imshow([sorted_positions], cmap=cmap, norm=norm, aspect='auto')
    
    # Dodanie etykiet osi
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(len(sorted_axes)))
    ax1.set_xticklabels(sorted_axes, rotation=20, ha='right')
    
    # Dodanie kolorowej skali z zakresem od -1 do 1
    cbar = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=0.2)
    # cbar.set_label('Normalized pozition (from -1 to 1)')
    
    # Dodanie wartości na komórkach heatmapy
    for i, pos in enumerate(sorted_positions):
        # Kolor tekstu zależy od wartości bezwzględnej
        color = 'black'
        if abs(pos) > 0.7:
            color = 'white'
        # Dodajmy oznaczenie kategorii obok wartości
        direction_marker = ""
        if sorted_directions[i] == 1:
            direction_marker = "+"  # Pozytywna kategoria
        elif sorted_directions[i] == -1:
            direction_marker = "-"  # Negatywna kategoria
        else:  # 0
            direction_marker = "="  # Zbalansowana kategoria
            
        ax1.text(i, 0, f"{pos:.2f} {direction_marker}", ha='center', va='center', color=color, fontweight='bold')
    
    ax1.set_title(title)
    
    # Dodanie pionowej linii w punkcie zero na skali kolorów
    cbar_ax = cbar.ax
    cbar_ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    # cbar_ax.text(0, 0.5, transform=cbar_ax.transAxes, va='center', ha='center', 
    #             color='black', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

    # # Panel 2: Wykres słupkowy siły polaryzacji
    # bars = ax2.barh(np.arange(len(sorted_positions)), sorted_positions, 
    #               color=[cmap(norm(pos)) for pos in sorted_positions])
    
    # # Dodanie etykiet osi
    # ax2.set_yticks(np.arange(len(sorted_axes)))
    # ax2.set_yticklabels([])
    # ax2.set_xlabel('')
    
    # # Ustaw zakres wykresu słupkowego zawsze od 0 do 1
    # ax2.set_xlim(0, max(sorted_positions) + 0.1)       
    
    # # Dodanie wartości na słupkach
    # for i, position in enumerate(sorted_positions):
    #     ax2.text(min(position + 0.05, 0.95), i, f"{position:.2f}", va='center')
    
    # ax2.set_title('Polarization strength')
    
    # # Dodanie siatki dla lepszej czytelności
    # ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # # Panel 3: Wykres słupkowy składowej ortogonalnej
    # orth_cmap = plt.cm.coolwarm  # Niebieski-Czerwony dla składowej ortogonalnej
    # orth_norm = plt.Normalize(-max(abs(min(sorted_orthogonal)), abs(max(sorted_orthogonal))), 
    #                         max(abs(min(sorted_orthogonal)), abs(max(sorted_orthogonal))))
    
    # bars_orth = ax3.barh(np.arange(len(sorted_orthogonal)), sorted_orthogonal, 
    #                   color=[orth_cmap(orth_norm(orth)) for orth in sorted_orthogonal])
    
    # # Dodanie etykiet osi
    # ax3.set_yticks(np.arange(len(sorted_axes)))
    # ax3.set_yticklabels([])
    # ax3.set_xlabel('')
    
    # # Ustaw zakres wykresu słupkowego od -max do max
    # max_abs_orth = max(abs(min(sorted_orthogonal)), abs(max(sorted_orthogonal)))
    # ax3.set_xlim(-max_abs_orth * 1.1, max_abs_orth * 1.1)
    
    # # Dodanie wartości na słupkach
    # for i, orth in enumerate(sorted_orthogonal):
    #     text_pos = orth + 0.05 * np.sign(orth) * max_abs_orth
    #     ax3.text(text_pos, i, f"{orth:.2f}", va='center')
    
    # ax3.set_title('orthogonal axis')
    
    # # Dodanie siatki i linii zero
    # ax3.grid(True, axis='x', linestyle='--', alpha=0.3)
    # ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Dodanie legendy z uwzględnieniem zbalansowanej kategorii
    neg_patch = Patch(color=cmap(norm(-0.8)), label='Negative')
    bal_patch = Patch(color=cmap(norm(0)), label='Balanced')
    pos_patch = Patch(color=cmap(norm(0.8)), label='Positive')
    legend_elements = [neg_patch, bal_patch, pos_patch]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    # Zapisanie lub wyświetlenie wykresu
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Wykres heatmap zapisany do: {output_path}")
    
    return fig


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import json
from scipy.spatial.distance import pdist, squareform, cosine
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd


class MultiAxisSemanticPolarizer:
    """
    Klasa do analizy polaryzacji semantycznej tekstu na wielu osiach jednocześnie,
    każda oś jest definiowana przez dwa przeciwstawne zbiory słów kluczowych.
    Zoptymalizowana wersja wykonująca pojedyncze rzutowanie MDS dla każdej analizy.
    """
    
    def __init__(self, random_seed=23, embedding_model="mistral:latest", host="localhost", port=11434, normalize_embeddings=True):
        """
        Inicjalizuje analizator wielowymiarowej polaryzacji semantycznej oparty na embedingach z Ollamy.
        
        Args:
            random_seed (int, optional): Ziarno losowości. Domyślnie 42.
            embedding_model (str, optional): Nazwa modelu embedingowego w Ollamie.
            host (str, optional): Host, na którym działa Ollama. Domyślnie "localhost".
            port (int, optional): Port, na którym nasłuchuje API Ollamy. Domyślnie 11434.
            normalize_embeddings (bool, optional): Czy normalizować embeddingi. Domyślnie True.
        """
        # Ustawienie ziarna dla powtarzalności wyników
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Konfiguracja modelu embedingowego Ollama
        self.embedding_model = embedding_model
        self.host = host
        self.port = port
        self.normalize_embeddings = normalize_embeddings
        print(f"Konfiguracja embedingów: model={embedding_model}, host={host}, port={port}")
        
        # Inicjalizacja zmiennych instancji
        self.semantic_features = None
        self.axes = {}  # Słownik osi semantycznych
        self.axes_data = {}  # Dane dla każdej osi
        self.is_trained = False
        
        # Cache embeddingów słów kluczowych
        self.word_embeddings_cache = {}
        
    def set_semantic_features(self, semantic_features: Dict[str, List[str]]):
        """
        Ustawia słownik kategorii semantycznych do analizy polaryzacji.
        
        Args:
            semantic_features (Dict[str, List[str]]): Słownik z kategoriami semantycznymi i ich słowami kluczowymi
                np. {"kategoria1": ["słowo1", "słowo2"], "kategoria2": ["słowo3", "słowo4"]}
        
        Returns:
            self: Obiekt klasy dla umożliwienia łańcuchowania metod
        """
        # Przetwórz semantic_features, aby obsłużyć długie teksty
        processed_features = preprocess_semantic_features(semantic_features)
        self.semantic_features = processed_features
        
        print(f"Ustawiono {len(processed_features)} kategorii semantycznych:")
        for category, phrases in processed_features.items():
            print(f"  {category}: {len(phrases)} fraz")
            for phrase in phrases:
                print(f"    - {phrase[:50]}{'...' if len(phrase) > 50 else ''}")
        
        # Identyfikacja par kategorii tworzących osie
        self.axes = identify_semantic_axes(self.semantic_features)
        
        return self
        
    def _compute_embeddings(self, words_list: List[str]) -> np.ndarray:
        """
        Oblicza embedingi dla listy słów przy użyciu skonfigurowanego API.
        
        Args:
            words_list (List[str]): Lista słów do analizy
            
        Returns:
            numpy.ndarray: Macierz embedingów
        """
        print(f"Obliczanie embeddingów dla {len(words_list)} elementów...")
        
        embeddings = []
        new_words = []
        
        # Sprawdzenie, które słowa są już w cache'u
        for word in words_list:
            if word in self.word_embeddings_cache:
                embeddings.append(self.word_embeddings_cache[word])
            else:
                new_words.append(word)
        
        # Obliczenie embeddingów dla nowych słów
        if new_words:
            print(f"Generowanie embeddingów dla {len(new_words)} nowych elementów...")
            
            new_embeddings = []
            for word in new_words:
                try:
                    # Wybierz właściwe API do generowania embeddingów
                    if OPEN_AI_API:
                        embedding = generate_embedding_openai(text=word, api_key=API_KEY_OPENAI)
                    elif VOYAGE_AI_API:
                        embedding = generate_embedding_voyage(text=word, api_key=API_KEY_VOYAGE)
                    elif GEMINI_AI_API:
                        embedding = generate_embedding_gemini(text=word, api_key=API_KEY_GEMINI)
                    elif OLLAMA:
                        embedding = generate_embedding_ollama(text=word, model=MODEL_OLLAMA)
                    else:
                        embedding = generate_embedding(text=word)
                    new_embeddings.append(embedding)
                    
                    # Dodanie do cache'u
                    self.word_embeddings_cache[word] = embedding
                except Exception as e:
                    print(f"Błąd przy generowaniu embeddingu dla '{word[:50]}...': {str(e)}")
                    # Dodaj pusty embedding w przypadku błędu
                    if len(embeddings) > 0 or len(new_embeddings) > 0:
                        # Użyj takiego samego rozmiaru jak poprzednie embeddingi
                        empty_embedding = np.zeros(
                            len(embeddings[0]) if embeddings else len(new_embeddings[0]))
                    else:
                        # Załóż domyślny rozmiar, jeśli to pierwszy embedding
                        empty_embedding = np.zeros(1536)  # Domyślny rozmiar dla OpenAI
                    new_embeddings.append(empty_embedding)
                    self.word_embeddings_cache[word] = empty_embedding
            
            # Dodanie nowych embeddingów do listy
            if len(new_embeddings) > 0:
                if isinstance(new_embeddings[0], np.ndarray) and isinstance(embeddings[0], list) if embeddings else False:
                    new_embeddings = [e.tolist() for e in new_embeddings]
                embeddings.extend(new_embeddings)
        
        # Konwersja listy list na macierz numpy
        return np.array(embeddings)
    
    def train(self):
        """
        Oblicza embeddingi i przygotowuje dane dla analizy polaryzacji.
        Nie wykonuje rzutowania MDS - to będzie wykonane podczas analizy polaryzacji.
        Dodaje obsługę trzeciej kategorii (zbalansowanej) jako centroid między negatywną i pozytywną.

        Returns:
            self: Obiekt klasy dla umożliwienia łańcuchowania metod
        """
        if not self.semantic_features or not self.axes:
            raise ValueError("Musisz najpierw ustawić kategorie semantyczne. Użyj metody set_semantic_features().")
        
        print(f"Rozpoczynam przygotowanie danych dla {len(self.axes)} osi semantycznych...")
        
        # Przygotuj dane dla każdej osi semantycznej
        for axis_name, axis_info in self.axes.items():
            print(f"\nPrzetwarzanie osi semantycznej: {axis_name}")
            pos_category = axis_info['positive']
            neg_category = axis_info['negative']
            
            pos_words = self.semantic_features[pos_category]
            neg_words = self.semantic_features[neg_category]
            
            # Dodaj obsługę zbalansowanej kategorii, jeśli istnieje
            balanced_words = []
            if 'balanced' in axis_info:
                balanced_category = axis_info['balanced']
                balanced_words = self.semantic_features[balanced_category]
                print(f"  Punkt zbalansowany ({balanced_category}): {len(balanced_words)} fraz")
        
            # Łączenie słów dla tej osi
            axis_words = pos_words + neg_words + balanced_words
            axis_labels = ['pos'] * len(pos_words) + ['neg'] * len(neg_words) + ['balanced'] * len(balanced_words)
            
            # Obliczanie embeddingów
            embeddings = self._compute_embeddings(axis_words)
            print(f"Obliczono embeddingi o wymiarze: {embeddings.shape}")
            
            # Indeksy dla każdego bieguna
            pos_indices = [i for i, label in enumerate(axis_labels) if label == 'pos']
            neg_indices = [i for i, label in enumerate(axis_labels) if label == 'neg']
            balanced_indices = [i for i, label in enumerate(axis_labels) if label == 'balanced']
            
            # Obliczanie centroidów w przestrzeni embeddingów
            pos_embeddings = embeddings[pos_indices] if pos_indices else np.array([])
            neg_embeddings = embeddings[neg_indices] if neg_indices else np.array([])
            balanced_embeddings = embeddings[balanced_indices] if balanced_indices else np.array([])
            
            # Obliczanie centroidów
            # pos_centroid = np.mean(pos_embeddings, axis=0) if len(pos_embeddings) > 0 else np.zeros(embeddings.shape[1])
            # neg_centroid = np.mean(neg_embeddings, axis=0) if len(neg_embeddings) > 0 else np.zeros(embeddings.shape[1])


            pos_centroid = pos_embeddings[0]
            neg_centroid = neg_embeddings[0]
            
            # Jeśli brak zbalansowanych słów, oblicz centroid jako średnią między negatywnym i pozytywnym
            if len(balanced_embeddings) == 0 and len(pos_embeddings) > 0 and len(neg_embeddings) > 0:
                print("  Tworzenie centroidu zbalansowanego jako średnia między negatywnym i pozytywnym")
                #balanced_centroid = (neg_centroid + pos_centroid) / 2
                balanced_centroid = np.mean([neg_centroid, pos_centroid], axis=0)

                #balanced_centroid = None # 
            # else:
            #     balanced_centroid = np.mean(balanced_embeddings, axis=0) if len(balanced_embeddings) > 0 else None
            
            # Zapisz dane osi
            axis_data = {
                "axis_name": axis_name,
                "pole_labels": {
                    "positive": pos_category,
                    "negative": neg_category
                },
                "all_words": axis_words,
                "word_labels": axis_labels,
                "embeddings": embeddings.tolist(),
                "neg_centroid": neg_centroid.tolist(),
                "pos_centroid": pos_centroid.tolist(),
                "pos_indices": pos_indices,
                "neg_indices": neg_indices,
                "balanced_indices": balanced_indices
            }
            
            # Dodaj dane centroidu zbalansowanego, nawet jeśli został utworzony automatycznie
            if balanced_centroid is not None:
                if 'balanced' in axis_info:
                    axis_data["pole_labels"]["balanced"] = axis_info['balanced']
                else:
                    # Utwórz sztuczną etykietę dla automatycznie utworzonego centroidu
                    axis_data["pole_labels"]["balanced"] = f"{axis_name}_balanced"
                
                axis_data["balanced_centroid"] = balanced_centroid.tolist()
            
            self.axes_data[axis_name] = axis_data
            
            print(f"  Biegun pozytywny ({pos_category}): {len(pos_words)} fraz")
            print(f"  Biegun negatywny ({neg_category}): {len(neg_words)} fraz")
            if balanced_centroid is not None:
                if len(balanced_embeddings) > 0:
                    print(f"  Punkt zbalansowany: {len(balanced_words)} fraz (z danych)")
                else:
                    print(f"  Punkt zbalansowany: utworzony automatycznie jako centroid")
        
        self.is_trained = True
        print(f"\nPrzygotowanie danych zakończone dla {len(self.axes)} osi semantycznych")
        
        return self
    
    def load_model(self, filepath):
        """
        Ładuje model z pliku JSON.
        
        Args:
            filepath (str): Ścieżka do pliku.
            
        Returns:
            Dict: Dane modelu.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Sprawdzenie, czy plik zawiera wymagane pola
            required_fields = ["semantic_features", "axes", "axes_data"]
            for field in required_fields:
                if field not in model_data:
                    print(f"Ostrzeżenie: Brak wymaganego pola '{field}' w pliku modelu.")
            
            # Odtworzenie zmiennych instancji z wczytanego modelu
            self.semantic_features = model_data.get("semantic_features", {})
            self.axes = model_data.get("axes", {})
            self.axes_data = model_data.get("axes_data", {})
            
            # Konfiguracja API, jeśli jest dostępna
            if "api_config" in model_data:
                config = model_data["api_config"]
                self.embedding_model = config.get("model", self.embedding_model)
                self.host = config.get("host", self.host)
                self.port = config.get("port", self.port)
                self.normalize_embeddings = config.get("normalize_embeddings", self.normalize_embeddings)
            
            self.is_trained = len(self.axes_data) > 0
            
            print(f"Model wczytany z pliku: {filepath}")
            print(f"Liczba kategorii semantycznych: {len(self.semantic_features)}")
            print(f"Liczba osi semantycznych: {len(self.axes)}")
            
            return model_data
            
        except Exception as e:
            print(f"Błąd podczas wczytywania modelu: {str(e)}")
            return None
    
    def save_model(self, filepath):
        """
        Zapisuje model do pliku JSON.
        
        Args:
            filepath (str): Ścieżka do pliku.
            
        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym przypadku.
        """
        if not self.is_trained:
            print("Ostrzeżenie: Model nie został wytrenowany.")
        
        try:
            # Przygotowanie danych modelu
            model_data = {
                "semantic_features": self.semantic_features,
                "axes": self.axes,
                "axes_data": self.axes_data,
                "api_config": {
                    "model": self.embedding_model,
                    "host": self.host,
                    "port": self.port,
                    "normalize_embeddings": self.normalize_embeddings
                }
            }
            
            # Zapisanie do pliku
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
                
            print(f"Model zapisany do pliku: {filepath}")
            return True
            
        except Exception as e:
            print(f"Błąd podczas zapisywania modelu: {str(e)}")
            return False
        
    def calculate_polarization_mds(self, text: str = None, embedding: np.ndarray = None, axis_name: str = None) -> Dict:
        """
        Oblicza polaryzację tekstu względem wybranej lub wszystkich osi semantycznych.
        Dla każdej osi wykonywane jest pojedyncze rzutowanie MDS uwzględniające 
        wszystkie słowa kluczowe, centroidy (w tym zbalansowany) oraz badany tekst.
        
        Args:
            text (str, optional): Tekst do analizy polaryzacji
            embedding (numpy.ndarray, optional): Embedding do analizy (alternatywnie do tekstu)
            axis_name (str, optional): Nazwa konkretnej osi do analizy. Jeśli None, analizowane są wszystkie osie.
                
        Returns:
            Dict: Wyniki analizy polaryzacji dla wszystkich lub wybranej osi
        """
        if text:
            print(f"Analizuję tekst: {text[:100]}...")
        
        if not self.is_trained:
            raise ValueError("Model nie został wytrenowany. Użyj metody train().")
        
        # Sprawdź, czy podano tekst lub embedding
        if embedding is None and text is None:
            raise ValueError("Musisz podać tekst lub embedding do analizy polaryzacji")
        
        # Jeśli podano tekst, wygeneruj embedding przy użyciu odpowiedniego API
        if embedding is None and text is not None:
            print(f"Generowanie embeddingu dla tekstu do analizy polaryzacji...")
            if OPEN_AI_API:
                embedding = generate_embedding_openai(text=text, api_key=API_KEY_OPENAI)
            elif VOYAGE_AI_API:
                embedding = generate_embedding_voyage(text=text, api_key=API_KEY_VOYAGE)
            elif GEMINI_AI_API:
                embedding = generate_embedding_gemini(text=text, api_key=API_KEY_GEMINI)
            elif OLLAMA:
                embedding = generate_embedding_ollama(text=text, model=MODEL_OLLAMA)
            else:
                embedding = generate_embedding(
                    text=text,
                )
        
        # Określenie, które osie analizować
        axes_to_analyze = [axis_name] if axis_name else list(self.axes_data.keys())
        
        # Słownik na wyniki dla wszystkich osi
        results = {}
        
        for axis in axes_to_analyze:
            if axis not in self.axes_data:
                print(f"Ostrzeżenie: Oś '{axis}' nie istnieje. Dostępne osie: {list(self.axes_data.keys())}")
                continue
                
            print(f"\nAnaliza polaryzacji dla osi: {axis}")
            axis_data = self.axes_data[axis]
            
            # Normalizacja embeddingu, jeśli potrzeba
            adjusted_embedding_norm = np.array(embedding)
            
            # Pobierz centroidy z osi
            neg_centroid = np.array(axis_data["neg_centroid"])
            pos_centroid = np.array(axis_data["pos_centroid"])
            
            # Sprawdź, czy mamy zbalansowany centroid
            has_balanced_centroid = "balanced_centroid" in axis_data
            balanced_centroid = np.array(axis_data["balanced_centroid"]) if has_balanced_centroid else None
            
            # 1. Wykonaj jedno rzutowanie MDS dla wszystkich słów i analizowanego tekstu
            # Połącz wszystkie embeddingi: słowa kluczowe + centroidy + analizowany tekst
            all_embeddings = []
            
            # Dodaj embeddingi słów kluczowych
            original_embeddings = np.array(axis_data["embeddings"])
            all_embeddings.append(original_embeddings)
            
            # Dodaj centroidy i analizowany tekst
            centroid_embeddings = [neg_centroid, pos_centroid]
            if has_balanced_centroid:
                centroid_embeddings.append(balanced_centroid)
            
            # Dodaj analizowany tekst
            all_embeddings.append(np.array(centroid_embeddings + [adjusted_embedding_norm]))
            
            # Połącz wszystkie embeddingi
            all_embeddings = np.vstack(all_embeddings)
            
            # Przygotuj etykiety dla wszystkich punktów
            word_labels = axis_data["word_labels"]
            centroid_labels = ["neg_centroid", "pos_centroid"]
            if has_balanced_centroid:
                centroid_labels.append("balanced_centroid")
            all_labels = word_labels + centroid_labels + ["analyzed_text"]
            
            # Indeksy dla punktów specjalnych
            n_words = len(original_embeddings)
            neg_centroid_idx = n_words
            pos_centroid_idx = n_words + 1
            balanced_centroid_idx = n_words + 2 if has_balanced_centroid else None
            analyzed_text_idx = n_words + len(centroid_labels)
            
            # Oblicz macierz odległości dla wszystkich punktów
            print(f"Obliczanie macierzy odległości kosinusowych...")
            all_distances = compute_cosine_distance_matrix(all_embeddings)
            
            # Początkowy wymiar danych
            original_dim = all_embeddings.shape[1]
            
            # Początkowa macierz odległości
            current_distances = all_distances
            n_components = 2 
            # Przeprowadzamy kolejne etapy redukcji
            print(f"Etap: Redukcja do {n_components} wymiarów...")
            mds = MDS(n_components=n_components, 
                    dissimilarity='precomputed', 
                    random_state=self.random_seed,
                    eps=1e-12,
                    max_iter=15000,  # Mniej iteracji dla etapów pośrednich
                    n_init=100)  # Więcej inicjalizacji dla ostatniego etapu
                
            # Wykonujemy MDS
            coords = mds.fit_transform(current_distances)
            
            # Raportujemy wartość stress
            print(f"  Wartość stress: {mds.stress_:.4f}")
            
            # Finalne współrzędne MDS
            all_mds_coords = coords
                    
            # 2. Wycentruj przestrzeń MDS tak, by środek był w [0, 0] ok
            centroid_center = (all_mds_coords[neg_centroid_idx] + all_mds_coords[pos_centroid_idx]) / 2
           
            all_mds_coords = all_mds_coords - centroid_center
            
            # 3. Pobierz współrzędne MDS dla wszystkich punktów po wycentrowaniu ok
            word_mds_coords = all_mds_coords[:n_words]
            neg_centroid_mds = all_mds_coords[neg_centroid_idx]
            pos_centroid_mds = all_mds_coords[pos_centroid_idx]
            balanced_centroid_mds = all_mds_coords[balanced_centroid_idx] if has_balanced_centroid else None
            text_position = all_mds_coords[analyzed_text_idx]
            
            # 4. Oblicz wektor osi (kierunek od negatywnego do pozytywnego centroidu)
            axis_vector = pos_centroid_mds - neg_centroid_mds
            axis_length = np.linalg.norm(axis_vector)
            
            # Normalizacja wektora osi
            if axis_length > 1e-10:
                axis_unit_vector = axis_vector / axis_length
            else:
                axis_unit_vector = np.array([1.0, 0.0])
            

            norma_wektora = np.linalg.norm(axis_unit_vector)
            print(f"Norma wektora: {norma_wektora}")
            # 5. Oblicz wektor ortogonalny (prostopadły) do głównej osi
            orthogonal_vector = np.array([-axis_unit_vector[1], axis_unit_vector[0]])
            
            # 6. Oblicz projekcję punktu tekstu na główną oś
            projection_on_main_axis = np.dot(text_position - neg_centroid_mds, axis_unit_vector)
            
            # 7. Oblicz projekcję ortogonalną (prostopadłą do głównej osi)
            orthogonal_projection = np.dot(text_position - neg_centroid_mds, orthogonal_vector)
            
            # 8. Oblicz punkt projekcji na głównej osi
            projection_point = neg_centroid_mds + projection_on_main_axis * axis_unit_vector
            
            # 9. Oblicz odległości od projekcji do centroidów
            distance_to_neg = np.linalg.norm(projection_point - neg_centroid_mds)
            distance_to_pos = np.linalg.norm(projection_point - pos_centroid_mds)


            print(f"Axis length: {axis_length}")
            print(f"---> Projekcja NEG: {distance_to_neg}")
            print(f"---> Projekcja POS: {distance_to_pos}")



            # 11. Oblicz znormalizowaną pozycję na osi
            if axis_length > 1e-10:
                # Znormalizowana pozycja od -1 (neg) do 1 (pos)
                # Dzielimy przez długość osi i przesuwamy zakres
                normalized_position = 2 * (projection_on_main_axis / axis_length) - 1
            else:
                normalized_position = 0.0

            print(f"--> Normalized position: {normalized_position}")


            # Dodaj odległość do zbalansowanego centroidu, jeśli istnieje
            if has_balanced_centroid:
                distance_to_balanced = np.linalg.norm(projection_point - balanced_centroid_mds)
                print(f"---> Projekcja BAL: {distance_to_balanced}")

                neg_distance = all_distances[neg_centroid_idx, analyzed_text_idx]
                pos_distance = all_distances[pos_centroid_idx, analyzed_text_idx]
                balanced_distance = all_distances[balanced_centroid_idx, analyzed_text_idx]
                sensitivity_barier = SENSITIVITY #SENSIVITY - czułość
                # sensitivity_barier = 0
                print(f"Sensitivity barier: {sensitivity_barier}")

                if sensitivity_barier > abs(normalized_position):
                    print(f"Krótki dystans do BALANCED!!!! {abs(normalized_position)} niz czułość: {sensitivity_barier}")
                # sensitivity_barier = (abs(neg_distance) + abs(pos_distance))*0.5 #SENSIVITY - czułość
                # if sensitivity_barier < abs(balanced_distance):
                #     print(f"Długi dystans do BALANCED!!!! {abs(balanced_distance)} niz czułość: {sensitivity_barier}")
            
            # 10. Określ kierunek na podstawie odległości (który centroid jest bliżej)
            if has_balanced_centroid:
                # Jeśli mamy trzy punkty, określamy czy tekst jest bliżej negatywnego, zbalansowanego czy pozytywnego
                distances = [
                    ("negative", distance_to_neg, axis_data["pole_labels"]["negative"]),
                    ("balanced", distance_to_balanced, axis_data["pole_labels"]["balanced"]),
                    ("positive", distance_to_pos, axis_data["pole_labels"]["positive"])
                ]
                closest = min(distances, key=lambda x: x[1])
                direction = closest[0]
                direction_label = closest[2]
                
                # Sprawdzamy czy klasyfikacja to "balanced" i czy wielkość bezwzględna > 10
                if sensitivity_barier > abs(normalized_position):
                    print(f"Przypisz kategorię BALANCED -  Normalized Position: {abs(balanced_distance)} < Sensitivity Barirer: {sensitivity_barier}")
                    direction = "balanced"
                    direction_label = axis_data["pole_labels"]["balanced"]
                else:
                    # Standardowe określenie kierunku (pozytywny/negatywny)
                    if distance_to_pos < distance_to_neg:
                        direction = "positive"
                        direction_label = axis_data["pole_labels"]["positive"]
                    else:
                        direction = "negative"
                        direction_label = axis_data["pole_labels"]["negative"]
            else:
                # Standardowe określenie kierunku (pozytywny/negatywny)
                if distance_to_pos < distance_to_neg:
                    direction = "positive"
                    direction_label = axis_data["pole_labels"]["positive"]
                else:
                    direction = "negative"
                    direction_label = axis_data["pole_labels"]["negative"]
            
            
            
            #12. Oblicz siłę polaryzacji
            if has_balanced_centroid:
                # Dla trzech punktów, obliczamy odległość od punktu zbalansowanego
                # i normalizujemy w stosunku do długości osi
                balanced_point_distance = np.linalg.norm(text_position - balanced_centroid_mds)
                min_distance = min(distance_to_neg, distance_to_balanced, distance_to_pos)
                total_distance = distance_to_neg + distance_to_balanced + distance_to_pos
                
                # Siła polaryzacji jest tym większa, im dalej od punktu zbalansowanego
                # i im bliżej do jednego z biegunów
                if direction == "balanced":
                    pass
                    strength = None
                    # # Jeśli jesteśmy najbliżej punktu zbalansowanego, siła polaryzacji powinna być niska
                    # strength = 1 - (3 * min_distance / total_distance) if total_distance > 1e-10 else 0.0
                    # strength = strength * 0.5  # Redukujemy siłę dla punktu zbalansowanego
                else:
                    # Jeśli jesteśmy bliżej jednego z biegunów, siła polaryzacji powinna być wyższa
                  #  strength = 1 - (3 * min_distance / total_distance) if total_distance > 1e-10 else 0.0
                    strength = None

            else:
                pass
            # #Standardowa siła polaryzacji dla dwóch punktów
            #     total_distance = distance_to_neg + distance_to_pos
            #     if total_distance > 1e-10:
            #         strength = abs(distance_to_neg - distance_to_pos) / total_distance
            #     else:
            #         strength = 0.0
            
            # 13. Oblicz odległości euklidesowe w przestrzeni MDS
            neg_euklidean_distance = np.linalg.norm(text_position - neg_centroid_mds)
            pos_euklidean_distance = np.linalg.norm(text_position - pos_centroid_mds)
            if has_balanced_centroid:
                balanced_euklidean_distance = np.linalg.norm(text_position - balanced_centroid_mds)
            
            # 14. Oblicz odległości kosinusowe w oryginalnej przestrzeni embeddingów
            neg_distance = all_distances[neg_centroid_idx, analyzed_text_idx]
            pos_distance = all_distances[pos_centroid_idx, analyzed_text_idx]
            if has_balanced_centroid:
                balanced_distance = all_distances[balanced_centroid_idx, analyzed_text_idx]
            
            # 15. Przygotuj wynik
            pos_indices = axis_data["pos_indices"]
            neg_indices = axis_data["neg_indices"]
            balanced_indices = axis_data.get("balanced_indices", [])
            
            result = {
                "axis_name": axis,
                "position_2d": text_position.tolist(),
                "projection_on_main_axis": float(projection_on_main_axis),
                "projection_point": projection_point.tolist(),
                "orthogonal_projection": float(orthogonal_projection),
                "normalized_position": float(normalized_position),
                "direction": direction,
                "direction_label": direction_label,
                "strength": None,
                "axis_length": float(axis_length),
                "distance_to_neg": float(distance_to_neg),
                "distance_to_pos": float(distance_to_pos),
                "neg_distance": float(neg_distance),
                "pos_distance": float(pos_distance),
                "neg_euklidean_distance": float(neg_euklidean_distance),
                "pos_euklidean_distance": float(pos_euklidean_distance),
                "neg_centroid_mds_2d": neg_centroid_mds.tolist(),
                "pos_centroid_mds_2d": pos_centroid_mds.tolist(),
                "axis_unit_vector": axis_unit_vector.tolist(),
                "orthogonal_vector": orthogonal_vector.tolist(),
                "mds_coords_2d": word_mds_coords.tolist(),
                "pole_labels": axis_data["pole_labels"],
                "all_words": axis_data["all_words"],
                "word_labels": axis_data["word_labels"],
                "stress_value": float(mds.stress_)
            }
            
            # Dodaj informacje o zbalansowanym centroidzie, jeśli istnieje
            if has_balanced_centroid:
                result["balanced_centroid_mds_2d"] = balanced_centroid_mds.tolist()
                result["distance_to_balanced"] = float(distance_to_balanced)
                result["balanced_distance"] = float(balanced_distance)
                result["balanced_euklidean_distance"] = float(balanced_euklidean_distance)
            
            # 16. Wyświetlenie wyniku
            print(f"  Pozycja MDS 2D: [{text_position[0]:.4f}, {text_position[1]:.4f}]")
            print(f"  Projekcja na główną oś: {projection_on_main_axis:.4f}")
            print(f"  Odległość projekcji do centroidu negatywnego: {distance_to_neg:.4f}")
            print(f"  Odległość projekcji do centroidu pozytywnego: {distance_to_pos:.4f}")
            if has_balanced_centroid:
                print(f"  Odległość projekcji do centroidu zbalansowanego: {distance_to_balanced:.4f}")
            print(f"  Kierunek polaryzacji: {direction} ({direction_label})")
            # print(f"  Siła polaryzacji: {strength:.4f}")
            print(f"  Znormalizowana pozycja: {normalized_position:.4f} (od -1 do 1)")
            print(f"  Składowa ortogonalna: {orthogonal_projection:.4f}")
            print(f"  Odległość kosinusowa do centroidu negatywnego: {neg_distance:.4f}")
            print(f"  Odległość kosinusowa do centroidu pozytywnego: {pos_distance:.4f}")
            if has_balanced_centroid:
                print(f"  Odległość kosinusowa do centroidu zbalansowanego: {balanced_distance:.4f}")
            print(f"  Wartość stress MDS: {mds.stress_}")
            
            # Dodanie wyniku do słownika
            results[axis] = result
        
        return results