from pydantic import BaseModel
from typing import Optional, List

class ExtractSchema(BaseModel):
    """Schemat do ekstrakcji danych z treści stron"""
    text: str
    is_democratic: bool
    is_republican: bool
    has_partisan_language: bool
    mentioned_topics: List[str]

class PolarizationResult(BaseModel):
    """Model do przechowywania wyników analizy polaryzacji"""
    title: Optional[str] = None
    url: str
    snippet: str
    source: Optional[str] = None
    position: Optional[int] = None
    search_query: str
    category: str
    perspective: str
    collection_timestamp: str
    content_type: Optional[str] = None
    date: Optional[str] = None
    is_democratic: Optional[bool] = None
    is_republican: Optional[bool] = None
    has_partisan_language: Optional[bool] = None
    mentioned_topics: Optional[List[str]] = None






from firecrawl import JsonConfig

# Konfiguracja API
SERPER_API_KEY = ''
FIRECRAWL_API_KEY = ''

# Konfiguracja dla Firecrawl
JSON_CONFIG = JsonConfig(
    schema=ExtractSchema
)

# Ustawienia pobierania
DEFAULT_RESULTS_PER_QUERY = 10
DEFAULT_DELAY_BETWEEN_QUERIES = 1.2
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT = 120000  # ms





import http.client
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import urllib.parse

class SerperPolarizationCollector:
    """Klasa do systematycznego zbierania danych o polaryzacji za pomocą Serper API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_connection = "google.serper.dev"
        self.results = {
            'democratic_perspective': [],
            'republican_perspective': [],
            'neutral_analysis': [],
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'api_used': 'serper.dev',
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0
            }
        }
    
    # Zorganizowane zestawy zapytań dla badania polaryzacji
    POLARIZATION_QUERIES = { 'democratic_perspective': [
        "Democratic policies on expanding voting access",
        "Progressive approach to economic equality",
        "Democratic climate policy and environmental justice",
        "Liberal framework for diversity and inclusion",
        "Universal healthcare proposals from Democrats",
        "Democratic position on reproductive healthcare access",
        "Immigration reform priorities under Democratic leadership",
        "Public education funding in Democratic platforms",
        "Progressive taxation models for wealth distribution",
        "Democratic approach to gun violence prevention",
        "LGBTQ+ rights legislation supported by Democrats",
        "Democratic environmental regulations and green economy",
        "Criminal justice reform from Democratic perspective",
        "Democratic labor policy and union protections",
        "Science-based public health initiatives by Democrats"
    ],

    'republican_perspective': [
        "Republican interpretation of constitutional liberties",
        "Conservative economic growth and deregulation policies",
        "Republican limited government philosophy",
        "Traditional values in Republican policy platforms",
        "Religious liberty protections advocated by Republicans",
        "Market-based healthcare reforms from Republicans",
        "Republican border security and immigration enforcement",
        "School choice initiatives and education freedom",
        "Republican tax reduction and fiscal responsibility",
        "Second Amendment protections in Republican platform",
        "Republican energy independence and domestic production",
        "Small business policies from Republican perspective",
        "Law enforcement support in Republican governance",
        "Republican position on education curriculum oversight",
        "American sovereignty in Republican foreign policy"
    ],

    # 'opposition_criticism': {
    #     'democrats_criticizing_republicans': [
    #         "Democratic critique of Republican voting legislation",
    #         "Progressive analysis of conservative economic impact",
    #         "Democratic response to Republican climate positions",
    #         "Liberal concerns about right-wing populism",
    #         "Democratic criticism of Republican healthcare policies",
    #         "Reproductive rights advocates on conservative legislation",
    #         "Democratic perspective on Republican immigration enforcement",
    #         "Public education advocates on school privatization"
    #     ],
    #     'republicans_criticizing_democrats': [
    #         "Conservative critique of progressive economic policies",
    #         "Republican perspective on liberal social values",
    #         "Conservative analysis of progressive criminal justice reform",
    #         "Republican criticism of Democratic spending priorities",
    #         "Conservative assessment of Democratic border policies",
    #         "Republican concerns about progressive education curriculum",
    #         "Conservative critique of green energy transition",
    #         "Republican perspective on Democratic national security approach"
    #     ]
    # },

    'contemporary_polarizing_issues': [
        "Democratic vs Republican approaches to AI regulation",
        "Partisan divide on social media content moderation",
        "Political polarization in pandemic response policies",
        "Party differences on election security vs voter access",
        "Ideological divide on transgender rights legislation",
        "Partisan approaches to China and global competition",
        "Democratic and Republican positions on student debt",
        "Political division on cryptocurrency regulation",
        "Partisan perspectives on police reform initiatives",
        "Party differences on Supreme Court reform proposals"
    ],

}

    
    def _make_search_request(self, query: str, num_results: int = 10) -> Optional[Dict]:
        """Wykonuje pojedyncze zapytanie do Serper API"""
        try:
            conn = http.client.HTTPSConnection(self.base_connection)
            
            payload = json.dumps({
                "q": query,
                "num": num_results,  # Liczba wyników
                "hl": "en",         # Język
                "gl": "us"          # Geolokalizacja (USA)
            })
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            conn.close()
            
            if res.status == 200:
                return json.loads(data.decode("utf-8"))
            else:
                print(f"❌ Błąd API {res.status} dla zapytania: {query}")
                return None
                
        except Exception as e:
            print(f"❌ Wyjątek dla zapytania '{query}': {e}")
            return None
    
    def _process_search_results(self, results: Dict, query: str, category: str, perspective: str) -> List[Dict]:
        """Przetwarza wyniki wyszukiwania i dodaje metadane"""
        processed_results = []
        
        # Przetwarzaj organiczne wyniki
        for item in results.get('organic', []):
            processed_item = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'source': item.get('source', ''),
                'position': item.get('position', 0),
                'search_query': query,
                'category': category,
                'perspective': perspective,
                'collection_timestamp': datetime.now().isoformat()
            }
            processed_results.append(processed_item)
        
        # Przetwarzaj news results jeśli istnieją
        for news_item in results.get('news', []):
            processed_item = {
                'title': news_item.get('title', ''),
                'url': news_item.get('link', ''),
                'snippet': news_item.get('snippet', ''),
                'source': news_item.get('source', ''),
                'date': news_item.get('date', ''),
                'search_query': query,
                'category': category,
                'perspective': perspective,
                'content_type': 'news',
                'collection_timestamp': datetime.now().isoformat()
            }
            processed_results.append(processed_item)
        
        return processed_results
    
    def collect_polarization_data(
        self, 
        categories: List[str] = None,
        results_per_query: int = 3,
        delay_between_queries: float = 1.0,
        max_retries: int = 3,
        preserve_full_text: bool = True
    ):
        """
        Zbiera systematyczne dane o polaryzacji politycznej
        
        Args:
            categories: kategorie do zbadania
            results_per_query: liczba wyników na zapytanie
            delay_between_queries: opóźnienie między zapytaniami
            max_retries: maksymalna liczba ponownych prób
            preserve_full_text: czy zachowywać pełne teksty (True = bez obcinania)
        """
        
        if categories is None:
            categories = ['democratic_perspective', 'republican_perspective']
        
        print(f"🔍 Rozpoczynam zbieranie danych polaryzacyjnych...")
        print(f"📊 Kategorie: {', '.join(categories)}")
        
        # Zbieranie danych demokratycznych
        if 'democratic_perspective' in categories:
            print(f"\n🔵 Zbieranie perspektywy demokratycznej...")
            self._collect_category_data(
                self.POLARIZATION_QUERIES['democratic_perspective'],
                'democratic_perspective',
                'pro_democratic',
                results_per_query,
                delay_between_queries,
                max_retries
            )
        
        # Zbieranie danych republikańskich
        if 'republican_perspective' in categories:
            print(f"\n🔴 Zbieranie perspektywy republikańskiej...")
            self._collect_category_data(
                self.POLARIZATION_QUERIES['republican_perspective'],
                'republican_perspective', 
                'pro_republican',
                results_per_query,
                delay_between_queries,
                max_retries
            )
        
        # Zbieranie krytyki przeciwników
        if 'opposition_criticism' in categories:
            print(f"\n⚔️ Zbieranie krytyki przeciwników...")
            
            print("  🔵➡️🔴 Demokraci krytykujący republikanów...")
            self._collect_category_data(
                self.POLARIZATION_QUERIES['opposition_criticism']['democrats_criticizing_republicans'],
                'democratic_perspective',
                'anti_republican',
                results_per_query,
                delay_between_queries,
                max_retries
            )
            
            print("  🔴➡️🔵 Republikanie krytykujący demokratów...")
            self._collect_category_data(
                self.POLARIZATION_QUERIES['opposition_criticism']['republicans_criticizing_democrats'],
                'republican_perspective',
                'anti_democratic', 
                results_per_query,
                delay_between_queries,
                max_retries
            )
        
        # Zbieranie analiz porównawczych
        if 'comparative_analysis' in categories:
            print(f"\n⚖️ Zbieranie analiz porównawczych...")
            self._collect_category_data(
                self.POLARIZATION_QUERIES['comparative_analysis'],
                'neutral_analysis',
                'comparative',
                results_per_query,
                delay_between_queries,
                max_retries
            )
        
        # Zapisanie wyników
        self._save_comprehensive_results(preserve_full_text)
    
    def _collect_category_data(
        self, 
        queries: List[str], 
        result_category: str,
        subcategory: str,
        results_per_query: int,
        delay: float,
        max_retries: int
    ):
        """Zbiera dane dla konkretnej kategorii zapytań"""
        
        for query in queries:
            self.results['metadata']['total_queries'] += 1
            
            # Logika ponownych prób
            for attempt in range(max_retries):
                try:
                    print(f"    🔍 '{query}' (próba {attempt + 1}/{max_retries})")
                    
                    search_results = self._make_search_request(query, results_per_query)
                    
                    if search_results:
                        processed_results = self._process_search_results(
                            search_results, query, subcategory, result_category
                        )
                        
                        self.results[result_category].extend(processed_results)
                        self.results['metadata']['successful_queries'] += 1
                        
                        print(f"      ✅ Zebrano {len(processed_results)} wyników")
                        break
                    else:
                        if attempt == max_retries - 1:
                            self.results['metadata']['failed_queries'] += 1
                            print(f"      ❌ Nie udało się po {max_retries} próbach")
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.results['metadata']['failed_queries'] += 1
                        print(f"      ❌ Błąd: {e}")
                    else:
                        print(f"      ⚠️ Próba {attempt + 1} nieudana, ponawianie...")
                
                # Opóźnienie między próbami
                time.sleep(delay)
            
            # Opóźnienie między zapytaniami
            time.sleep(delay)
    
    def _save_comprehensive_results(self, preserve_full_text: bool = True):
        """Zapisuje kompleksowe wyniki z analizą"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Aktualizuj statystyki końcowe
        self.results['metadata'].update({
            'democratic_results': len(self.results['democratic_perspective']),
            'republican_results': len(self.results['republican_perspective']),
            'neutral_results': len(self.results['neutral_analysis']),
            'total_results': sum([
                len(self.results['democratic_perspective']),
                len(self.results['republican_perspective']),
                len(self.results['neutral_analysis'])
            ])
        })
        
        # Stwórz folder na wyniki
        results_folder = Path(f"serper_polarization_{timestamp}")
        results_folder.mkdir(exist_ok=True)
        
        # Zapisz główny plik JSON
        main_file = results_folder / "complete_polarization_data.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Zapisz osobne pliki dla każdej perspektywy
        for perspective in ['democratic_perspective', 'republican_perspective', 'neutral_analysis']:
            if self.results[perspective]:
                perspective_file = results_folder / f"{perspective}.json"
                with open(perspective_file, 'w', encoding='utf-8') as f:
                    json.dump(self.results[perspective], f, indent=2, ensure_ascii=False)
        
        # Stwórz raport analizy
        self._create_analysis_report(results_folder, timestamp)
        
        # Stwórz pliki CSV dla łatwiejszej analizy
        self._create_csv_exports(results_folder, truncate_text=not preserve_full_text)
        
        print(f"\n✨ Zakończono zbieranie danych!")
        print(f"📁 Wyniki zapisane w folderze: {results_folder}")
        print(f"📊 Statystyki:")
        print(f"   • Łączne zapytania: {self.results['metadata']['total_queries']}")
        print(f"   • Udane zapytania: {self.results['metadata']['successful_queries']}")
        print(f"   • Nieudane zapytania: {self.results['metadata']['failed_queries']}")
        print(f"   • Łączne wyniki: {self.results['metadata']['total_results']}")
    
    def _create_analysis_report(self, folder: Path, timestamp: str):
        """Tworzy szczegółowy raport analizy"""
        report_file = folder / "analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RAPORT ANALIZY POLARYZACJI POLITYCZNEJ\n")
            f.write("=" * 50 + "\n\n")
            
            meta = self.results['metadata']
            f.write(f"Data zbierania: {meta['collection_date']}\n")
            f.write(f"API użyte: {meta['api_used']}\n")
            f.write(f"Łączne zapytania: {meta['total_queries']}\n")
            f.write(f"Udane zapytania: {meta['successful_queries']}\n")
            f.write(f"Nieudane zapytania: {meta['failed_queries']}\n")
            f.write(f"Sukces rate: {(meta['successful_queries']/meta['total_queries']*100):.1f}%\n\n")
            
            # Statystyki perspektyw
            f.write("ROZKŁAD PERSPEKTYW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Perspektywa demokratyczna: {meta['democratic_results']} wyników\n")
            f.write(f"Perspektywa republikańska: {meta['republican_results']} wyników\n")
            f.write(f"Analizy neutralne: {meta['neutral_results']} wyników\n\n")
            
            # Analiza źródeł
            f.write("ANALIZA ŹRÓDEŁ:\n")
            f.write("-" * 15 + "\n")
            
            for perspective_name, perspective_data in [
                ('Demokratyczna', self.results['democratic_perspective']),
                ('Republikańska', self.results['republican_perspective']),
                ('Neutralna', self.results['neutral_analysis'])
            ]:
                if perspective_data:
                    f.write(f"\n{perspective_name}:\n")
                    sources = {}
                    for item in perspective_data:
                        source = item.get('source', 'Unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
                        f.write(f"  {source}: {count} wyników\n")
    
    def _create_csv_exports(self, folder: Path, truncate_text: bool = False, max_length: int = 500):
        """
        Tworzy eksporty CSV dla łatwiejszej analizy w Excel/innych narzędziach
        
        Args:
            folder: folder docelowy
            truncate_text: czy obcinać długie teksty
            max_length: maksymalna długość tekstu (jeśli truncate_text=True)
        """
        import csv
        
        for perspective in ['democratic_perspective', 'republican_perspective', 'neutral_analysis']:
            if self.results[perspective]:
                # Stwórz dwie wersje: pełną i skróconą
                csv_file_full = folder / f"{perspective}_full.csv"
                csv_file_truncated = folder / f"{perspective}_truncated.csv"
                
                # Wszystkie dostępne pola z danych
                if self.results[perspective]:
                    fieldnames = list(self.results[perspective][0].keys())
                else:
                    continue
                
                # Pełna wersja CSV
                with open(csv_file_full, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in self.results[perspective]:
                        writer.writerow(item)
                
                # Wersja skrócona (jeśli potrzebna)
                if truncate_text:
                    with open(csv_file_truncated, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for item in self.results[perspective]:
                            # Ogranicz długość tekstów
                            row = {}
                            for k, v in item.items():
                                if isinstance(v, str) and len(v) > max_length:
                                    # Inteligentne obcinanie - zachowaj kompletne zdania
                                    truncated = v[:max_length]
                                    last_period = truncated.rfind('.')
                                    last_space = truncated.rfind(' ')
                                    
                                    if last_period > max_length * 0.8:  # Jeśli kropka jest blisko końca
                                        row[k] = truncated[:last_period + 1]
                                    elif last_space > max_length * 0.9:  # Jeśli spacja jest blisko końca
                                        row[k] = truncated[:last_space] + "..."
                                    else:
                                        row[k] = truncated + "..."
                                else:
                                    row[k] = v
                            writer.writerow(row)

    def analyze_text_lengths(self):
        """Analizuje długości tekstów w zebranych danych"""
        print("\n📊 ANALIZA DŁUGOŚCI TEKSTÓW:")
        print("=" * 40)
        
        for perspective_name, perspective_data in [
            ('Demokratyczna', self.results['democratic_perspective']),
            ('Republikańska', self.results['republican_perspective']),
            ('Neutralna', self.results['neutral_analysis'])
        ]:
            if perspective_data:
                snippets = [item.get('snippet', '') for item in perspective_data if item.get('snippet')]
                if snippets:
                    lengths = [len(s) for s in snippets]
                    print(f"\n{perspective_name} ({len(snippets)} tekstów):")
                    print(f"  • Średnia długość: {sum(lengths)/len(lengths):.0f} znaków")
                    print(f"  • Najkrótszy: {min(lengths)} znaków")
                    print(f"  • Najdłuższy: {max(lengths)} znaków")
                    print(f"  • Teksty >500 znaków: {sum(1 for l in lengths if l > 500)}")
                    print(f"  • Teksty >1000 znaków: {sum(1 for l in lengths if l > 1000)}")
                    
                    # Przykład najdłuższego tekstu
                    longest_idx = lengths.index(max(lengths))
                    longest_text = snippets[longest_idx]
                    print(f"  • Najdłuższy tekst (fragment):")
                    print(f"    '{longest_text[:150]}...'")

    def get_full_texts_sample(self, perspective: str = 'democratic_perspective', limit: int = 3):
        """
        Pokazuje przykłady pełnych tekstów bez obcięć
        
        Args:
            perspective: perspektywa do pokazania
            limit: liczba przykładów
        """
        print(f"\n📄 PRZYKŁADY PEŁNYCH TEKSTÓW - {perspective.upper()}:")
        print("=" * 60)
        
        data = self.results.get(perspective, [])
        for i, item in enumerate(data[:limit]):
            print(f"\n--- PRZYKŁAD {i+1} ---")
            print(f"Tytuł: {item.get('title', 'Brak tytułu')}")
            print(f"Źródło: {item.get('source', 'Nieznane')}")
            print(f"URL: {item.get('url', 'Brak URL')}")
            print(f"Zapytanie: {item.get('search_query', 'Nieznane')}")
            
            snippet = item.get('snippet', '')
            print(f"Długość tekstu: {len(snippet)} znaków")
            print(f"Pełny tekst:")
            print(f"'{snippet}'")
            print("-" * 60)
    
    def export_full_texts_only(self, filename_prefix: str = "full_texts"):
        """Eksportuje tylko pełne teksty bez metadanych dla analizy NLP"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for perspective in ['democratic_perspective', 'republican_perspective', 'neutral_analysis']:
            if self.results[perspective]:
                # Zwykły plik tekstowy z pełnymi snippet'ami
                txt_file = f"{filename_prefix}_{perspective}_{timestamp}.txt"
                
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"PEŁNE TEKSTY - {perspective.upper()}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, item in enumerate(self.results[perspective], 1):
                        snippet = item.get('snippet', '')
                        if snippet.strip():  # Tylko jeśli tekst nie jest pusty
                            f.write(f"TEKST #{i}\n")
                            f.write(f"Źródło: {item.get('source', 'Nieznane')}\n")
                            f.write(f"Zapytanie: {item.get('search_query', 'Nieznane')}\n")
                            f.write(f"Długość: {len(snippet)} znaków\n")
                            f.write("-" * 30 + "\n")
                            f.write(f"{snippet}\n")
                            f.write("\n" + "=" * 50 + "\n\n")
                
                print(f"✅ Pełne teksty zapisane do: {txt_file}")
    
    # Nowa metoda dodana na potrzeby integracji z Firecrawl
    def get_collected_urls(self) -> List[str]:
        """Zwraca listę wszystkich zebranych URL-i do dalszej analizy"""
        urls = []
        
        for perspective in ['democratic_perspective', 'republican_perspective', 'neutral_analysis']:
            for item in self.results[perspective]:
                if 'url' in item and item['url'] not in urls:
                    urls.append(item['url'])
        
        return urls



import asyncio
from typing import List, Dict, Any, Optional
from firecrawl import AsyncFirecrawlApp, JsonConfig
from datetime import datetime
import json
from pathlib import Path

class FirecrawlPolarizationAnalyzer:
    """Klasa do głębszej analizy treści stron z użyciem Firecrawl"""
    
    def __init__(self, api_key: str, json_config: JsonConfig):
        self.api_key = api_key
        self.json_config = json_config
        self.app = AsyncFirecrawlApp(api_key=api_key)
        self.results = {
            'urls_analyzed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'start_time': datetime.now().isoformat(),
            'analyses': []
        }
    
    async def analyze_url(self, url: str, timeout: int = 120000) -> Optional[Dict[str, Any]]:
        """Analizuje pojedynczy URL za pomocą Firecrawl"""
        try:
            print(f"🔍 Analizuję: {url}")
            
            response = await self.app.scrape_url(
                url=url,
                formats=['json'],
                only_main_content=True,
                json_options=self.json_config,
                timeout=timeout
            )
            
            # Jeśli analiza się powiodła
            if response and hasattr(response, 'json') and response.json:
                print(f"✅ Analiza zakończona: {url}")
                
                # Konwertuj dane na serializowalny format
                json_data = self._ensure_serializable(response.json)
                
                # Zapisz wynik wraz z metadanymi
                result = {
                    'url': url,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'data': json_data
                }
                
                self.results['analyses'].append(result)
                self.results['successful_analyses'] += 1
                return result
            else:
                print(f"⚠️ Brak wyników dla: {url}")
                self.results['failed_analyses'] += 1
                return None
                
        except Exception as e:
            print(f"❌ Błąd analizy {url}: {str(e)}")
            self.results['failed_analyses'] += 1
            return None
    
    def _ensure_serializable(self, data):
        """Upewnia się, że dane są serializowalne do JSON"""
        if isinstance(data, dict):
            return {k: self._ensure_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._ensure_serializable(item) for item in data]
        elif callable(data):  # Funkcje
            return str(data)
        elif hasattr(data, 'to_dict'):  # Obiekty Pydantic
            return self._ensure_serializable(data.to_dict())
        elif hasattr(data, '__dict__'):  # Inne obiekty
            return self._ensure_serializable(data.__dict__)
        else:
            try:
                # Sprawdź czy serializowalne
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                # Jeśli nie, zamień na string
                return 
    
    async def analyze_urls_batch(self, urls: List[str], batch_size: int = 5, timeout: int = 120000):
        """Analizuje partie URL-i równolegle"""
        self.results['urls_analyzed'] = len(urls)
        
        # Przetwarzaj URLs w partiach, aby uniknąć przeciążenia
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            tasks = [self.analyze_url(url, timeout) for url in batch]
            await asyncio.gather(*tasks)
            
            # Krótka przerwa między partiami
            if i + batch_size < len(urls):
                print(f"⏱️ Przerwa między partiami ({i+batch_size}/{len(urls)})")
                await asyncio.sleep(2)
    
    def save_results(self, output_folder: str = None):
        """Zapisuje wyniki analizy"""
        # Zaktualizuj metadane
        self.results['end_time'] = datetime.now().isoformat()
        self.results['success_rate'] = (
            self.results['successful_analyses'] / self.results['urls_analyzed'] 
            if self.results['urls_analyzed'] > 0 else 0
        )
        
        # Utworzenie nazwy folderu lub użycie dostarczonej
        if not output_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = f"firecrawl_analysis_{timestamp}"
        
        folder_path = Path(output_folder)
        folder_path.mkdir(exist_ok=True, parents=True)
        
        # Zapewnij, że wszystkie dane są serializowalne
        serializable_results = self._ensure_serializable(self.results)
        
        # Zapisz pełne wyniki
        results_file = folder_path / "firecrawl_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Zapisz raport statystyczny
        stats_file = folder_path / "analysis_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("RAPORT ANALIZY FIRECRAWL\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Rozpoczęto: {self.results['start_time']}\n")
            f.write(f"Zakończono: {self.results['end_time']}\n")
            f.write(f"Analizowane URL-e: {self.results['urls_analyzed']}\n")
            f.write(f"Udane analizy: {self.results['successful_analyses']}\n")
            f.write(f"Nieudane analizy: {self.results['failed_analyses']}\n")
            f.write(f"Wskaźnik sukcesu: {self.results['success_rate']*100:.1f}%\n\n")
            
            # Podsumowanie wykrytych trendów
            democratic_count = 0
            republican_count = 0
            partisan_count = 0
            
            for analysis in self.results['analyses']:
                data = analysis.get('data', {})
                if data.get('is_democratic'):
                    democratic_count += 1
                if data.get('is_republican'):
                    republican_count += 1
                if data.get('has_partisan_language'):
                    partisan_count += 1
            
            f.write("PODSUMOWANIE TRENDÓW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Strony demokratyczne: {democratic_count}\n")
            f.write(f"Strony republikańskie: {republican_count}\n")
            f.write(f"Strony z wyraźnie partyjnym językiem: {partisan_count}\n")
        
        print(f"\n✨ Wyniki analizy Firecrawl zapisane w: {folder_path}")
        print(f"📊 Statystyki: {self.results['successful_analyses']}/{self.results['urls_analyzed']} udanych analiz")

import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional




class IntegratedPolarizationAnalyzer:
    """Klasa integrująca zbieranie danych z Serper i analizę z Firecrawl"""
    
    def __init__(self):
        self.serper_collector = SerperPolarizationCollector(api_key=SERPER_API_KEY)
        self.firecrawl_analyzer = FirecrawlPolarizationAnalyzer(
            api_key=FIRECRAWL_API_KEY,
            json_config=JSON_CONFIG
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = Path(f"polarization_analysis_{self.timestamp}")
        self.output_folder.mkdir(exist_ok=True)
    
    def collect_serper_data(
        self,
        categories: List[str] = None,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
        delay_between_queries: float = DEFAULT_DELAY_BETWEEN_QUERIES,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """Zbierz dane z Serper API"""
        print("\n🔍 ETAP 1: ZBIERANIE DANYCH Z SERPER API")
        print("=" * 60)
        
        if categories is None:
            categories = [
                'democratic_perspective', 
                'republican_perspective', 

            ]
        
        # Wykonaj zbieranie danych z Serper
        self.serper_collector.collect_polarization_data(
            categories=categories,
            results_per_query=results_per_query,
            delay_between_queries=delay_between_queries,
            max_retries=max_retries,
            preserve_full_text=True
        )
        
        # Utwórz podfolder dla wyników Serper
        serper_folder = self.output_folder / "serper_results"
        serper_folder.mkdir(exist_ok=True)
        
        # Zapisz wyniki Serper do podfolderu
        self._save_serper_results(serper_folder)
        
        # Przeanalizuj zebraną zawartość
        self.serper_collector.analyze_text_lengths()
        
        return self.serper_collector.get_collected_urls()
    
    def _save_serper_results(self, folder: Path):
        """Zapisuje wyniki z Serper do określonego folderu"""
        # Zapisz główny plik JSON
        main_file = folder / "complete_polarization_data.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(self.serper_collector.results, f, indent=2, ensure_ascii=False)
        
        # Zapisz osobne pliki dla każdej perspektywy
        for perspective in ['democratic_perspective', 'republican_perspective', 'neutral_analysis']:
            if self.serper_collector.results[perspective]:
                perspective_file = folder / f"{perspective}.json"
                with open(perspective_file, 'w', encoding='utf-8') as f:
                    json.dump(self.serper_collector.results[perspective], f, indent=2, ensure_ascii=False)
    
    async def analyze_with_firecrawl(
        self, 
        urls: List[str] = None, 
        batch_size: int = 5,
        timeout: int = DEFAULT_TIMEOUT
    ):
        """Analizuj strony za pomocą Firecrawl"""
        print("\n🧠 ETAP 2: GŁĘBOKA ANALIZA TREŚCI Z FIRECRAWL")
        print("=" * 60)
        
        # Jeśli nie dostarczono URL-i, pobierz je z wyników Serper
        if urls is None:
            urls = self.serper_collector.get_collected_urls()
        
        # Utwórz podfolder dla wyników Firecrawl
        firecrawl_folder = self.output_folder / "firecrawl_results"
        firecrawl_folder.mkdir(exist_ok=True)
        
        # Wykonaj analizę stron
        print(f"📊 Analizuję {len(urls)} stron w partiach po {batch_size}...")
        await self.firecrawl_analyzer.analyze_urls_batch(
            urls=urls,
            batch_size=batch_size,
            timeout=timeout
        )
        
        # Zapisz wyniki Firecrawl
        self.firecrawl_analyzer.save_results(str(firecrawl_folder))
    
    def combine_results(self):
        """Łączy wyniki z obu źródeł w jeden zintegrowany zestaw danych"""
        print("\n🔄 ETAP 3: INTEGRACJA WYNIKÓW")
        print("=" * 60)
        
        # Przygotuj strukturę dla zintegrowanych wyników
        integrated_results = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'serper_queries': self.serper_collector.results['metadata']['total_queries'],
                'firecrawl_analyses': self.firecrawl_analyzer.results['urls_analyzed'],
                'integrated_items': 0
            },
            'perspectives': {
                'democratic': [],
                'republican': [],
                'neutral': []
            }
        }
        
        # Słownik mapujący URL-e na wyniki analizy Firecrawl
        firecrawl_by_url = {}
        for analysis in self.firecrawl_analyzer.results['analyses']:
            firecrawl_by_url[analysis['url']] = analysis['data']
        
        # Funkcja pomocnicza do przetwarzania perspektywy
        def process_perspective(perspective_name, firecrawl_category):
            for item in self.serper_collector.results[perspective_name]:
                url = item.get('url')
                integrated_item = item.copy()  # Kopiuj dane z Serper
                
                # Jeśli mamy analizę Firecrawl dla tego URL, dodaj ją
                if url in firecrawl_by_url:
                    firecrawl_data = firecrawl_by_url[url]
                    integrated_item.update({
                        'firecrawl_analysis': {
                            'text': firecrawl_data.get('text'),
                            'is_democratic': firecrawl_data.get('is_democratic'),
                            'is_republican': firecrawl_data.get('is_republican'),
                            'has_partisan_language': firecrawl_data.get('has_partisan_language'),
                            'mentioned_topics': firecrawl_data.get('mentioned_topics')
                        }
                    })
                
                integrated_results['perspectives'][firecrawl_category].append(integrated_item)
                integrated_results['metadata']['integrated_items'] += 1
        
        # Przetwórz dane z każdej perspektywy
        process_perspective('democratic_perspective', 'democratic')
        process_perspective('republican_perspective', 'republican')
        process_perspective('neutral_analysis', 'neutral')
        
        # Zapisz zintegrowane wyniki
        integrated_file = self.output_folder / "integrated_polarization_results.json"
        with open(integrated_file, 'w', encoding='utf-8') as f:
            json.dump(integrated_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Zintegrowane wyniki zapisane w: {integrated_file}")
        print(f"📊 Zintegrowane elementy: {integrated_results['metadata']['integrated_items']}")
    
    def generate_final_report(self):
        """Generuje końcowy raport z całej analizy"""
        report_file = self.output_folder / "final_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Raport Analizy Polaryzacji Politycznej\n\n")
            f.write(f"*Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Podsumowanie\n\n")
            f.write(f"- **Zbieranie danych**: {self.serper_collector.results['metadata']['total_queries']} zapytań w Serper API\n")
            f.write(f"- **Udane zapytania**: {self.serper_collector.results['metadata']['successful_queries']} ({self.serper_collector.results['metadata']['successful_queries']/self.serper_collector.results['metadata']['total_queries']*100:.1f}%)\n")
            f.write(f"- **Łączne wyniki**: {self.serper_collector.results['metadata']['total_results']} elementów\n")
            f.write(f"- **Analizy Firecrawl**: {self.firecrawl_analyzer.results['successful_analyses']} udanych analiz z {self.firecrawl_analyzer.results['urls_analyzed']} URL-i\n\n")
            
            f.write("## Rozkład Perspektyw\n\n")
            f.write("| Perspektywa | Liczba wyników | Procent |\n")
            f.write("|-------------|----------------|--------|\n")
            
            total = self.serper_collector.results['metadata']['total_results']
            dem = self.serper_collector.results['metadata']['democratic_results']
            rep = self.serper_collector.results['metadata']['republican_results']
            neu = self.serper_collector.results['metadata']['neutral_results']
            
            f.write(f"| Demokratyczna | {dem} | {dem/total*100:.1f}% |\n")
            f.write(f"| Republikańska | {rep} | {rep/total*100:.1f}% |\n")
            f.write(f"| Neutralna | {neu} | {neu/total*100:.1f}% |\n\n")
            
            f.write("## Kluczowe Wnioski z Analizy Firecrawl\n\n")
            
            # Zliczanie wyników z Firecrawl
            dem_count = 0
            rep_count = 0
            partisan_count = 0
            
            for analysis in self.firecrawl_analyzer.results['analyses']:
                data = analysis.get('data', {})
                if data.get('is_democratic'):
                    dem_count += 1
                if data.get('is_republican'):
                    rep_count += 1
                if data.get('has_partisan_language'):
                    partisan_count += 1
            
            total_analyzed = self.firecrawl_analyzer.results['successful_analyses']
            
            f.write(f"- Strony o charakterze demokratycznym: {dem_count} ({dem_count/total_analyzed*100:.1f}% analizowanych)\n")
            f.write(f"- Strony o charakterze republikańskim: {rep_count} ({rep_count/total_analyzed*100:.1f}% analizowanych)\n")
            f.write(f"- Strony z wyraźnie partyjnym językiem: {partisan_count} ({partisan_count/total_analyzed*100:.1f}% analizowanych)\n\n")
            
            f.write("## Dostępne Pliki\n\n")
            f.write("- `serper_results/` - Wyniki wyszukiwania z Serper API\n")
            f.write("- `firecrawl_results/` - Wyniki analizy treści z Firecrawl\n")
            f.write("- `integrated_polarization_results.json` - Zintegrowane wyniki z obu źródeł\n")
        
        print(f"\n✨ Raport końcowy zapisany w: {report_file}")





import asyncio

async def main():
    """Główna funkcja uruchamiająca zintegrowaną analizę polaryzacji"""
    print("🚀 ZINTEGROWANA ANALIZA POLARYZACJI POLITYCZNEJ")
    print("=" * 60)
    
    # Inicjalizacja zintegrowanego analizatora
    analyzer = IntegratedPolarizationAnalyzer()
    
    # Etap 1: Zbieranie danych z Serper
    urls = analyzer.collect_serper_data(
        categories=[
            'democratic_perspective',
            'republican_perspective', 
        ],
        results_per_query=10,
        delay_between_queries=1.2,
        max_retries=2
    )
    
    # # Etap 2: Analiza treści z Firecrawl
    # # Możemy ograniczyć liczbę analizowanych URL-i, jeśli jest ich dużo
    # if len(urls) > 30:
    #     print(f"⚠️ Ograniczam analizę do 30 URL-i z {len(urls)} zebranych")
    #     urls = urls[:]

    # await analyzer.analyze_with_firecrawl(
    #     urls=urls,
    #     batch_size=5,  # Analiza 5 URL-i jednocześnie
    #     timeout=120000  # 2 minuty na stronę
    # )
    
    # # # Etap 3: Łączenie wyników
    # analyzer.combine_results()
    
    # # Etap 4: Generowanie końcowego raportu
    # analyzer.generate_final_report()
    
    print("\n✅ ANALIZA ZAKOŃCZONA")
    print("=" * 60)
    print("Wszystkie wyniki zostały zapisane w folderze wynikowym.")

# Uruchomienie asynchronicznego programu głównego
if __name__ == "__main__":
    asyncio.run(main())