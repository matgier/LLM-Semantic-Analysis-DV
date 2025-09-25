#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uproszczony moduł do analizy tekstu - CTTR, entropia, 20 słów kluczowych
oraz Jensen-Shannon Divergence między rozkładami TF-IDF
"""

import re
import math
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set

class TextAnalyzer:
    """
    Uproszczona klasa do przetwarzania i analizy tekstu.
    """
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Inicjalizacja analizatora tekstu.
        
        Args:
            remove_stopwords: Czy usuwać stop words
        """
        self.remove_stopwords = remove_stopwords
        
        # Podstawowe stop words w języku angielskim
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'or', 'but', 'not', 'have', 'had',
            'can', 'could', 'should', 'would', 'been', 'being', 'their',
            'them', 'they', 'this', 'these', 'those', 'who', 'which', 'what',
            'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Podstawowe czyszczenie tekstu.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizacja tekstu na słowa.
        """
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        
        if self.remove_stopwords:
            tokens = [token for token in tokens 
                     if token and token not in self.stopwords and len(token) > 1]
        else:
            tokens = [token for token in tokens if token and len(token) > 1]
        
        return tokens
    
    def get_word_frequencies(self, text: str) -> Dict[str, int]:
        """
        Oblicza częstotliwości słów w tekście.
        """
        tokens = self.tokenize(text)
        return dict(Counter(tokens))
    
    def calculate_cttr(self, text: str) -> float:
        """
        Oblicza Corrected TTR = unique_words / sqrt(2 * total_words).
        """
        tokens = self.tokenize(text)
        
        if not tokens:
            return 0.0
        
        unique_words = len(set(tokens))
        total_words = len(tokens)
        
        return unique_words / math.sqrt(2 * total_words)
    
    def calculate_entropy(self, text: str) -> float:
        """
        Oblicza entropię Shannona dla rozkładu słów w tekście.
        """
        word_frequencies = self.get_word_frequencies(text)
        
        if not word_frequencies:
            return 0.0
        
        total_words = sum(word_frequencies.values())
        probabilities = [freq / total_words for freq in word_frequencies.values()]
        
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def get_key_words(self, text: str, n: int = 20) -> List[Tuple[str, int]]:
        """
        Zwraca n najważniejszych słów w tekście na podstawie częstości.
        """
        word_freq = self.get_word_frequencies(text)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:n]
    
    def calculate_tf(self, text: str) -> Dict[str, float]:
        """
        Oblicza Term Frequency (TF) dla słów w tekście.
        
        TF(t,d) = count(t,d) / |d|
        """
        word_frequencies = self.get_word_frequencies(text)
        total_words = sum(word_frequencies.values())
        
        if total_words == 0:
            return {}
        
        tf_scores = {word: freq / total_words 
                    for word, freq in word_frequencies.items()}
        
        return tf_scores
    
    def calculate_idf(self, all_texts: Dict[str, str]) -> Dict[str, float]:
        """
        Oblicza Inverse Document Frequency (IDF) dla wszystkich słów w korpusie.
        
        IDF(t) = log(N / df(t))
        """
        # Zbiór wszystkich unikalnych słów w korpusie
        all_words = set()
        documents_containing_word = Counter()
        
        # Dla każdego dokumentu znajdź unikalne słowa
        for text in all_texts.values():
            unique_words_in_doc = set(self.tokenize(text))
            all_words.update(unique_words_in_doc)
            
            # Zlicz w ilu dokumentach występuje każde słowo
            for word in unique_words_in_doc:
                documents_containing_word[word] += 1
        
        # Oblicz IDF dla każdego słowa
        total_documents = len(all_texts)
        idf_scores = {}
        
        for word in all_words:
            df = documents_containing_word[word]
            # Dodaj 1 do df aby uniknąć dzielenia przez 0
            idf_scores[word] = math.log(total_documents / (df + 1)) + 1  # +1 dla wygładzenia
        
        return idf_scores
    
    def calculate_tfidf(self, text: str, idf_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Oblicza TF-IDF dla słów w tekście.
        
        TF-IDF(t,d) = TF(t,d) * IDF(t)
        """
        tf_scores = self.calculate_tf(text)
        tfidf_scores = {}
        
        for word, tf in tf_scores.items():
            if word in idf_scores:
                tfidf_scores[word] = tf * idf_scores[word]
        
        return tfidf_scores
    
    def calculate_kl_divergence(self, p: Dict[str, float], q: Dict[str, float], 
                              all_words: Set[str]) -> float:
        """
        Oblicza Kullback-Leibler Divergence między dwoma rozkładami.
        
        KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
        """
        kl_div = 0.0
        epsilon = 1e-10  # Małe epsilon dla uniknięcia log(0)
        
        for word in all_words:
            p_i = p.get(word, 0) + epsilon
            q_i = q.get(word, 0) + epsilon
            
            kl_div += p_i * math.log2(p_i / q_i)
        
        return kl_div
    
    def calculate_jensen_shannon_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Oblicza Jensen-Shannon Divergence między dwoma rozkładami.
        
        JSD(P||Q) = 0.5 * [KL(P||M) + KL(Q||M)]
        gdzie M = 0.5 * (P + Q)
        """
        # Zbierz wszystkie słowa z obu rozkładów
        all_words = set(p.keys()).union(set(q.keys()))
        
        # Utwórz rozkład średni M
        m = {}
        for word in all_words:
            m[word] = 0.5 * (p.get(word, 0) + q.get(word, 0))
        
        # Oblicz KL dla P||M i Q||M
        kl_p_m = self.calculate_kl_divergence(p, m, all_words)
        kl_q_m = self.calculate_kl_divergence(q, m, all_words)
        
        # Oblicz JSD
        jsd = 0.5 * (kl_p_m + kl_q_m)
        
        return jsd
    
    def normalize_distribution(self, dist: Dict[str, float]) -> Dict[str, float]:
        """
        Normalizuje rozkład prawdopodobieństwa, aby suma wynosiła 1.
        """
        total = sum(dist.values())
        if total == 0:
            return dist
        
        return {k: v/total for k, v in dist.items()}
    
    def analyze_category(self, text: str) -> Dict[str, any]:
        """
        Przeprowadza analizę kategorii tekstu.
        """
        tokens = self.tokenize(text)
        
        if not tokens:
            return {
                'error': 'Brak tokenów do analizy',
                'total_words': 0,
                'unique_words': 0,
                'cttr': 0.0,
                'entropy': 0.0,
                'key_words': [],
                'tf': {}
            }
        
        unique_words = set(tokens)
        total_words = len(tokens)
        
        # Obliczenia
        cttr = self.calculate_cttr(text)
        entropy = self.calculate_entropy(text)
        key_words = self.get_key_words(text, 20)
        tf = self.calculate_tf(text)  # Dodajemy tf do wyników
        
        return {
            'total_words': total_words,
            'unique_words': len(unique_words),
            'cttr': cttr,
            'entropy': entropy,
            'key_words': key_words,
            'tf': tf  # Będzie używane do obliczenia TF-IDF i JSD
        }
    
    def analyze_all_categories(self, categories: Dict[str, List[str]]) -> Dict[str, Dict[str, any]]:
        """
        Analizuje wszystkie kategorie tekstów.
        """
        results = {}
        processed_texts = {}
        
        # Przygotuj teksty i wykonaj podstawowe analizy
        for category, texts in categories.items():
            combined_text = ' '.join(texts)
            processed_texts[category] = combined_text
            results[category] = self.analyze_category(combined_text)
        
        # Oblicz IDF dla całego korpusu
        idf_scores = self.calculate_idf(processed_texts)
        
        # Oblicz TF-IDF dla każdej kategorii
        for category, analysis in results.items():
            tfidf = self.calculate_tfidf(processed_texts[category], idf_scores)
            results[category]['tfidf'] = tfidf
        
        # Normalizuj rozkłady TF-IDF
        for category, analysis in results.items():
            results[category]['tfidf_normalized'] = self.normalize_distribution(analysis['tfidf'])
        
        return results
    
    def calculate_jsd_matrix(self, results: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, float]]:
        """
        Oblicza macierz Jensen-Shannon Divergence między wszystkimi parami kategorii.
        """
        categories = list(results.keys())
        jsd_matrix = {}
        
        for i, cat1 in enumerate(categories):
            jsd_matrix[cat1] = {}
            
            for j, cat2 in enumerate(categories):
                if i == j:
                    jsd_matrix[cat1][cat2] = 0.0  # JSD między identycznymi rozkładami = 0
                else:
                    p = results[cat1]['tfidf_normalized']
                    q = results[cat2]['tfidf_normalized']
                    jsd = self.calculate_jensen_shannon_divergence(p, q)
                    jsd_matrix[cat1][cat2] = jsd
        
        return jsd_matrix
    
    def print_metrics_table(self, results: Dict[str, Dict[str, any]], name_of_the_file: str):
        """
        Wypisuje wyniki CTTR i entropii w formie tabeli.
        """
        # Nagłówek tabeli
        output = ""

        # Nagłówek tabeli
        output += "\n===== TABELA WYNIKÓW METRYK =====\n"
        output += f"{'Kategoria':<35} | {'CTTR':<10} | {'Entropia':<10} | {'Liczba słów':<12} | {'Unikalne słowa':<15}\n"
        output += "-" * 90 + "\n"

        # Dane
        for category, analysis in sorted(results.items()):
            output += f"{category:<35} | {analysis['cttr']:<10.4f} | {analysis['entropy']:<10.4f} | {analysis['total_words']:<12} | {analysis['unique_words']:<15}\n"

        # Zapis do pliku
        with open(f"{name_of_the_file}_CTTR_entropy_metrics.txt", "w", encoding="utf-8") as file:
            file.write(output)


    def print_key_words(self, results: Dict[str, Dict[str, any]], name_of_the_file: str):
        """
        Wypisuje 20 słów kluczowych dla każdej kategorii.
        """
        output = "\n===== SŁOWA KLUCZOWE DLA KATEGORII =====\n"

        print("\n\n===== SŁOWA KLUCZOWE DLA KATEGORII =====")

        for category, analysis in results.items():  # przykładowa struktura
            output += f"\n=== {category} ===\n"
            output += "20 słów kluczowych:\n"

            key_words = analysis['key_words']

            for i in range(0, len(key_words), 2):
                if i + 1 < len(key_words):
                    word1, freq1 = key_words[i]
                    word2, freq2 = key_words[i + 1]
                    output += f"{i+1:2d}. {word1:<20}: {freq1:<4}    {i+2:2d}. {word2:<20}: {freq2:<4}\n"
                else:
                    word, freq = key_words[i]
                    output += f"{i+1:2d}. {word:<20}: {freq:<4}\n"

        # Po pętli zapis całości
        with open(f"{name_of_the_file}_keywords_report.txt", "w", encoding="utf-8") as file:
            file.write(output)

    
    def print_jsd_matrix(self, jsd_matrix: Dict[str, Dict[str, float]]):
        """
        Wypisuje macierz Jensen-Shannon Divergence w formie tabeli.
        """
        categories = sorted(jsd_matrix.keys())
        
        print("\n\n===== MACIERZ JENSEN-SHANNON DIVERGENCE =====")
        print("(Miara podobieństwa między rozkładami TF-IDF, niższe wartości = większe podobieństwo)")
        
        # Nagłówek z nazwami kategorii
        header = "Kategoria"
        for cat in categories:
            # Skróć nazwę kategorii, jeśli jest za długa
            short_cat = cat[:12] + ".." if len(cat) > 14 else cat
            header += f" | {short_cat:<14}"
        print(header)
        print("-" * (len(categories) * 17 + 10))
        
        # Wiersze z wartościami JSD
        for cat1 in categories:
            short_cat1 = cat1[:12] + ".." if len(cat1) > 14 else cat1
            row = f"{short_cat1:<10}"
            
            for cat2 in categories:
                jsd = jsd_matrix[cat1][cat2]
                row += f" | {jsd:<14.4f}"
            
            print(row)

