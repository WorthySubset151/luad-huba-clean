"""Warstwa analityczna LUAD-HUBA (headless).

Czyste obliczenia survival (Kaplan-Meier, Cox) zwracające dane, nie wykresy.
Wspólny rdzeń dla frontendów: Streamlit owija dane w Plotly, terminal renderuje
je jako tabele. Jedno źródło prawdy metodologii.
"""

__author__ = "Łukasz Połaski"
