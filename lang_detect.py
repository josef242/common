#!/usr/bin/env python3
"""
lang_detect.py
Language detection utilities for filtering during tokenization.
Supports multiple backends with fallback options.
"""

import logging
from typing import Optional, Tuple
from functools import lru_cache

# Try to import language detection libraries in order of preference
LANG_DETECT_AVAILABLE = False
LANG_DETECT_BACKEND = None

try:
    # FastText is fastest and most accurate for short texts
    import fasttext
    # Download model with: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    LANG_DETECT_BACKEND = "fasttext"
    LANG_DETECT_AVAILABLE = True
except ImportError:
    try:
        # Fallback to langdetect (slower but no model download needed)
        from langdetect import detect_langs, LangDetectException
        LANG_DETECT_BACKEND = "langdetect"
        LANG_DETECT_AVAILABLE = True
    except ImportError:
        try:
            # Final fallback to langid
            import langid
            LANG_DETECT_BACKEND = "langid"
            LANG_DETECT_AVAILABLE = True
        except ImportError:
            logging.warning("No language detection library available. Install fasttext, langdetect, or langid.")


class LanguageDetector:
    """Unified interface for language detection with multiple backends."""
    
    def __init__(self, backend: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize language detector.
        
        Parameters
        ----------
        backend : str, optional
            Force specific backend ('fasttext', 'langdetect', 'langid')
        model_path : str, optional
            Path to fasttext model (only for fasttext backend)
        """
        self.backend = backend or LANG_DETECT_BACKEND
        self.model = None
        self.initialized = False
        
        if not LANG_DETECT_AVAILABLE:
            raise RuntimeError(
                "No language detection library available. "
                "Install one of: pip install fasttext-wheel, langdetect, or langid"
            )
        
        if self.backend == "fasttext":
            if model_path is None:
                model_path = "lid.176.bin"  # Default fasttext language model
            try:
                # Suppress fasttext warnings
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(model_path)
                self.initialized = True
            except Exception as e:
                logging.warning(f"Failed to load fasttext model from {model_path}: {e}")
                logging.warning("Download with: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
                # Fallback to next available backend
                if "langdetect" in globals():
                    self.backend = "langdetect"
                elif "langid" in globals():
                    self.backend = "langid"
                else:
                    raise
        
        if self.backend in ["langdetect", "langid"]:
            self.initialized = True
    
    @lru_cache(maxsize=10000)
    def detect(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Parameters
        ----------
        text : str
            Text to analyze
        threshold : float
            Minimum confidence threshold (not used by all backends)
            
        Returns
        -------
        lang_code : str
            ISO language code (e.g., 'en', 'es', 'zh')
        confidence : float
            Confidence score [0, 1]
        """
        if not self.initialized or not text or len(text.strip()) < 10:
            return "unknown", 0.0
        
        try:
            if self.backend == "fasttext":
                # FastText returns (('__label__en',), array([0.99]))
                predictions = self.model.predict(text.replace('\n', ' '), k=1)
                if predictions[0]:
                    lang = predictions[0][0].replace('__label__', '')
                    conf = float(predictions[1][0])
                    return lang, conf
                    
            elif self.backend == "langdetect":
                # Returns list of Language objects with lang and prob
                results = detect_langs(text)
                if results:
                    best = results[0]
                    return best.lang, best.prob
                    
            elif self.backend == "langid":
                # Returns (lang, confidence)
                lang, conf = langid.classify(text)
                return lang, conf
                
        except Exception as e:
            logging.debug(f"Language detection failed: {e}")
            
        return "unknown", 0.0
    
    def is_english(self, text: str, threshold: float = 0.8) -> bool:
        """
        Check if text is English with confidence above threshold.
        
        Parameters
        ----------
        text : str
            Text to check
        threshold : float
            Minimum confidence to consider text as English
            
        Returns
        -------
        bool
            True if text is detected as English with sufficient confidence
        """
        lang, conf = self.detect(text)
        return lang == 'en' and conf >= threshold


# Convenience function for simple English detection
_detector = None

def is_english(text: str, threshold: float = 0.8, backend: Optional[str] = None) -> bool:
    """
    Quick check if text is English.
    
    This function maintains a singleton detector for efficiency.
    """
    global _detector
    if _detector is None:
        try:
            _detector = LanguageDetector(backend=backend)
        except Exception:
            # If no language detection available, return True (don't filter)
            return True
    
    return _detector.is_english(text, threshold)