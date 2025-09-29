# tests/test_utils.py
"""
Tests para el módulo utils.py

Prueba la función winsound_beep en diferentes plataformas
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Añadir src al path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 🚀 Opción 2: Mock winsound en Linux
if sys.platform != "win32":
    sys.modules['winsound'] = types.SimpleNamespace(Beep=lambda freq, dur: None)


def test_winsound_beep_on_windows():
    """Test winsound_beep cuando winsound está disponible (Windows)."""
    # Mock del módulo winsound
    mock_winsound = MagicMock()
    mock_winsound.Beep = MagicMock()

    with patch.dict('sys.modules', {'winsound': mock_winsound}):
        import importlib
        import utils
        importlib.reload(utils)
        from utils import winsound_beep

        # Llamamos a la función
        winsound_beep(1000, 200)

        # Verificamos que se llamó Beep con los parámetros correctos
        mock_winsound.Beep.assert_called_once_with(1000, 200)


def test_winsound_beep_on_non_windows():
    """Test winsound_beep en plataforma no-Windows (simulado)."""
    with patch('sys.platform', 'linux'):
        from utils import winsound_beep
        try:
            winsound_beep(1500, 300)
            assert True  # No debe fallar
        except Exception as e:
            pytest.fail(f"winsound_beep no debería fallar en plataformas no-Windows: {e}")


def test_winsound_beep_exception_handling():
    """Test que winsound_beep maneja excepciones correctamente."""
    mock_winsound = MagicMock()
    mock_winsound.Beep = MagicMock(side_effect=RuntimeError("Error de audio"))

    with patch.dict('sys.modules', {'winsound': mock_winsound}):
        import importlib
        import utils
        importlib.reload(utils)
        from utils import winsound_beep

        try:
            winsound_beep(1000, 200)
            assert True
        except Exception:
            pytest.fail("winsound_beep no manejó la excepción correctamente")


def test_winsound_beep_with_zero_params():
    """Test winsound_beep con parámetros cero."""
    mock_winsound = MagicMock()

    with patch.dict('sys.modules', {'winsound': mock_winsound}):
        import importlib
        import utils
        importlib.reload(utils)
        from utils import winsound_beep

        winsound_beep(0, 0)

        mock_winsound.Beep.assert_called_once_with(0, 0)
