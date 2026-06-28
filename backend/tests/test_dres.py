from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.dres import DresClient


class TestNormalizeVerdict:
    def test_412_is_wrong(self):
        mock_response = MagicMock()
        mock_response.status_code = 412
        mock_response.text = '{"submission": "WRONG"}'
        mock_response.json.return_value = {"submission": "WRONG"}

        result = DresClient.normalize_verdict(mock_response)
        assert result["status"] == "wrong"

    def test_202_is_pending(self):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.text = ""
        mock_response.json.return_value = {}

        result = DresClient.normalize_verdict(mock_response)
        assert result["status"] == "pending"

    def test_correct_verdict(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"submission": "CORRECT"}'
        mock_response.json.return_value = {"submission": "CORRECT"}

        result = DresClient.normalize_verdict(mock_response)
        assert result["status"] == "correct"

    def test_wrong_verdict_in_text(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.text = '{"submission": "WRONG"}'
        mock_response.json.return_value = {"submission": "WRONG"}

        result = DresClient.normalize_verdict(mock_response)
        assert result["status"] == "wrong"

    def test_error_on_non_ok(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.ok = False
        mock_response.text = '{"description": "Server error"}'
        mock_response.json.return_value = {"description": "Server error"}

        result = DresClient.normalize_verdict(mock_response)
        assert result["status"] == "error"
        assert "Server error" in result["message"]


class TestPickActiveEvaluationId:
    def test_active_preferred(self):
        evals = [
            {"id": "1", "status": "INACTIVE"},
            {"id": "2", "status": "ACTIVE"},
            {"id": "3", "status": "INACTIVE"},
        ]
        assert DresClient.pick_active_evaluation_id(evals) == "2"

    def test_first_fallback(self):
        evals = [
            {"id": "1", "status": "INACTIVE"},
            {"id": "2", "status": "INACTIVE"},
        ]
        assert DresClient.pick_active_evaluation_id(evals) == "1"

    def test_empty(self):
        assert DresClient.pick_active_evaluation_id([]) is None

    def test_none(self):
        assert DresClient.pick_active_evaluation_id(None) is None


class TestDresClientInit:
    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="Missing DRES URL"):
            DresClient("")

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="must start with http"):
            DresClient("ftp://example.com")

    def test_valid_url(self):
        client = DresClient("https://dres.example.com/")
        assert client.base_url == "https://dres.example.com"

    def test_trailing_slash_stripped(self):
        client = DresClient("http://localhost:8080///")
        assert client.base_url == "http://localhost:8080"
