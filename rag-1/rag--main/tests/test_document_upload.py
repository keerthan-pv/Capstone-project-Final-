"""Security tests for module5.document_upload."""
from __future__ import annotations

import pytest

from module5.document_upload import (
    DocumentUploader, UploadValidationError, _safe_basename, extract_document_answer,
    normalize_upload_query,
)
from langchain_core.documents import Document


class TestSafeBasename:
    def test_strips_directory_components(self):
        assert _safe_basename("../../etc/passwd") == "passwd"
        assert _safe_basename("C:\\Users\\bob\\report.pdf") == "report.pdf"

    def test_drops_control_characters(self):
        assert _safe_basename("file\x00name.txt") == "filename.txt"

    def test_empty_input_yields_default(self):
        assert _safe_basename("") == "uploaded"


class TestUploadValidation:
    def test_rejects_oversized_upload(self):
        uploader = DocumentUploader(max_bytes=100)
        with pytest.raises(UploadValidationError, match="too large"):
            uploader.parse_file(b"x" * 200, "file.txt")

    def test_rejects_empty_upload(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="Empty"):
            uploader.parse_file(b"", "file.txt")

    def test_rejects_unsupported_extension(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="Unsupported"):
            uploader.parse_file(b"contents", "malicious.exe")

    def test_rejects_pdf_with_bad_magic_bytes(self):
        uploader = DocumentUploader()
        # Looks like a PDF by extension but the bytes are not a PDF.
        with pytest.raises(UploadValidationError, match="not a valid PDF"):
            uploader.parse_file(b"this is not a pdf", "fake.pdf")

    def test_rejects_docx_with_bad_magic_bytes(self):
        uploader = DocumentUploader()
        with pytest.raises(UploadValidationError, match="not a valid DOCX"):
            uploader.parse_file(b"this is not a zip", "fake.docx")

    def test_accepts_valid_text_upload(self):
        uploader = DocumentUploader()
        docs = uploader.parse_file(
            b"This is a paragraph of more than twenty characters in length.\n\nAnother paragraph here too, also long enough.",
            "notes.txt",
        )
        assert len(docs) >= 1
        assert all(d.metadata["source"] == "notes.txt" for d in docs)


def test_project_question_uses_the_explicit_project_list_not_project_management():
    answer = extract_document_answer(
        "what are the projects he have done?",
        [Document(page_content=(
            "Experience includes software development and project management. "
            "Other projects include BunkWell Attendance Manager, Employee Management App, "
            "and Home Automation System."
        ))],
    )

    assert "BunkWell Attendance Manager" in answer
    assert "Employee Management App" in answer
    assert "project management" not in answer.lower()


def test_general_upload_answer_returns_document_evidence_not_prompt_rules():
    answer = extract_document_answer(
        "What is the candidate's CGPA?",
        [Document(page_content=(
            "JSS Science and Technology University, Mysore. "
            "Bachelor of Engineering, Computer Science – 9.3 CGPA."
        ))],
    )

    assert "9.3 CGPA" in answer
    assert "short, direct answer" not in answer.lower()
    assert "strict rules" not in answer.lower()


def test_merged_resume_heading_and_shorthand_are_normalized():
    normalized = normalize_upload_query("what r the ProgrammingLanguages he know?")
    assert "what are the" in normalized.casefold()
    assert "programming" in normalized.casefold()
    assert "languages" in normalized.casefold()
    answer = extract_document_answer(
        "what r the ProgrammingLanguages he know?",
        [Document(page_content="Programming Languages: Python, C, Dart, JavaScript, Go.")],
    )
    assert "Python, C, Dart" in answer


def test_college_question_matches_a_university_named_in_the_document():
    answer = extract_document_answer(
        "What is his college?",
        [Document(page_content="JSS Science and Technology University, Mysore.")],
    )
    assert "JSS Science and Technology University" in answer


def test_resume_field_questions_return_only_the_requested_value():
    document = Document(page_content=(
        "Others - Technical Lead | Engineering In Kannada - 56k+ Youtube | Built Two Startups\n"
        "Programming Languages: Python, C, Dart, Javascript, Go\n"
        "Technologies/Frameworks: Flutter, Django REST Framework, MySQL"
    ))

    assert extract_document_answer("What is his YouTube channel name?", [document]) == (
        "The YouTube channel is Engineering In Kannada."
    )
    assert extract_document_answer("Which programming languages does he know?", [document]) == (
        "Programming languages: Python, C, Dart, Javascript, Go"
    )
    assert extract_document_answer("Which frameworks does he use?", [document]).startswith(
        "Technologies/frameworks: Flutter, Django REST Framework, MySQL"
    )
