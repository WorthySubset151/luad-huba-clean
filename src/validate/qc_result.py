"""Struktury danych reprezentujące wyniki kontroli jakości kohorty."""

__author__ = "Łukasz Połaski"

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Severity(str, Enum):
    """Poziom istotności problemu wykrytego podczas kontroli jakości."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QCCategory(str, Enum):
    """Kategoria problemu wykrytego podczas walidacji kohorty."""

    MISSING_STAR_FILE = "missing_star_file"
    ORPHAN_STAR_FILE = "orphan_star_file"
    MISSING_CLINICAL = "missing_clinical"
    DUPLICATE_SAMPLE = "duplicate_sample"
    MISSING_SURVIVAL = "missing_survival"


@dataclass
class QCIssue:
    """Pojedynczy problem wykryty podczas kontroli jakości.

    Atrybuty:
        severity: Poziom istotności (error blokuje pipeline, warning informuje).
        category: Kategoria problemu z ``QCCategory``.
        message: Czytelny opis problemu.
        context: Dodatkowe dane diagnostyczne (np. identyfikatory próbek).
    """

    severity: Severity
    category: QCCategory
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializuje problem do słownika gotowego do zapisu w JSON."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class QCReport:
    """Zbiorczy raport kontroli jakości kohorty.

    Gromadzi listę problemów i udostępnia statystyki oraz serializację
    do formatu JSON na potrzeby logowania i wizualizacji w panelu QC.
    """

    issues: list[QCIssue] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, issue: QCIssue) -> None:
        """Dodaje pojedynczy problem do raportu."""
        self.issues.append(issue)

    def extend(self, issues: list[QCIssue]) -> None:
        """Dodaje listę problemów do raportu."""
        self.issues.extend(issues)

    def by_severity(self, severity: Severity) -> list[QCIssue]:
        """Zwraca problemy o wskazanym poziomie istotności."""
        return [i for i in self.issues if i.severity == severity]

    def by_category(self, category: QCCategory) -> list[QCIssue]:
        """Zwraca problemy z wskazanej kategorii."""
        return [i for i in self.issues if i.category == category]

    @property
    def n_errors(self) -> int:
        """Liczba problemów o istotności ERROR."""
        return len(self.by_severity(Severity.ERROR))

    @property
    def n_warnings(self) -> int:
        """Liczba problemów o istotności WARNING."""
        return len(self.by_severity(Severity.WARNING))

    @property
    def has_errors(self) -> bool:
        """Czy raport zawiera przynajmniej jeden problem typu ERROR."""
        return self.n_errors > 0

    def summary(self) -> dict[str, int]:
        """Zwraca podsumowanie liczbowe per poziom istotności."""
        return {
            "total": len(self.issues),
            "errors": self.n_errors,
            "warnings": self.n_warnings,
            "info": len(self.by_severity(Severity.INFO)),
        }

    def to_dict(self) -> dict:
        """Serializuje cały raport do słownika gotowego do zapisu w JSON."""
        return {
            "created_at": self.created_at,
            "summary": self.summary(),
            "issues": [issue.to_dict() for issue in self.issues],
        }
