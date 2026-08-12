"""Results tables and how they get rendered.

Columns are not declared anywhere: a table's columns are the union of the keys its rows return,
in first-seen order, so a metric that starts reporting a new number needs no change here. Rows
that are missing a column render blank rather than dropping the row — one variant lacking a
fused cloud should not cost you the rest of the table.
"""

import csv
import io
from pathlib import Path

FORMATS = ("csv", "md", "tex")


class Table:
    def __init__(self, name: str):
        self.name = name
        self.rows: list[dict] = []
        self.columns: list[str] = []

    def add(self, row: dict) -> None:
        for key in row:
            if key not in self.columns:
                self.columns.append(key)
        self.rows.append(dict(row))

    def __len__(self) -> int:
        return len(self.rows)

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=self.columns, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        return buf.getvalue()

    def to_markdown(self, precision: int = 4) -> str:
        cells = [[_fmt(row.get(c, ""), precision) for c in self.columns] for row in self.rows]
        widths = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c)
                  for i, c in enumerate(self.columns)]
        lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(self.columns, widths)) + " |",
                 "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        lines += ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"
                  for row in cells]
        return "\n".join(lines)

    def to_latex(self, precision: int = 4, caption: str = None) -> str:
        head = " & ".join(_tex_escape(c) for c in self.columns) + r" \\"
        body = [" & ".join(_tex_escape(_fmt(row.get(c, ""), precision)) for c in self.columns)
                + r" \\" for row in self.rows]
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\begin{tabular}{" + "l" * len(self.columns) + "}", r"\toprule",
                 head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
        if caption:
            lines.append(r"\caption{" + _tex_escape(caption) + "}")
        lines += [r"\label{tab:" + self.name + "}", r"\end{table}"]
        return "\n".join(lines)

    def write(self, out_dir: Path, formats=("csv", "md")) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        render = {"csv": self.to_csv, "md": self.to_markdown,
                  "tex": lambda: self.to_latex(caption=f"{self.name} evaluation")}
        written = []
        for fmt in formats:
            if fmt not in render:
                raise ValueError(f"unknown format {fmt!r}; known: {FORMATS}")
            path = out_dir / f"{self.name}.{fmt}"
            path.write_text(render[fmt]() + "\n")
            written.append(path)
        return written


def _fmt(value, precision: int) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}f}" if 1e-3 <= abs(value) < 1e6 or value == 0 \
            else f"{value:.{precision}g}"
    return str(value)


def _tex_escape(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text
