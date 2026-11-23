"""Render a DeepDetect classification evaluation in a Visdom text window."""

from html import escape


def write_confusion_table(vis, test_key, matrix):
    labels = [escape(str(label)) for label in matrix["labels"]]
    headings = "".join(f'<th scope="col">{label}</th>' for label in labels)
    rows = []
    for i, (label, row) in enumerate(zip(labels, matrix["rows"], strict=True)):
        cells = []
        for j, count in enumerate(row["counts"]):
            if i == j:
                cells.append(
                    '<td style="background-color:#d8efdd;color:#143d20">'
                    f"<strong>{int(count)}</strong></td>"
                )
            else:
                cells.append(f"<td>{int(count)}</td>")
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    caption = escape(str(matrix.get("test_name") or test_key))
    if "iteration" in matrix:
        caption += f' — iteration {int(matrix["iteration"])}'
    vis.text(
        f'<table class="table table-condensed table-bordered"><caption>{caption}</caption>'
        f'<thead><tr><th>Actual ↓ / Predicted →</th>{headings}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>',
        win=f"confusion-{test_key}", opts={"title": f"Confusion matrix - {test_key}"},
    )
