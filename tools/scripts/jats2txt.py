"""Convert a JATS full-text XML (Europe PMC) to plain text, keeping section titles and tables."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def text_of(el):
    return " ".join("".join(el.itertext()).split())


def walk(el, out, depth=0):
    tag = el.tag
    if tag == "title":
        out.append("\n" + "#" * min(depth, 4) + " " + text_of(el))
        return
    if tag == "p":
        out.append(text_of(el))
        return
    if tag in ("table-wrap", "fig"):
        out.append("\n[" + tag + "] " + " ".join(text_of(c) for c in el if c.tag in ("label", "caption")))
        for row in el.iter("tr"):
            out.append(" | ".join(text_of(c) for c in row))
        return
    for c in el:
        walk(c, out, depth + (1 if tag == "sec" else 0))


def main(path):
    root = ET.parse(path).getroot()
    out = []
    for part in ("front", "body", "back"):
        node = root.find(part)
        if node is not None:
            walk(node, out)
    dst = Path(path).with_suffix(".txt")
    dst.write_text("\n".join(out), encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main(sys.argv[1])
