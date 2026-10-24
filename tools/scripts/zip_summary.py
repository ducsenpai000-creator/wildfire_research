"""Summarise a zip archive by directory (names and sizes only; contents are not decoded)."""
import collections
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as z:
    infos = [i for i in z.infolist() if not i.is_dir()]
    groups = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for i in infos:
        d = i.filename.rsplit("/", 1)[0] if "/" in i.filename else "."
        g = groups[d]
        g[0] += 1
        g[1] += i.file_size
        g[2][i.filename.rsplit(".", 1)[-1].lower()] += 1
    print(f"total files: {len(infos)}")
    for d in sorted(groups):
        n, size, ext = groups[d]
        print(f"{n:6d} {size / 2**20:9.1f} MB  {d}  {dict(ext)}")
    bad = z.testzip()
    print("CRC check:", "OK" if bad is None else f"first bad file {bad}")
