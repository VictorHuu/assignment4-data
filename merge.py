from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders


def ascii_safe(s):
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    return s.encode("ascii", "replace").decode("ascii")


def sanitize_http_headers(http_headers):
    if http_headers is None:
        return None

    # 优先保留原始状态行
    statusline = getattr(http_headers, "statusline", None) or getattr(http_headers, "protocol", None) or "HTTP/1.1 200 OK"

    cleaned = []
    for name, value in http_headers.headers:
        cleaned.append((ascii_safe(name), ascii_safe(value)))

    return StatusAndHeaders(
        statusline=ascii_safe(statusline),
        headers=cleaned,
        protocol=getattr(http_headers, "protocol", "HTTP/1.1"),
    )


inputs = [
    "subsampled_positive_urls_part1.warc.gz",
    "subsampled_positive_urls_part2.warc.gz",
]
output = "merged.warc.gz"

written = 0
sanitized = 0
skipped = 0

with open(output, "wb") as out_f:
    writer = WARCWriter(out_f, gzip=True)

    for path in inputs:
        with open(path, "rb") as in_f:
            for idx, record in enumerate(ArchiveIterator(in_f), 1):
                try:
                    if record.http_headers is not None:
                        try:
                            record.http_headers.to_ascii_bytes()
                        except UnicodeEncodeError:
                            record.http_headers = sanitize_http_headers(record.http_headers)
                            sanitized += 1

                    # 关键：让 warcio 重新计算长度
                    record.length = None

                    writer.write_record(record)
                    written += 1

                except Exception as e:
                    skipped += 1
                    rec_id = record.rec_headers.get_header("WARC-Record-ID") if record.rec_headers else "UNKNOWN"
                    rec_type = record.rec_headers.get_header("WARC-Type") if record.rec_headers else "UNKNOWN"
                    print(f"SKIP file={path} idx={idx} type={rec_type} id={rec_id} err={e!r}")

print(f"done: written={written}, sanitized={sanitized}, skipped={skipped}")