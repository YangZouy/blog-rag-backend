"""从 Hexo 仓库中提取嵌入的 PDF 并将其转换为分块。

PDF 文件位于 `source/` 目录下（通常为 `source/files/`）。文本提取优先使用
PyMuPDF，若失败则回退至 pdfplumber。纯图片型 PDF（无文本层）会被标记为
`ocr=True`，以便导入步骤决定是进行 OCR 还是跳过；默认情况下，我们仍会索引
其中可用的少量文本，并标记该分块以便日后查找。
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple
import fitz  # PyMuPDF；运行时硬依赖，缺失应在模块导入时即报错
import pdfplumber  # 兜底提取器；同上

from data.parse_hexo import DocumentChunk, _splitter

# 私用区(PUA)项目符号：PDF 里常见的“•/◆/§”被嵌成私用区码点（如 U+F071），
# get_text 提取出来是乱码，会污染向量。统一还原成列表标记 "- "。
_PUA_BULLETS = {"\uf071", "\uf0b7", "\uf0a7", "\uf06c", "\uf0d8", "\uf0fc", "\uf0a8"}


def _is_pua(cp: int) -> bool:
    """是否落在 Unicode 私用区（BMP + 两个补充平面）。"""
    return (
        0xE000 <= cp <= 0xF8FF
        or 0xF0000 <= cp <= 0xFFFFD
        or 0x100000 <= cp <= 0x10FFFD
    )


def _clean_pdf_text(text: str) -> str:
    """清洗 PDF 提取文本：去私用区乱码、合并中文硬折行、折叠空白。

    只做“提取噪声”层面的清洗，不改语义。诊断显示红宝书唯一脏点是
    U+F071(×11) 列表符；dom&bom 系列本身干净。中文硬折行合并对所有
    PDF 都能提升嵌入质量（PDF 会在行宽处插入换行，把词切断）。
    """
    if not text:
        return text
    # 1) 私用区字符：项目符号还原为 "\n- "，其余 PUA 乱码直接删除
    out: List[str] = []
    for ch in text:
        if ch in _PUA_BULLETS:
            out.append("\n- ")
        elif _is_pua(ord(ch)):
            continue
        else:
            out.append(ch)
    text = "".join(out)
    # 2) 合并中文硬折行：两个 CJK 字符之间的单个换行是排版折行，去掉；
    #    段落空行(\n\n)与列表标记保留
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    # 3) 折叠多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


"""
这里返回的是tuple，固定结构的一组返回值
如果换成列表的话，更像同类元素的集合，可读性差一些
"""
def _extract_text(path: str) -> Tuple[str, bool]:
    """Return (text, needs_ocr). Empty text + needs_ocr=True means image-only."""
    text = ""
    try:
        doc = fitz.open(path)
        # get_text仅针对文字有效，如果是拍照扫描版pdf，无法提取出的内容
        text = "\n".join(page.get_text() for page in doc)
    except Exception:
        pass  # fitz 提取失败（损坏/扫描件）→ 回退 pdfplumber

    # pdfplumber兜底
    if not text.strip():
        try:
            with pdfplumber.open(path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            pass

    # 清洗提取噪声（私用区乱码 / 中文硬折行 / 多余空白）
    text = _clean_pdf_text(text)
    text = text.strip()
    # 如果兜底后依旧无法提取text，说明需要ocr了
    return text, (len(text) == 0)


def _pdf_url(repo_path: str, pdf_path: str) -> str:
    from core.config import get_settings

    s = get_settings()
    base = s.SITE_URL.rstrip("/")
    rel = os.path.relpath(pdf_path, os.path.join(repo_path, "source"))
    rel = rel.replace(os.sep, "/")
    return f"{base}/{rel}"


def parse_pdfs(repo_path: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    source_dir = os.path.join(repo_path, "source")
    if not os.path.isdir(source_dir):
        return chunks

    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, fname)
            # 将文件名拆成主名+扩展名
            slug = os.path.splitext(fname)[0]
            text, needs_ocr = _extract_text(path)
            if not text:
                # still register a placeholder chunk so the PDF is discoverable
                chunks.append(
                    DocumentChunk(
                        slug=slug,
                        title=slug,
                        url=_pdf_url(repo_path, path),
                        content="",
                        doc_type="pdf",
                        chunk_index=0,
                        ocr=True,
                    )
                )
                continue
            pieces = _splitter.split_text(text)
            for i, piece in enumerate(pieces):
                chunks.append(
                    DocumentChunk(
                        slug=slug,
                        title=slug,
                        url=_pdf_url(repo_path, path),
                        content=piece,
                        doc_type="pdf",
                        chunk_index=i,
                        ocr=needs_ocr,
                    )
                )
    return chunks
