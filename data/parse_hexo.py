"""将 Hexo 源仓库解析为分块文档。

每篇 Markdown 文章使用 python-frontmatter 读取；URL 根据 frontmatter 中的
`permalink`/`slug` 生成（回退至 Hexo 默认路由 /YYYY/MM/DD/slug/）。
文章被切分为约 500 字符且带有重叠的分块，每个分块共享文章的元数据，
以便引用能链接回原文。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 中文优先的分隔符顺序：段落 → 句子 → 子句 → 字符
_ZH_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]

# 递归字符切分（PDF / 兜底用）
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=_ZH_SEPARATORS,
)

# 文章正文切块：单段过长时切分，500–800 字、80–120 重叠
_POST_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=_ZH_SEPARATORS,
)

"""
系统内部统一的知识片段，方便后面做embedding、存储、检索和生成答案

@dataclass是python标准库里的一个装饰器，简化纯数据类写法
"""
@dataclass
class DocumentChunk:
    # 文章短标识符
    slug: str
    title: str
    url: str
    content: str
    # 按照内容来源/类型分
    doc_type: str = "post"  # "post" | "pdf" | "web"
    date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    chunk_index: int = 0
    # 表示这个内容是否需要OCR
    ocr: bool = False
    # 所属章节标题（按 #/##/### 拆分得到），用于检索嵌入与判分上下文
    section: Optional[str] = None
    # 检索相似度分数（查询时由 retriever 填充，入库时为 None）
    score: Optional[float] = None
    description: Optional[str] = None

    def embed_text(self) -> str:
        """用于生成向量的组合文本：标题 / 标签 / 章节 / 正文一起嵌入，
        避免只嵌入孤立正文导致语义漂移（例如 Hyper-V 片段被误判为 RAG）。"""
        parts = []
        if self.title:
            parts.append(f"文章标题：{self.title}")
        if self.tags:
            parts.append("标签：" + "、".join(self.tags))
        if self.section:
            parts.append(f"章节：{self.section}")
        parts.append(self.content)
        return "\n".join(parts)

    # 将DocumentChunk对象转成普通字典，
    def to_payload(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "doc_type": self.doc_type,
            "date": self.date,
            "tags": self.tags,
            "chunk_index": self.chunk_index,
            "ocr": self.ocr,
            "section": self.section,
            "description": self.description
        }

    """
    声明类方法，表示不是绑定到某个对象实例上，而是绑定到类本身
    """
    @classmethod
    def from_payload(cls, p: dict) -> "DocumentChunk":
        return cls(
            slug=p["slug"],
            title=p["title"],
            url=p["url"],
            content=p["content"],
            doc_type=p.get("doc_type", "post"),
            date=p.get("date"),
            tags=p.get("tags", []),
            chunk_index=p.get("chunk_index", 0),
            ocr=p.get("ocr", False),
            section=p.get("section"),
            score=p.get("score"),
            description=p.get("description")
        )


def _norm_title(t: str) -> str:
    """标题归一化用于匹配：去首尾空白、合并内部空白、小写。
    Hexo search.xml 的标题与 frontmatter title 理应一致，归一化只为容错。"""
    return re.sub(r"\s+", " ", (t or "").strip()).lower()


# search.xml（hexo-generator-search 产物）里 title→真实 URL 的映射，按 repo_path 缓存。
_SEARCH_URL_CACHE: dict[str, dict[str, str]] = {}


def _load_search_url_map(repo_path: str) -> dict[str, str]:
    """从 Hexo 生成的 search.xml 读取「标题 → 真实相对 URL」映射。

    这是 URL 的权威来源：Hexo 用 hexo-permalink-pinyin 生成的 slug 才是线上
    真实路由。后端自己用 pypinyin 猜拼音会因多音字（自律 zi-lu vs zi-lv、
    离职 chi-zhi vs li-zhi）和全角符号残留字节导致 slug 不一致 → 推荐阅读 404。
    优先查这张表，猜拼音仅作兜底。

    在 repo_path 下按优先级查找：public/search.xml（hexo generate 产物）→
    .deploy_git/search.xml（hexo deploy 产物）。找不到或解析失败返回空表。
    """
    if repo_path in _SEARCH_URL_CACHE:
        return _SEARCH_URL_CACHE[repo_path]

    mapping: dict[str, str] = {}
    candidates = [
        os.path.join(repo_path, "public", "search.xml"),
        os.path.join(repo_path, ".deploy_git", "search.xml"),
    ]
    xml_path = next((p for p in candidates if os.path.isfile(p)), None)
    if xml_path is None:
        _SEARCH_URL_CACHE[repo_path] = mapping
        return mapping

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        for entry in tree.getroot().findall("entry"):
            title_el = entry.find("title")
            url_el = entry.find("url")
            url = None
            if url_el is not None and (url_el.text or "").strip():
                url = url_el.text.strip()
            else:
                link_el = entry.find("link")
                if link_el is not None:
                    url = (link_el.get("href") or "").strip() or None
            if title_el is not None and title_el.text and url:
                mapping[_norm_title(title_el.text)] = url
    except Exception:
        # 解析失败不应阻断建索引，退回拼音兜底。
        mapping = {}

    _SEARCH_URL_CACHE[repo_path] = mapping
    return mapping


def _pinyin_slug(text: str) -> str:
    """复刻 hexo-permalink-pinyin：中文转拼音、空格转连字符、整体小写。
    多音字可能与 Hexo 有细微差异；若文章 frontmatter 显式写了 permalink，
    则优先使用 permalink，不依赖此函数。"""
    try:
        from pypinyin import slug as _fn

        value = _fn(text, separator="-", strict=False).lower()
    except Exception:
        value = text.lower()
    value = re.sub(r"\s+", "-", value)
    # 只保留 [a-z0-9-]：清理全角符号（：、）经拼音转换后残留的非 ASCII 字节，
    # 与 hexo-permalink-pinyin 剥符号的行为对齐。仅兜底用，主路径走 search.xml。
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def _build_url(meta: dict, date, url_slug: str, is_post: bool,
               title: str = "", repo_path: str = "") -> str:
    s = get_settings()
    base = s.SITE_URL.rstrip("/")
    # 0) Hexo search.xml 权威 URL（线上真实路由，规避拼音猜测偏差）最优先。
    if title and repo_path:
        real = _load_search_url_map(repo_path).get(_norm_title(title))
        if real:
            return real if real.startswith("http") else f"{base}{real}"
    # 1) frontmatter 显式 permalink 次优先（精确，避免拼音多音字偏差）
    permalink = meta.get("permalink") or meta.get("url")
    if permalink:
        return permalink if permalink.startswith("http") else f"{base}{permalink}"
    # 2) 文章(post)：Hexo 默认 /YYYY/MM/DD/title/ 路由
    if is_post and date is not None:
        y = m = day = None
        if hasattr(date, "year"):
            y, m, day = date.year, date.month, date.day
        else:
            # frontmatter date 常被读成字符串 '2026-07-20 17:10:14'，提取 Y/M/D
            match = re.match(r"\s*(\d{4})-(\d{1,2})-(\d{1,2})", str(date))
            if match:
                y, m, day = (int(g) for g in match.groups())
        if y is not None:
            return f"{base}/{y:04d}/{m:02d}/{day:02d}/{url_slug}/"
    # 3) 页面(page) 及其它：直接用目录路径 /slug/（如 /about/、/projects/.../）
    return f"{base}/{url_slug}/"


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_WIKI_IMG_RE = re.compile(r"!\[\[[^\]]*\]\]")


def _split_by_headings(body: str):
    """按 #/##/### 标题切分正文，返回 [(section, text), ...]。
    section 取最近的 ##/### 标题；标题行本身保留在 text 首部，保证嵌入带上下文。"""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        return [(None, body)]
    out = []
    current_section = None
    pre = body[: matches[0].start()].strip()
    if pre:
        out.append((None, pre))
    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        seg = body[start:end].strip()
        if level >= 2:
            current_section = heading
        text = f"{'#' * level} {heading}\n\n{seg}".strip() if seg else f"{'#' * level} {heading}"
        out.append((current_section, text))
    return out


def _split_post_into_chunks(slug, title, url, date, tags, body, doc_type="post", desc=None):
    """把一篇文章切成带 section 的 chunk：先按标题分段，单段过长再切分；
    禁止只包含标题、图片链接或极短代码碎片的块入库。"""
    out = []
    idx = 0
    for section, text in _split_by_headings(body):
        if not text or len(text.strip()) < 50:
            continue
        no_img = _WIKI_IMG_RE.sub("", _IMG_RE.sub("", text))
        if len(no_img.strip()) < 40:
            continue
        if len(text) <= 700:
            out.append(_make_chunk(slug, title, url, date, tags, section, text, idx, doc_type, desc))
            idx += 1
        else:
            for piece in _POST_SPLITTER.split_text(text):
                if len(piece.strip()) < 40:
                    continue
                out.append(_make_chunk(slug, title, url, date, tags, section, piece, idx, doc_type, desc))
                idx += 1
    return out

def _make_chunk(slug, title, url, date, tags, section, content, idx, doc_type="post", desc=None):
    return DocumentChunk(
        slug=slug,
        title=title,
        url=url,
        content=content,
        doc_type=doc_type,
        date=date,
        tags=tags or [],
        chunk_index=idx,
        section=section,
        description=desc
    )


def parse_hexo_repo(repo_path: str) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    source_dir = os.path.join(repo_path, "source")
    if not os.path.isdir(source_dir):
        return chunks
    # os.walk：python标准库中用来递归遍历目录树
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith(".md"):
                continue
            path = os.path.join(root, fname)
            try:
                # 第三方库：python-frontmatter
                import frontmatter

                with open(path, encoding="utf-8") as fh:
                    # 读整个md文件 把头部元数据解析出来 把正文内容分离出来
                    post = frontmatter.load(fh)
            except Exception:
                continue

            meta = post.metadata
            desc = meta.get("description")
            if meta.get("published") is False or meta.get("draft"):
                continue
            body = post.content.strip()
            if not body:
                continue

            # 相对 source 的路径（统一为正斜杠、去掉扩展名），用作：
            #  - 全局唯一 slug（避免所有 index.md 都被算成 "index" 导致点 ID 冲突）
            #  - 判断该文件是「文章(post)」还是「页面(page)」
            rel = os.path.relpath(path, source_dir).replace("\\", "/")
            rel_noext = rel[:-3] if rel.endswith(".md") else rel
            is_post = rel_noext == "_posts" or rel_noext.startswith("_posts/")

            title = meta.get("title") or os.path.splitext(fname)[0]
            raw_date = meta.get("date")
            date_str = str(raw_date) if raw_date is not None else None
            raw_tags = meta.get("tags") or []
            if isinstance(raw_tags, str):
                # Hexo frontmatter 有时把 tags 写成 "a, b, c" 字符串而非 YAML 列表
                tags = [t.strip() for t in re.split(r"[,，、]", raw_tags) if t.strip()]
            else:
                tags = [str(t).strip() for t in raw_tags if str(t).strip()]

            # URL 段：index.md 页面用其所在目录（如 about、projects/ai-content-distribution）；
            # 普通文章/页面用标题拼音
            base_name = os.path.splitext(fname)[0].lower()
            if base_name == "index":
                url_slug = rel_noext.rsplit("/", 1)[0] if "/" in rel_noext else rel_noext
            else:
                url_slug = _pinyin_slug(title)
            # slug：index.md 页面用目录路径（如 about），其余用相对路径，均保证唯一
            slug = url_slug if base_name == "index" else rel_noext
            doc_type = "post" if is_post else "page"
            url = _build_url(meta, raw_date, url_slug, is_post,
                             title=title, repo_path=repo_path)

            chunks.extend(
                _split_post_into_chunks(slug, title, url, date_str, tags, body, doc_type, desc)
            )
    return chunks
