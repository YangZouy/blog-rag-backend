"""将 Hexo 源仓库解析为分块文档。

每篇 Markdown 文章使用 python-frontmatter 读取；URL 根据 frontmatter 中的
`permalink`/`slug` 生成（回退至 Hexo 默认路由 /YYYY/MM/DD/slug/）。
文章被切分为约 500 字符且带有重叠的分块，每个分块共享文章的元数据，
以便引用能链接回原文。
"""
from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import frontmatter  # 运行时硬依赖：缺失应在模块导入时即报 ModuleNotFoundError，而非静默 0 chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import get_settings

logger = logging.getLogger("blog-rag")
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
    # 相对路径 /YYYY/MM/DD/slug/
    path: Optional[str] = None
    doc_type: str = "post"
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
        避免只嵌入孤立正文导致语义漂移"""
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
            "path": self.path,
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
            path=p.get("path"),
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

def _strip_origin(url: str, base: str) -> str:
    """把绝对 URL 归一成相对路径（/xxx/）。已是相对则原样返回。"""
    if url.startswith("/"):
        return url
    base = (base or "").rstrip("/")
    if base and url.startswith(base):
        rel = url[len(base):]
        return rel if rel.startswith("/") else "/" + rel
    return url  # 其它域名兜底原样

def _fetch_live_search_xml(site_url: str) -> dict[str, str]:
    """在线拉取 {SITE_URL}/search.xml → 标题→相对URL 映射。失败(离线/未部署)返回空。"""
    if not site_url:
        return {}
    xml_url = site_url.rstrip("/") + "/search.xml"
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        req = urllib.request.Request(xml_url, headers={"User-Agent": "blog-rag-ingest/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        mapping: dict[str, str] = {}
        for entry in root.findall("entry"):
            t = entry.find("title")
            u = entry.find("url")
            url = None
            if u is not None and (u.text or "").strip():
                url = u.text.strip()
            elif entry.find("link") is not None:
                url = (entry.find("link").get("href") or "").strip() or None
            if t is not None and t.text and url:
                mapping[_norm_title(t.text)] = _strip_origin(url, site_url)
        return mapping
    except Exception:
        return {}

def _parse_search_xml(xml_path: str, site_url: str) -> dict[str, str]:
    """解析 search.xml → 标题→相对URL 映射。失败返回空 dict。"""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        mapping: dict[str, str] = {}
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
                mapping[_norm_title(title_el.text)] = _strip_origin(url, site_url)
        return mapping
    except Exception:
        return {}

# search.xml（hexo-generator-search 产物）里 title→真实 URL 的映射，按 repo_path 缓存。
_SEARCH_URL_CACHE: dict[str, dict[str, str]] = {}

def _load_search_url_map(repo_path: str) -> dict[str, str]:
    """标题→相对URL 映射，权威来源优先级：

    ① 在线 {SITE_URL}/search.xml  —— 唯一权威源。用户真正访问的是线上站点，
       线上路由才是"真的"；本地 `hexo g` 产物只是预览态，可能因构建机时区不同而分叉。
    ② 本地 public/search.xml      —— 离线兜底（刚 hexo g 的产物，可能与线上不一致）。
    ③ 本地 .deploy_git/search.xml —— 最后兜底（历史部署快照，可能陈旧）。

    ⚠️ 已知分叉源：Hexo permalink 含 :year/:month/:day 时，生成结果依赖构建机系统时区。
       若 CI（UTC）与本地（UTC+8）不一致，凌晨 0-8 点发布的文章会差一天。
       根治方式是在 CI workflow 里设 job 级 `env: TZ: Asia/Shanghai`，而非改这里的优先级。
    """
    if repo_path in _SEARCH_URL_CACHE:
        return _SEARCH_URL_CACHE[repo_path]

    s = get_settings()

    # ① 在线 search.xml（权威：线上真实路由）
    live = _fetch_live_search_xml(s.SITE_URL)
    if live:
        logger.info("URL 源 = 在线 search.xml (%s)，%d 条", s.SITE_URL, len(live))
        _SEARCH_URL_CACHE[repo_path] = live
        return live

    # ②③ 本地文件兜底
    for rel, note in (
        (os.path.join("public", "search.xml"), "本地 hexo g 产物，可能与线上不一致"),
        (os.path.join(".deploy_git", "search.xml"), "历史部署快照，可能陈旧"),
    ):
        xml_path = os.path.join(repo_path, rel)
        if os.path.isfile(xml_path):
            mapping = _parse_search_xml(xml_path, s.SITE_URL)
            if mapping:
                logger.warning("在线 search.xml 不可用，改用 %s（%s），%d 条", rel, note, len(mapping))
                _SEARCH_URL_CACHE[repo_path] = mapping
                return mapping

    logger.error("所有 search.xml 源均不可用 —— 本次入库的文章将全部没有链接")
    _SEARCH_URL_CACHE[repo_path] = {}
    return {}

def _build_url(meta: dict, date, url_slug: str, is_post: bool,
               title: str = "", repo_path: str = "") -> str:
    """返回文章绝对 URL。无权威来源时返回 ""（宁可没链接，也不给 404 链接）。

    date 参数保留仅为兼容调用方签名，已不参与 URL 推导。
    """
    s = get_settings()
    base = s.SITE_URL.rstrip("/")
    # ① Hexo search.xml 权威 URL（线上真实路由）
    if title and repo_path:
        real = _load_search_url_map(repo_path).get(_norm_title(title))
        if real:
            return real if real.startswith("http") else f"{base}{real}"
    # ② frontmatter 显式 permalink
    permalink = meta.get("permalink") or meta.get("url")
    if permalink:
        return permalink if permalink.startswith("http") else f"{base}{permalink}"
    # ③ index.md 页面：目录即路由（about/index.md → /about/），确定性规则非猜测
    if not is_post and url_slug:
        return f"{base}/{url_slug}/"
    # ④ 无权威来源 → 空 URL。chunk 仍入库参与检索，只是不出现在「推荐阅读」
    logger.warning("no authoritative URL for %r (not in search.xml, no permalink)", title)
    return ""

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
    s = get_settings()
    path = _strip_origin(url, s.SITE_URL) if url else None
    return DocumentChunk(
        slug=slug,
        title=title,
        url=url,
        content=content,
        path=path,
        doc_type=doc_type,
        date=date,
        tags=tags or [],
        chunk_index=idx,
        section=section,
        description=desc,
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
                with open(path, encoding="utf-8") as fh:
                    # 读整个md文件 把头部元数据解析出来 把正文内容分离出来
                    post = frontmatter.load(fh)
            except Exception:
                logger.warning("跳过无法解析的 Markdown：%s", path)
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

            base_name = os.path.splitext(fname)[0].lower()
            is_index_page = base_name == "index"
            if is_index_page:
                # index.md 的路由 = 所在目录（about/index.md → /about/），确定性可靠
                url_slug = rel_noext.rsplit("/", 1)[0] if "/" in rel_noext else rel_noext
            else:
                url_slug = ""   # 不再猜测；URL 只能来自 search.xml 或 permalink
            slug = url_slug if is_index_page else rel_noext
            doc_type = "post" if is_post else "page"
            url = _build_url(meta, raw_date, url_slug, is_post,
                             title=title, repo_path=repo_path)

            chunks.extend(
                _split_post_into_chunks(slug, title, url, date_str, tags, body, doc_type, desc)
            )
    missing = sorted({c.title for c in chunks if not c.url})
    if missing:
        logger.warning(
            "以下 %d 篇未在 search.xml 中匹配到，已入库但无链接"
            "（多半是刚 push、GitHub Actions 还没部署完，等 1~2 分钟后重跑即可）：\n%s",
            len(missing), "\n".join("  - " + t for t in missing),
        )
    return chunks

