import traceback
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
import time
import random
import os
import glob
from collections import Counter
from itertools import combinations
from difflib import SequenceMatcher
import platform
import datetime
import copy
from io import BytesIO
from paths import get_project_paths
import re
from difflib import SequenceMatcher

if not st.session_state.get("authentication_status"):
    st.info("请先登录系统 🔒")
    st.switch_page("Home.py")
    st.stop()

def get_api_key(provider_name: str) -> str:
    try:
        return st.secrets[provider_name]
    except Exception:
        return os.environ.get(provider_name, "")

# =======================================================================
# 0. 页面状态初始化 & 项目路径
# =======================================================================

USER_ROOT = st.session_state.get("user_root_dir", "")
USERNAME = st.session_state.get("username")

def init_analysis_session_state():
    defaults = {
        #"active_project_selector": "default_project",

        # 本页核心数据
        "open_codes": None,
        "axial_codes_df": pd.DataFrame(columns=['code', 'category', 'confidence', 'reasoning', 'status']),

        # 本页缓存
        "clusters_cache": None,
        "merge_groups": None,
        "alignment_results": None,
        "sortable_items": [],
        "kanban_meta": {},
        "page_num_align": 0,
        "peer_source_token": None,
        "last_alignment_signature": None,

        # 轴心分析上下文
        "kanban_ai_mode": "默认",
        "kanban_research_domain": "",
        "kanban_research_topic": "",

        "merge_mode": "默认",
        "merge_output_mode": "先预览后确认",
        "merge_groups": [],
        "merge_auto_groups": [],
        "merge_review_groups": [],

        # 自动保存显示
        "last_autosave_time": "",
        "last_autosave_reason": "",

        # 撤销
        "history_stack": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def hard_reset_analysis_project():
    st.session_state.open_codes = None
    st.session_state.axial_codes_df = pd.DataFrame(columns=['code', 'category', 'confidence', 'reasoning', 'status'])
    st.session_state.history_stack = []

    st.session_state.clusters_cache = None
    st.session_state.merge_groups = None
    st.session_state.alignment_results = None
    st.session_state.sortable_items = []
    st.session_state.kanban_meta = {}
    st.session_state.page_num_align = 0

    st.session_state.kanban_ai_mode = "默认"
    st.session_state.kanban_research_domain = ""
    st.session_state.kanban_research_topic = ""

    st.session_state.merge_mode = "默认"
    st.session_state.merge_output_mode = "先预览后确认"
    st.session_state.merge_groups = []
    st.session_state.merge_auto_groups = []
    st.session_state.merge_review_groups = []

    st.session_state.last_autosave_time = ""
    st.session_state.last_autosave_reason = ""

    st.session_state.peer_source_token = None
    st.session_state.last_alignment_signature = None

    for k in [
        'primary_uploader',
        'peer_uploader',
        'analysis_recovery_file',
        'main_data_load_key',
        'peer_data_load_key',
        'peer_uploaded_file',
        'peer_file_path',
        'kanban_research_domain_input_tab2',
        'kanban_research_topic_input_tab2',
        'kanban_research_domain_input_tab3',
        'kanban_research_topic_input_tab3',
    ]:
        if k in st.session_state:
            del st.session_state[k]

def reset_analysis_state():
    st.session_state.clusters_cache = None
    st.session_state.merge_groups = None
    st.session_state.alignment_results = None
    st.session_state.sortable_items = []
    st.session_state.kanban_meta = {}
    st.session_state.page_num_align = 0
    st.session_state.peer_source_token = None
    st.session_state.last_alignment_signature = None

init_analysis_session_state()

def sync_shared_text_state(shared_key, widget_key, mirror_keys=None):
    value = st.session_state.get(widget_key, "")
    st.session_state[shared_key] = value

    if mirror_keys:
        for k in mirror_keys:
            st.session_state[k] = value

def init_shared_widget_value(shared_key, widget_key):
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(shared_key, "")

def render_shared_domain_input(label, widget_key, mirror_keys=None, placeholder=""):
    init_shared_widget_value("kanban_research_domain", widget_key)
    st.text_input(
        label,
        key=widget_key,
        placeholder=placeholder,
        on_change=sync_shared_text_state,
        args=("kanban_research_domain", widget_key, mirror_keys or []),
    )

def render_shared_topic_input(label, widget_key, mirror_keys=None, placeholder=""):
    init_shared_widget_value("kanban_research_topic", widget_key)
    st.text_input(
        label,
        key=widget_key,
        placeholder=placeholder,
        on_change=sync_shared_text_state,
        args=("kanban_research_topic", widget_key, mirror_keys or []),
    )
    
# 先确定当前项目与路径（后面数据加载会直接用到）
existing_projects = []
if USER_ROOT and os.path.exists(USER_ROOT):
    existing_projects = [
        d for d in os.listdir(USER_ROOT)
        if os.path.isdir(os.path.join(USER_ROOT, d))
    ]

if not existing_projects:
    st.error("❌ 没有可用项目")
    st.stop()

if st.session_state.get("active_project_selector") not in existing_projects:
    st.session_state["active_project_selector"] = existing_projects[0]

selected_project = st.session_state["active_project_selector"]
DIRS = get_project_paths(USERNAME, selected_project)

INPUT_DIR = DIRS["opening_final"]
RECOVERY_DIR = DIRS["analysis_autosave"]
FINAL_DIR = DIRS["analysis_final"]

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.session_state.api_key = get_api_key("QWEN_API_KEY")

if not st.session_state.api_key:
    st.error("未读取到系统 API Key，请检查 .streamlit/secrets.toml 或环境变量配置。")
    st.stop()

api_ready = bool(st.session_state.api_key)

# 尝试导入拖拽库
try:
    from streamlit_sortables import sort_items
    HAS_SORTABLE = True
except ImportError:
    HAS_SORTABLE = False

# =======================================================================
# 1. 核心工具函数 & 数据加载模块
# =======================================================================

def ensure_recovery_dir():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR, exist_ok=True)

def load_from_jsonl(filepath):
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip():
                        records.append(json.loads(line))
                except:
                    continue

    flat_codes = []
    for r in records:
        idx = r.get('original_row_index')
        codes_list = r.get('generated_codes', [])
        source_file = r.get('source_file', 'unknown')
        original_ids = r.get('original_ids', [])
        batch_id = r.get('batch_id', None)

        if isinstance(codes_list, list):
            for c in codes_list:
                if isinstance(c, dict):
                    flat_codes.append({
                        'source_file': source_file,
                        'original_row_index': idx,
                        'original_ids': c.get('original_ids', original_ids),
                        'batch_id': c.get('batch_id', batch_id),
                        'original_code': c.get('original_code', c.get('code')),
                        'peer_code': c.get('peer_code', None),
                        'aligned_code': c.get('aligned_code', c.get('code')),
                        'code': c.get('code'),
                        'quote': c.get('quote'),
                        'confidence': c.get('confidence', 0)
                    })
    return pd.DataFrame(flat_codes)

def load_from_uploaded_jsonl(uploaded_file):
    records = []
    uploaded_file.seek(0)
    for raw_line in uploaded_file:
        try:
            line = raw_line.decode("utf-8").strip()
            if line:
                records.append(json.loads(line))
        except:
            continue

    flat_codes = []
    for r in records:
        idx = r.get('original_row_index')
        codes_list = r.get('generated_codes', [])
        source_file = r.get('source_file', 'unknown')
        original_ids = r.get('original_ids', [])
        batch_id = r.get('batch_id', None)

        if isinstance(codes_list, list):
            for c in codes_list:
                if isinstance(c, dict):
                    flat_codes.append({
                        'source_file': source_file,
                        'original_row_index': idx,
                        'original_ids': c.get('original_ids', original_ids),
                        'batch_id': c.get('batch_id', batch_id),
                        'original_code': c.get('original_code', c.get('code')),
                        'peer_code': c.get('peer_code', None),
                        'aligned_code': c.get('aligned_code', c.get('code')),
                        'code': c.get('code'),
                        'quote': c.get('quote'),
                        'confidence': c.get('confidence', 0)
                    })
    return pd.DataFrame(flat_codes)

def save_current_progress(df):
    """保存全量状态"""
    if df.empty:
        return None
    ensure_recovery_dir()
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"VisualAnalysis_Full_{date_str}.jsonl"
    filepath = os.path.join(RECOVERY_DIR, filename)

    if 'original_row_index' in df.columns:
        grouped = df.groupby('original_row_index')
        with open(filepath, "w", encoding="utf-8") as f:
            for idx, group in grouped:
                first_row = group.iloc[0]
                codes_list = []
                for _, row in group.iterrows():
                    codes_list.append({
                        "original_ids": row.get('original_ids', []),
                        "batch_id": row.get('batch_id', None),
                        "original_code": row.get('original_code'),
                        "peer_code": row.get('peer_code'),
                        "aligned_code": row.get('aligned_code'),
                        "code": row['code'],
                        "quote": row['quote'],
                        "confidence": row.get('confidence', 0)
                    })
                record = {
                    "original_row_index": int(idx) if pd.notna(idx) else None,
                    "original_ids": first_row.get('original_ids', []),
                    "batch_id": first_row.get('batch_id', None),
                    "source_file": first_row.get('source_file', 'unknown'),
                    "generated_codes": codes_list,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return filename
    return None

def build_final_analysis_table():
    """
    当前页面输出给下一页和 analysis_final 的标准表：
    - 内部仍使用 original_code / peer_code / aligned_code / code
    - 输出时统一整理为：
      original_row_index / quote / original_opening_code / peer_opening_code /
      aligned_opening_code / code / recommend_axial_code / source_file /
      confidence / original_ids / batch_id
    """
    if st.session_state.open_codes is None or st.session_state.open_codes.empty:
        return pd.DataFrame()

    final_df = st.session_state.open_codes.copy()

    if (
        'axial_codes_df' in st.session_state
        and st.session_state.axial_codes_df is not None
        and not st.session_state.axial_codes_df.empty
        and 'code' in st.session_state.axial_codes_df.columns
        and 'category' in st.session_state.axial_codes_df.columns
    ):
        axial_map = (
            st.session_state.axial_codes_df[['code', 'category']]
            .dropna(subset=['code'])
            .drop_duplicates(subset=['code'], keep='last')
            .rename(columns={'category': 'recommend_axial_code'})
        )

        if 'recommend_axial_code' in final_df.columns:
            final_df = final_df.drop(columns=['recommend_axial_code'])

        final_df = final_df.merge(axial_map, on='code', how='left')
    elif 'recommend_axial_code' not in final_df.columns:
        final_df['recommend_axial_code'] = ""

    def get_series(df_source, primary, fallback=None):
        if primary in df_source.columns:
            return df_source[primary]
        if fallback and fallback in df_source.columns:
            return df_source[fallback]
        return pd.Series([""] * len(df_source), index=df_source.index)

    export_df = pd.DataFrame({
        'original_row_index': get_series(final_df, 'original_row_index'),
        'quote': get_series(final_df, 'quote'),
        'original_opening_code': get_series(final_df, 'original_code', 'code'),
        'peer_opening_code': get_series(final_df, 'peer_code'),
        'aligned_opening_code': get_series(final_df, 'aligned_code', 'code'),
        'code': get_series(final_df, 'code'),
        'recommend_axial_code': get_series(final_df, 'recommend_axial_code'),
        'source_file': get_series(final_df, 'source_file'),
        'confidence': get_series(final_df, 'confidence'),
        'original_ids': get_series(final_df, 'original_ids'),
        'batch_id': get_series(final_df, 'batch_id'),
    })

    preferred_cols = [
        'original_row_index',
        'quote',
        'original_opening_code',
        'peer_opening_code',
        'aligned_opening_code',
        'code',
        'recommend_axial_code',
        'source_file',
        'confidence',
        'original_ids',
        'batch_id'
    ]
    return export_df[preferred_cols]

def final_table_to_excel(final_df):
    import datetime
    import os

    output = BytesIO()

    # ========= 1️⃣ 构造维度sheet =========
    meta = st.session_state.get("kanban_meta", {})
    fixed_headers = {"❓ 待定区", "🗑️ 回收站"}

    dims_data = []
    for dim, info in meta.items():
        if dim in fixed_headers:
            continue
        dims_data.append({
            "dimension": dim,
            "definition": info.get("definition", "")
        })

    dims_df = pd.DataFrame(dims_data)

    # ========= 2️⃣ 写Excel（两个sheet） =========
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="analysis_result", index=False)
        dims_df.to_excel(writer, sheet_name="axial_dimensions", index=False)

    output.seek(0)

    # ========= 3️⃣ 生成文件名 =========
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aligned_merge_category_{time_str}.xlsx"

    return output.getvalue(), filename

def save_final_outputs(excel_data, filename="page4_input.xlsx"):
    """
    保存当前页面标准输出到项目目录 analysis_final
    """
    os.makedirs(FINAL_DIR, exist_ok=True)
    save_path = os.path.join(FINAL_DIR, filename)

    try:
        with open(save_path, "wb") as f:
            f.write(excel_data)
        return True, save_path
    except Exception as e:
        return False, str(e)

def autosave_final_snapshot(reason=""):
    """
    关键操作后自动保存：
    - 保存恢复用 jsonl 到 analysis_autosave
    - 保存标准输入表到 analysis_final/page4_input.xlsx
    """
    if st.session_state.open_codes is None or st.session_state.open_codes.empty:
        return False, "暂无数据"

    try:
        save_current_progress(st.session_state.open_codes)
    except Exception:
        pass

    final_df = build_final_analysis_table()
    if final_df.empty:
        return False, "暂无数据"

    excel_data, _ = final_table_to_excel(final_df)
    ok, msg = save_final_outputs(excel_data, filename="page4_input.xlsx")

    if ok:
        st.session_state["last_autosave_reason"] = reason
        st.session_state["last_autosave_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return ok, msg

# [UPDATED] 撤销系统 (Undo System)
def push_history(action_name="Unknown Action"):
    """在修改数据前调用，保存当前状态快照"""
    if 'history_stack' not in st.session_state:
        st.session_state.history_stack = []

    if len(st.session_state.history_stack) >= 10:
        st.session_state.history_stack.pop(0)

    snapshot = {
        'open_codes': st.session_state.open_codes.copy(deep=True) if isinstance(st.session_state.open_codes, pd.DataFrame) else None,
        'axial_codes_df': st.session_state.axial_codes_df.copy(deep=True) if isinstance(st.session_state.axial_codes_df, pd.DataFrame) else pd.DataFrame(),
        'sortable_items': copy.deepcopy(st.session_state.get('sortable_items', [])),
        'kanban_meta': copy.deepcopy(st.session_state.get('kanban_meta', {})),
        'clusters_cache': copy.deepcopy(st.session_state.get('clusters_cache')),
        'merge_groups': copy.deepcopy(st.session_state.get('merge_groups')),
        'alignment_results': copy.deepcopy(st.session_state.get('alignment_results')),
        'desc': action_name,
        'time': time.strftime("%H:%M:%S")
    }
    st.session_state.history_stack.append(snapshot)

def normalize_to_list(value):
    """把 original_ids 这类字段统一转成 list，避免 None / str / 单值导致 set(None) 报错"""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [x for x in value if pd.notna(x)]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [x for x in parsed if pd.notna(x)]
            except Exception:
                pass
        return [text]

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [value]


def get_peer_source_token(peer_source):
    """给队友文件生成一个稳定 token，用来判断用户是不是换了文件"""
    if peer_source is None:
        return None

    if isinstance(peer_source, str):
        try:
            mtime = os.path.getmtime(peer_source)
        except Exception:
            mtime = 0
        return f"path::{os.path.abspath(peer_source)}::{mtime}"

    name = getattr(peer_source, "name", "uploaded")
    size = getattr(peer_source, "size", 0)
    return f"upload::{name}::{size}"


def clear_alignment_state(clear_peer_code=True):
    """重新对比前，先清空旧结果、旧分页、旧 peer_code"""
    st.session_state.alignment_results = None
    st.session_state.page_num_align = 0
    st.session_state.last_alignment_signature = None

    if clear_peer_code and st.session_state.get("open_codes") is not None and not st.session_state.open_codes.empty:
        st.session_state.open_codes = st.session_state.open_codes.copy()
        if "peer_code" in st.session_state.open_codes.columns:
            st.session_state.open_codes["peer_code"] = None

def align_records_by_ids_and_batch(df_mine, df_theirs, overlap_threshold=0.66, match_threshold=0.6):
    """
    基于 original_ids 和 batch_id 对齐队友编码。
    - 优先 original_ids（完全或部分重叠）
    - 部分重叠用 overlap_ratio >= overlap_threshold 判定
    - original_ids 不匹配时用 batch_id 匹配
    - fallback 可选文本相似度
    """
    from difflib import SequenceMatcher

    if df_mine is None or df_theirs is None or df_mine.empty or df_theirs.empty:
        return []

    theirs_records = df_theirs.to_dict('records')
    alignment = []
    mine_records = df_mine.to_dict('records')

    # 建立索引
    theirs_by_batch = {}
    theirs_by_ids = {}

    for r in theirs_records:
        bid = r.get('batch_id')
        if bid is not None:
            theirs_by_batch.setdefault(bid, []).append(r)

        oids = normalize_to_list(r.get('original_ids'))
        if oids:
            key = frozenset(oids)
            theirs_by_ids.setdefault(key, []).append(r)

    for my_row in mine_records:
        my_code = str(my_row.get('code', '') or '')
        my_quote = str(my_row.get('quote', '') or '')
        my_bid = my_row.get('batch_id')
        my_oids = set(normalize_to_list(my_row.get('original_ids')))

        best_match = None
        best_score = 0.0
        match_type = None

        # --------- 优先 original_ids 匹配 ---------
        for their_oids_set, records in theirs_by_ids.items():
            their_oids = set(their_oids_set)
            intersection = len(my_oids & their_oids)
            union = len(my_oids | their_oids)
            if union == 0:
                continue
            overlap_ratio = intersection / union
            if overlap_ratio == 1.0:
                # 完全匹配，直接用第一个
                best_match = records[0]
                best_score = 1.0
                match_type = 'full'
                break
            elif overlap_ratio >= overlap_threshold and overlap_ratio > best_score:
                best_match = records[0]
                best_score = overlap_ratio
                match_type = 'partial'

        # --------- batch_id 回退匹配 ---------
        if best_match is None and my_bid is not None:
            candidates = theirs_by_batch.get(my_bid, [])
            for r in candidates:
                # 可加文本相似度判断
                if not my_quote and not r.get('quote', ''):
                    ratio = 1.0
                else:
                    ratio = SequenceMatcher(None, my_quote, str(r.get('quote', ''))).ratio()
                if ratio > best_score:
                    best_match = r
                    best_score = ratio
                    match_type = 'batch'

        # --------- 状态判定 ---------
        status = "unique"
        their_code = None
        if best_match is not None and best_score >= match_threshold:
            their_code = str(best_match.get('code', ''))
            if my_code.strip() == their_code.strip():
                status = "agreed"
            else:
                status = "conflict"

        alignment.append({
            "quote": my_quote,
            "my_code": my_code,
            "their_code": their_code,
            "status": status,
            "score": best_score,
            "match_type": match_type,
            "raw_row_idx": my_row.get('original_row_index'),
            "raw_original_ids": my_row.get('original_ids'),
            "batch_id": my_bid
        })

    return alignment

def perform_undo():
    """执行撤销"""
    if 'history_stack' in st.session_state and st.session_state.history_stack:
        last_state = st.session_state.history_stack.pop()
        st.session_state.open_codes = last_state.get('open_codes')
        st.session_state.axial_codes_df = last_state.get('axial_codes_df', pd.DataFrame())
        st.session_state.sortable_items = last_state.get('sortable_items', [])
        st.session_state.kanban_meta = last_state.get('kanban_meta', {})
        st.session_state.clusters_cache = last_state.get('clusters_cache')
        st.session_state.merge_groups = last_state.get('merge_groups')
        st.session_state.alignment_results = last_state.get('alignment_results')
        try:
            autosave_final_snapshot("撤销操作")
        except Exception:
            pass
        st.toast(f"已撤销: {last_state['desc']}")
        time.sleep(0.5)
        st.rerun()
    else:
        st.warning("没有可撤销的操作")

@st.cache_data(show_spinner=False)
def normalize_code_text(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[，。、“”‘’；：,.!?！？（）()\[\]{}\-_/\\]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text

def char_ngrams(text, n=2):
    text = normalize_code_text(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def lexical_similarity(a, b):
    a_norm = normalize_code_text(a)
    b_norm = normalize_code_text(b)
    if not a_norm or not b_norm:
        return 0.0

    seq_score = SequenceMatcher(None, a_norm, b_norm).ratio()

    grams_a = char_ngrams(a_norm, 2)
    grams_b = char_ngrams(b_norm, 2)
    inter = len(grams_a & grams_b)
    union = max(1, len(grams_a | grams_b))
    jaccard = inter / union

    contain_bonus = 0.12 if (a_norm in b_norm or b_norm in a_norm) else 0.0
    score = 0.45 * seq_score + 0.45 * jaccard + contain_bonus
    return min(1.0, score)

def get_embeddings_dashscope(text_list, api_key):
    if not text_list:
        return []
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            resp = client.embeddings.create(
                model="text-embedding-v2",
                input=batch
            )
            batch_emb = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_emb)
        except Exception as e:
            st.error(f"Embedding API Error: {e}")
            return []
    return np.array(all_embeddings)

def perform_clustering(codes, embeddings, n_clusters=None, distance_threshold=0.6):
    if len(codes) < 2:
        return {0: codes}

    if n_clusters is not None:
        n_clusters = max(1, min(int(n_clusters), len(codes)))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )

    labels = clustering.fit_predict(embeddings)
    clusters = {}
    for code, label in zip(codes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(code)
    return clusters

def extract_code_from_sortable_item(item_text):
    text = str(item_text)
    return text.split(" (x")[0].strip()

def build_sortable_label(code, freq_map):
    return f"{code} (x{int(freq_map.get(code, 1))})"

def make_unique_header(base_name, used_names):
    name = (base_name or "").strip() or "未命名维度"
    candidate = name
    idx = 2
    while candidate in used_names:
        candidate = f"{name}{idx}"
        idx += 1
    used_names.add(candidate)
    return candidate

def save_current_kanban_result(sortable_items, fixed_headers):
    new_recs = []
    for g in sortable_items:
        cat = g["header"]
        if cat in fixed_headers:
            continue

        for it in g.get("items", []):
            code = extract_code_from_sortable_item(it)
            new_recs.append({
                'code': code,
                'category': cat,
                'confidence': 5,
                'reasoning': 'AI推荐+人工调整' if st.session_state.get("clusters_cache") else '人工拖拽',
                'status': 'Accepted'
            })

    if not new_recs:
        return 0

    ndf = pd.DataFrame(new_recs)
    st.session_state.axial_codes_df = pd.concat([
        st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].isin(ndf['code'])],
        ndf
    ], ignore_index=True)
    return len(new_recs)

def parse_ai_json_content(content):
    if not content:
        return []

    text = content.strip()

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    start_obj = text.find("{")
    end_obj = text.rfind("}")

    if start_arr != -1 and end_arr != -1:
        text = text[start_arr:end_arr + 1]
    elif start_obj != -1 and end_obj != -1:
        text = text[start_obj:end_obj + 1]

    parsed = json.loads(text)

    if isinstance(parsed, dict):
        if isinstance(parsed.get("clusters"), list):
            return parsed["clusters"]
        return [parsed]

    if isinstance(parsed, list):
        return parsed

    return []

def generate_ai_dimension_meta(cluster_payload, api_key, research_domain="", research_topic=""):
    if not cluster_payload:
        return {}

    research_domain = (research_domain or "").strip()
    research_topic = (research_topic or "").strip()

    prompt = f"""
你正在帮助研究者进行中文质性研究的“轴心编码”。

研究领域：{research_domain if research_domain else "未提供"}
研究主题：{research_topic if research_topic else "未提供"}

下面给你若干组已经聚好的开放编码，请你结合上述研究领域与研究主题，为每一组生成：

1. 一个简短、像研究维度的中文名称（2~8个字）
2. 一句非常短的定义（8~20个字）

请注意：
- 这不是普通的语义聚类命名
- 必须尽量贴合“研究主题”进行轴心归类
- 维度名称应具有研究解释性，而不是仅仅概括字面意思

请严格返回 JSON 数组，不要解释，不要 markdown，不要代码块。
格式如下：
[
  {{
    "cluster_id": "0",
    "header": "初始蜜月期",
    "definition": "初入组织时的积极体验"
  }}
]

要求：
- 名称必须像研究维度，不要用“其他”“杂项”“维度1”
- 定义必须是一句话，简洁、能直接显示在看板列头下
- 用词尽量概括，不要照抄编码
- 必须覆盖每个 cluster_id

聚类数据如下：
{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        res = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = res.choices[0].message.content.strip()
        parsed = parse_ai_json_content(content)

        meta = {}
        for item in parsed:
            cid = str(item.get("cluster_id", "")).strip()
            if cid:
                meta[cid] = {
                    "header": str(item.get("header", "")).strip(),
                    "definition": str(item.get("definition", "")).strip()
                }
        return meta
    except Exception:
        return {}

def build_ai_kanban(df, api_key, mode="默认", research_domain="", research_topic=""):
    if df is None or df.empty or 'code' not in df.columns:
        return [], {}

    freq_map = df['code'].dropna().astype(str).value_counts().to_dict()
    codes = list(freq_map.keys())

    if not codes:
        return [], {}

    n_codes = len(codes)

    if n_codes <= 4:
        target_clusters = n_codes
    else:
        if mode == "更细分":
            target_clusters = min(max(4, round(n_codes * 0.5)), min(10, n_codes))
        elif mode == "更收敛":
            target_clusters = min(max(3, round(n_codes * 0.25)), min(8, n_codes))
        else:
            target_clusters = min(max(4, round(n_codes * 0.35)), min(9, n_codes))

    target_clusters = max(1, min(int(target_clusters), n_codes))

    embs = get_embeddings_dashscope(codes, api_key)
    if len(embs) == 0:
        return [], {}

    raw_clusters = perform_clustering(codes, embs, n_clusters=target_clusters)

    cluster_items = []
    for lbl, items in raw_clusters.items():
        ordered_codes = sorted(items, key=lambda x: (-freq_map.get(x, 0), x))
        cluster_items.append({
            "cluster_id": str(lbl),
            "codes": [{"code": c, "count": int(freq_map.get(c, 0))} for c in ordered_codes],
            "weight": int(sum(freq_map.get(c, 0) for c in ordered_codes))
        })

    cluster_items = sorted(cluster_items, key=lambda x: (-x["weight"], x["cluster_id"]))

    ai_meta = generate_ai_dimension_meta(
        [{"cluster_id": g["cluster_id"], "codes": g["codes"]} for g in cluster_items],
        api_key,
        research_domain=research_domain,
        research_topic=research_topic
    )

    used_names = {"❓ 待定区", "🗑️ 回收站"}
    sortable_items = [{"header": "❓ 待定区", "items": []}]
    kanban_meta = {
        "❓ 待定区": {"definition": "暂未归类的编码", "locked": True},
        "🗑️ 回收站": {"definition": "不纳入当前轴心结果", "locked": True},
    }

    for cluster in cluster_items:
        cluster_id = cluster["cluster_id"]
        codes_in_group = [x["code"] for x in cluster["codes"]]
        fallback_name = codes_in_group[0] if codes_in_group else f"维度{cluster_id}"

        ai_info = ai_meta.get(cluster_id, {})
        header = make_unique_header(ai_info.get("header") or fallback_name, used_names)
        definition = (ai_info.get("definition") or f"围绕“{fallback_name}”形成的相关经验").strip()

        sortable_items.append({
            "header": header,
            "items": [build_sortable_label(c, freq_map) for c in codes_in_group]
        })
        kanban_meta[header] = {
            "definition": definition,
            "locked": False
        }

    sortable_items.append({"header": "🗑️ 回收站", "items": []})
    return sortable_items, kanban_meta

def find_synonym_groups(codes, embeddings, threshold=0.85):
    if len(codes) < 2:
        return {}
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    groups = {}
    for i, (code, label) in enumerate(zip(codes, labels)):
        if label not in groups:
            groups[label] = {"codes": [], "indices": []}
        groups[label]["codes"].append(code)
        groups[label]["indices"].append(i)
    result_groups = {}
    for lbl, data in groups.items():
        if len(data["codes"]) > 1:
            if len(data["indices"]) > 1:
                group_emb = embeddings[data["indices"]]
                sim_matrix = cosine_similarity(group_emb)
                avg_sim = np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
            else:
                avg_sim = 1.0
            result_groups[lbl] = {"codes": data["codes"], "score": avg_sim}
    return result_groups

# =========================
# TAB 2 智能标签清洗：新增函数
# =========================

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def get_merge_strategy(mode="默认"):
    """
    三档模式不是只改阈值，而是改策略：
    - 严格：少自动合并，多人工确认
    - 默认：平衡
    - 宽松：多召回候选，但不等于更多自动保存
    """
    strategy_map = {
        "严格": {
            "candidate_topk": 4,
            "full_pair_limit": 220,
            "max_candidate_pairs": 48,
            "embedding_threshold": 0.84,
            "lexical_threshold": 0.89,
            "fusion_threshold": 0.85,
            "contain_embedding_floor": 0.80,
            "component_max_size": 5,
            "max_components": 16,
            "auto_confidence": 0.90,
            "review_confidence": 0.42,
            "mode_hint": "严格模式下，自动合并只保留高把握结果；边界样本优先进入 review。"
        },
        "默认": {
            "candidate_topk": 6,
            "full_pair_limit": 220,
            "max_candidate_pairs": 72,
            "embedding_threshold": 0.79,
            "lexical_threshold": 0.84,
            "fusion_threshold": 0.81,
            "contain_embedding_floor": 0.74,
            "component_max_size": 6,
            "max_components": 20,
            "auto_confidence": 0.84,
            "review_confidence": 0.38,
            "mode_hint": "默认模式下，自动合并与人工确认保持平衡。"
        },
        "宽松": {
            "candidate_topk": 10,
            "full_pair_limit": 220,
            "max_candidate_pairs": 120,
            "embedding_threshold": 0.72,
            "lexical_threshold": 0.76,
            "fusion_threshold": 0.74,
            "contain_embedding_floor": 0.68,
            "component_max_size": 8,
            "max_components": 24,
            "auto_confidence": 0.86,
            "review_confidence": 0.34,
            "mode_hint": "宽松模式表示尽量多召回候选，但更多结果应进入 review，而不是自动保存。"
        }
    }
    return strategy_map.get(mode, strategy_map["默认"]).copy()

def normalize_merge_decision(decision):
    text = str(decision or "").strip().lower()

    if (
        text in {"auto_merge", "direct_merge", "merge", "auto", "yes"}
        or "直接合并" in text
        or "可直接合并" in text
        or "自动合并" in text
    ):
        return "auto_merge"

    if (
        text in {"review", "manual", "uncertain", "check"}
        or "人工确认" in text
        or "建议确认" in text
        or "review" in text
    ):
        return "review"

    if (
        text in {"reject", "no_merge", "no", "different"}
        or "不应合并" in text
        or "不能合并" in text
        or "reject" in text
    ):
        return "reject"

    return "review"

def build_candidate_pairs(codes, embeddings, freq_map, top_percent=0.10, mode="默认"):
    """
    构建候选同义编码对（pair-level）
    - 使用动态阈值，只保留 top N% 高相似对
    - 不直接应用 merge
    """
    strategy = get_merge_strategy(mode)
    if len(codes) < 2 or len(embeddings) == 0:
        return []

    sim_matrix = cosine_similarity(embeddings)
    candidate_pairs = []

    # 扁平化编码对
    n = len(codes)
    for i in range(n):
        for j in range(i + 1, n):
            fusion_sim = 0.72 * sim_matrix[i, j] + 0.28 * lexical_similarity(codes[i], codes[j])
            candidate_pairs.append({
                "code_a": codes[i],
                "code_b": codes[j],
                "fusion_similarity": fusion_sim,
                "embedding_similarity": sim_matrix[i, j],
                "lexical_similarity": lexical_similarity(codes[i], codes[j]),
            })

    # 动态阈值：保留 top N%
    candidate_pairs = sorted(candidate_pairs, key=lambda x: -x["fusion_similarity"])
    top_k = max(1, int(len(candidate_pairs) * top_percent))
    return candidate_pairs[:top_k]

def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def default_merged_name(codes, freq_map):
    codes = dedupe_preserve_order(codes)
    if not codes:
        return ""
    return sorted(codes, key=lambda x: (-freq_map.get(x, 0), len(x), x))[0]

def build_candidate_components(candidate_pairs, codes, embeddings, freq_map, mode="默认"):
    """
    把 pair 候选组织成候选小组：
    - 先按图的连通分量组织
    - 再把过大的组按 embedding 再拆小
    这样第二层 LLM 可以按“小组”复核，而不是逐 pair 判断
    """
    if not candidate_pairs:
        return []

    strategy = get_merge_strategy(mode)
    max_group_size = strategy["component_max_size"]

    code_to_idx = {str(c): i for i, c in enumerate(codes)}

    # 建图
    adj = {}
    for p in candidate_pairs:
        a = str(p["code_a"])
        b = str(p["code_b"])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # 找连通分量
    visited = set()
    raw_components = []

    for node in adj.keys():
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)

        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)

        comp = sorted(comp, key=lambda x: (-freq_map.get(x, 0), x))
        raw_components.append(comp)

        def split_component_rec(code_list):
            code_list = dedupe_preserve_order(code_list)

            # 🚀 新增颗粒度保护
            MIN_GROUP_SIZE_FOR_SPLIT = 3
            if len(code_list) <= max_group_size or len(code_list) <= MIN_GROUP_SIZE_FOR_SPLIT:
                return [sorted(code_list, key=lambda x: (-freq_map.get(x, 0), x))]

            idxs = [code_to_idx[c] for c in code_list if c in code_to_idx]
            if len(idxs) < 2:
                chunks = [code_list[i:i + max_group_size] for i in range(0, len(code_list), max_group_size)]
                return [sorted(ch, key=lambda x: (-freq_map.get(x, 0), x)) for ch in chunks]

            sub_embs = embeddings[idxs]
            n_clusters = max(2, int(np.ceil(len(code_list) / max_group_size)))
            n_clusters = min(n_clusters, max(2, len(code_list) - 1))

            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(sub_embs)

                buckets = {}
                for code, label in zip(code_list, labels):
                    buckets.setdefault(int(label), []).append(code)

                parts = []
                for _, bucket_codes in buckets.items():
                    if len(bucket_codes) > max_group_size and len(bucket_codes) >= MIN_GROUP_SIZE_FOR_SPLIT:
                        parts.extend(split_component_rec(bucket_codes))
                    else:
                        parts.append(sorted(bucket_codes, key=lambda x: (-freq_map.get(x, 0), x)))
                return parts

            except Exception:
                chunks = [code_list[i:i + max_group_size] for i in range(0, len(code_list), max_group_size)]
                return [sorted(ch, key=lambda x: (-freq_map.get(x, 0), x)) for ch in chunks]

    split_components = []
    for comp in raw_components:
        split_components.extend(split_component_rec(comp))

    # 过滤只剩 1 个标签的小组（1 个标签无需合并）
    split_components = [c for c in split_components if len(c) >= 2]

    # 按组的重要性排序：组内标签总频次优先，其次组大小
    split_components = sorted(
        split_components,
        key=lambda grp: (-sum(freq_map.get(c, 0) for c in grp), -len(grp), grp[0])
    )[:strategy["max_components"]]

    results = []
    for idx, grp in enumerate(split_components, 1):
        results.append({
            "component_id": f"C{idx}",
            "codes": grp
        })

    return results

def review_candidate_groups_with_llm(candidate_components, candidate_pairs, api_key, freq_map,
                                     research_domain="", research_topic="", mode="默认"):
    """
    第二层：按“小组”做 LLM 复核，而不是逐 pair 判断。
    输出结构仍然兼容：
    - merge_groups
    - merge_auto_groups
    - merge_review_groups
    """
    if not candidate_components:
        return {"pairs": candidate_pairs, "components": [], "auto": [], "review": []}

    strategy = get_merge_strategy(mode)

    pair_lookup = {}
    for p in candidate_pairs:
        key = tuple(sorted([str(p["code_a"]), str(p["code_b"])]))
        pair_lookup[key] = p

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    auto_groups = []
    review_groups = []
    auto_idx = 1
    review_idx = 1

    for comp in candidate_components:
        component_id = comp["component_id"]
        comp_codes = dedupe_preserve_order(comp.get("codes", []))
        comp_set = set(comp_codes)

        internal_pairs = [
            p for p in candidate_pairs
            if p["code_a"] in comp_set and p["code_b"] in comp_set
        ]
        internal_pairs = sorted(
            internal_pairs,
            key=lambda x: (-x["fusion_similarity"], -x["embedding_similarity"], x["code_a"], x["code_b"])
        )[:12]

        pair_evidence = []
        for p in internal_pairs:
            pair_evidence.append({
                "code_a": p["code_a"],
                "code_b": p["code_b"],
                "embedding_similarity": p["embedding_similarity"],
                "lexical_similarity": p["lexical_similarity"],
                "fusion_similarity": p["fusion_similarity"],
                "signals": p.get("signals", []),
                "candidate_sources": p.get("candidate_sources", [])
            })

        prompt = f"""
你正在做中文质性研究中的“标签清洗 / 同类合并”复核。

研究领域：{research_domain if research_domain else "未提供"}
研究主题：{research_topic if research_topic else "未提供"}
当前策略：{mode}
策略说明：{strategy["mode_hint"]}

⚠️ 请注意：
- 尽量保留小组的原始颗粒度（尤其是 ≤3 个编码的小组不要轻易合并）
- 只有高度同义、语义完全一致的编码才可合并
- 不跨不同上下文或分析维度合并
- 每个编码最多出现在一个组
- merged_name 尽量简洁、稳定、可作为最终统一标签名
- 输出 JSON 对象，不要解释，不要 markdown，不要代码块

请严格返回这个格式：
{{
  "component_id": "{component_id}",
  "auto_groups": [
    {{
      "codes": ["标签A", "标签B"],
      "merged_name": "统一标签名",
      "confidence": 0.91,
      "reasoning": "一句话说明"
    }}
  ],
  "review_groups": [
    {{
      "codes": ["标签C", "标签D", "标签E"],
      "merged_name": "建议统一标签名",
      "confidence": 0.68,
      "reasoning": "一句话说明"
    }}
  ],
  "reject_codes": ["标签F"]
}}

候选小组编码：
{json.dumps(comp_codes, ensure_ascii=False, indent=2)}

组内关系证据：
{json.dumps(pair_evidence, ensure_ascii=False, indent=2)}
""".strip()

        try:
            res = client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = res.choices[0].message.content.strip()
            parsed = parse_ai_json_content(content)

            if isinstance(parsed, list) and parsed:
                obj = parsed[0] if isinstance(parsed[0], dict) else {}
            elif isinstance(parsed, dict):
                obj = parsed
            else:
                obj = {}

            used_codes = set()

            # -------------------------
            # 先处理 auto_groups
            # -------------------------
            for g in obj.get("auto_groups", []) if isinstance(obj.get("auto_groups", []), list) else []:
                codes_in_group = [c for c in dedupe_preserve_order(g.get("codes", [])) if c in comp_set and c not in used_codes]
                if len(codes_in_group) < 2:
                    continue

                confidence = safe_float(g.get("confidence"), 0.0)
                if confidence < strategy["auto_confidence"]:
                    continue

                merged_name = str(g.get("merged_name", "") or "").strip() or default_merged_name(codes_in_group, freq_map)
                reasoning = str(g.get("reasoning", "") or "").strip()

                auto_groups.append({
                    "group_id": f"A{auto_idx}",
                    "codes": codes_in_group,
                    "merged_name": merged_name,
                    "score": confidence,
                    "reasoning": reasoning,
                    "decision": "auto"
                })
                auto_idx += 1
                used_codes.update(codes_in_group)

            # -------------------------
            # 再处理 review_groups
            # -------------------------
            for g in obj.get("review_groups", []) if isinstance(obj.get("review_groups", []), list) else []:
                codes_in_group = [c for c in dedupe_preserve_order(g.get("codes", [])) if c in comp_set and c not in used_codes]
                if len(codes_in_group) < 2:
                    continue

                confidence = safe_float(g.get("confidence"), 0.0)
                if confidence < strategy["review_confidence"]:
                    continue

                merged_name = str(g.get("merged_name", "") or "").strip() or default_merged_name(codes_in_group, freq_map)
                reasoning = str(g.get("reasoning", "") or "").strip()

                review_groups.append({
                    "group_id": f"R{review_idx}",
                    "codes": codes_in_group,
                    "merged_name": merged_name,
                    "score": confidence,
                    "reasoning": reasoning,
                    "decision": "review"
                })
                review_idx += 1
                used_codes.update(codes_in_group)

            # -------------------------
            # 如果 LLM 没给出 review，但这个 component 本身就很像，兜底给 review
            # -------------------------
            remaining = [c for c in comp_codes if c not in used_codes]
            if len(remaining) >= 2 and not obj.get("auto_groups") and not obj.get("review_groups"):
                review_groups.append({
                    "group_id": f"R{review_idx}",
                    "codes": remaining,
                    "merged_name": default_merged_name(remaining, freq_map),
                    "score": 0.50,
                    "reasoning": "模型未输出明确分组，保留为人工确认候选",
                    "decision": "review"
                })
                review_idx += 1

        except Exception:
            # LLM 异常时，保守兜底：整组进入 review
            if len(comp_codes) >= 2:
                review_groups.append({
                    "group_id": f"R{review_idx}",
                    "codes": comp_codes,
                    "merged_name": default_merged_name(comp_codes, freq_map),
                    "score": 0.50,
                    "reasoning": "LLM异常，保留为人工确认候选",
                    "decision": "review"
                })
                review_idx += 1

    return {
        "pairs": candidate_pairs,
        "components": candidate_components,
        "auto": auto_groups,
        "review": review_groups
    }

def review_candidate_pairs_with_llm(candidate_pairs, api_key):
    """
    LLM 审核：只判断高度同义编码
    - 不提供长引文
    - 明确规则：保留颗粒度，不跨轴心合并
    """
    if not candidate_pairs:
        return []

    client = OpenAI(api_key=api_key)
    all_results = []

    # 构造 prompt
    batch_payload = [
        {"code_a": p["code_a"], "code_b": p["code_b"], "fusion_sim": p["fusion_similarity"]}
        for p in candidate_pairs
    ]

    prompt = f"""
你正在做中文质性研究的标签同义合并判断：
- 仅合并高度同义编码
- 不跨不同轴心或分析维度
- 不提供引文
- 输出 JSON 格式：pair_id, decision(auto_merge/review/reject)

待审核编码对：
{json.dumps(batch_payload, ensure_ascii=False, indent=2)}
    """.strip()

    res = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    parsed = parse_ai_json_content(res.choices[0].message.content)
    for idx, item in enumerate(parsed, 1):
        item["pair_id"] = f"P{idx}"
        all_results.append(item)

    return all_results

def should_auto_merge_pair(pair_item, review_item, mode="默认"):
    strategy = get_merge_strategy(mode)
    decision = review_item.get("decision")
    same_phenomenon = str(review_item.get("same_phenomenon", "")).lower()
    confidence = safe_float(review_item.get("confidence"), 0.0)

    if decision != "auto_merge":
        return False
    if same_phenomenon == "no":
        return False
    if confidence < strategy["auto_confidence"]:
        return False

    strong_semantic = pair_item.get("embedding_similarity", 0.0) >= strategy.get("auto_embedding_floor", 0.0)
    strong_lexical = pair_item.get("lexical_similarity", 0.0) >= strategy.get("auto_lexical_floor", 0.0)

    if pair_item.get("same_norm"):
        return True
    if strong_semantic or strong_lexical:
        return True

    return False

def build_groups_from_edges(edge_details, freq_map, group_type="auto"):
    """
    把 pair 级结果整理成 group 级结果
    """
    if not edge_details:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in edge_details:
        union(e["code_a"], e["code_b"])

    raw_groups = {}
    for node in list(parent.keys()):
        root = find(node)
        raw_groups.setdefault(root, set()).add(node)

    groups = []
    prefix = "A" if group_type == "auto" else "R"

    for idx, (_, code_set) in enumerate(raw_groups.items(), 1):
        codes = sorted(list(code_set), key=lambda x: (-freq_map.get(x, 0), x))
        related_edges = [
            e for e in edge_details
            if e["code_a"] in code_set and e["code_b"] in code_set
        ]

        name_votes = [
            str(e.get("merged_name", "")).strip()
            for e in related_edges
            if str(e.get("merged_name", "")).strip()
        ]
        if name_votes:
            merged_name = Counter(name_votes).most_common(1)[0][0]
        else:
            merged_name = codes[0]

        reasoning_list = []
        for e in related_edges:
            r = str(e.get("reasoning", "")).strip()
            if r and r not in reasoning_list:
                reasoning_list.append(r)

        score_vals = [safe_float(e.get("confidence"), 0.0) for e in related_edges]
        avg_score = round(float(np.mean(score_vals)) if score_vals else 0.0, 3)

        groups.append({
            "group_id": f"{prefix}{idx}",
            "codes": codes,
            "merged_name": merged_name,
            "score": avg_score,
            "reasoning": "；".join(reasoning_list[:3]),
            "decision": group_type
        })

    groups = sorted(groups, key=lambda x: (-x["score"], -len(x["codes"]), x["group_id"]))
    return groups

def build_merge_review_results(candidate_pairs, llm_reviews, freq_map, mode="默认"):
    """
    综合第一层候选 + 第二层复核，产出：
    - merge_groups
    - merge_auto_groups
    - merge_review_groups
    """
    strategy = get_merge_strategy(mode)
    review_map = {r["pair_id"]: r for r in llm_reviews if r.get("pair_id")}

    pair_results = []
    auto_edges = []
    review_edges = []

    for pair in candidate_pairs:
        rv = review_map.get(pair["pair_id"], {
            "pair_id": pair["pair_id"],
            "decision": "review",
            "merged_name": "",
            "same_phenomenon": "uncertain",
            "confidence": 0.50,
            "reasoning": "缺少 LLM 结果，默认人工确认"
        })

        item = {
            **pair,
            "decision": normalize_merge_decision(rv.get("decision")),
            "merged_name": str(rv.get("merged_name", "") or "").strip(),
            "same_phenomenon": str(rv.get("same_phenomenon", "") or "").strip().lower(),
            "confidence": safe_float(rv.get("confidence"), 0.0),
            "reasoning": str(rv.get("reasoning", "") or "").strip()
        }
        pair_results.append(item)

        if should_auto_merge_pair(pair, item, mode=mode):
            auto_edges.append(item)
        elif item["decision"] in {"auto_merge", "review"} and item["confidence"] >= strategy["review_confidence"]:
            item["decision"] = "review"
            review_edges.append(item)

    auto_groups = build_groups_from_edges(auto_edges, freq_map, group_type="auto")
    review_groups = build_groups_from_edges(review_edges, freq_map, group_type="review")

    return {
        "pairs": pair_results,
        "auto": auto_groups,
        "review": review_groups
    }

def apply_merge_groups(groups, action_name="智能标签合并"):
    """
    把 group 结果真正应用到 st.session_state.open_codes['code']
    """
    if not groups:
        return 0

    replace_map = {}
    applied_group_count = 0

    for g in groups:
        codes = [str(x).strip() for x in g.get("codes", []) if str(x).strip()]
        merged_name = str(g.get("merged_name", "")).strip()

        if not merged_name or len(codes) < 2:
            continue

        for c in codes:
            replace_map[c] = merged_name
        applied_group_count += 1

    if not replace_map:
        return 0

    push_history(action_name)
    st.session_state.open_codes['code'] = st.session_state.open_codes['code'].replace(replace_map)
    autosave_final_snapshot(action_name)
    return applied_group_count

def apply_high_confidence_merge(candidate_pairs, llm_reviews, st_state, confidence_threshold=0.9):
    """
    仅应用高置信度自动合并，其他保留给人工确认/轴心分析
    """
    high_conf = [r for r in llm_reviews if r.get("decision") == "auto_merge" and r.get("confidence", 0) >= confidence_threshold]

    replace_map = {r["code_a"]: r["merged_name"] for r in high_conf}
    replace_map.update({r["code_b"]: r["merged_name"] for r in high_conf})

    if not replace_map:
        return 0

    st_state.open_codes['code'] = st_state.open_codes['code'].replace(replace_map)
    return len(high_conf)

def drop_groups_by_codes(code_list):
    """
    应用/忽略某组后，把涉及到这些 code 的候选从当前建议中移除，避免脏状态
    """
    used = set(code_list or [])

    st.session_state.merge_auto_groups = [
        g for g in st.session_state.get("merge_auto_groups", [])
        if not (used & set(g.get("codes", [])))
    ]

    st.session_state.merge_review_groups = [
        g for g in st.session_state.get("merge_review_groups", [])
        if not (used & set(g.get("codes", [])))
    ]

def clear_merge_scan_state():
    st.session_state.merge_groups = []
    st.session_state.merge_auto_groups = []
    st.session_state.merge_review_groups = []

def render_merge_group_quotes(df_source, code_list, max_rows=8):
    if df_source is None or df_source.empty or not code_list:
        st.caption("暂无引文")
        return

    sub = df_source[df_source['code'].isin(code_list)][['code', 'quote']].drop_duplicates()
    sub = sub.head(max_rows)

    if sub.empty:
        st.caption("暂无引文")
    else:
        st.dataframe(sub, width="stretch", hide_index=True)

def align_records_by_quote(df_mine, df_theirs, match_threshold=0.6):
    theirs_records = df_theirs.to_dict('records')
    alignment = []
    mine_records = df_mine.to_dict('records')

    theirs_buckets = {}
    for r in theirs_records:
        q_len = len(str(r.get('quote', '')))
        bucket_id = q_len // 10
        if bucket_id not in theirs_buckets:
            theirs_buckets[bucket_id] = []
        theirs_buckets[bucket_id].append(r)

    for my_row in mine_records:
        my_quote = str(my_row.get('quote', ''))
        my_len = len(my_quote)
        my_bucket = my_len // 10
        my_code = str(my_row.get('code', ''))

        best_match = None
        best_ratio = 0

        candidates = []
        for b in [my_bucket - 1, my_bucket, my_bucket + 1]:
            if b in theirs_buckets:
                candidates.extend(theirs_buckets[b])

        if not candidates:
            candidates = theirs_records

        my_char_set = set(my_quote)

        for their_row in candidates:
            their_quote = str(their_row.get('quote', ''))

            if not my_quote and not their_quote:
                ratio = 1.0
            else:
                their_char_set = set(their_quote)
                intersection = len(my_char_set & their_char_set)
                union = len(my_char_set | their_char_set)
                jaccard = intersection / union if union > 0 else 0

                if jaccard < 0.3:
                    continue

                ratio = SequenceMatcher(None, my_quote, their_quote).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = their_row

        status = "unique"
        their_code = None
        if best_ratio >= match_threshold:
            their_code = str(best_match.get('code', ''))
            if my_code.strip() == their_code.strip():
                status = "agreed"
            else:
                status = "conflict"

        alignment.append({
            "quote": my_quote,
            "my_code": my_code,
            "their_code": their_code,
            "status": status,
            "similarity": best_ratio,
            "raw_row_idx": my_row.get('original_row_index')
        })
    return alignment

def generate_html_tag_cloud(df):
    if df.empty or 'code' not in df.columns:
        return "无数据"
    counts = df['code'].value_counts()
    if counts.empty:
        return "无有效标签"
    max_count = counts.max()
    min_count = counts.min()
    tags_html = ""
    colors = ['#4a90e2', '#50e3c2', '#b8e986', '#f5a623', '#f8e71c', '#d0021b', '#9013fe', '#4a4a4a']
    for code, count in counts.items():
        size = 14 if max_count == min_count else 12 + (count - min_count) / (max_count - min_count) * 24
        color = random.choice(colors)
        tags_html += f"""<span style="font-size: {size}px; color: {color}; margin: 5px; padding: 5px; 
            display: inline-block; border: 1px solid #eee; border-radius: 5px; background-color: #fafafa;"
            title="出现频次: {count}">{code}</span>"""
    return f"<div style='line-height: 2.0; text-align: center; padding: 20px; background: white; border-radius: 10px; border: 1px solid #eee;'>{tags_html}</div>"

# =======================================================================
# 2. 页面配置与 CSS
# =======================================================================
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #FDFBF5; }

    .quote-box {
        background-color: #f9f9f9;
        border-left: 4px solid #B0C4DE;
        padding: 10px;
        margin-bottom: 10px;
        font-family: "SimSun", "Times New Roman", serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
    }

    .quote-label {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 4px;
        font-weight: bold;
    }

    /* 拖拽工作区：统一列宽与卡片宽度 */
    .stSortable {
        display: flex !important;
        gap: 14px !important;
        align-items: flex-start !important;
        width: 100% !important;
    }

    .stSortable > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    .stSortable > div > div {
        background-color: #E6F7FF !important;
        border: 1px solid #69C0FF !important;
        color: #003a8c !important;
        border-radius: 6px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 10px !important;
        margin-bottom: 8px !important;
        white-space: normal !important;
        line-height: 1.4 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        max-width: 100% !important;
    }

    .custom-card-hint {
        background-color: #E6F7FF;
        border: 1px solid #91D5FF;
        border-radius: 6px;
        padding: 12px;
        color: #0050B3;
        font-size: 1rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title(f"🧩 区域 3：分析工作台，清洗、对齐与归类 ｜ 项目准备: {selected_project}")

with st.sidebar:
    st.header("📁 项目管理")

    if st.session_state.get("active_project_selector") not in existing_projects:
        st.session_state["active_project_selector"] = existing_projects[0]

    st.selectbox(
        "选择项目",
        options=existing_projects,
        key="active_project_selector",
        on_change=hard_reset_analysis_project
    )

    selected_project = st.session_state["active_project_selector"]
    DIRS = get_project_paths(USERNAME, selected_project)

    INPUT_DIR = DIRS["opening_final"]
    RECOVERY_DIR = DIRS["analysis_autosave"]
    FINAL_DIR = DIRS["analysis_final"]

    ensure_recovery_dir()
    os.makedirs(FINAL_DIR, exist_ok=True)

    st.info("系统会在关键操作后自动保存到 analysis_final/page4_input.xlsx。")
    if st.session_state.get("last_autosave_time"):
        st.caption(f"最近自动保存：{st.session_state['last_autosave_time']}｜{st.session_state.get('last_autosave_reason', '')}")

    st.divider()
    st.subheader("↩️ 操作控制")

    if st.session_state.history_stack:
        if st.button(f"↩️ 撤销上一步 ({len(st.session_state.history_stack)})", type="primary", use_container_width=True):
            perform_undo()
    else:
        st.button("↩️ 撤销 (无记录)", disabled=True, use_container_width=True)

    st.divider()
    st.subheader("📥 恢复进度")

    jsonl_files = glob.glob(os.path.join(RECOVERY_DIR, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)

    if jsonl_files:
        selected_file = st.selectbox(
            "选择历史文件",
            [os.path.basename(f) for f in jsonl_files],
            index=0,
            key="analysis_recovery_file"
        )
        if st.button("🔄 载入选中文件", use_container_width=True):
            filepath = os.path.join(RECOVERY_DIR, selected_file)
            df_loaded = load_from_jsonl(filepath)
            if not df_loaded.empty:
                st.session_state.open_codes = df_loaded

                for col in ['original_code', 'peer_code', 'aligned_code', 'original_row_index']:
                    if col not in st.session_state.open_codes.columns:
                        if col == 'peer_code':
                            st.session_state.open_codes[col] = None
                        elif col == 'original_row_index':
                            st.session_state.open_codes[col] = range(len(st.session_state.open_codes))
                        else:
                            st.session_state.open_codes[col] = st.session_state.open_codes['code']

                reset_analysis_state()
                autosave_final_snapshot("载入历史进度")
                st.success(f"成功载入 {len(df_loaded)} 条记录！")
                st.rerun()
            else:
                st.warning("该文件为空或格式不正确")
    else:
        st.caption("暂无历史存档")

    st.divider()
    if st.button("🗑 清空当前项目", use_container_width=True):
        hard_reset_analysis_project()
        st.success("已清空当前项目页面状态")
        time.sleep(0.5)
        st.rerun()

# 数据加载
data_missing = st.session_state.open_codes is None or st.session_state.open_codes.empty
data_invalid = False
if not data_missing:
    if 'code' not in st.session_state.open_codes.columns:
        data_invalid = True

if data_missing or data_invalid:
    if data_invalid:
        st.warning("⚠️ 数据格式错误（缺少 'code' 列）。")
    else:
        st.info("请选择数据来源后开始分析。")

    t1, t2 = st.tabs(["📁 从云端目录导入", "📤 本地上传"])

    with t1:
        server_files = []
        if os.path.exists(INPUT_DIR):
            server_files = [
                f for f in os.listdir(INPUT_DIR)
                if f.endswith((".xlsx", ".csv"))
            ]

        if server_files:
            target_f = st.selectbox(
                "选择上一阶段文件：",
                ["-- 请选择 --"] + server_files,
                key="main_data_load_key"
            )

            if st.button("✅ 确认载入数据并开始", type="primary", key="load_main_from_cloud"):
                if target_f != "-- 请选择 --":
                    p = os.path.join(INPUT_DIR, target_f)
                    df_load = pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)

                    if 'code' not in df_load.columns:
                        st.error("所选文件缺少 'code' 列。")
                        st.stop()

                    if 'original_row_index' not in df_load.columns:
                        df_load['original_row_index'] = range(len(df_load))
                    if 'original_code' not in df_load.columns:
                        df_load['original_code'] = df_load['code']
                    if 'peer_code' not in df_load.columns:
                        df_load['peer_code'] = None
                    if 'aligned_code' not in df_load.columns:
                        df_load['aligned_code'] = df_load['code']

                    st.session_state.open_codes = df_load
                    reset_analysis_state()
                    autosave_final_snapshot("载入主分析数据")
                    st.success(f"✅ 已从云端载入：{target_f}")
                    st.rerun()
        else:
            st.warning("opening_final 文件夹为空。")

    with t2:
        uploaded_file = st.file_uploader(
            "📤 上传开放编码结果表 (Excel/CSV)",
            type=['xlsx', 'csv'],
            key="primary_uploader"
        )

        if uploaded_file:
            try:
                df_load = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

                if 'code' not in df_load.columns:
                    st.error("文件缺少 'code' 列。")
                    st.stop()

                if 'original_row_index' not in df_load.columns:
                    df_load['original_row_index'] = range(len(df_load))
                if 'original_code' not in df_load.columns:
                    df_load['original_code'] = df_load['code']
                if 'peer_code' not in df_load.columns:
                    df_load['peer_code'] = None
                if 'aligned_code' not in df_load.columns:
                    df_load['aligned_code'] = df_load['code']

                st.session_state.open_codes = df_load
                reset_analysis_state()
                autosave_final_snapshot("载入主分析数据")
                st.success("✅ 本地文件载入成功！")
                st.rerun()

            except Exception as e:
                st.error(f"读取失败：{e}")
                st.stop()

    st.warning("无可用主分析数据")
    st.stop()

# 确保列存在
df = st.session_state.open_codes
for col in ['original_code', 'peer_code', 'aligned_code', 'original_row_index']:
    if col not in df.columns:
        if col == 'peer_code':
            df[col] = None
        elif col == 'original_row_index':
            df[col] = range(len(df))
        else:
            df[col] = df['code']
st.session_state.open_codes = df

unique_codes = df['code'].dropna().unique().tolist()

# =======================================================================
# 主选项卡布局
# =======================================================================
tab_align, tab_clean, tab_kanban = st.tabs(["🤝 队友对齐 (分歧解决)", "🧹 标签清洗 (同义合并)", "🧱 积木归类 (轴心分析)"])

# -----------------------------------------------------------------------
# TAB 1: 队友对齐
# -----------------------------------------------------------------------
with tab_align:
    st.caption("上传队友的编码文件，AI将自动对齐并列出差异。")
    st.markdown("### 导入队友编码文件")

    pt1, pt2 = st.tabs(["📁 从云端目录导入", "📤 本地上传"])

    with pt1:
        peer_server_files = []
        if os.path.exists(INPUT_DIR):
            peer_server_files = [
                f for f in os.listdir(INPUT_DIR)
                if f.endswith(('.xlsx', '.csv', '.jsonl'))
            ]

        if peer_server_files:
            target_peer_f = st.selectbox(
                "选择队友文件：",
                ["-- 请选择 --"] + peer_server_files,
                key="peer_data_load_key"
            )

            if st.button("✅ 载入队友文件", type="primary", key="load_peer_from_cloud"):
                if target_peer_f != "-- 请选择 --":
                    st.session_state.peer_file_path = os.path.join(INPUT_DIR, target_peer_f)
                    if "peer_uploaded_file" in st.session_state:
                        del st.session_state["peer_uploaded_file"]
                    st.success(f"✅ 已选择云端文件：{target_peer_f}")
        else:
            st.warning("opening_final 文件夹中没有可用的队友文件。")

    with pt2:
        uploaded_peer_file = st.file_uploader(
            "上传队友文件",
            type=['xlsx', 'csv', 'jsonl'],
            key="peer_uploader"
        )
        if uploaded_peer_file is not None:
            st.session_state.peer_uploaded_file = uploaded_peer_file
            if "peer_file_path" in st.session_state:
                del st.session_state["peer_file_path"]
            st.success(f"✅ 已上传本地文件：{uploaded_peer_file.name}")

    peer_source = None
    if st.session_state.get("peer_uploaded_file") is not None:
        peer_source = st.session_state.peer_uploaded_file
    elif st.session_state.get("peer_file_path"):
        peer_source = st.session_state.peer_file_path

    rerun_compare_clicked = st.button("🔄 重新对比", key="rerun_alignment_compare")

    if peer_source is not None:
        try:
            # 1) 读取队友文件
            if isinstance(peer_source, str):
                if peer_source.endswith('.csv'):
                    df_peer = pd.read_csv(peer_source)
                elif peer_source.endswith('.jsonl'):
                    df_peer = load_from_jsonl(peer_source)
                else:
                    df_peer = pd.read_excel(peer_source)
            else:
                if peer_source.name.endswith('.csv'):
                    df_peer = pd.read_csv(peer_source)
                elif peer_source.name.endswith('.jsonl'):
                    df_peer = load_from_uploaded_jsonl(peer_source)
                else:
                    df_peer = pd.read_excel(peer_source)

            if 'code' not in df_peer.columns:
                st.error("队友文件缺少 'code' 列")
                st.stop()

            if 'quote' not in df_peer.columns:
                st.error("队友文件缺少 'quote' 列")
                st.stop()

            # 2) 判断是否为新文件
            current_peer_token = get_peer_source_token(peer_source)
            peer_changed = current_peer_token != st.session_state.get("peer_source_token")

            if peer_changed:
                st.session_state.peer_source_token = current_peer_token
                clear_alignment_state(clear_peer_code=True)

            # 3) 自动对比触发条件
            should_compare = (
                st.session_state.get("alignment_results") is None
                or peer_changed
                or rerun_compare_clicked
            )

            if rerun_compare_clicked:
                clear_alignment_state(clear_peer_code=True)

            # 4) 执行对比
            if should_compare:
                with st.spinner("正在快速比对... (已启用性能优化)"):
                    results = align_records_by_ids_and_batch(
                        st.session_state.open_codes,
                        df_peer,
                        overlap_threshold=0.66,
                        match_threshold=0.6
                    ) or []

                    st.session_state.alignment_results = results
                    st.session_state.last_alignment_signature = (
                        f"{current_peer_token}|mine_rows={len(st.session_state.open_codes)}|peer_rows={len(df_peer)}"
                    )

                    # 同步 peer_code 到主表
                    open_codes = st.session_state.open_codes.copy()
                    updates = 0

                    if results:
                        push_history("同步队友编码")

                    for r in results:
                        raw_row_idx = r.get('raw_row_idx')
                        their_code = r.get('their_code')
                        quote = r.get('quote', '')

                        if raw_row_idx is not None and 'original_row_index' in open_codes.columns:
                            mask = open_codes['original_row_index'].astype(str) == str(raw_row_idx)
                            if mask.any():
                                open_codes.loc[mask, 'peer_code'] = their_code
                                updates += int(mask.sum())
                                continue

                        if quote and 'quote' in open_codes.columns:
                            mask = open_codes['quote'].astype(str) == str(quote)
                            if mask.any():
                                open_codes.loc[mask, 'peer_code'] = their_code
                                updates += int(mask.sum())

                    st.session_state.open_codes = open_codes

                    if updates > 0:
                        autosave_final_snapshot("同步队友编码")
                        st.toast(f"已同步 {updates} 条队友数据")

            # 5) 展示结果时兜底成 list，避免 NoneType 报错
            results = st.session_state.get("alignment_results") or []
            conflicts = [r for r in results if r.get('status') == 'conflict']

            if conflicts:
                st.warning(f"发现 {len(conflicts)} 处分歧")
                page_size = 4

                if 'page_num_align' not in st.session_state:
                    st.session_state.page_num_align = 0

                max_page = max(0, (len(conflicts) - 1) // page_size)
                if st.session_state.page_num_align > max_page:
                    st.session_state.page_num_align = 0

                start_idx = st.session_state.page_num_align * page_size
                current_batch = conflicts[start_idx:start_idx + page_size]

                st.progress(min(1.0, (start_idx + len(current_batch)) / len(conflicts)))

                for i, item in enumerate(current_batch):
                    idx_real = start_idx + i
                    with st.container(border=True):
                        st.markdown(f"<div class='quote-box'>{item['quote']}</div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"👤 我: **{item['my_code']}**")
                            if st.button("👈 保留我的", key=f"k_my_{idx_real}", width="stretch"):
                                item['status'] = 'resolved'
                                st.rerun()

                        with c2:
                            st.warning(f"👥 他: **{item['their_code']}**")
                            if st.button("👉 采纳队友", key=f"k_th_{idx_real}", width="stretch"):
                                push_history(f"采纳队友编码: {item['their_code']}")
                                mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                       (st.session_state.open_codes['original_code'] == item['my_code'])
                                if mask.any():
                                    st.session_state.open_codes.loc[mask, 'aligned_code'] = item['their_code']
                                    st.session_state.open_codes.loc[mask, 'code'] = item['their_code']
                                    autosave_final_snapshot("采纳队友编码")
                                    item['status'] = 'resolved'
                                    st.success("已更新")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("定位失败")

                        ai_k = f"ai_adv_{idx_real}"
                        custom_code = st.text_input("✏️ 修改为", value=st.session_state.get(ai_k, item['my_code']), key=f"inp_{idx_real}")
                        ca, cb = st.columns([1, 2])

                        with ca:
                            if st.button("🤖 问AI", key=f"ask_{idx_real}", disabled=not api_ready):
                                prompt = f"引文：{item['quote']}\n标签A：{item['my_code']}\n标签B：{item['their_code']}\n请给出一个最准确的简短标签："
                                try:
                                    client = OpenAI(
                                        api_key=st.session_state.api_key,
                                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                                    )
                                    res = client.chat.completions.create(
                                        model="qwen-plus",
                                        messages=[{"role": "user", "content": prompt}]
                                    )
                                    st.session_state[ai_k] = res.choices[0].message.content.strip()
                                    st.rerun()
                                except Exception:
                                    st.error("API Error")

                        with cb:
                            if st.button("✅ 应用修改", key=f"app_{idx_real}", type="primary", width="stretch"):
                                push_history(f"修改编码为: {custom_code}")
                                mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                       (st.session_state.open_codes['original_code'] == item['my_code'])
                                if mask.any():
                                    st.session_state.open_codes.loc[mask, 'aligned_code'] = custom_code
                                    st.session_state.open_codes.loc[mask, 'code'] = custom_code
                                    autosave_final_snapshot("修改分歧编码")
                                    item['status'] = 'resolved'
                                    st.success("已更新")
                                    time.sleep(0.5)
                                    st.rerun()

                cp1, cp2 = st.columns(2)
                if st.session_state.page_num_align > 0:
                    if cp1.button("⬅️ 上一页"):
                        st.session_state.page_num_align -= 1
                        st.rerun()

                if start_idx + page_size < len(conflicts):
                    if cp2.button("下一页 ➡️"):
                        st.session_state.page_num_align += 1
                        st.rerun()

            else:
                st.success("🎉 所有分歧已解决！")

        except Exception:
            st.error("❌ 发生异常（调试模式）")
            st.code(traceback.format_exc())

    st.divider()
    if st.button("完成对齐，进入标签清洗 ➡️", key="to_clean_stage", use_container_width=True):
        ok, msg = autosave_final_snapshot("完成队友对齐")
        if ok:
            st.success("已自动保存到 analysis_final/page4_input.xlsx，请点击上方【🧹 标签清洗（同义合并）】继续。")
        else:
            st.error(f"自动保存失败：{msg}")

# -----------------------------------------------------------------------
# TAB 2: 同义合并（两层策略版）
# -----------------------------------------------------------------------
with tab_clean:
    st.markdown("#### 🧹 标签标准化")
    st.caption("第一层先召回候选，第二层再由大语言模型复核。宽松 = 更多候选提示，不等于更多自动保存。")

    top1, top2 = st.columns([1.2, 1.2])

    with top1:
        st.selectbox(
            "清洗策略",
            options=["严格", "默认", "宽松"],
            key="merge_mode"
        )

    with top2:
        st.selectbox(
            "输出方式",
            options=["先预览后确认", "自动应用高置信合并"],
            key="merge_output_mode"
        )

    ctx1, ctx2 = st.columns(2)
    with ctx1:
        render_shared_domain_input(
            "研究领域（用于二层复核）",
            widget_key="kanban_research_domain_input_tab2",
            mirror_keys=["kanban_research_domain_input_tab3"],
            placeholder="例如：组织社会化 / 教育社会学 / 平台劳动研究"
        )

    with ctx2:
        render_shared_topic_input(
            "研究主题（用于二层复核）",
            widget_key="kanban_research_topic_input_tab2",
            mirror_keys=["kanban_research_topic_input_tab3"],
            placeholder="例如：新员工组织融入过程中的情感体验与适应机制"
        )

    scan_col1, scan_col2 = st.columns([1.4, 1.0])

    with scan_col1:
        scan_clicked = st.button("🚀 智能扫描候选", type="primary", use_container_width=True)

    with scan_col2:
        if st.button("🧹 清空本轮建议", use_container_width=True):
            clear_merge_scan_state()
            st.rerun()

    if scan_clicked:
        if not api_ready:
            st.error("需 API Key")
        else:
            with st.spinner("第一层召回候选中..."):
                u_codes = df['code'].dropna().astype(str).unique().tolist()
                freq_map = df['code'].dropna().astype(str).value_counts().to_dict()

                embs = get_embeddings_dashscope(u_codes, st.session_state.api_key)
                if len(embs) == 0:
                    clear_merge_scan_state()
                    st.error("Embedding 生成失败")
                    st.stop()

                candidate_pairs = build_candidate_pairs(
                    u_codes,
                    embs,
                    freq_map=freq_map,
                    mode=st.session_state.get("merge_mode", "默认")
                )

                candidate_components = build_candidate_components(
                    candidate_pairs,
                    codes=u_codes,
                    embeddings=embs,
                    freq_map=freq_map,
                    mode=st.session_state.get("merge_mode", "默认")
                )

            if not candidate_pairs or not candidate_components:
                clear_merge_scan_state()
                st.info("当前没有召回到可疑重复标签。")
            else:
                with st.spinner("第二层组级语义复核中..."):
                    research_domain = (
                        st.session_state.get("kanban_research_domain_input_tab2", "").strip()
                        or st.session_state.get("kanban_research_domain_input_tab3", "").strip()
                        or st.session_state.get("kanban_research_domain", "").strip()
                    )

                    research_topic = (
                        st.session_state.get("kanban_research_topic_input_tab2", "").strip()
                        or st.session_state.get("kanban_research_topic_input_tab3", "").strip()
                        or st.session_state.get("kanban_research_topic", "").strip()
                    )

                    st.session_state["kanban_research_domain"] = research_domain
                    st.session_state["kanban_research_topic"] = research_topic

                    merge_result = review_candidate_groups_with_llm(
                        candidate_components=candidate_components,
                        candidate_pairs=candidate_pairs,
                        api_key=st.session_state.api_key,
                        freq_map=freq_map,
                        research_domain=research_domain,
                        research_topic=research_topic,
                        mode=st.session_state.get("merge_mode", "默认")
                    )

                st.session_state.merge_groups = merge_result
                st.session_state.merge_auto_groups = merge_result["auto"]
                st.session_state.merge_review_groups = merge_result["review"]

                if (
                    st.session_state.get("merge_output_mode") == "自动应用高置信合并"
                    and st.session_state.merge_auto_groups
                ):
                    applied = apply_merge_groups(
                        st.session_state.merge_auto_groups,
                        action_name="自动应用高置信合并"
                    )
                    clear_merge_scan_state()
                    st.success(f"已自动应用 {applied} 组高置信合并")
                    time.sleep(0.5)
                    st.rerun()

                st.rerun()

    merge_result = st.session_state.get("merge_groups", {})
    merge_auto_groups = st.session_state.get("merge_auto_groups", []) or []
    merge_review_groups = st.session_state.get("merge_review_groups", []) or []

    pair_count = len(merge_result.get("pairs", [])) if isinstance(merge_result, dict) else 0

    if pair_count > 0 or merge_auto_groups or merge_review_groups:
        m1, m2, m3 = st.columns(3)
        m1.metric("候选对", pair_count)
        m2.metric("可自动合并", len(merge_auto_groups))
        m3.metric("建议人工确认", len(merge_review_groups))

        batch1, batch2 = st.columns(2)

        with batch1:
            if st.button(
                "✅ 应用全部自动合并",
                use_container_width=True,
                disabled=not bool(merge_auto_groups)
            ):
                applied = apply_merge_groups(
                    merge_auto_groups,
                    action_name=f"批量自动合并（{st.session_state.get('merge_mode', '默认')}）"
                )
                clear_merge_scan_state()
                st.success(f"已应用 {applied} 组自动合并")
                time.sleep(0.5)
                st.rerun()

        with batch2:
            st.caption("自动合并只处理高置信结果；其余保留给人工确认。")

    if merge_auto_groups:
        st.markdown("##### 🤖 可自动合并")
        for group in merge_auto_groups:
            gid = group["group_id"]
            codes = group.get("codes", [])
            merged_name = group.get("merged_name", "")
            score = group.get("score", 0.0)
            reasoning = group.get("reasoning", "")

            with st.container(border=True):
                info_col, act_col = st.columns([3, 1])

                with info_col:
                    st.write(f"**{gid}**｜建议统一为：**{merged_name}**｜置信度：`{score:.2f}`")
                    st.caption("候选标签：" + " / ".join(codes))
                    if reasoning:
                        st.caption("判断依据：" + reasoning)

                    with st.expander("📄 查看引文", expanded=False):
                        render_merge_group_quotes(df, codes)

                with act_col:
                    if st.button("✅ 应用本组", key=f"apply_auto_{gid}", use_container_width=True):
                        applied = apply_merge_groups(
                            [group],
                            action_name=f"应用自动合并 {gid}"
                        )
                        drop_groups_by_codes(codes)
                        st.success(f"已应用 {applied} 组")
                        time.sleep(0.5)
                        st.rerun()

    if merge_review_groups:
        st.markdown("##### 👀 建议人工确认")
        all_codes_options = sorted(df['code'].dropna().astype(str).unique().tolist())

        for group in merge_review_groups:
            gid = group["group_id"]
            default_codes = group.get("codes", [])
            default_name = group.get("merged_name", "") or (default_codes[0] if default_codes else "")
            score = group.get("score", 0.0)
            reasoning = group.get("reasoning", "")

            with st.container(border=True):
                col_info, col_act = st.columns([3, 1])

                with col_info:
                    st.write(f"**{gid}**｜建议人工确认｜参考置信度：`{score:.2f}`")
                    if reasoning:
                        st.caption("判断依据：" + reasoning)

                    keep = st.multiselect(
                        "包含标签",
                        options=sorted(list(set(all_codes_options + default_codes))),
                        default=default_codes,
                        key=f"review_keep_{gid}"
                    )

                    if keep:
                        with st.expander("📄 查看引文", expanded=True):
                            render_merge_group_quotes(df, keep)

                with col_act:
                    freqs = df[df['code'].isin(keep)]['code'].value_counts()
                    rec_name = freqs.idxmax() if not freqs.empty else default_name

                    new_n = st.text_input(
                        "合并为",
                        value=rec_name,
                        key=f"review_name_{gid}"
                    )

                    if st.button("✅ 确认合并", key=f"confirm_review_{gid}", use_container_width=True):
                        if len(keep) < 2:
                            st.warning("至少选择 2 个标签")
                        else:
                            applied = apply_merge_groups(
                                [{
                                    "codes": keep,
                                    "merged_name": new_n
                                }],
                                action_name=f"人工确认合并 {gid}"
                            )
                            drop_groups_by_codes(keep)
                            st.success(f"已应用 {applied} 组")
                            time.sleep(0.5)
                            st.rerun()

                    if st.button("🚫 忽略本组", key=f"ignore_review_{gid}", use_container_width=True):
                        drop_groups_by_codes(default_codes)
                        st.info("已忽略本组")
                        time.sleep(0.3)
                        st.rerun()

    if pair_count == 0 and not merge_auto_groups and not merge_review_groups:
        st.info("暂无智能清洗建议。")

    st.divider()
    if st.button("完成清洗，进入积木归类 ➡️", key="to_kanban_stage", use_container_width=True):
        ok, msg = autosave_final_snapshot("完成标签清洗")
        if ok:
            st.success("已自动保存到 analysis_final/page4_input.xlsx，请点击上方【🧱 积木归类（轴心分析）】继续。")
        else:
            st.error(f"自动保存失败：{msg}")

# -----------------------------------------------------------------------
# TAB 3: 积木归类
# -----------------------------------------------------------------------
with tab_kanban:
    if not HAS_SORTABLE:
        st.error("需安装 streamlit-sortables")
    else:
        st.markdown("### 🧱 积木归类（轴心分析）")
        st.caption("先输入研究领域与研究主题，再由 AI 生成结构概览，并在下方拖拽修订。")

        ctx1, ctx2 = st.columns(2)
        with ctx1:
            render_shared_domain_input(
                "研究领域",
                widget_key="kanban_research_domain_input_tab3",
                mirror_keys=["kanban_research_domain_input_tab2"],
                placeholder="例如：组织社会化 / 教育社会学 / 平台劳动研究"
            )

        with ctx2:
            render_shared_topic_input(
                "研究主题",
                widget_key="kanban_research_topic_input_tab3",
                mirror_keys=["kanban_research_topic_input_tab2"],
                placeholder="例如：新员工组织融入过程中的情感体验与适应机制"
            )

        top_a, top_b, top_c, top_d = st.columns([1.4, 1.0, 1.0, 1.5])

        with top_a:
            recommend_clicked = st.button(
                "🤖 AI推荐维度",
                type="primary",
                use_container_width=True,
                disabled=not api_ready
            )

        with top_b:
            st.selectbox(
                "模式",
                options=["默认", "更细分", "更收敛"],
                key="kanban_ai_mode"
            )

        with top_c:
            retry_clicked = st.button(
                "🔄 重新生成",
                use_container_width=True,
                disabled=not api_ready
            )

        with top_d:
            save_clicked = st.button(
                "✅ 确认当前归类",
                type="primary",
                use_container_width=True,
                disabled=not bool(st.session_state.get("sortable_items"))
            )

        with st.expander("📊 数据预览", expanded=False):
            st.html(generate_html_tag_cloud(df))

        if recommend_clicked or retry_clicked:
            research_domain = (
                st.session_state.get("kanban_research_domain_input_tab3", "").strip()
                or st.session_state.get("kanban_research_domain_input_tab2", "").strip()
                or st.session_state.get("kanban_research_domain", "").strip()
            )

            research_topic = (
                st.session_state.get("kanban_research_topic_input_tab3", "").strip()
                or st.session_state.get("kanban_research_topic_input_tab2", "").strip()
                or st.session_state.get("kanban_research_topic", "").strip()
            )

            st.session_state["kanban_research_domain"] = research_domain
            st.session_state["kanban_research_topic"] = research_topic

            if not research_domain or not research_topic:
                st.warning("请先填写研究领域和研究主题，再进行 AI 推荐。")
            elif not api_ready:
                st.error("需 API Key")
            else:
                if st.session_state.get("sortable_items"):
                    push_history("AI推荐维度")

                status_box = st.empty()
                status_box.info("正在结合研究主题分析开放编码…")
                time.sleep(0.2)

                status_box.info("正在生成研究导向的维度名…")
                sortable_items, kanban_meta = build_ai_kanban(
                    df,
                    st.session_state.api_key,
                    mode=st.session_state.get("kanban_ai_mode", "默认"),
                    research_domain=research_domain,
                    research_topic=research_topic
                )

                if sortable_items:
                    status_box.info("正在分配编码到维度…")
                    time.sleep(0.2)

                    st.session_state.sortable_items = sortable_items
                    st.session_state.kanban_meta = kanban_meta
                    st.session_state.clusters_cache = True

                    status_box.success("AI 已生成结构概览与拖拽工作区")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    status_box.error("生成失败，请重试。")

        sortable_items = st.session_state.get("sortable_items") or []
        kanban_meta = st.session_state.get("kanban_meta", {}) or {}
        fixed_headers = {"❓ 待定区", "🗑️ 回收站"}

        if not sortable_items:
            st.markdown(
                """
                <div style="
                    border:1px dashed #C9C9C9;
                    background:#FFFDF7;
                    border-radius:14px;
                    padding:28px 24px;
                    margin-top:12px;
                    color:#555;">
                    <div style="font-size:1.05rem; font-weight:700; margin-bottom:10px;">看板尚未生成</div>
                    <div style="line-height:1.9;">
                        • 请先填写研究领域与研究主题<br>
                        • 点击 <b>AI推荐维度</b> 后会直接生成归类结构<br>
                        • 上方只看结构概览，不展示具体标签<br>
                        • 下方人工修订区用于真正拖拽修改
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("#### 🧭 AI归类结构概览")
            st.caption("这里只用于快速理解当前结构，不展示具体标签。")

            overview_groups = sortable_items
            n_cols = min(3, len(overview_groups)) if overview_groups else 1

            for row_start in range(0, len(overview_groups), n_cols):
                row_groups = overview_groups[row_start:row_start + n_cols]
                cols = st.columns(n_cols)

                for idx, group in enumerate(row_groups):
                    header = group["header"]
                    definition = kanban_meta.get(header, {}).get("definition", "")
                    item_count = len(group.get("items", []))

                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div style="
                                background:#FAFAFA;
                                border:1px solid #E5E5E5;
                                border-radius:12px;
                                padding:12px 14px;
                                margin-bottom:10px;
                                min-height:92px;">
                                <div style="font-weight:700; color:#222; margin-bottom:4px;">
                                    {header}
                                </div>
                                <div style="font-size:0.86rem; color:#666; margin-bottom:6px;">
                                    {item_count} 个编码
                                </div>
                                <div style="font-size:0.9rem; color:#444; line-height:1.5;">
                                    {definition if definition else '&nbsp;'}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.markdown("---")

            st.markdown("#### ✋ 人工修订区")
            st.caption("下面才是实际工作区：拖拽标签、改名、新增或删除维度。")

            current_view = [
                {"header": g["header"], "items": g.get("items", [])}
                for g in sortable_items
            ]

            result_view = sort_items(
                current_view,
                multi_containers=True,
                direction='vertical',
                key="kb"
            )

            if result_view != current_view:
                push_history("拖拽积木分类")
                result_map = {g["header"]: g["items"] for g in result_view}

                new_full_state = []
                for g in st.session_state.sortable_items:
                    new_full_state.append({
                        "header": g["header"],
                        "items": result_map.get(g["header"], g.get("items", []))
                    })

                st.session_state.sortable_items = new_full_state
                st.rerun()

            with st.expander("⚙️ 维度管理", expanded=False):
                st.caption("可改名、新增、删除维度；待定区与回收站固定保留。")

                rename_map = {}

                for idx, g in enumerate(st.session_state.sortable_items):
                    header = g["header"]
                    locked = header in fixed_headers
                    definition = kanban_meta.get(header, {}).get("definition", "")
                    item_count = len(g.get("items", []))

                    col_r1, col_r2, col_r3 = st.columns([4, 4, 1.2])

                    with col_r1:
                        rename_val = st.text_input(
                            f"维度 {idx + 1}",
                            value=header,
                            key=f"rename_dim_{idx}",
                            disabled=locked
                        )
                        rename_map[header] = rename_val

                    with col_r2:
                        st.text_input(
                            "定义",
                            value=definition,
                            key=f"dim_def_{idx}",
                            disabled=True
                        )

                    with col_r3:
                        st.markdown(
                            f"<div style='padding-top:32px; color:#666; text-align:center;'>{item_count} 个</div>",
                            unsafe_allow_html=True
                        )
                        if (not locked) and st.button("🗑 删除", key=f"del_dim_{idx}", use_container_width=True):
                            push_history(f"删除分类: {header}")

                            moved_items = []
                            new_state = []
                            for item in st.session_state.sortable_items:
                                if item["header"] == header:
                                    moved_items.extend(item.get("items", []))
                                else:
                                    new_state.append(item)

                            for item in new_state:
                                if item["header"] == "❓ 待定区":
                                    item["items"] = item.get("items", []) + moved_items

                            st.session_state.sortable_items = new_state
                            if header in st.session_state.kanban_meta:
                                del st.session_state.kanban_meta[header]
                            st.rerun()

                cm1, cm2 = st.columns([2, 1])

                with cm1:
                    if st.button("✅ 应用名称修改", use_container_width=True):
                        proposed = []
                        seen = set()
                        valid = True
                        err_msg = ""

                        for g in st.session_state.sortable_items:
                            old_name = g["header"]
                            new_name = old_name if old_name in fixed_headers else rename_map.get(old_name, old_name).strip()

                            if not new_name:
                                valid = False
                                err_msg = "维度名称不能为空"
                                break
                            if new_name in seen:
                                valid = False
                                err_msg = "维度名称不能重复"
                                break

                            seen.add(new_name)
                            proposed.append((old_name, new_name))

                        if not valid:
                            st.warning(err_msg)
                        else:
                            push_history("修改分类名称")

                            old_meta = copy.deepcopy(st.session_state.get("kanban_meta", {}))
                            old_items = {g["header"]: g.get("items", []) for g in st.session_state.sortable_items}

                            new_state = []
                            new_meta = {}

                            for old_name, new_name in proposed:
                                new_state.append({
                                    "header": new_name,
                                    "items": old_items.get(old_name, [])
                                })
                                new_meta[new_name] = old_meta.get(
                                    old_name,
                                    {"definition": "", "locked": new_name in fixed_headers}
                                )

                            st.session_state.sortable_items = new_state
                            st.session_state.kanban_meta = new_meta
                            st.rerun()

                with cm2:
                    new_dim = st.text_input("新建维度", key="new_dim_input")
                    if st.button("➕ 添加", use_container_width=True):
                        new_dim = new_dim.strip()
                        headers = [g["header"] for g in st.session_state.sortable_items]

                        if not new_dim:
                            st.warning("请输入维度名称")
                        elif new_dim in headers:
                            st.warning("该维度已存在")
                        else:
                            push_history(f"添加分类: {new_dim}")

                            insert_at = max(1, len(st.session_state.sortable_items) - 1)
                            st.session_state.sortable_items.insert(insert_at, {
                                "header": new_dim,
                                "items": []
                            })
                            st.session_state.kanban_meta[new_dim] = {
                                "definition": "人工新增维度",
                                "locked": False
                            }
                            st.rerun()

        if save_clicked:
            push_history("保存轴心归类")
            saved_count = save_current_kanban_result(st.session_state.sortable_items, fixed_headers)
            if saved_count:
                autosave_final_snapshot("确认当前归类")
                st.success(f"已确认 {saved_count} 条归类结果！")
            else:
                st.warning("当前没有可保存的归类结果。")

        st.divider()
        if st.button(
            "进入下一页：轴心编码 ➡️",
            key="go_axial_page",
            type="primary",
            use_container_width=True,
            disabled=not bool(st.session_state.get("sortable_items"))
        ):
            save_current_kanban_result(st.session_state.get("sortable_items", []), fixed_headers)
            ok, msg = autosave_final_snapshot("进入下一页：轴心编码")
            if ok:
                st.switch_page("pages/4_Axial_Coding.py")
            else:
                st.error(f"自动保存失败：{msg}")
