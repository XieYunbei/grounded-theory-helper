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

# 尝试导入拖拽库
try:
    from streamlit_sortables import sort_items
    HAS_SORTABLE = True
except ImportError:
    HAS_SORTABLE = False

# =======================================================================
# 0. 核心工具函数 & 数据加载模块
# =======================================================================

RECOVERY_DIR = "recovery_data_visual_analysis"

def ensure_recovery_dir():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR)

def load_from_jsonl(filepath):
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip(): records.append(json.loads(line))
                except: continue
    
    flat_codes = []
    for r in records:
        idx = r.get('original_row_index')
        codes_list = r.get('generated_codes', [])
        source_file = r.get('source_file', 'unknown')
        if isinstance(codes_list, list):
            for c in codes_list:
                if isinstance(c, dict):
                    flat_codes.append({
                        'source_file': source_file,
                        'original_row_index': idx,
                        # 完整保留四列状态
                        'original_code': c.get('original_code', c.get('code')),
                        'peer_code': c.get('peer_code', None),
                        'aligned_code': c.get('aligned_code', c.get('code')),
                        'code': c.get('code'), # 最终列
                        'quote': c.get('quote'),
                        'confidence': c.get('confidence', 0)
                    })
    return pd.DataFrame(flat_codes)

def save_current_progress(df):
    """保存全量状态"""
    if df.empty: return None
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
                        "original_code": row.get('original_code'),
                        "peer_code": row.get('peer_code'),
                        "aligned_code": row.get('aligned_code'),
                        "code": row['code'], 
                        "quote": row['quote'],
                        "confidence": row.get('confidence', 0)
                    })
                record = {
                    "original_row_index": int(idx) if pd.notna(idx) else None,
                    "source_file": first_row.get('source_file', 'unknown'),
                    "generated_codes": codes_list,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return filename
    return None

# [NEW] 撤销系统 (Undo System)
def push_history(action_name="Unknown Action"):
    """在修改数据前调用，保存当前状态快照"""
    if 'history_stack' not in st.session_state:
        st.session_state.history_stack = []
    
    # 限制栈深度，防止内存爆炸 (存最近 10 步)
    if len(st.session_state.history_stack) >= 10:
        st.session_state.history_stack.pop(0)
    
    snapshot = {
        'open_codes': st.session_state.open_codes.copy(deep=True),
        'axial_codes_df': st.session_state.axial_codes_df.copy(deep=True),
        'desc': action_name,
        'time': time.strftime("%H:%M:%S")
    }
    st.session_state.history_stack.append(snapshot)

def perform_undo():
    """执行撤销"""
    if 'history_stack' in st.session_state and st.session_state.history_stack:
        last_state = st.session_state.history_stack.pop()
        st.session_state.open_codes = last_state['open_codes']
        st.session_state.axial_codes_df = last_state['axial_codes_df']
        st.toast(f"已撤销: {last_state['desc']}")
        time.sleep(0.5)
        st.rerun()
    else:
        st.warning("没有可撤销的操作")

@st.cache_data(show_spinner=False)
def get_embeddings_dashscope(text_list, api_key):
    if not text_list: return []
    client = OpenAI(
        api_key=api_key, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
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
    if len(codes) < 2: return {0: codes}
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold, 
        metric='cosine', 
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    clusters = {}
    for code, label in zip(codes, labels):
        if label not in clusters: clusters[label] = []
        clusters[label].append(code)
    return clusters

def find_synonym_groups(codes, embeddings, threshold=0.85):
    if len(codes) < 2: return {}
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    groups = {}
    for i, (code, label) in enumerate(zip(codes, labels)):
        if label not in groups: groups[label] = {"codes": [], "indices": []}
        groups[label]["codes"].append(code)
        groups[label]["indices"].append(i)
    result_groups = {}
    for lbl, data in groups.items():
        if len(data["codes"]) > 1:
            if len(data["indices"]) > 1:
                group_emb = embeddings[data["indices"]]
                sim_matrix = cosine_similarity(group_emb)
                avg_sim = np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
            else: avg_sim = 1.0
            result_groups[lbl] = {"codes": data["codes"], "score": avg_sim}
    return result_groups

# [OPTIMIZED] 优化后的对齐算法 (加速版)
def align_records_by_quote(df_mine, df_theirs, match_threshold=0.6):
    theirs_records = df_theirs.to_dict('records')
    alignment = []
    mine_records = df_mine.to_dict('records')
    
    # 预处理：构建由引文长度索引的列表，减少遍历范围
    # 简单分桶：按长度分桶，步长为 10
    theirs_buckets = {}
    for r in theirs_records:
        q_len = len(str(r.get('quote', '')))
        bucket_id = q_len // 10
        if bucket_id not in theirs_buckets: theirs_buckets[bucket_id] = []
        theirs_buckets[bucket_id].append(r)
    
    for my_row in mine_records:
        my_quote = str(my_row.get('quote', ''))
        my_len = len(my_quote)
        my_bucket = my_len // 10
        my_code = str(my_row.get('code', ''))
        
        best_match = None
        best_ratio = 0
        
        # 只搜索长度相近的桶 (前后各扩1个桶)
        candidates = []
        for b in [my_bucket-1, my_bucket, my_bucket+1]:
            if b in theirs_buckets:
                candidates.extend(theirs_buckets[b])
        
        # 如果桶策略漏了（或者quote极短/极长），则全量兜底？
        # 为了性能，这里假设引文长度差异不会太大。如果candidates为空，则扩大搜索或全量
        if not candidates: 
            candidates = theirs_records # Fallback
            
        # 进一步优化：字面交集预筛 (Jaccard Pre-filter)
        # 只有当字符交集大于一定比例才进行 SequenceMatcher
        my_char_set = set(my_quote)
        
        for their_row in candidates:
            their_quote = str(their_row.get('quote', ''))
            
            # 快速 Jaccard 检查
            if not my_quote and not their_quote:
                ratio = 1.0
            else:
                their_char_set = set(their_quote)
                intersection = len(my_char_set & their_char_set)
                union = len(my_char_set | their_char_set)
                jaccard = intersection / union if union > 0 else 0
                
                # 如果 Jaccard 连 0.3 都不到，SequenceMatcher 肯定也很低，跳过
                if jaccard < 0.3: 
                    continue
                    
                # 昂贵的计算
                ratio = SequenceMatcher(None, my_quote, their_quote).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = their_row
        
        status = "unique"
        their_code = None
        if best_ratio >= match_threshold:
            their_code = str(best_match.get('code', ''))
            # 此时比较的是 my_code (可能已修改) 和 their_code
            # 注意：对齐时主要看差异，这里标记 conflict
            if my_code.strip() == their_code.strip(): status = "agreed"
            else: status = "conflict"
        
        alignment.append({
            "quote": my_quote, "my_code": my_code, "their_code": their_code,
            "status": status, "similarity": best_ratio,
            "raw_row_idx": my_row.get('original_row_index')
        })
    return alignment

def generate_html_tag_cloud(df):
    if df.empty or 'code' not in df.columns: return "无数据"
    counts = df['code'].value_counts()
    if counts.empty: return "无有效标签"
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

def reset_analysis_state():
    keys = ['embeddings', 'clusters_cache', 'sortable_items', 'merge_groups', 'alignment_results', 'page_num_align']
    for k in keys:
        if k in st.session_state: del st.session_state[k]

# =======================================================================
# 1. 页面配置与 CSS
# =======================================================================
st.set_page_config(page_title="分析工作台", layout="wide")

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
    .stSortable > div > div {
        background-color: #E6F7FF !important; border: 1px solid #69C0FF !important; color: #003a8c !important; 
        border-radius: 6px !important; font-size: 1.1rem !important; font-weight: 600 !important;
        padding: 10px !important; margin-bottom: 8px !important;
        white-space: normal !important; line-height: 1.4 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .custom-card-hint {
        background-color: #E6F7FF; border: 1px solid #91D5FF; border-radius: 6px;
        padding: 12px; color: #0050B3; font-size: 1rem; margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧩 分析工作台：清洗、对齐与归类")

# 状态初始化
if 'axial_codes_df' not in st.session_state:
    st.session_state.axial_codes_df = pd.DataFrame(columns=['code', 'category', 'confidence', 'reasoning', 'status'])
if 'clusters_cache' not in st.session_state: st.session_state.clusters_cache = None
if 'history_stack' not in st.session_state: st.session_state.history_stack = []

# 数据加载
data_missing = 'open_codes' not in st.session_state or st.session_state.open_codes is None or st.session_state.open_codes.empty
data_invalid = False
if not data_missing:
    if 'code' not in st.session_state.open_codes.columns: data_invalid = True

if data_missing or data_invalid:
    if data_invalid: st.warning("⚠️ 数据格式错误（缺少 'code' 列）。")
    else: st.info("👋 请上传数据以开始分析。")
    uploaded_file = st.file_uploader("📂 上传开放编码结果表 (Excel/CSV)", type=['xlsx', 'csv'], key="primary_uploader")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): df_load = pd.read_csv(uploaded_file)
            else: df_load = pd.read_excel(uploaded_file)
            if 'code' not in df_load.columns: st.error("文件缺少 'code' 列。"); st.stop()
            
            # [INIT] 初始化4列结构
            if 'original_row_index' not in df_load.columns:
                df_load['original_row_index'] = range(len(df_load))
                
            if 'original_code' not in df_load.columns: df_load['original_code'] = df_load['code']
            if 'peer_code' not in df_load.columns: df_load['peer_code'] = None
            if 'aligned_code' not in df_load.columns: df_load['aligned_code'] = df_load['code']
            
            st.session_state.open_codes = df_load
            reset_analysis_state() 
            st.success("✅ 数据加载成功！")
            time.sleep(1); st.rerun()
        except Exception as e: st.error(f"读取失败: {e}"); st.stop()
    else: st.stop()

# 确保列存在
df = st.session_state.open_codes
for col in ['original_code', 'peer_code', 'aligned_code', 'original_row_index']:
    if col not in df.columns:
        if col == 'peer_code': df[col] = None
        elif col == 'original_row_index': df[col] = range(len(df))
        else: df[col] = df['code']
st.session_state.open_codes = df 

unique_codes = df['code'].dropna().unique().tolist()

# 侧边栏与API
with st.sidebar:
    st.header("📂 进度管理")
    
    # [NEW] 撤销按钮
    if st.session_state.history_stack:
        if st.button(f"↩️ 撤销上一步 ({len(st.session_state.history_stack)})", type="primary", width="stretch"):
            perform_undo()
    else:
        st.button("↩️ 撤销 (无记录)", disabled=True, width="stretch")
    
    st.divider()
    
    if os.path.exists(RECOVERY_DIR):
        jsonl_files = glob.glob(os.path.join(RECOVERY_DIR, "*.jsonl"))
        jsonl_files.sort(key=os.path.getmtime, reverse=True)
        if jsonl_files:
            with st.expander("📥 加载历史进度", expanded=True):
                selected_file = st.selectbox("选择历史文件", [os.path.basename(f) for f in jsonl_files], index=0)
                if st.button("🔄 载入选中文件", width="stretch"):
                    filepath = os.path.join(RECOVERY_DIR, selected_file)
                    df_loaded = load_from_jsonl(filepath)
                    if not df_loaded.empty:
                        st.session_state.open_codes = df_loaded
                        reset_analysis_state() 
                        st.success(f"成功载入！")
                        time.sleep(1); st.rerun()
    
    st.divider()
    with st.expander("🔑 API Key 设置", expanded=not bool(st.session_state.get('api_key'))):
        val = st.session_state.get('api_key', '')
        new_key = st.text_input("DashScope Key", value=val, type="password")
        if new_key != val:
            st.session_state.api_key = new_key
            st.success("Key 已更新")
            
api_ready = bool(st.session_state.get('api_key'))

# =======================================================================
# 主选项卡布局
# =======================================================================
tab_align, tab_clean, tab_kanban = st.tabs(["🤝 队友对齐 (分歧解决)", "🧹 标签清洗 (同义合并)", "🧱 积木归类 (轴心分析)"])

# -----------------------------------------------------------------------
# TAB 1: 队友对齐
# -----------------------------------------------------------------------
with tab_align:
    st.caption("上传队友的编码文件，AI将自动对齐并列出差异。")
    file_peer = st.file_uploader("上传队友文件", type=['xlsx', 'csv', 'jsonl'])
    
    if file_peer:
        try:
            if file_peer.name.endswith('.csv'): df_peer = pd.read_csv(file_peer)
            elif file_peer.name.endswith('.jsonl'): df_peer = pd.read_json(file_peer, lines=True)
            else: df_peer = pd.read_excel(file_peer)
            
            if 'alignment_results' not in st.session_state or st.button("🔄 重新对比"):
                with st.spinner("正在快速比对... (已启用性能优化)"):
                    results = align_records_by_quote(df, df_peer)
                    st.session_state.alignment_results = results
                    
                    # 强力回填 peer_code
                    push_history("同步队友编码") # 保存状态
                    updates = 0
                    for r in results:
                        if r['raw_row_idx'] is not None and str(r['raw_row_idx']) in df['original_row_index'].astype(str).values:
                             mask = df['original_row_index'].astype(str) == str(r['raw_row_idx'])
                             st.session_state.open_codes.loc[mask, 'peer_code'] = r['their_code']
                             updates += 1
                        elif r['quote']:
                             mask = st.session_state.open_codes['quote'] == r['quote']
                             st.session_state.open_codes.loc[mask, 'peer_code'] = r['their_code']
                             updates += 1
                    
                    if updates > 0:
                        save_current_progress(st.session_state.open_codes)
                        st.toast(f"已同步 {updates} 条队友数据")
            
            results = st.session_state.alignment_results
            conflicts = [r for r in results if r['status'] == 'conflict']
            
            if conflicts:
                st.warning(f"发现 {len(conflicts)} 处分歧")
                page_size = 4
                if 'page_num_align' not in st.session_state: st.session_state.page_num_align = 0
                start_idx = st.session_state.page_num_align * page_size
                current_batch = conflicts[start_idx:start_idx+page_size]
                
                st.progress(min(1.0, (start_idx + len(current_batch)) / len(conflicts)))
                
                for i, item in enumerate(current_batch):
                    idx_real = start_idx + i
                    with st.container(border=True):
                        st.markdown(f"<div class='quote-box'>{item['quote']}</div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"👤 我: **{item['my_code']}**")
                            if st.button("👈 保留我的", key=f"k_my_{idx_real}", width="stretch"):
                                item['status'] = 'resolved'; st.rerun()
                        with c2:
                            st.warning(f"👥 他: **{item['their_code']}**")
                            if st.button("👉 采纳队友", key=f"k_th_{idx_real}", width="stretch"):
                                push_history(f"采纳队友编码: {item['their_code']}") # Undo
                                mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                       (st.session_state.open_codes['original_code'] == item['my_code'])
                                if mask.any():
                                    st.session_state.open_codes.loc[mask, 'aligned_code'] = item['their_code']
                                    st.session_state.open_codes.loc[mask, 'code'] = item['their_code']
                                    save_current_progress(st.session_state.open_codes)
                                    item['status'] = 'resolved'; st.success("已更新"); time.sleep(0.5); st.rerun()
                                else: st.error("定位失败")
                        
                        ai_k = f"ai_adv_{idx_real}"
                        custom_code = st.text_input("✏️ 修改为", value=st.session_state.get(ai_k, item['my_code']), key=f"inp_{idx_real}")
                        ca, cb = st.columns([1, 2])
                        with ca:
                            if st.button("🤖 问AI", key=f"ask_{idx_real}", disabled=not api_ready):
                                prompt = f"引文：{item['quote']}\n标签A：{item['my_code']}\n标签B：{item['their_code']}\n请给出一个最准确的简短标签："
                                try:
                                    client = OpenAI(api_key=st.session_state.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                                    res = client.chat.completions.create(model="qwen-plus", messages=[{"role":"user","content":prompt}])
                                    st.session_state[ai_k] = res.choices[0].message.content.strip()
                                    st.rerun()
                                except: st.error("API Error")
                        with cb:
                            if st.button("✅ 应用修改", key=f"app_{idx_real}", type="primary", width="stretch"):
                                push_history(f"修改编码为: {custom_code}") # Undo
                                mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                       (st.session_state.open_codes['original_code'] == item['my_code'])
                                if mask.any():
                                    st.session_state.open_codes.loc[mask, 'aligned_code'] = custom_code
                                    st.session_state.open_codes.loc[mask, 'code'] = custom_code
                                    save_current_progress(st.session_state.open_codes)
                                    item['status'] = 'resolved'; st.success("已更新"); time.sleep(0.5); st.rerun()

                cp1, cp2 = st.columns(2)
                if st.session_state.page_num_align > 0:
                    if cp1.button("⬅️ 上一页"): st.session_state.page_num_align -= 1; st.rerun()
                if start_idx + page_size < len(conflicts):
                    if cp2.button("下一页 ➡️"): st.session_state.page_num_align += 1; st.rerun()
            else:
                st.success("🎉 所有分歧已解决！")
        except Exception as e: st.error(f"Error: {e}")
    
    st.divider()
    if st.button("💾 手动保存对齐进度", key="save_align_manual", width="stretch"):
         fn = save_current_progress(st.session_state.open_codes)
         st.success(f"已保存")

# -----------------------------------------------------------------------
# TAB 2: 同义合并
# -----------------------------------------------------------------------
with tab_clean:
    c1, c2 = st.columns([2, 1])
    c1.markdown("#### 🧹 标签标准化"); c1.caption("合并语义重复的标签。此操作仅更新【最终清洗编码】列。")
    merge_threshold = c2.slider("相似度阈值", 0.7, 0.99, 0.85)
    
    if c2.button("🚀 扫描重复", type="primary"):
        if not api_ready: st.error("需 API Key")
        else:
            with st.spinner("分析中..."):
                u_codes = df['code'].dropna().unique().tolist()
                embs = get_embeddings_dashscope(u_codes, st.session_state.api_key)
                if len(embs)>0:
                    st.session_state.merge_groups = find_synonym_groups(u_codes, embs, merge_threshold)
                    st.rerun()

    if 'merge_groups' in st.session_state and st.session_state.merge_groups:
        groups = st.session_state.merge_groups
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]['score'], reverse=True)
        
        for gid, data in sorted_groups:
            codes = data["codes"]
            with st.container(border=True):
                col_info, col_act = st.columns([3, 1])
                with col_info:
                    st.write(f"**建议组** (相似度 {data['score']:.2f})")
                    all_codes_options = sorted(list(set(df['code'].dropna().unique().tolist() + codes)))
                    keep = st.multiselect("包含标签", all_codes_options, default=codes, key=f"ms_{gid}")
                    if keep:
                        with st.expander("📄 查看引文", expanded=True):
                            filtered_df = df[df['code'].isin(keep)][['code', 'quote']]
                            n_sample = min(5, len(filtered_df))
                            if n_sample > 0:
                                sub = filtered_df.sample(n_sample)
                                st.dataframe(sub, width="stretch", hide_index=True)
                with col_act:
                    freqs = df[df['code'].isin(keep)]['code'].value_counts()
                    rec_name = freqs.idxmax() if not freqs.empty else ""
                    new_n = st.text_input("合并为", value=rec_name, key=f"nn_{gid}")
                    if st.button("✅ 合并", key=f"bm_{gid}"):
                        push_history(f"合并标签: {keep} -> {new_n}") # Undo
                        st.session_state.open_codes['code'] = st.session_state.open_codes['code'].replace(keep, new_n)
                        save_current_progress(st.session_state.open_codes)
                        del st.session_state.merge_groups[gid]
                        st.success("已合并"); time.sleep(0.5); st.rerun()
        if not sorted_groups: st.success("暂无建议")
        
    st.divider()
    if st.button("💾 手动保存清洗进度", key="save_clean_manual", width="stretch"):
         fn = save_current_progress(st.session_state.open_codes)
         st.success(f"已保存")

# -----------------------------------------------------------------------
# TAB 3: 积木归类
# -----------------------------------------------------------------------
with tab_kanban:
    if not HAS_SORTABLE: st.error("需安装 streamlit-sortables")
    else:
        st.markdown("""<div class="custom-card-hint">🧱 <b>轴心编码工作台</b></div>""", unsafe_allow_html=True)
        cv1, cv2 = st.columns(2)
        with cv1: st.html(generate_html_tag_cloud(df))
        with cv2:
            top = df['code'].value_counts().head(10).reset_index()
            top.columns = ['code', 'count']
            c = alt.Chart(top).mark_bar().encode(
                x='count', y=alt.Y('code', sort='-x'), tooltip=['code','count']
            ).properties(height=200)
            st.altair_chart(c, width="stretch")

        st.divider()
        if st.session_state.clusters_cache is None:
            if st.button("🔄 初始化/重置 积木堆"):
                if not api_ready: st.error("需 API Key")
                else:
                    with st.spinner("聚类中..."):
                        uc = df['code'].dropna().unique().tolist()
                        embs = get_embeddings_dashscope(uc, st.session_state.api_key)
                        if len(embs)>0:
                            cl = perform_clustering(uc, embs, distance_threshold=0.4)
                            k_data = []
                            leftover = []
                            for lbl, items in cl.items():
                                freqs = {c: len(df[df['code']==c]) for c in items}
                                items_freq = [f"{c} (x{freqs[c]})" for c in items]
                                if len(items)>=2:
                                    rep = max(freqs, key=freqs.get)
                                    k_data.append({'header': f"{rep}", 'items': items_freq})
                                else: leftover.extend(items_freq)
                            k_data.insert(0, {'header': '❓ 待定区', 'items': leftover})
                            k_data.append({'header': '🗑️ 回收站', 'items': []})
                            st.session_state.sortable_items = k_data
                            st.session_state.clusters_cache = True
                            st.rerun()
        
        if 'sortable_items' in st.session_state:
            with st.expander("🔧 维度管理", expanded=True):
                c_m1, c_m2 = st.columns([2, 1])
                headers = [g['header'] for g in st.session_state.sortable_items]
                with c_m1:
                    edit_df = pd.DataFrame(headers, columns=["分类名称"])
                    edited_df = st.data_editor(edit_df, width="stretch", hide_index=True, key="hed")
                    if st.button("✅ 应用名称修改", width="stretch"):
                        push_history("修改分类名称") # Undo
                        new_h = edited_df["分类名称"].tolist()
                        if len(new_h) == len(set(new_h)):
                            new_state = []
                            old_map = {g['header']: g['items'] for g in st.session_state.sortable_items}
                            for h in new_h:
                                new_state.append({'header': h, 'items': old_map.get(h, [])})
                            for old_h, old_i in old_map.items():
                                if old_h not in new_h and old_i: new_state[0]['items'].extend(old_i)
                            st.session_state.sortable_items = new_state
                            st.rerun()
                with c_m2:
                    new_dim = st.text_input("新建维度")
                    if st.button("➕ 添加", width="stretch"):
                        if new_dim and new_dim not in headers:
                            push_history(f"添加分类: {new_dim}") # Undo
                            st.session_state.sortable_items.insert(1, {'header': new_dim, 'items': []})
                            st.rerun()

            view_opts = st.multiselect("显示维度", headers, default=headers[:6])
            curr_view = [g for g in st.session_state.sortable_items if g['header'] in view_opts]
            res = sort_items(curr_view, multi_containers=True, direction='vertical', key="kb")
            
            if res != curr_view:
                push_history("拖拽积木分类") # Undo
                res_map = {g['header']: g['items'] for g in res}
                new_full_state = []
                for g in st.session_state.sortable_items:
                    if g['header'] in res_map: g['items'] = res_map[g['header']]
                    new_full_state.append(g)
                st.session_state.sortable_items = new_full_state
                st.rerun()

            if st.button("💾 保存归类结果 (至 Page 3)", type="primary", width="stretch"):
                push_history("保存轴心归类") # Undo
                new_recs = []
                for g in st.session_state.sortable_items:
                    cat = g['header']
                    if '回收' in cat or '待定' in cat: continue
                    for it in g['items']:
                        code = it.split(' (x')[0]
                        new_recs.append({
                            'code': code, 'category': cat, 'confidence': 5, 
                            'reasoning': '人工拖拽', 'status': 'Accepted'
                        })
                if new_recs:
                    ndf = pd.DataFrame(new_recs)
                    st.session_state.axial_codes_df = pd.concat([
                        st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].isin(ndf['code'])],
                        ndf
                    ], ignore_index=True)
                    st.success(f"已保存 {len(new_recs)} 条结果！")

# =======================================================================
# 5. 全量清洗清单 (包含所有版本)
# =======================================================================
st.divider()
st.subheader("5️⃣ 开放编码清洗清单 (Full Traceability)")
st.caption("全流程追溯：原始 -> 对齐后 -> 最终清洗。您可以直接在此表格修改【最终清洗编码】。")

if not st.session_state.open_codes.empty:
    display_df = st.session_state.open_codes.copy()
    
    col_config = {
        "quote": st.column_config.TextColumn("原始引文", disabled=True, width="medium"),
        "original_code": st.column_config.TextColumn("👤 我的原始", disabled=True),
        "peer_code": st.column_config.TextColumn("👥 队友原始", disabled=True),
        "aligned_code": st.column_config.TextColumn("🤝 对齐后 (Tab1)", disabled=True),
        "code": st.column_config.TextColumn("✅ 最终清洗 (Tab2)", disabled=False)
    }
    
    final_cols = ['quote', 'original_code', 'peer_code', 'aligned_code', 'code']
    final_cols = [c for c in final_cols if c in display_df.columns]
    
    edited_clean_df = st.data_editor(
        display_df[final_cols], 
        column_config=col_config,
        use_container_width=True,
        key="clean_editor"
    )
    
    if st.button("💾 保存清单修改", type="primary", key="save_list_edit"):
        push_history("修改清洗清单") # Undo
        st.session_state.open_codes['code'] = edited_clean_df['code']
        fn = save_current_progress(st.session_state.open_codes)
        st.success(f"修改已保存！")
        time.sleep(0.5)
        st.rerun()

# =======================================================================
# 6. 结果展示 (轴心编码)
# =======================================================================
st.divider()
st.subheader("6️⃣ 已确认的轴心编码")
if not st.session_state.axial_codes_df.empty:
    st.dataframe(st.session_state.axial_codes_df[['category', 'code']], width="stretch")
