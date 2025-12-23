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

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def ensure_recovery_dir():
    ensure_dir(RECOVERY_DIR)

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
    
    # 确保 open_codes 结构是标准的4列
    temp_df = df.copy()
    if 'original_row_index' not in temp_df.columns: temp_df['original_row_index'] = range(len(temp_df))
    
    if 'original_row_index' in temp_df.columns:
        grouped = temp_df.groupby('original_row_index')
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

def save_analysis_progress(analysis_df, sortable_items):
    """保存当前的积木归类状态 (analysis_df 和 sortable_items)"""
    ensure_dir(ANALYSIS_STATE_DIR)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"AxialAnalysisState_{date_str}.json"
    filepath = os.path.join(ANALYSIS_STATE_DIR, filename)

    # 1. Prepare analysis_df (drop large embeddings)
    df_to_save = analysis_df.drop(columns=['embedding'], errors='ignore').to_dict('records')
    
    # 2. Combine all state data
    state_data = {
        'analysis_df_records': df_to_save,
        'sortable_items': sortable_items,
        'timestamp': datetime.datetime.now().isoformat()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=4)
        
    return filename

def process_analysis_state_data(state_data):
    """核心逻辑：从解析后的 JSON 字典中恢复 session state"""
    try:
        df_loaded = pd.DataFrame(state_data.get('analysis_df_records', []))
        if df_loaded.empty:
             st.error("加载的分析数据为空。")
             return False

        # 核心恢复步骤
        st.session_state.analysis_df = df_loaded
        st.session_state.sortable_items = state_data.get('sortable_items', [])
        st.session_state.clusters_cache = True # 标记为已载入

        # 确保 embedding 列存在
        if 'embedding' not in st.session_state.analysis_df.columns:
             st.session_state.analysis_df['embedding'] = None
        
        return True
    except Exception as e:
        st.error(f"恢复状态失败: {e}")
        return False

def load_analysis_progress_from_file(filename):
    """载入历史积木归类状态 (从本地文件)"""
    filepath = os.path.join(ANALYSIS_STATE_DIR, filename)
    if not os.path.exists(filepath):
        st.error(f"文件 {filename} 不存在。")
        return False
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state_data = json.load(f)
            
        return process_analysis_state_data(state_data)

    except Exception as e:
        st.error(f"载入失败: {e}")
        return False

def load_analysis_progress_from_uploaded_file(uploaded_file):
    """载入历史积木归类状态 (从上传文件，支持 JSON/Excel/CSV)"""
    file_name = uploaded_file.name
    
    try:
        if file_name.endswith('.json'):
            file_content = uploaded_file.read().decode("utf-8")
            state_data = json.loads(file_content)
            
            if process_analysis_state_data(state_data):
                st.toast("JSON状态文件载入成功！")
                return True
            return False

        elif file_name.endswith(('.xlsx', '.xls', '.csv')):
            
            if file_name.endswith('.csv'):
                df_import = pd.read_csv(uploaded_file)
            else:
                df_import = pd.read_excel(uploaded_file)
                
            required_cols = ['code', 'final_category'] 
            if not all(col in df_import.columns for col in required_cols):
                st.error(f"导入的 Excel/CSV 文件缺少必要的列。请确保文件包含以下列: {required_cols}")
                return False

            if 'analysis_df' not in st.session_state or st.session_state.analysis_df.empty:
                st.error("请先点击【初始化/重置 分析数据】按钮，获取 AI 聚类结果后再导入 Excel/CSV 文件进行分类覆盖。")
                return False
            
            # 核心覆盖逻辑
            current_df = st.session_state.analysis_df.copy()
            
            # 确保 code 列是字符串类型进行比对
            df_import['code'] = df_import['code'].astype(str) 
            
            # 创建一个用于映射的 Series
            update_map = df_import.set_index('code')['final_category']
            
            # 使用 update_map 更新 analysis_df
            st.session_state.analysis_df['final_category'] = st.session_state.analysis_df['code'].apply(
                lambda x: update_map.get(x) if x in update_map.index else (current_df.loc[current_df['code'] == x, 'final_category'].iloc[0] if (current_df['code'] == x).any() else None)
            )
            
            st.session_state.analysis_df.fillna({'final_category': None}, inplace=True)
            
            st.toast("Excel/CSV 分类结果已成功覆盖！")
            return True

        else:
            st.error("不支持的文件类型。")
            return False

    except Exception as e:
        st.error(f"处理上传文件失败: 错误信息: {e}")
        return False


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

# 词云图（颜色深浅表示频次）
def generate_html_tag_cloud_color_coded(df):
    """生成一个颜色深浅表示频次的词云图（频次越高，颜色越深）。"""
    if df.empty or 'code' not in df.columns: return "无数据"
    counts = df['code'].value_counts()
    if counts.empty: return "无有效标签"
    max_count = counts.max()
    min_count = counts.min()
    tags_html = ""
    
    # 基础颜色 (例如蓝色调)
    base_color_start = np.array([195, 232, 255]) # 浅蓝
    base_color_end = np.array([0, 58, 140]) # 深蓝
    
    # 统一大小，颜色按频次变化
    size = 18 
    
    for code, count in counts.items():
        # 根据频次计算插值比例 (0 到 1)
        if max_count == min_count:
            ratio = 0.5
        else:
            ratio = (count - min_count) / (max_count - min_count)
            
        # 线性插值计算颜色
        r = int(base_color_start[0] + (base_color_end[0] - base_color_start[0]) * ratio)
        g = int(base_color_start[1] + (base_color_end[1] - base_color_start[1]) * ratio)
        b = int(base_color_start[2] + (base_color_end[2] - base_color_start[2]) * ratio)
        
        color = f"#{r:02x}{g:02x}{b:02x}"
        
        tags_html += f"""<span style="font-size: {size}px; color: {color}; margin: 5px; padding: 5px; 
            display: inline-block; border: 1px solid #ccc; border-radius: 5px; font-weight: bold; background-color: #f0f8ff;"
            title="出现频次: {count}">{code}</span>"""
            
    return f"<div style='line-height: 2.0; text-align: center; padding: 20px; background: white; border-radius: 10px; border: 1px solid #eee; max-height: 250px; overflow-y: auto;'>{tags_html}</div>"

def reset_analysis_state():
    # 移除 analysis_df 会导致下次需要重新跑聚类
    keys = ['embeddings', 'clusters_cache', 'sortable_items', 'merge_groups', 'alignment_results', 'page_num_align', 'analysis_df']
    for k in keys:
        if k in st.session_state: del st.session_state[k]

# 优化后的对齐算法 (加速版) - 保持不变
def align_records_by_quote(df_mine, df_theirs, match_threshold=0.6):
    theirs_records = df_theirs.to_dict('records')
    alignment = []
    mine_records = df_mine.to_dict('records')
    
    # 预处理：构建由引文长度索引的列表，减少遍历范围
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
        
        candidates = []
        # 优化：只检查长度相近的引文
        for b in [my_bucket-1, my_bucket, my_bucket+1]:
            if b in theirs_buckets:
                candidates.extend(theirs_buckets[b])
        
        if not candidates: 
            # 如果没找到相近长度的，则退回到全量搜索
            candidates = theirs_records
            
        my_char_set = set(my_quote)
        
        for their_row in candidates:
            their_quote = str(their_row.get('quote', ''))
            
            if not my_quote and not their_quote:
                ratio = 1.0
            else:
                # 优化：通过 Jaccard 相似度快速排除明显不匹配的
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
            # 无论是否达到 match_threshold，只要找到最佳匹配，就记录其相似度和队友代码
            if my_code.strip() == their_code.strip(): status = "agreed"
            else: status = "conflict"
        
        alignment.append({
            "quote": my_quote, "my_code": my_code, "their_code": their_code,
            "status": status, "similarity": best_ratio,
            "raw_row_idx": my_row.get('original_row_index')
        })
    return alignment

def display_merge_groups(df, groups_to_display, mode_key):
    """封装标签合并的交互显示逻辑"""
    if not groups_to_display:
        st.success("暂无建议或已处理完毕。")
        return

    # 仅对要展示的 groups 进行排序
    sorted_groups = sorted(groups_to_display.items(), key=lambda x: x[1]['score'], reverse=True)
    
    for gid, data in sorted_groups:
        codes = data["codes"]
        with st.container(border=True):
            col_info, col_act = st.columns([3, 1])
            with col_info:
                st.write(f"**建议组** (相似度 {data['score']:.2f})")
                
                # Check if code is still in the main df before listing
                current_active_codes = st.session_state.open_codes['code'].dropna().unique().tolist()
                default_codes = [c for c in codes if c in current_active_codes]
                all_codes_options = sorted(list(set(current_active_codes + codes)))
                
                keep = st.multiselect(
                    "包含标签", 
                    all_codes_options, 
                    default=default_codes, 
                    key=f"ms_{gid}_{mode_key}"
                )
                
                if keep:
                    with st.expander("📄 查看引文", expanded=False):
                        filtered_df = df[df['code'].isin(keep)][['code', 'quote']]
                        n_sample = min(5, len(filtered_df))
                        if n_sample > 0:
                            sub = filtered_df.sample(n_sample)
                            st.dataframe(sub, width="stretch", hide_index=True)
            with col_act:
                freqs = df[df['code'].isin(keep)]['code'].value_counts()
                rec_name = freqs.idxmax() if not freqs.empty else ""
                new_n = st.text_input("合并为", value=rec_name, key=f"nn_{gid}_{mode_key}")
                if st.button("✅ 合并", key=f"bm_{gid}_{mode_key}"):
                    # 检查是否真的有变化
                    if new_n and new_n not in keep and keep:
                        push_history(f"合并标签: {keep} -> {new_n}") # Undo
                        # 确保只替换在 keep 列表且在当前 df 中的 code
                        replace_codes = [c for c in keep if c in st.session_state.open_codes['code'].values]
                        st.session_state.open_codes['code'] = st.session_state.open_codes['code'].replace(replace_codes, new_n)
                        save_current_progress(st.session_state.open_codes)
                        
                        # 从 session_state.merge_groups 中删除已处理的组
                        if gid in st.session_state.merge_groups:
                            del st.session_state.merge_groups[gid]
                            
                        st.success("已合并"); time.sleep(0.5); st.rerun()
                    else:
                        st.warning("操作无效：新标签名为空或在新标签名已经在被合并列表中。")

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
if 'page_num_align' not in st.session_state: st.session_state.page_num_align = 0


# 数据加载
data_missing = 'open_codes' not in st.session_state or st.session_state.open_codes is None or st.session_state.open_codes.empty
data_invalid = False
if not data_missing:
    if 'code' not in st.session_state.open_codes.columns: data_invalid = True

if data_missing or data_invalid:
    if data_invalid: st.error("⚠️ 数据格式错误（缺少 'code' 列）。")
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
            if 'quote' not in df_load.columns: df_load['quote'] = df_load['code'] # 假定 quote 缺失时，用 code 替代
            
            st.session_state.open_codes = df_load
            reset_analysis_state() 
            st.success("✅ 数据加载成功！")
            time.sleep(1); st.rerun()
        except Exception as e: st.error(f"读取失败: {e}"); st.stop()
    else: st.stop()

# 确保列存在 (防止用户跳过初始化步骤)
df = st.session_state.open_codes
for col in ['original_code', 'peer_code', 'aligned_code', 'original_row_index', 'quote']:
    if col not in df.columns:
        if col == 'peer_code': df[col] = None
        elif col == 'original_row_index': df[col] = range(len(df))
        elif col == 'quote': df[col] = df['code']
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
# 2. 主选项卡布局
# =======================================================================
tab_align, tab_clean, tab_kanban = st.tabs(["🤝 队友对齐 (分歧解决)", "🧹 标签清洗 (同义合并)", "🧱 积木归类 (轴心分析)"])

# -----------------------------------------------------------------------
# TAB 1: 队友对齐
# -----------------------------------------------------------------------
with tab_align:
    st.caption("上传队友的编码文件，AI将自动对齐并列出差异。")
    file_peer = st.file_uploader("上传队友文件", type=['xlsx', 'csv'])
    
    if file_peer:
        try:
            if file_peer.name.endswith('.csv'): df_peer = pd.read_csv(file_peer)
            else: df_peer = pd.read_excel(file_peer)
            
            # 确保 df_peer 有 quote 和 code 列
            if 'quote' not in df_peer.columns: 
                 st.warning("队友文件缺少 'quote' 列，无法对齐。请确保文件结构完整。")
                 file_peer = None
                 st.stop()
            if 'code' not in df_peer.columns: df_peer['code'] = df_peer['quote'] # 假设 code 缺失时用 quote 代替
            
            # 重新比对按钮，确保计算结果是最新的
            if 'alignment_results' not in st.session_state or st.button("🔄 重新对比", key="re_align_btn"):
                with st.spinner("正在快速比对..."):
                    # 使用稍微低的阈值确保所有潜在匹配都被捕获
                    results = align_records_by_quote(df, df_peer, match_threshold=0.6)
                    st.session_state.alignment_results = results
                    
                    # 强力回填 peer_code（同步队友的编码）
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
                        st.toast(f"已同步 {updates} 条队友数据")

            # --------------------- 模式选择 ---------------------
            st.divider()
            st.markdown("#### ⚙️ 队友对齐操作模式")
            align_mode = st.radio(
                "选择自动化程度：",
                ["半自动 (手动确认)", "分段模式 (部分自动)", "自动 (全部采纳)"],
                horizontal=True,
                key="align_mode_select"
            )
            st.divider()
            
            results = st.session_state.alignment_results
            conflicts_all = [r for r in results if r['status'] == 'conflict']
            conflicts = [] # [FIX] Initialize variable to avoid NameError in 'Automatic' mode

            # --------------------- 自动模式 ---------------------
            if align_mode == "自动 (全部采纳)":
                if st.button("🚀 运行全自动对齐/采纳", type="primary", key="auto_align_run"):
                    push_history("全自动队友对齐")
                    total_aligned = 0
                    
                    # 自动采纳逻辑：所有相似度 >= 0.6 的分歧，默认采纳【队友】的编码（因为对齐的目的是统一）
                    for item in conflicts_all:
                        if item['similarity'] >= 0.6: 
                            target_code = item['their_code'] 
                            
                            # 更新 open_codes 中的 code 和 aligned_code
                            mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                   (st.session_state.open_codes['original_code'] == item['my_code'])

                            if mask.any():
                                st.session_state.open_codes.loc[mask, 'aligned_code'] = target_code
                                st.session_state.open_codes.loc[mask, 'code'] = target_code
                                item['status'] = 'auto_resolved'
                                total_aligned += 1
                    
                    save_current_progress(st.session_state.open_codes)
                    st.success(f"🎉 运行结束，已自动处理 {total_aligned} 处分歧。")
                    st.session_state.alignment_results = results # 更新状态
                    st.rerun()
                
                st.info("⚠️ **注意：** 此模式将对所有引文相似度 $\ge 0.6$ 的分歧进行自动决策（采纳队友编码）。")

            # --------------------- 分段模式 ---------------------
            elif align_mode == "分段模式 (部分自动)":
                st.markdown("##### 分段模式界限设置")
                col_a, col_m = st.columns(2)
                threshold_auto = col_a.slider("自动采纳阈值 ($\geq$):", 0.7, 1.0, 0.95, key="align_auto_thresh")
                threshold_manual = col_m.slider("人工复核阈值 (介于):", 0.0, threshold_auto, 0.60, key="align_manual_thresh")
                
                if st.button("🚀 运行分段处理 (自动采纳高置信区)", type="primary", key="segment_align_run"):
                    push_history("分段模式队友对齐")
                    auto_resolved_count = 0
                    
                    # A. 处理高置信度区域（自动通过）
                    for item in conflicts_all:
                        if item['status'] == 'conflict' and item['similarity'] >= threshold_auto:
                            target_code = item['their_code']
                            
                            mask = (st.session_state.open_codes['quote'] == item['quote']) & \
                                   (st.session_state.open_codes['original_code'] == item['my_code'])

                            if mask.any():
                                st.session_state.open_codes.loc[mask, 'aligned_code'] = target_code
                                st.session_state.open_codes.loc[mask, 'code'] = target_code
                                item['status'] = 'auto_resolved'
                                auto_resolved_count += 1
                    
                    save_current_progress(st.session_state.open_codes)
                    st.success(f"🎉 高置信度区域已处理！已自动采纳 {auto_resolved_count} 条记录。")
                    st.session_state.alignment_results = results
                    st.rerun()

                # B. 人工复核区域：显示给用户交互 (similarity 介于两个阈值之间)
                conflicts = [r for r in conflicts_all if r['similarity'] >= threshold_manual and r['similarity'] < threshold_auto]
                
                if conflicts:
                    st.warning(f"🤖 AI 不确定区域：仍需人工复核 {len(conflicts)} 处分歧")
                else:
                    st.success("🎉 分段模式下，人工复核区域已清空！")
                    
            # --------------------- 半自动模式 (手动确认) ---------------------
            else: # align_mode == "半自动 (手动确认)"
                conflicts = conflicts_all
                if conflicts:
                    st.warning(f"📢 发现 {len(conflicts)} 处分歧，请手动复核：")
                else:
                    st.success("🎉 所有分歧已解决！")
            
            # --------------------- 交互展示逻辑（应用于分段/半自动模式下的 conflicts） ---------------------
            
            if conflicts:
                page_size = 4
                start_idx = st.session_state.page_num_align * page_size
                current_batch = conflicts[start_idx:start_idx+page_size]
                
                st.progress(min(1.0, (start_idx + len(current_batch)) / len(conflicts)))
                
                for i, item in enumerate(current_batch):
                    idx_real = start_idx + i
                    with st.container(border=True):
                        st.markdown(f"<div class='quote-box'>{item['quote']}</div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"👤 我: **{item['my_code']}** (相似度: {item['similarity']:.2f})")
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
                                else: st.error("定位失败")

                cp1, cp2 = st.columns(2)
                if st.session_state.page_num_align > 0:
                    if cp1.button("⬅️ 上一页"): st.session_state.page_num_align -= 1; st.rerun()
                if start_idx + page_size < len(conflicts):
                    if cp2.button("下一页 ➡️"): st.session_state.page_num_align += 1; st.rerun()
            
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
    # merge_threshold 用于扫描，不用于自动/分段模式的决策
    merge_threshold = c2.slider("扫描相似度阈值", 0.7, 0.99, 0.85, key="scan_thresh_clean_tab2")
    
    if c2.button("🚀 扫描重复", type="primary"):
        if not api_ready: st.error("需 API Key")
        else:
            with st.spinner("分析中..."):
                # 确保使用最新的 code 列表
                u_codes = df['code'].dropna().unique().tolist()
                if len(u_codes) < 2:
                    st.success("标签数量不足 2 个或已无重复标签，无需扫描。")
                    if 'merge_groups' in st.session_state: del st.session_state.merge_groups
                else:
                    embs = get_embeddings_dashscope(u_codes, st.session_state.api_key)
                    if len(embs) > 0:
                        # find_synonym_groups 内部使用了 1-threshold 作为距离
                        st.session_state.merge_groups = find_synonym_groups(u_codes, embs, merge_threshold)
                        st.rerun()

    st.divider()
    st.markdown("#### ⚙️ 标签清洗操作模式")
    clean_mode = st.radio(
        "选择自动化程度：",
        ["半自动 (手动确认)", "分段模式 (部分自动)", "自动 (全部采纳)"],
        horizontal=True,
        key="clean_mode_select"
    )
    st.divider()

    # 确保 groups 变量已定义并从 session_state 获取
    groups = st.session_state.get('merge_groups', {})
    
    # 无论 groups 是否为空，都先显示提示
    if not groups:
        st.info("请先点击上方的【🚀 扫描重复】按钮，以获取相似标签建议。")
    
    # --------------------- 自动模式 (阈值显示已修复) ---------------------
    if clean_mode == "自动 (全部采纳)":
        st.markdown("##### 自动合并设置")
        # [FIX] 滑动条已移至此处，无论 groups 是否为空都显示
        threshold_auto_mode = st.slider(
            "自动合并阈值 (相似度 $\geq$):", 
            0.7, 1.0, 0.90, 
            key="clean_full_auto_thresh"
        )
        
        if st.button("🚀 运行全自动合并", type="primary", key="auto_clean_run", disabled=not groups):
            if groups:
                push_history("全自动标签合并")
                total_merged = 0
                
                groups_to_merge = {k: v for k, v in groups.items() if v['score'] >= threshold_auto_mode}
                
                if not groups_to_merge:
                    st.warning(f"没有相似度 $\ge {threshold_auto_mode}$ 的标签组可供合并。请尝试降低阈值或重新扫描。")
                else:
                    for gid, data in groups_to_merge.items():
                        codes_to_replace = data["codes"]
                        freqs = df[df['code'].isin(codes_to_replace)]['code'].value_counts()
                        new_n = freqs.idxmax() if not freqs.empty else codes_to_replace[0]
                        
                        st.session_state.open_codes['code'] = st.session_state.open_codes['code'].replace(codes_to_replace, new_n)
                        if gid in st.session_state.merge_groups: del st.session_state.merge_groups[gid]
                        total_merged += len(codes_to_replace)
                    
                    save_current_progress(st.session_state.open_codes)
                    st.session_state.merge_groups = st.session_state.merge_groups
                    st.success(f"🎉 运行结束，已自动合并 {total_merged} 个标签（阈值 $\ge {threshold_auto_mode}$）。")
                    time.sleep(1)
                    st.rerun()

        st.info(f"⚠️ **注意：** 此模式将自动合并所有相似度 $\ge {threshold_auto_mode}$ 的标签组，无需人工逐一确认。")


    # --------------------- 分段模式 (阈值显示已修复) ---------------------
    elif clean_mode == "分段模式 (部分自动)":
        
        st.markdown("##### 分段模式界限设置")
        col_a, col_m = st.columns(2)
        # [FIX] 滑动条已移至此处，无论 groups 是否为空都显示
        threshold_auto = col_a.slider("自动合并阈值 ($\geq$):", 0.85, 1.0, 0.90, key="clean_auto_thresh")
        threshold_manual = col_m.slider("人工复核阈值 (介于):", 0.70, threshold_auto, 0.80, key="clean_manual_thresh")
        
        if st.button("🚀 运行分段处理 (自动合并高置信区)", type="primary", key="segment_clean_run", disabled=not groups):
            if groups:
                push_history("分段模式标签合并")
                auto_merged_count = 0
                
                # A. 找出要自动处理的组
                high_conf_groups = {k: v for k, v in groups.items() if v['score'] >= threshold_auto}
                
                if not high_conf_groups:
                    st.warning(f"本次运行没有发现相似度 $\ge {threshold_auto}$ 的标签组。")
                else:
                    for gid, data in high_conf_groups.items():
                        codes_to_replace = data["codes"]
                        freqs = df[df['code'].isin(codes_to_replace)]['code'].value_counts()
                        new_n = freqs.idxmax() if not freqs.empty else codes_to_replace[0]
                        
                        st.session_state.open_codes['code'] = st.session_state.open_codes['code'].replace(codes_to_replace, new_n)
                        if gid in st.session_state.merge_groups: del st.session_state.merge_groups[gid]
                        auto_merged_count += len(codes_to_replace)
                    
                    save_current_progress(st.session_state.open_codes)
                    st.session_state.merge_groups = st.session_state.merge_groups
                    st.success(f"🎉 高置信度区域已处理！已自动合并 {auto_merged_count} 个标签。")
                    time.sleep(0.5)
                st.rerun()

        # B. 人工复核区域：显示给用户交互 (similarity 介于两个阈值之间)
        st.divider() # 增加分割线
        
        if groups:
            current_groups = st.session_state.get('merge_groups', {})
            manual_conf_groups = {
                k: v for k, v in current_groups.items() 
                if v['score'] >= threshold_manual and v['score'] < threshold_auto
            }

            if manual_conf_groups:
                st.warning(f"🤖 人工复核区：相似度在 **[{threshold_manual:.2f}, {threshold_auto:.2f})** 之间，共需复核 {len(manual_conf_groups)} 组建议：")
                display_merge_groups(df, manual_conf_groups, "segment")
            else:
                st.success(f"🎉 当前无相似度在 **[{threshold_manual:.2f}, {threshold_auto:.2f})** 范围内的组需要人工复核！")

        
    # --------------------- 半自动模式 (现有逻辑) ---------------------
    else: # clean_mode == "半自动 (手动确认)"
        if groups:
            st.warning(f"📢 发现 {len(groups)} 组建议，请手动复核：")
            # 在半自动模式下，显示所有 groups
            display_merge_groups(df, groups, "semi_auto")
        
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
        
        # [NEW] 积木工作区进度管理
        st.divider()
        st.subheader("💾 积木工作区进度")
        
        c_load_save, c_load_dropdown = st.columns([1, 2])
        
        # 保存按钮
        with c_load_save:
            save_disabled = 'analysis_df' not in st.session_state or st.session_state.analysis_df.empty
            if st.button("💾 保存当前积木状态", type="secondary", disabled=save_disabled, key="save_axial_state_manual"):
                if 'analysis_df' in st.session_state and 'sortable_items' in st.session_state:
                    fn = save_analysis_progress(st.session_state.analysis_df, st.session_state.sortable_items)
                    st.success(f"积木状态已保存为 {fn}！")
                else:
                    st.warning("无数据可保存。请先初始化分析数据。")

        # 载入下拉框 (支持上传 JSON/Excel/CSV)
        with c_load_dropdown:
            uploaded_state_file = st.file_uploader(
                "📥 上传积木状态文件 (.json/.xlsx/.csv)", 
                type=['json', 'xlsx', 'xls', 'csv'], 
                key="uploaded_axial_state"
            )

            if uploaded_state_file is not None:
                if st.button(f"🔄 载入上传文件: {uploaded_state_file.name}", key="load_uploaded_axial_state", type="primary", use_container_width=True):
                    if load_analysis_progress_from_uploaded_file(uploaded_state_file): 
                        st.success(f"已成功载入上传文件：{uploaded_state_file.name}。")
                        time.sleep(0.5); st.rerun()
            
            st.markdown("---") # 分隔符
            ensure_dir(ANALYSIS_STATE_DIR)
            analysis_files = glob.glob(os.path.join(ANALYSIS_STATE_DIR, "*.json"))
            analysis_files.sort(key=os.path.getmtime, reverse=True)
            analysis_file_names = [os.path.basename(f) for f in analysis_files]

            if analysis_file_names:
                selected_file = st.selectbox("或选择历史积木状态 (本地)", analysis_file_names, index=0)
                if st.button("🔄 载入本地选中积木状态", key="load_axial_state_manual_local", type="secondary", use_container_width=True):
                    if load_analysis_progress_from_file(selected_file): 
                        st.success(f"已成功载入 {selected_file}。")
                        time.sleep(0.5); st.rerun()
            else:
                st.info("暂无本地历史积木状态可载入。")
        
        st.divider()
        
        # 模式切换 - 移除散点图模式 (C)
        analysis_mode = st.radio(
            "📊 分析模式选择",
            ["拖拽看板 (启发式)", "Data Editor 分组 (稳定版)"],
            index=1, 
            horizontal=True, key="analysis_mode"
        )
        
        cv1, cv2 = st.columns(2)
        with cv1: st.html(generate_html_tag_cloud_color_coded(df))
        with cv2: st.html(generate_html_tag_cloud_color_coded(df)) 
        
        # --- 模式切换下的初始化/重置按钮 ---
        st.divider()
        if st.session_state.clusters_cache is None or st.button("🔄 初始化/重置 分析数据"):
            if not api_ready: st.error("需 API Key")
            else:
                with st.spinner("聚类中..."):
                    uc = df['code'].dropna().unique().tolist()
                    embs = get_embeddings_dashscope(uc, st.session_state.api_key)
                    if len(embs)>0:
                        cl = perform_clustering(uc, embs, distance_threshold=0.4)
                        
                        # Data Editor 模式所需的数据准备 (新增 embeddings 缓存)
                        cluster_map = {item: lbl for lbl, items in cl.items() for item in items}
                        emb_map = {code: embs[i] for i, code in enumerate(uc)}
                        
                        temp_df = df.copy()
                        temp_df['cluster_id'] = temp_df['code'].apply(lambda x: f"AI Group {cluster_map.get(x, 'NA')}")
                        if 'final_category' not in temp_df.columns:
                            # 从已保存的轴心编码中同步现有分类
                            code_to_cat = st.session_state.axial_codes_df.set_index('code')['category'].to_dict()
                            temp_df['final_category'] = temp_df['code'].apply(lambda x: code_to_cat.get(x))

                        unique_code_df = temp_df.drop_duplicates(subset=['code'])
                        unique_code_df = unique_code_df[['code', 'cluster_id', 'final_category']].sort_values(by='cluster_id').reset_index(drop=True)
                        unique_code_df['embedding'] = unique_code_df['code'].apply(lambda x: emb_map.get(x))
                        
                        st.session_state.analysis_df = unique_code_df
                        
                        # 看板模式数据准备
                        k_data = []
                        leftover = []
                        for lbl, items in cl.items():
                            freqs = {c: len(df[df['code']==c]) for c in items}
                            items_freq = [f"{c} (x{freqs[c]})" for c in items]
                            if len(items)>=2:
                                rep = max(freqs, key=freqs.get)
                                # 确保不与 Data Editor 模式的 AI Group 混淆
                                k_data.append({'header': f"AI 建议组: {rep}", 'items': items_freq}) 
                            else: leftover.extend(items_freq)
                        
                        k_data.insert(0, {'header': '❓ 待定区', 'items': leftover})
                        k_data.append({'header': '🗑️ 回收站', 'items': []})
                        st.session_state.sortable_items = k_data
                        st.session_state.clusters_cache = True
                        st.rerun()

        # --- MODE 1: 拖拽看板 (启发式) ---
        if analysis_mode == "拖拽看板 (启发式)":
            if 'sortable_items' not in st.session_state:
                st.warning("请先点击【初始化/重置 分析数据】按钮或【载入历史积木状态】。")
            
            else: # 如果已初始化，显示拖拽看板
                
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
                                    # 保留原有内容
                                    new_state.append({'header': h, 'items': old_map.get(h, [])}) 
                                
                                # 处理被删除的维度：将内容移到待定区
                                existing_headers = [g['header'] for g in st.session_state.sortable_items]
                                for old_h, old_i in old_map.items():
                                    if old_h not in new_h and old_h in existing_headers:
                                        # 假设 '❓ 待定区' 永远是第一个
                                        if new_state and new_state[0]['header'] == '❓ 待定区':
                                            new_state[0]['items'].extend(old_i)
                                
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
                    # 重新构建完整的 sortable_items 状态
                    for g in st.session_state.sortable_items:
                        if g['header'] in res_map: g['items'] = res_map[g['header']]
                        new_full_state.append(g)
                    st.session_state.sortable_items = new_full_state
                    st.rerun()

                if st.button("💾 保存归类结果 (至 Page 3)", type="primary", width="stretch"):
                    push_history("保存轴心归类") # Undo
                    new_recs = []
                    
                    # 从 sortable_items 中提取结果
                    for g in st.session_state.sortable_items:
                        cat = g['header']
                        # 忽略辅助区
                        if '回收' in cat or '待定' in cat or 'AI 建议组' in cat: continue
                        
                        for it in g['items']:
                            code = it.split(' (x')[0].strip() # 提取标签名
                            new_recs.append({
                                'code': code, 'category': cat, 'confidence': 5, 
                                'reasoning': '人工拖拽', 'status': 'Accepted'
                            })
                    if new_recs:
                        ndf = pd.DataFrame(new_recs)
                        # 更新轴心编码表：保留未处理的，更新已处理的
                        st.session_state.axial_codes_df = pd.concat([
                            st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].isin(ndf['code'])],
                            ndf
                        ], ignore_index=True)
                        st.success(f"已保存 {len(new_recs)} 条结果！")

        # --- MODE 2: Data Editor 分组 (稳定版) ---
        elif analysis_mode == "Data Editor 分组 (稳定版)":
            if 'analysis_df' in st.session_state and not st.session_state.analysis_df.empty:
                st.info("💡 **稳定模式：** 拖拽不稳定时使用。表格已按【AI 建议分组】折叠，点击展开查看，然后批量设置【最终归类】。")
                
                # 维度管理 for Mode A
                with st.expander("🔧 维度管理 (新增轴心分类)", expanded=True):
                    current_categories = st.session_state.axial_codes_df['category'].dropna().unique().tolist()
                    new_dim = st.text_input("输入新的轴心分类名称", key="new_dim_a_input")
                    if st.button("➕ 添加新分类", key="add_new_cat_a"):
                        if new_dim and new_dim not in current_categories:
                            # 添加一个临时记录，确保新分类进入 axial_codes_df，从而出现在下拉菜单中
                            new_row = pd.DataFrame([{'code': f'NEW_TEMP_CODE_{int(time.time())}', 'category': new_dim, 'confidence': 0, 'reasoning': 'User Added', 'status': 'Pending'}])
                            st.session_state.axial_codes_df = pd.concat([st.session_state.axial_codes_df, new_row], ignore_index=True)
                            st.success(f"已添加分类：{new_dim}")
                            time.sleep(0.5)
                            st.rerun() # 刷新以更新下拉框选项

                # Get all categories including newly added ones
                current_categories = st.session_state.axial_codes_df['category'].dropna().unique().tolist()
                
                # 排除 embedding 列
                df_to_edit = st.session_state.analysis_df.drop(columns=['embedding'], errors='ignore')
                
                edited_analysis_df = st.data_editor(
                    df_to_edit,
                    column_config={
                        "code": "开放编码",
                        "cluster_id": st.column_config.TextColumn("AI 建议分组", disabled=True),
                        "final_category": st.column_config.SelectboxColumn(
                            "✅ 最终归类",
                            options=["(未归类)"] + current_categories, # 选项包括所有已创建的分类和未归类
                            required=True
                        )
                    },
                    hide_index=True,
                    num_rows="dynamic",
                    column_order=("cluster_id", "code", "final_category"),
                    key="data_editor_mode",
                    use_container_width=True
                )

                if st.button("💾 保存 Data Editor 归类结果", type="primary", width="stretch"):
                    push_history("保存 Data Editor 轴心归类") # Undo
                    new_recs = []
                    # 确保处理 edited_analysis_df 得到的结果
                    for _, row in edited_analysis_df.iterrows():
                        cat = row['final_category']
                        code = row['code']
                        if cat and cat != "(未归类)":
                             new_recs.append({
                                'code': code, 'category': cat, 'confidence': 5, 
                                'reasoning': 'Data Editor 分组确认', 'status': 'Accepted'
                            })
                    
                    if new_recs:
                        ndf = pd.DataFrame(new_recs)
                        # 更新轴心编码表，同时移除临时记录
                        st.session_state.axial_codes_df = pd.concat([
                            st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].isin(ndf['code'])],
                            ndf
                        ], ignore_index=True)
                        
                        # 清理临时代码
                        st.session_state.axial_codes_df = st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].str.startswith('NEW_TEMP_CODE_')]
                        
                        # 更新 analysis_df 的 final_category 状态
                        for code, category in zip(ndf['code'], ndf['category']):
                            st.session_state.analysis_df.loc[st.session_state.analysis_df['code'] == code, 'final_category'] = category
                        
                        st.success(f"已保存 {len(new_recs)} 条结果！")
            else:
                st.warning("请先点击【初始化/重置 分析数据】按钮或【载入历史积木状态】。")

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
    # 移除临时代码 (如果有)
    display_axial_df = st.session_state.axial_codes_df[~st.session_state.axial_codes_df['code'].str.startswith('NEW_TEMP_CODE_')]
    
    st.dataframe(display_axial_df[['category', 'code']], width="stretch")
