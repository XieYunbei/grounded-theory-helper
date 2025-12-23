# pages/1_Data_Upload.py (FIXED FOR EXCEL PROVENANCE)
import streamlit as st
import pandas as pd
import docx 
import re
import os
import datetime

# =======================================================================
# 辅助函数：智能分块读取 (保持不变)
# =======================================================================
CHUNK_SIZE = 800 

def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def auto_save_data(df, prefix="Processed"):
    """自动保存处理后的数据"""
    ensure_save_dir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.xlsx"
    filepath = os.path.join(SAVE_DIR, filename)
    try:
        df.to_excel(filepath, index=False)
        return filepath
    except Exception as e:
        st.error(f"自动保存失败: {e}")
        return None

def clean_content_smart(text, matched_keyword=None):
    """ 智能清洗 (仅去除格式噪音，不删内容) """
    if not isinstance(text, str) or not text: return ""
    
    # 1. 切掉显性关键词
    if matched_keyword and text.startswith(matched_keyword):
        text = text[len(matched_keyword):]
    
    # 2. 去除类似 (00:00): 的时间戳
    text = re.sub(r"^.{0,15}?[\[\(（]?\d{1,2}:\d{1,2}(:\d{1,2})?.*?[\]\)）]?\s*[:：]?", "", text)
    
    # 3. 兜底去除开头的冒号
    if not matched_keyword:
        text = re.sub(r"^[^0-9\n]{1,10}?[:：]", "", text)

    return text.strip().strip(":：")

def split_sentences(text):
    if not isinstance(text, str) or not text: return []
    pattern = r"([。！？!?]|\.\.\.+[”\"']?)"
    chunks = re.split(pattern, text)
    sentences = []
    current = ""
    for chunk in chunks:
        current += chunk
        if re.match(pattern, chunk):
            sentences.append(current.strip())
            current = ""
    if current.strip(): sentences.append(current.strip())
    return [s for s in sentences if s]

# --- Excel 解析逻辑 ---
def parse_excel_file(file):
    """
    智能解析 Excel
    1. 寻找 ID 列
    2. 寻找 内容 列
    3. 默认角色设为 A (受访者)
    """
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Excel 读取失败: {e}")
        return []

    # 1. 智能猜测 ID 列
    possible_id_names = ['被试编号', '被试ID', '编号', 'Participant_ID', 'Participant', 'ID', 'Subject', 'Name']
    id_col = None
    for col in df.columns:
        if any(name.lower() in str(col).lower() for name in possible_id_names):
            id_col = col
            break
    
    # 2. 智能猜测 内容 列 (排除ID列后，找最长的字符列，或者叫 Content/Text 的列)
    content_col = None
    possible_content_names = ['content', 'text', '内容', '文本', '回答', 'Answer', 'Response']
    
    # A. 优先找名字匹配的
    for col in df.columns:
        if col == id_col: continue
        if any(name.lower() in str(col).lower() for name in possible_content_names):
            content_col = col
            break
    
    # B. 没找到名字，找第一列非ID的列
    if not content_col:
        remaining_cols = [c for c in df.columns if c != id_col]
        if remaining_cols:
            content_col = remaining_cols[0]

    if not id_col or not content_col:
        st.warning(f"⚠️ 文件 `{file.name}` 结构识别存疑。\n自动识别 ID列: `{id_col}` | 内容列: `{content_col}`。\n建议表头包含：'被试编号' 和 '内容'。")
        if not content_col: return []

    # 3. 转换格式
    parsed_data = []
    for idx, row in df.iterrows():
        raw_text = str(row[content_col])
        # Excel 数据通常比较短，可能不需要切分句子，但为了统一，还是过一遍分句
        sents = split_sentences(raw_text)
        
        # ID 构造
        user_id = str(row[id_col]) if id_col else f"Row{idx+1}"
        
        for s in sents:
            parsed_data.append({
                "global_id": f"A-{user_id}-{idx+1}", # 构造唯一ID
                "role_code": "A", # 默认为受访者
                "content": s,
                "source_file": file.name,
                "file_index": 999 # Excel 默认放最后
            })
            
    return parsed_data

# --- Word/Txt 解析逻辑 (保留原逻辑) ---
def parse_lines_optimized(lines, q_list, a_list, force_non_q_to_a=False):
    parsed = []
    current_role = "N"
    q_list = sorted([str(k).strip() for k in q_list if str(k).strip()], key=len, reverse=True)
    a_list = sorted([str(k).strip() for k in a_list if str(k).strip()], key=len, reverse=True)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        line = line.replace('\ufeff', '')
        
        detected_role = None
        content = line
        matched_kw = None
        
        for kw in q_list:
            if line.startswith(kw):
                detected_role = "Q"; matched_kw = kw; break
        
        if not detected_role:
            for kw in a_list:
                if line.startswith(kw):
                    detected_role = "A"; matched_kw = kw; break
            if not detected_role and force_non_q_to_a:
                detected_role = "A"; matched_kw = None 
        
        if detected_role:
            content = clean_content_smart(line, matched_kw)
            current_role = detected_role
        else:
            if current_role != "N": content = line 
            else: current_role = "A"; content = clean_content_smart(line, None)

        sents = split_sentences(content)
        for s in sents:
            parsed.append({"role": current_role, "content": s})
    return parsed

def read_txt_docx(file):
    lines = []
    try:
        if file.name.endswith('.docx'):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                if para.text.strip(): lines.append(para.text.strip())
        elif file.name.endswith('.txt'):
            content = file.getvalue().decode("utf-8")
            lines = content.splitlines()
    except Exception as e:
        st.error(f"读取失败: {e}")
    return lines

# =======================================================================
# 1. 页面配置
# =======================================================================
st.set_page_config(page_title="数据预处理", layout="wide")
st.markdown("""
<style>
    .stApp { font-family: "Microsoft YaHei", sans-serif; }
    div[data-testid="stCheckbox"] label span p { font-size: 18px !important; font-weight: bold; color: #d63031; }
    .big-caption { font-size: 16px !important; color: #666; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("区域1: 数据导入与预处理中心 📥")

# Session State
if 'atomic_df' not in st.session_state: st.session_state.atomic_df = None
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'q_keywords' not in st.session_state: st.session_state.q_keywords = ["访谈者", "主持人", "Q"]
if 'a_keywords' not in st.session_state: st.session_state.a_keywords = ["受访者", "A"]

# =======================================================================
# 步骤 1: 导入设置 (针对非Excel数据)
# =======================================================================
with st.container(border=True):
    st.subheader("🛠️ 步骤 1: 解析规则配置 (仅针对 Word/Txt)")
    
    col1, col2 = st.columns(2)
    def tag_manager(label, key_prefix, s_list):
        c_in, c_btn = col1.columns([3,1]) if key_prefix=='q' else col2.columns([3,1])
        with c_in: new = st.text_input(label, key=f"{key_prefix}_in", label_visibility="collapsed", placeholder=f"输入{label}...")
        with c_btn: 
            if st.button("➕", key=f"{key_prefix}_add"): 
                if new and new not in s_list: s_list.append(new); st.rerun()
        if s_list: st.caption(" | ".join(s_list))

    with col1:
        st.info("🎤 访谈者 (Interviewer)")
        tag_manager("关键词", "q", st.session_state.q_keywords)
    with col2:
        st.success("👤 受访者 (Interviewee)")
        tag_manager("关键词", "a", st.session_state.a_keywords)
    
    force_mode = st.checkbox("🔘 开启【非访谈者即受访者】模式 (推荐)", value=True)

# =======================================================================
# 步骤 2: 文件上传
# =======================================================================
with st.container(border=True):
    st.subheader("📂 步骤 2: 数据上传")
    st.info("支持 .xlsx (自动识别ID列), .docx, .txt (自动角色切分)")
    
    uploaded_files = st.file_uploader("拖拽文件到此处", type=["xlsx", "xls", "txt", "docx"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("🚀 开始解析 (Parse All)", type="primary"):
            all_data = []
            progress = st.progress(0, text="正在解析...")
            
            for i, f in enumerate(uploaded_files):
                # A. Excel 处理分支
                if f.name.endswith(('.xlsx', '.xls')):
                    units = parse_excel_file(f)
                    all_data.extend(units)
                
                # B. Word/Txt 处理分支
                else:
                    lines = read_txt_docx(f)
                    units = parse_lines_optimized(lines, st.session_state.q_keywords, st.session_state.a_keywords, force_mode)
                    f_idx = f"{i+1:02d}"
                    for r_i, u in enumerate(units):
                        gid = f"{u['role']}-{f_idx}-{r_i+1:03d}"
                        all_data.append({
                            "global_id": gid, "role_code": u['role'], "content": u['content'],
                            "source_file": f.name, "file_index": i
                        })
                
                progress.progress((i+1)/len(uploaded_files))
            
            if all_data:
                st.session_state.atomic_df = pd.DataFrame(all_data)
                # 清除旧的滑块状态
                keys_to_del = [k for k in st.session_state.keys() if k.startswith(("slider_", "num_s_", "num_e_"))]
                for k in keys_to_del: del st.session_state[k]
                st.session_state.processed_df = None
                
                st.success(f"解析完成！共获取 {len(all_data)} 条数据。")
                st.rerun()
            else:
                st.error("未能解析出有效数据。如果是Excel，请检查是否包含'被试编号'表头。")

# =======================================================================
# 步骤 3: 裁剪与组块
# =======================================================================
if st.session_state.atomic_df is not None:
    st.divider()
    st.subheader("✂️ 步骤 3: 数据裁剪与组块")
    
    df = st.session_state.atomic_df
    
    # 1. 物理裁剪 (仅针对 Word/Txt 来源的文件，Excel 通常不需要)
    files = df['source_file'].unique()
    non_excel_files = [f for f in files if not f.endswith(('.xlsx', '.xls'))]
    
    if non_excel_files:
        with st.expander("✂️ 对话文件首尾裁剪 (Word/Txt)", expanded=True):
            trimmed_dfs = []
            # 先分离 Excel 数据直接保留
            excel_df = df[df['source_file'].str.endswith(('.xlsx', '.xls'))]
            if not excel_df.empty: trimmed_dfs.append(excel_df)
            
            for f_name in non_excel_files:
                sub_df = df[df['source_file'] == f_name].reset_index(drop=True)
                total = len(sub_df)
                if total > 0:
                    k_s = f"num_s_{f_name}"; k_e = f"num_e_{f_name}"; k_sl = f"slider_{f_name}"
                    if k_s not in st.session_state: st.session_state[k_s]=0; st.session_state[k_e]=total; st.session_state[k_sl]=(0,total)
                    
                    def update_slide(): st.session_state[k_s], st.session_state[k_e] = st.session_state[k_sl]
                    def update_num(): 
                        s,e = st.session_state[k_s], st.session_state[k_e]
                        if s>e: s=e
                        st.session_state[k_sl] = (s,e)
                    
                    st.markdown(f"**{f_name}**")
                    c1,c2,c3 = st.columns([1,4,1])
                    with c1: st.number_input("始",0,total,key=k_s,on_change=update_num,label_visibility="collapsed")
                    with c2: st.slider("",0,total,key=k_sl,on_change=update_slide,label_visibility="collapsed")
                    with c3: st.number_input("终",0,total,key=k_e,on_change=update_num,label_visibility="collapsed")
                    
                    s,e = st.session_state[k_sl]
                    trimmed_dfs.append(sub_df.iloc[s:e])
            
            df_trimmed = pd.concat(trimmed_dfs).reset_index(drop=True)
    else:
        df_trimmed = df # 全是 Excel，无需裁剪

    # 2. 组块设置
    st.markdown("#### 📦 智能组块 (Batching)")
    col_b1, col_b2 = st.columns([1, 3])
    with col_b1:
        batch_size = st.number_input("目标字数/组块", value=800, step=100)
    
    if st.button("⚡ 生成最终数据并保存", type="primary"):
        # 执行组块
        batch_ids = []
        curr_id = 1; curr_len = 0
        
        for idx, row in df_trimmed.iterrows():
            l = len(str(row['content']))
            if curr_len > batch_size:
                curr_id += 1; curr_len = 0
            batch_ids.append(curr_id)
            curr_len += l
            
        df_trimmed['batch_id'] = batch_ids
        st.session_state.processed_df = df_trimmed
        
        # 自动保存
        saved_path = auto_save_data(df_trimmed)
        if saved_path:
            st.success(f"✅ 数据已自动备份至: `{saved_path}`")
        st.rerun()

# =======================================================================
# 步骤 4: 结果与导出
# =======================================================================
if st.session_state.processed_df is not None:
    st.divider()
    st.subheader("✅ 步骤 4: 准备就绪")
    
    df_final = st.session_state.processed_df
    
    # 颜色区分
    def color_row(row):
        color = '#e9ecef' if row['role_code'] == 'Q' else '#d4edda'
        return [f'background-color: {color}; color: black; border-bottom: 1px solid white;'] * len(row)

    st.dataframe(
        df_final[['batch_id', 'global_id', 'role_code', 'content', 'source_file']].style.apply(color_row, axis=1),
        use_container_width=True, height=400
    )
    
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        if st.button("🗑️ 重新处理"): st.session_state.processed_df = None; st.rerun()
    with c2:
        # 手动保存按钮 (其实已经自动保存了，但提供一个显性下载)
        csv = df_final.to_csv(index=False).encode('utf-8-sig')
        st.download_button("💾 下载处理后的数据 (.csv)", csv, "processed_data.csv", "text/csv")
    with c3:
        st.session_state.final_coding_data = df_final
        st.button("➡️ 前往编码 (Go to Coding)", type="primary", on_click=lambda: st.switch_page("pages/2_Open_Coding.py"))
