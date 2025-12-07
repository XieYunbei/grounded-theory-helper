import streamlit as st
import pandas as pd
import time 
from openai import OpenAI 
import json
import os 
import glob
from io import BytesIO 
import datetime

# =======================================================================
# 0. 数据持久化与恢复模块 (Data Persistence)
# =======================================================================

RECOVERY_DIR = "recovery_opening_coding"

def ensure_recovery_dir():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR)

def get_current_filename(theme):
    """
    生成文件名：Opening_主题_日期.jsonl
    """
    # 清洗文件名中的非法字符
    safe_theme = "".join([c for c in theme if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_theme: safe_theme = "Untitled_Project"
    
    date_str = datetime.datetime.now().strftime("%Y%m%d") 
    return f"Opening_{safe_theme}_{date_str}.jsonl"

def save_record_to_jsonl(record_dict, filename):
    """
    追加写入单条处理记录 (包含该行生成的所有编码)
    """
    ensure_recovery_dir()
    filepath = os.path.join(RECOVERY_DIR, filename)
    
    # 补充时间戳
    record_dict['timestamp'] = datetime.datetime.now().isoformat()
    
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"自动保存失败: {e}")

def load_from_jsonl(filepath):
    """
    从 JSONL 读取数据，并将其扁平化为 open_codes 需要的格式
    """
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip():
                        records.append(json.loads(line))
                except:
                    continue
    
    # 将记录转换为 DataFrame 所需的扁平列表
    flat_codes = []
    processed_indices = set()
    file_sources = set() # 用于校验
    
    for r in records:
        idx = r.get('original_row_index')
        if idx is not None:
            processed_indices.add(idx)
            
        # 提取该行对应的编码列表
        codes_list = r.get('generated_codes', [])
        source_file = r.get('source_file', 'unknown')
        file_sources.add(source_file)
        
        if isinstance(codes_list, list):
            for c in codes_list:
                if isinstance(c, dict):
                    flat_codes.append({
                        'source_file': source_file,
                        'code': c.get('code'),
                        'quote': c.get('quote'),
                        'confidence': c.get('confidence', 0),
                        'original_row_index': idx
                    })
    
    return pd.DataFrame(flat_codes), processed_indices, file_sources

# =======================================================================
# 1. 核心逻辑函数区
# =======================================================================

def call_qwen_api(api_key, model_id, prompt_text, temperature=0.1):
    try:
        if model_id in ["qwen-max", "qwen-plus", "deepseek-v3", "deepseek-r1", "kimi-k2-thinking", "glm-4.6"]:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            client_key = api_key 
        elif model_id.startswith("gpt"):
            base_url = "https://api.openai.com/v1"
            client_key = st.session_state.get('openai_key', api_key) 
        elif model_id.startswith("gemini"):
            base_url = "https://api.gemini.com/v1" 
            client_key = st.session_state.get('gemini_key', api_key) 
        else:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            client_key = api_key

        client = OpenAI(api_key=client_key, base_url=base_url)
        
        response = client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "你是一位严谨的扎根理论研究专家，请严格遵守用户指令。"},
                {"role": "user", "content": prompt_text}
            ],
        )

        usage = response.usage
        total_tokens = getattr(usage, "total_tokens", 0)
        
        return {"success": True, "text": response.choices[0].message.content, "tokens": total_tokens}
    except Exception as e:
        return {"success": False, "error": f"API Exception: {str(e)}", "tokens": 0}

# (V51) Meta-Prompt
def create_background_meta_prompt(core_theme):
    return f"""
你是一位专精于扎根理论方法论的顶尖专家。用户正在研究核心主题：“{core_theme}”。

你的任务是：为后续的编码工作制定一套**操作化判别标准**。

请严格、且仅输出以下 JSON 格式：

{{
  "definition_logic": "纳入标准：请用200字左右定义，什么样的文本才算属于这个主题？",
  "exclusion_logic": "排除标准：请用200字左右定义，什么样即使沾边但也必须排除的内容？（必须包含具体的边界情况或混淆概念）"
}}
"""

# (V53) Final Coding Prompt
def create_final_coding_prompt(core_theme, definition_logic, exclusion_logic, text_to_code):
    return f"""
你是严谨的扎根理论专家。任务是对[待处理文段]进行开放性编码。

1. 核心焦点
{core_theme}

2. 判别标准 (必须严格执行)
* 纳入标准:{definition_logic}
* 排除标准:{exclusion_logic}

3. 编码铁律
铁律一 语义纯化：Code必须是语义完整且最简短的词组。删除原文中不包含核心意义的语言赘述（如口头禅、连接词、冗余的主语）。
铁律二 细致拆分：一段话包含多个独立的动作或意义，必须拆分成多条。严禁合并。
铁律三 贴地性原则：Code 必须是低级、具象的描述性短语，拒绝抽象概念。
铁律四 精准引用：Quote 必须是原文的精准复制，不能改写。

4. 编码步骤
步骤1 判别：逐句阅读，对照判别标准，识别所有符合纳入标准的文本片段。
步骤2 初次切分：对识别出的片段执行原子化拆分，生成一个初始代码列表。
步骤3 穷尽性审计：
    * 重新核对：将你生成的初始代码列表与[待处理文段]进行对比。
    * 检查遗漏：检查原始文段中是否还有任何符合纳入标准的、但未被编码的并列词、转折句或对立概念（例如：既要A又要B）。
    * 补充：如果发现遗漏，请立即补充完整。
步骤4 清洗：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行铁律一（语义纯化）和铁律三（贴地性原则），生成最终 Code。
步骤5 格式化：生成JSON。
步骤6 进行置信度confidence评分：进行五点评分，1分为非常不确定，2分为比较确定，3分为有点确定，4分为比较确定，5分为非常确定。

5. 输出格式
只输出一个JSON数组，每个对象必须包含 code 、quote和confidence。
多条编码示例:
[
  {{
    "code": "(第一个编码标签)",
    "quote": "(支撑编码1的原文片段)",
    "confidence": 5
  }},
  {{
    "code": "(第二个编码标签)",
    "quote": "(支撑编码2的原文片段)",
    "confidence": 4
  }}
]
零条编码示例: []

[待处理文段]:
{text_to_code}

提醒：严格遵守判别标准与编码步骤，按照规定JSON格式输出！不输出其他内容！
"""

def extract_json(text, start_char='[', end_char=']'):
    try:
        if start_char == '[': start_index = text.find('[')
        else: start_index = text.find('{')
        if end_char == ']': end_index = text.rfind(']')
        else: end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = text[start_index : end_index + 1]
            return json.loads(json_str)
        else: return None
    except Exception as e:
        return f"JSON解析错误: {e}. 原始文本: {text}"

@st.cache_data 
def to_excel(df_raw, df_codes, df_meta):
    output = BytesIO()
    if df_raw is None: df_raw = pd.DataFrame()
    if df_codes is None: df_codes = pd.DataFrame()
    if df_meta is None: df_meta = pd.DataFrame()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_raw.to_excel(writer, index=False, sheet_name='raw_data')
        df_codes.to_excel(writer, index=False, sheet_name='open_codes')
        df_meta.to_excel(writer, index=False, sheet_name='project_meta')
    processed_data = output.getvalue()
    return processed_data

def get_manual_prompt_template():
    return f"""
你是严谨的扎根理论专家。任务是对[待处理文段]进行开放性编码。

1. 核心焦点
[请在此处输入核心焦点研究主题]

2. 判别标准 (必须严格执行)
* 纳入标准:[请在此处粘贴纳入标准]
* 排除标准:[请在此处粘贴排除排除标准]

3. 编码铁律
铁律一 语义纯化：Code必须是语义完整且最简短的词组。仅删除原文中不包含核心意义的语言赘述（如口头禅、连接词、冗余的主语）。当“意图”是主题的核心时，保留意图词。
铁律二 细致拆分：一段话包含多个独立的动作或意义，必须拆分成多条。严禁合并。
铁律三 贴地性原则：Code 必须是低级、具象的描述性短语，拒绝抽象概念。
铁律四 精准引用：Quote 必须是原文的精准复制，不能改写。

4. 编码步骤
步骤1 判别：逐句阅读，对照动态判别标准，识别所有符合纳入标准的文本片段。
步骤2 初次切分：对识别出的片段执行原子化拆分，生成一个初始代码列表。
步骤3 穷尽性审计：检查遗漏的并列词或转折句，并补充。
步骤4 清洗：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行语义纯化和贴地性原则，生成最终 Code。
步骤5 格式化：生成JSON。
步骤6 进行置信度confidence评分：进行五点评分。

5. 输出格式
只输出一个JSON数组，每个对象必须包含 code 、quote和confidence。
[待处理文段]:
{{text_to_code}}

提醒：严格遵守判别标准与编码步骤，按照规定JSON格式输出！不输出其他内容！
"""


# =======================================================================
# 2. Streamlit 页面布局
# =======================================================================

# 初始化 Session State
if 'prompt_mode' not in st.session_state: st.session_state.prompt_mode = "1. 智能向导 (全自动)" 
if 'custom_prompt' not in st.session_state: st.session_state.custom_prompt = get_manual_prompt_template()
if 'definition_logic' not in st.session_state: st.session_state.definition_logic = ""
if 'exclusion_logic' not in st.session_state: st.session_state.exclusion_logic = ""
if 'open_codes' not in st.session_state: st.session_state.open_codes = pd.DataFrame(columns=['source_file', 'code', 'quote', 'confidence', 'original_row_index'])
if 'core_theme' not in st.session_state: st.session_state.core_theme = "（请在此输入您的研究主题）" 
if 'selected_model' not in st.session_state: st.session_state.selected_model = "qwen-plus"
if 'openai_key' not in st.session_state: st.session_state.openai_key = "" 
if 'gemini_key' not in st.session_state: st.session_state.gemini_key = "" 
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'temperature' not in st.session_state: st.session_state.temperature = 0.1
if 'total_token_usage' not in st.session_state: st.session_state.total_token_usage = 0

st.set_page_config(page_title="区域2: 开放性编码", layout="wide")

# 获取当前原始数据 (用于校验)
df = st.session_state.raw_data if 'raw_data' in st.session_state and st.session_state.raw_data is not None else None

# [NEW] 侧边栏：历史存档恢复 (与轴心编码一致的逻辑)
with st.sidebar:
    st.header("📂 进度管理")
    st.warning("⚠️ 注意：为了保证断点续传的准确性，请勿在研究过程中随意修改上传文件的文件名或行顺序。")
    st.info("系统会自动将编码结果保存到 `recovery_opening_coding` 文件夹。")
    
    ensure_recovery_dir()
    # 扫描 jsonl 文件，按时间倒序
    jsonl_files = glob.glob(os.path.join(RECOVERY_DIR, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)
    
    if jsonl_files:
        st.subheader("📥 恢复进度")
        selected_file = st.selectbox("选择历史文件", [os.path.basename(f) for f in jsonl_files], index=0)
        
        if st.button("🔄 载入选中文件"):
            filepath = os.path.join(RECOVERY_DIR, selected_file)
            loaded_df, processed_indices, file_sources = load_from_jsonl(filepath)
            
            if not loaded_df.empty:
                # [NEW] 数据源校验
                if df is not None and 'source_file' in df.columns:
                    current_files = set(df['source_file'].unique())
                    if not file_sources.issubset(current_files):
                        st.warning(f"⚠️ 警告：存档中的源文件 ({file_sources}) 与当前上传的文件 ({current_files}) 不完全匹配。这可能导致行索引错位。")
                
                # 1. 载入到 session_state (全量覆盖)
                st.session_state.open_codes = loaded_df
                
                # [NEW] 进度显示
                total_rows = len(df) if df is not None else "Unknown"
                processed_count = len(processed_indices)
                st.success(f"✅ 成功恢复 {len(loaded_df)} 条编码记录！")
                st.info(f"📊 进度状态: 已处理 {processed_count} / {total_rows} 行")
                
                time.sleep(1)
                st.rerun()
            else:
                st.warning("该文件为空或格式不包含有效数据")
    else:
        st.caption("暂无历史存档")
        
    st.divider()
    # [NEW] 清空/重置按钮
    if st.button("🗑️ 清空当前进度 (重新开始)", type="secondary", help="这将清空所有已生成的编码结果，允许你从头开始运行。"):
        st.session_state.open_codes = pd.DataFrame(columns=['source_file', 'code', 'quote', 'confidence', 'original_row_index'])
        st.success("已清空进度。")
        time.sleep(1)
        st.rerun()

st.title("区域2: 开放性编码 Prompt生成与执行区 🛠️")

# =======================================================================
# 3. 配置区域
# =======================================================================
with st.container(border=True):
    st.subheader("步骤 1: 配置模式与规则")
    
    col_key, col_model = st.columns(2)
    with col_key:
        st.markdown("###### 🔑 密钥配置")
        api_key_input = st.text_input("DashScope Key (Qwen/DeepSeek/GLM)", type="password", value=st.session_state.get('api_key', ''), label_visibility="collapsed", help="用于 Qwen, DeepSeek, GLM")
        if api_key_input: st.session_state.api_key = api_key_input
        st.session_state.openai_key = st.text_input("OpenAI Key (GPT-4o)", type="password", value=st.session_state.get('openai_key', ''), help="用于 GPT-4o")
        st.session_state.gemini_key = st.text_input("Gemini Key (Gemini)", type="password", value=st.session_state.get('gemini_key', ''), help="用于 Gemini")
        st.markdown("""<small>[获取DashScope Key](https://bailian.console.aliyun.com/?tab=model#/api-key)</small>""", unsafe_allow_html=True)
        st.markdown("""<small>[领取学生300元优惠券](https://university.aliyun.com/?userCode=r3yteowb)</small>""", unsafe_allow_html=True)
    
    with col_model:
        st.markdown("###### 🧠 模型选择")
        model_options = {
            "👑 Qwen-Max (阿里旗舰)": "qwen-max",
            "🌟 GPT-4o (全球旗舰)": "gpt-4o",
            "🚀 Gemini 2.5 Pro (Google 旗舰)": "gemini-2.5-pro",
            "💎 GLM-4.6 (智谱AI旗舰)": "glm-4.6",
            "🔥 DeepSeek-V3 (逻辑强)": "deepseek-v3",
            "⚖️ Qwen-Plus (平衡推荐)": "qwen-plus",
        }
        model_ids = list(model_options.values())
        try: default_index = model_ids.index(st.session_state.selected_model)
        except ValueError: default_index = 0 
        selected_model_name = st.selectbox("选择模型", options=model_options.keys(), index=default_index, label_visibility="collapsed")
        st.session_state.selected_model = model_options[selected_model_name]

    st.divider()
    mode_options = ["1. 智能向导 (全自动)", "2. 外部辅助 (傻瓜版)", "3. 高级自定义 (完全手动)"]
    st.session_state.prompt_mode = st.radio("选择工作模式", mode_options, horizontal=True)

    st.markdown("#### 1. 核心研究主题")
    core_theme_input = st.text_input("研究主题", value=st.session_state.core_theme, label_visibility="collapsed")
    st.session_state.core_theme = core_theme_input

    # --- 模式 A: 智能向导 ---
    if st.session_state.prompt_mode == "1. 智能向导 (全自动)":
        if st.button("🤖 一键生成判别标准", type="primary"):
            if not st.session_state.api_key: st.error("请输入 DashScope Key！"); st.stop()
            elif not core_theme_input or "请在" in core_theme_input: st.error("请输入有效主题！"); st.stop()
            
            with st.spinner("正在分析..."):
                meta_prompt = create_background_meta_prompt(st.session_state.core_theme)
                api_res = call_qwen_api(st.session_state.api_key, st.session_state.selected_model, meta_prompt, temperature=0.3)
                
                if api_res["success"]:
                    st.session_state.total_token_usage += api_res["tokens"]
                    data = extract_json(api_res["text"], start_char='{', end_char='}')
                    if isinstance(data, dict):
                        st.session_state.definition_logic = data.get('definition_logic', '')
                        st.session_state.exclusion_logic = data.get('exclusion_logic', '')
                        st.success("标准生成成功！请在下方确认。")
                    else: st.error(f"生成失败: {data}")
                else: st.error(api_res["error"])

    # --- 模式 B: 外部辅助 (傻瓜版) ---
    elif st.session_state.prompt_mode == "2. 外部辅助 (傻瓜版)":
        st.info("📋 **傻瓜模式：** 利用网页版 AI 强大的推理能力生成标准，然后将结果粘贴回来。")
        help_prompt = f"""我正在做关于【{st.session_state.core_theme}】的扎根理论编码。
请为我制定两个标准：1. 纳入标准 (Definition Logic)：请用一句话定义，什么样的文本才算属于这个主题？ 2. 排除标准 (Exclusion Logic)：请用一句话定义，什么样即使沾边但也必须排除的内容？
请严格按照 “1. 纳入标准：...” 和 “2. 排除标准：...” 的格式直接给出这两段话，不要其他废话。"""
        
        with st.expander("📋 点击展开：复制求助指令", expanded=True):
            st.code(help_prompt, language="text")

    # --- 模式 3: 高级自定义 (完全手动) ---
    else:
        st.warning("🛠️ **专家模式：** 您完全控制 Prompt。")
        
        uploaded_prompt_file = st.file_uploader("📥 上传您的 Prompt (.txt) 文件", type=["txt"])
        if uploaded_prompt_file:
            string_data = uploaded_prompt_file.getvalue().decode("utf-8")
            st.session_state.custom_prompt = string_data
            st.success("Prompt 文件读取成功！")
        
        st.session_state.custom_prompt = st.text_area("完整 Prompt 编辑器 (包含 {text_to_code})", value=st.session_state.custom_prompt, height=400)
    
    # --- 公共区域：显示/编辑标准 ---
    st.divider()

    if st.session_state.prompt_mode in ["1. 智能向导 (全自动)", "2. 外部辅助 (傻瓜版)"]:
        col_def, col_exc = st.columns(2)
        with col_def:
            st.session_state.definition_logic = st.text_area("✅ 纳入标准 (Definition)", value=st.session_state.definition_logic, height=100)
        with col_exc:
            st.session_state.exclusion_logic = st.text_area("❌ 排除标准 (Exclusion)", value=st.session_state.exclusion_logic, height=100)
            
    # --- Prompt Saving Feature ---
    prompt_to_save = ""
    if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
        prompt_to_save = st.session_state.custom_prompt
        save_label = "💾 下载自定义 Prompt (.txt)"
        filename_prefix = "CustomPrompt"
    elif st.session_state.definition_logic and st.session_state.exclusion_logic:
        prompt_to_save = create_final_coding_prompt(
            st.session_state.core_theme, 
            st.session_state.definition_logic, 
            st.session_state.exclusion_logic, 
            "{text_to_code}" 
        )
        save_label = "💾 下载最终编码 Prompt (.txt)"
        filename_prefix = "FinalCodingPrompt"
    
    if prompt_to_save:
        timestamp = time.strftime("%Y%m%d%H%M")
        filename = f"{filename_prefix}_{st.session_state.core_theme}_{timestamp}.txt"
        st.download_button(
            label=save_label,
            data=prompt_to_save,
            file_name=filename,
            mime="text/plain",
            key="download_final_prompt",
            type="secondary"
        )


# =======================================================================
# 4. 执行区域
# =======================================================================
can_run = False
if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
    can_run = "{text_to_code}" in st.session_state.custom_prompt
elif st.session_state.definition_logic and st.session_state.exclusion_logic:
    can_run = True

if can_run:
    if df is None:
        st.warning("⚠️ 请先在“1_Data_Upload”页面上传数据。")
        st.stop()
        
    with st.container(border=True):
        st.subheader("步骤 2: 执行开放性编码")
        st.dataframe(df, height=150)
        
        temperature_input = st.slider("温度 (Temperature) - 推荐 0.1", 0.0, 1.0, value=st.session_state.temperature, step=0.05)
        st.session_state.temperature = temperature_input
        
        with st.expander(f"点击查看 Prompt 预览 (将发送给 {st.session_state.selected_model})"):
            if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
                st.code(st.session_state.custom_prompt, language="markdown")
            else:
                preview_prompt = create_final_coding_prompt(st.session_state.core_theme, st.session_state.definition_logic, st.session_state.exclusion_logic, "[待处理文本]")
                st.code(preview_prompt, language="markdown")

        col1, col2 = st.columns(2)
        with col1:
            num_to_test = st.number_input("测试条数", 1, 50, 3)
            if st.button("▶️ 测试运行"):
                if not st.session_state.api_key and not st.session_state.openai_key and not st.session_state.gemini_key: st.error("请至少输入一个 API 密钥！"); st.stop()
                with st.spinner("测试中..."):
                    test_results = []
                    # [FIX] 测试运行时也要考虑是否已处理？通常测试运行不需要持久化保存
                    for i, row in df.head(num_to_test).iterrows():
                        if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
                            prompt = st.session_state.custom_prompt.format(text_to_code=row['text_content'])
                        else:
                            prompt = create_final_coding_prompt(st.session_state.core_theme, st.session_state.definition_logic, st.session_state.exclusion_logic, row['text_content'])
                        
                        res = call_qwen_api(st.session_state.api_key, st.session_state.selected_model, prompt, st.session_state.temperature)
                        if res["success"]:
                            st.session_state.total_token_usage += res["tokens"]
                            codes = extract_json(res["text"], start_char='[', end_char=']')
                            
                            clean_codes = []
                            if isinstance(codes, list):
                                for c in codes:
                                    if isinstance(c, dict) and 'code' in c: 
                                        if 'quote' not in c: c['quote'] = "（AI未返回Quote）"
                                        if 'confidence' not in c: c['confidence'] = 0
                                        clean_codes.append(c)
                                    elif isinstance(c, str):
                                        clean_codes.append({"code": c, "quote": "（AI未返回Quote）", "confidence": 0})
                            test_results.extend(clean_codes)
                        else: st.error(res["error"])
                    st.dataframe(test_results)

        with col2:
            st.markdown(f"**累计Token:** `{st.session_state.total_token_usage}`")
            
            is_running = st.session_state.get('is_processing', False)
            if is_running:
                if st.button("⏹️ 停止处理", type="primary"): 
                    st.session_state.stop_requested = True; st.rerun()
            else:
                if st.button("🚀 批量处理 (智能跳过)", type="primary"): 
                    st.session_state.is_processing = True; st.session_state.stop_requested = False; st.rerun()

            if st.session_state.get('is_processing', False):
                progress_bar = st.progress(0, text="准备中...")
                log_container = st.empty()
                log_messages = []
                
                # [CRITICAL FIX] 智能跳过逻辑：基于 original_row_index
                # 确保 'original_row_index' 存在且为整数类型，避免类型不匹配导致 isin 失效
                if 'original_row_index' in st.session_state.open_codes.columns:
                    # 将列转为数值型，处理可能的 None/NaN
                    processed_series = pd.to_numeric(st.session_state.open_codes['original_row_index'], errors='coerce').dropna()
                    processed = processed_series.unique()
                else: processed = []
                
                to_process = df[~df.index.isin(processed)]
                total = len(to_process)
                
                if total == 0:
                    st.success("🎉 所有数据已处理完毕（包含历史恢复的数据）。")
                    st.session_state.is_processing = False
                    st.rerun()

                count = 0
                for i, row in to_process.iterrows():
                    if st.session_state.stop_requested: st.error("已停止"); st.session_state.is_processing = False; st.rerun(); break
                    
                    if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
                        prompt = st.session_state.custom_prompt.format(text_to_code=row['text_content'])
                    else:
                        prompt = create_final_coding_prompt(st.session_state.core_theme, st.session_state.definition_logic, st.session_state.exclusion_logic, row['text_content'])
                    
                    res = call_qwen_api(st.session_state.api_key, st.session_state.selected_model, prompt, st.session_state.temperature)
                    
                    log_msg = ""
                    if res["success"]:
                        st.session_state.total_token_usage += res["tokens"]
                        codes = extract_json(res["text"], start_char='[', end_char=']')
                        
                        clean_codes = []
                        if isinstance(codes, list):
                            for c in codes:
                                if isinstance(c, dict) and 'code' in c:
                                    if 'quote' not in c: c['quote'] = "（AI未返回Quote）"
                                    if 'confidence' not in c: c['confidence'] = 0
                                    clean_codes.append(c)
                                elif isinstance(c, str):
                                    clean_codes.append({"code": c, "quote": "（AI未返回Quote）", "confidence": 0})

                        if clean_codes:
                            code_str = ", ".join([f"[{c['code']} ({c['confidence']}/5)]" for c in clean_codes])
                            log_msg = f"✅ 行{i} | 🪙{res['tokens']} | 🏷️ {code_str}"
                            
                            # 1. 更新 Session State DataFrame
                            new_df = pd.DataFrame(clean_codes)
                            new_df['source_file'] = row.get('source_file', 'unknown')
                            new_df['original_row_index'] = i
                            st.session_state.open_codes = pd.concat([st.session_state.open_codes, new_df], ignore_index=True)
                            
                            # 2. [NEW] 立即持久化保存到 JSONL (Recovery)
                            record_to_save = {
                                "original_row_index": i,
                                "source_file": row.get('source_file', 'unknown'),
                                "text_content": row['text_content'], # 原始文本也存一下，方便核对
                                "generated_codes": clean_codes,
                                "model": st.session_state.selected_model
                            }
                            filename = get_current_filename(st.session_state.core_theme)
                            save_record_to_jsonl(record_to_save, filename)
                            
                        else: 
                            log_msg = f"⚪ 行{i} | 🪙{res['tokens']} | 无相关内容"
                    else: 
                        log_msg = f"❌ API错误: {res['error']}"

                    if log_msg: log_messages.append(log_msg)
                    log_container.text_area("日志", value="\n".join(reversed(log_messages)), height=250)
                    
                    count += 1
                    progress_bar.progress(count / total, text=f"进度: {count}/{total} (正在处理第 {i} 行)")
                
                if not st.session_state.stop_requested:
                    st.success("完成！"); st.session_state.is_processing = False; st.rerun()

# =======================================================================
# 5. 结果预览
# =======================================================================
if not st.session_state.open_codes.empty:
    with st.container(border=True):
        st.subheader("步骤 3: 结果预览与保存")
        
        cols = ['source_file', 'code', 'quote', 'confidence', 'original_row_index']
        for c in cols: 
            if c not in st.session_state.open_codes.columns: st.session_state.open_codes[c] = None
            
        edited = st.data_editor(
            st.session_state.open_codes, 
            column_order=['source_file', 'code', 'quote', 'confidence'],
            disabled=['source_file'],
            num_rows="dynamic", key="editor", height=400
        )
        st.session_state.open_codes = edited
        
        st.markdown("#### 保存项目")
        meta_bg = "Custom" if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)" else f"纳入：{st.session_state.definition_logic}\n排除：{st.session_state.exclusion_logic}"
            
        excel_data = to_excel(
            df, 
            edited, 
            pd.DataFrame({"core_theme":[st.session_state.core_theme], "bg":[meta_bg]})
        )
        st.download_button("🚀 下载项目 (.xlsx)", data=excel_data, file_name=f"Project_{st.session_state.core_theme}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.page_link("pages/3_Axial_Coding.py", label="下一步 (轴心编码)", icon="➡️")

