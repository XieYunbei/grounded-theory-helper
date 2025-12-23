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

def save_batch_record(record_dict, filename):
    """
    保存单个 Batch 的处理结果
    """
    ensure_recovery_dir()
    filepath = os.path.join(RECOVERY_DIR, filename)
    record_dict['timestamp'] = datetime.datetime.now().isoformat()
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"自动保存失败: {e}")

def load_from_jsonl(filepath):
    """
    适配 Batch 结构的恢复逻辑
    """
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    if line.strip(): records.append(json.loads(line))
                except: continue
    
    flat_codes = []
    processed_batches = set()
    
    for r in records:
        b_id = r.get('batch_id')
        if b_id is not None:
            processed_batches.add(b_id)
            
        codes_list = r.get('final_codes', []) 
        for c in codes_list:
            flat_codes.append(c)
    
    return pd.DataFrame(flat_codes), processed_batches

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
        error_str = str(e)
        # [FIX] 增加友好的错误提示
        if "401" in error_str or "Incorrect API key" in error_str:
            return {"success": False, "error": "⚠️ API Key 无效 (401)：请检查密钥是否复制完整、是否有多余空格，或账户是否欠费。", "tokens": 0}
        else:
            return {"success": False, "error": f"API Exception: {error_str}", "tokens": 0}

# [FIXED] 修复了参数定义，现在可以接收 start_char 了
def extract_json(text, start_char='[', end_char=']'):
    try:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        return [] if start_char == '[' else {}
    except:
        return [] if start_char == '[' else {}

def reconstruct_quote_and_validate(ai_item, atomic_lookup):
    """ 
    守门员逻辑: 
    1. 接收 AI 返回的 IDs
    2. 校验 ID 是否存在且非 Q 开头
    3. 拼接原文
    """
    raw_ids = ai_item.get('ids', [])
    if isinstance(raw_ids, str): raw_ids = [raw_ids]
    
    valid_ids = []
    quote_parts = []
    source_files = set()
    
    for uid in raw_ids:
        # 校验1: ID是否存在
        if uid not in atomic_lookup.index: continue
        
        try:
            row = atomic_lookup.loc[uid]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
        except: continue

        # 校验2: 身份协议 (Q不编)
        if str(uid).startswith("Q-") or row['role_code'] == 'Q': continue
            
        valid_ids.append(uid)
        quote_parts.append(str(row['content']))
        source_files.add(row['source_file'])
        
    if not valid_ids: return None
        
    return {
        "code": ai_item.get('code', 'Unnamed Code'),
        "quote": "".join(quote_parts), 
        "original_ids": valid_ids,
        "source_file": list(source_files)[0] if source_files else "Unknown",
        "confidence": ai_item.get('confidence', 3)
    }

# Meta-Prompt
def create_background_meta_prompt(core_theme):
    return f"""
你是一位专精于扎根理论方法论的顶尖专家。用户正在研究核心主题：“{core_theme}”。
你的任务是：为后续的编码工作制定一套**操作化判别标准**。
请严格、且仅输出以下 JSON 格式：
{{
  "definition_logic": "纳入标准：请用200字左右定义，什么样的文本才算属于这个主题？",
  "exclusion_logic": "排除标准：请用200字左右定义，什么样即使沾边但也必须排除的内容？"
}}
"""

# Final Coding Prompt
def create_final_coding_prompt(core_theme, definition_logic, exclusion_logic, batch_text):
    return f"""
你是严谨的扎根理论专家。你正在处理经过原子化切分的访谈数据。每行文本都带有唯一ID，代表一个物理上的最小语境行。你的任务是对提供的[待处理文段]进行开放性编码。

一、核心焦点
{core_theme}

二、判别标准
* 纳入标准: {definition_logic}
* 排除标准: {exclusion_logic}

三、身份协议-必须严格执行
* 输入文本每一行都带有 ID，例如 [Q-01-001] 或 [A-01-001]。
* [Q-...] 开头的行：是访谈者/主持人。这些行仅作为理解语境的背景信息。严禁对这些行生成编码！
* [A-...] 开头的行：是受访者。你只能对这些行进行编码。

四、编码原则
原则一：语义纯化：Code必须是语义完整且最简短的词组。删除原文中不包含核心意义的语言赘述（如口头禅、连接词、冗余的主语）。
原则二：语义挖掘：有时一行短句可能包含多个独立的动作、情感或观点。不要合并意义！必须对同一行 ID 生成多条不同的 Code，精准捕捉每一个微小的意义单元。
原则三：语境重组: 务必审视上下文。如果相邻的几行共同构成可编码的独立单元，请将这些 ID 打包，赋予同一个 Code。
原则四：贴地性原则：Code 必须是低级、具象的描述性短语，拒绝抽象概念。

五、编码步骤
1.扫描: 阅读文本，利用 Q 端理解语境，锁定 A 端内容。
3.意义单元界定:
    * 判断当前行是否包含多个独立意义？若有，进行语义挖掘（原则二）
    * 判断当前行是否需要联系上文才能读懂？若需，进行语境重组（原则三）
3.穷尽性审计：
    * 重新核对：将你生成的初始代码列表与[待处理文段]进行对比。
    * 检查遗漏：检查原始文段中是否还有任何符合纳入标准的、但未被编码的并列词、转折句或对立概念（例如：既要A又要B）。
    * 补充：如果发现遗漏，请立即补充完整。
4.提炼与命名：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行原则一（语义纯化）和原则四（贴地性原则），生成最终 Code。
5.零引文：不要返回原文 Quote，仅返回 IDs。
6.进行置信度confidence评分：进行五点评分，1分为非常不确定，2分为比较确定，3分为有点确定，4分为比较确定，5分为非常确定。
7.格式化：生成JSON。

六、输出格式
只输出一个JSON数组，每个对象必须包含 code 、ids和confidence。
多条编码示例:
[
  {{
    "code": "(第一个编码标签)",
    "ids": ["A-01-005", "A-01-006"], 
    "confidence": 5
  }},
  {{
    "code": "(第二个编码标签)",
    "ids": ["A-01-006"], 
    "confidence": 4
  }}
]
零条编码示例: []

[待处理文段]:
{batch_text}

提醒：严格遵守判别标准与编码步骤，按照规定JSON格式输出！不输出其他内容！
"""

def get_manual_prompt_template():
    return """
你是严谨的扎根理论专家。你正在处理经过原子化切分的访谈数据。每行文本都带有唯一ID，代表一个物理上的最小语境行。你的任务是对提供的[待处理文段]进行开放性编码。

1. 核心焦点
[请在此处输入核心焦点研究主题]

2. 判别标准
* 纳入标准: [请粘贴纳入标准]
* 排除标准: [请粘贴排除标准]

三、身份协议-必须严格执行
* 输入文本每一行都带有 ID，例如 [Q-01-001] 或 [A-01-001]。
* [Q-...] 开头的行：是访谈者/主持人。这些行仅作为理解语境的背景信息。严禁对这些行生成编码！
* [A-...] 开头的行：是受访者。你只能对这些行进行编码。

四、编码原则
原则一：语义纯化：Code必须是语义完整且最简短的词组。删除原文中不包含核心意义的语言赘述（如口头禅、连接词、冗余的主语）。
原则二：语义挖掘：有时一行短句可能包含多个独立的动作、情感或观点。不要合并意义！必须对同一行 ID 生成多条不同的 Code，精准捕捉每一个微小的意义单元。
原则三：语境重组: 务必审视上下文。如果相邻的几行共同构成可编码的独立单元，请将这些 ID 打包，赋予同一个 Code。
原则四：贴地性原则：Code 必须是低级、具象的描述性短语，拒绝抽象概念。

五、编码步骤
1.扫描: 阅读文本，利用 Q 端理解语境，锁定 A 端内容。
3.意义单元界定:
    * 判断当前行是否包含多个独立意义？若有，进行语义挖掘（原则二）
    * 判断当前行是否需要联系上文才能读懂？若需，进行语境重组（原则三）
3.穷尽性审计：
    * 重新核对：将你生成的初始代码列表与[待处理文段]进行对比。
    * 检查遗漏：检查原始文段中是否还有任何符合纳入标准的、但未被编码的并列词、转折句或对立概念（例如：既要A又要B）。
    * 补充：如果发现遗漏，请立即补充完整。
4.提炼与命名：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行原则一（语义纯化）和原则四（贴地性原则），生成最终 Code。
5.零引文：不要返回原文 Quote，仅返回 IDs。
6.进行置信度confidence评分：进行五点评分，1分为非常不确定，2分为比较确定，3分为有点确定，4分为比较确定，5分为非常确定。
7.格式化：生成JSON。

六、输出格式
只输出一个JSON数组，每个对象必须包含 code 、ids和confidence。
多条编码示例:
[
  {{
    "code": "(第一个编码标签)",
    "ids": ["A-01-005", "A-01-006"], 
    "confidence": 5
  }},
  {{
    "code": "(第二个编码标签)",
    "ids": ["A-01-006"], 
    "confidence": 4
  }}
]
零条编码示例: []

[待处理文段]:
{batch_text}

提醒：严格遵守判别标准与编码步骤，按照规定JSON格式输出！不输出其他内容！
"""

# [FIX] 移除了 @st.cache_data 以解决 unhashable type: list 错误
def to_excel(df_raw, df_codes, df_meta):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if df_raw is not None: df_raw.to_excel(writer, index=False, sheet_name='raw_data')
        if df_codes is not None: 
            # 兼容处理：将 list 类型的 ids 转为字符串保存
            df_save = df_codes.copy()
            if 'original_ids' in df_save.columns:
                df_save['original_ids'] = df_save['original_ids'].astype(str)
            df_save.to_excel(writer, index=False, sheet_name='open_codes')
        if df_meta is not None: df_meta.to_excel(writer, index=False, sheet_name='project_meta')
    return output.getvalue()

# =======================================================================
# 2. 页面与数据加载逻辑
# =======================================================================

# Session State
if 'prompt_mode' not in st.session_state: st.session_state.prompt_mode = "1. 智能向导 (全自动)" 
if 'custom_prompt' not in st.session_state: st.session_state.custom_prompt = get_manual_prompt_template()
if 'definition_logic' not in st.session_state: st.session_state.definition_logic = ""
if 'exclusion_logic' not in st.session_state: st.session_state.exclusion_logic = ""
if 'open_codes' not in st.session_state: st.session_state.open_codes = pd.DataFrame(columns=['source_file', 'code', 'quote', 'confidence', 'original_ids', 'batch_id'])
if 'core_theme' not in st.session_state: st.session_state.core_theme = "（请在此输入您的研究主题）" 
if 'selected_model' not in st.session_state: st.session_state.selected_model = "qwen-plus"
if 'openai_key' not in st.session_state: st.session_state.openai_key = "" 
if 'gemini_key' not in st.session_state: st.session_state.gemini_key = "" 
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'temperature' not in st.session_state: st.session_state.temperature = 0.1
if 'total_token_usage' not in st.session_state: st.session_state.total_token_usage = 0
if 'processed_batches' not in st.session_state: st.session_state.processed_batches = set()

st.set_page_config(page_title="区域2: 开放性编码", layout="wide")

# --- 数据源获取逻辑 ---
df_atomic = None
atomic_lookup = None

if 'final_coding_data' in st.session_state and st.session_state.final_coding_data is not None:
    df_atomic = st.session_state.final_coding_data
else:
    st.warning("⚠️ 未检测到 Step 1 的处理数据。请上传 Step 1 下载的 Processed_xxxx.xlsx 文件。")
    uploaded_file = st.file_uploader("📂 上传数据表", type=["xlsx", "csv"]) # 增加 csv 支持
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_atomic = pd.read_csv(uploaded_file)
            else:
                df_atomic = pd.read_excel(uploaded_file)
            
            if 'global_id' in df_atomic.columns and 'batch_id' in df_atomic.columns:
                st.session_state.final_coding_data = df_atomic
                st.success("✅ 数据加载成功！")
                st.rerun() # 刷新页面
            else:
                st.error("表格格式错误：缺少 global_id 或 batch_id 列。")
                df_atomic = None
        except Exception as e:
            st.error(f"读取失败: {e}")

if df_atomic is None:
    st.stop() 

# 建立索引以便快速查找 (Gatekeeper 用)
atomic_lookup = df_atomic.set_index('global_id')

# 侧边栏：历史存档恢复
with st.sidebar:
    st.header("📂 进度管理")
    
    # 撤回功能
    if st.session_state.processed_batches:
        # 获取最新的 batch_id
        last_batch = sorted(list(st.session_state.processed_batches))[-1]
        if st.button(f"↩️ 撤回 Batch {last_batch}", type="secondary"):
            st.session_state.open_codes = st.session_state.open_codes[st.session_state.open_codes['batch_id'] != last_batch]
            st.session_state.processed_batches.remove(last_batch)
            st.warning(f"已撤回 Batch {last_batch}。")
            st.rerun()
            
    st.divider()
    
    st.warning("⚠️ 注意：为了保证断点续传的准确性，请勿在研究过程中随意修改上传文件的文件名或行顺序。")
    ensure_recovery_dir()
    jsonl_files = glob.glob(os.path.join(RECOVERY_DIR, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)
    
    if jsonl_files:
        st.subheader("📥 恢复进度")
        selected_file = st.selectbox("选择历史文件", [os.path.basename(f) for f in jsonl_files], index=0)
        
        if st.button("🔄 载入选中文件"):
            filepath = os.path.join(RECOVERY_DIR, selected_file)
            loaded_df, processed_set = load_from_jsonl(filepath)
            
            if not loaded_df.empty:
                st.session_state.open_codes = loaded_df
                st.session_state.processed_batches = processed_set
                st.success(f"✅ 成功恢复 {len(loaded_df)} 条编码记录！")
                time.sleep(1)
                st.rerun()
            else:
                st.warning("该文件为空或格式不包含有效数据")
    else:
        st.caption("暂无历史存档")
        
    st.divider()
    if st.button("🗑️ 清空当前进度 (重新开始)", type="secondary"):
        st.session_state.open_codes = pd.DataFrame(columns=['source_file', 'code', 'quote', 'confidence', 'original_ids', 'batch_id'])
        st.session_state.processed_batches = set()
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
    mode_options = ["1. 智能向导 (全自动)", "2. 外部辅助 (推荐，需用到网页端，适用最新大模型) ", "3. 高级自定义 (完全手动)"]
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

    # --- 模式 B: 外部辅助 ---
    elif st.session_state.prompt_mode == "2. 外部辅助 (推荐，需用到网页端，适用最新大模型) ":
        st.info("📋 **火箭模式：** 利用网页版 AI 强大的推理能力生成标准，然后将结果粘贴回来。")
        help_prompt = f"""我正在做关于【{st.session_state.core_theme}】的扎根理论编码。
请为我制定两个标准：1. 纳入标准：请用一句话定义，什么样的文本才算属于这个主题？ 2. 排除标准：请用一句话定义，什么样即使沾边但也必须排除的内容？
请严格按照 “1. 纳入标准：...” 和 “2. 排除标准：...” 的格式直接给出这两段话，不要其他废话。"""
        with st.expander("📋 点击展开：复制求助指令", expanded=True):
            st.code(help_prompt, language="text")

    # --- 模式 3: 高级自定义 ---
    else:
        st.warning("🛠️ **专家模式：** 您完全控制 Prompt。注意：请使用 `{batch_text}` 作为文本占位符。")
        st.session_state.custom_prompt = st.text_area("完整 Prompt 编辑器", value=st.session_state.custom_prompt, height=400)
    
    # --- 公共区域：显示/编辑标准 ---
    st.divider()
    if st.session_state.prompt_mode in ["1. 智能向导 (全自动)", "2. 外部辅助 (推荐，需用到网页端，适用最新大模型) "]:
        col_def, col_exc = st.columns(2)
        with col_def:
            st.session_state.definition_logic = st.text_area("✅ 纳入标准 (Definition)", value=st.session_state.definition_logic, height=100)
        with col_exc:
            st.session_state.exclusion_logic = st.text_area("❌ 排除标准 (Exclusion)", value=st.session_state.exclusion_logic, height=100)

# =======================================================================
# 4. 执行区域
# =======================================================================
can_run = False
if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
    can_run = "{batch_text}" in st.session_state.custom_prompt 
    if not can_run and "{text_to_code}" in st.session_state.custom_prompt:
        st.error("检测到旧版占位符 `{text_to_code}`，请替换为 `{batch_text}` 以适配新的组块逻辑。")
elif st.session_state.definition_logic and st.session_state.exclusion_logic:
    can_run = True

if can_run:
    if df_atomic is None:
        st.warning("⚠️ 请先加载数据。")
        st.stop()
        
    with st.container(border=True):
        st.subheader("步骤 2: 批量编码执行")
        
        # 准备数据
        unique_batches = sorted(df_atomic['batch_id'].unique())
        pending_batches = [b for b in unique_batches if b not in st.session_state.processed_batches]
        
        st.markdown(f"**任务统计**: 总组块 `{len(unique_batches)}` | 已完成 `{len(st.session_state.processed_batches)}` | 待处理 `{len(pending_batches)}`")
        
        c_p_preview = st.expander("👀 查看当前 Batch Prompt 预览")
        
        col_act1, col_act2, col_act3 = st.columns([1, 1, 3])
        
        if col_act1.button("▶️ 开始/继续", type="primary", disabled=len(pending_batches)==0):
            st.session_state.is_coding = True
            st.rerun()
            
        if col_act2.button("test (测试1条)"):
            st.session_state.is_coding = True
            st.session_state.test_mode = True
            st.rerun()
            
        if st.session_state.get('is_coding', False):
            if st.button("⏹️ 暂停/停止"): 
                st.session_state.is_coding = False
                st.rerun()
            
            progress_bar = st.progress(0, text="初始化...")
            log_container = st.empty()
            log_messages = []
            
            total = len(pending_batches)
            if total == 0:
                st.success("🎉 所有组块已处理完毕。")
                st.session_state.is_processing = False
                st.rerun()

            count = 0
            for i, batch_id in enumerate(pending_batches):
                if st.session_state.stop_requested: st.error("已停止"); st.session_state.is_processing = False; st.rerun(); break
                if not st.session_state.is_coding: break
                
                # 1. 组装 Batch Text
                batch_rows = df_atomic[df_atomic['batch_id'] == batch_id]
                batch_text_lines = []
                for _, r in batch_rows.iterrows():
                    batch_text_lines.append(f"[{r['global_id']}] {r['content']}")
                batch_text_full = "\n".join(batch_text_lines)

                # Prompt 预览更新 (仅第一条)
                if i == 0:
                    if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
                         preview = st.session_state.custom_prompt.format(batch_text=batch_text_full)
                    else:
                         preview = create_final_coding_prompt(st.session_state.core_theme, st.session_state.definition_logic, st.session_state.exclusion_logic, batch_text_full)
                    c_p_preview.code(preview)

                # 2. 构造 Prompt
                if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
                    prompt = st.session_state.custom_prompt.format(batch_text=batch_text_full)
                else:
                    prompt = create_final_coding_prompt(st.session_state.core_theme, st.session_state.definition_logic, st.session_state.exclusion_logic, batch_text_full)
                
                # 3. 调用 API
                res = call_qwen_api(st.session_state.api_key, st.session_state.selected_model, prompt, st.session_state.temperature)
                
                log_msg = ""
                if res["success"]:
                    st.session_state.total_token_usage += res["tokens"]
                    raw_codes = extract_json(res["text"])
                    
                    final_codes_for_batch = []
                    # 守门员校验
                    if isinstance(raw_codes, list):
                        for item in raw_codes:
                            clean_item = reconstruct_quote_and_validate(item, atomic_lookup)
                            if clean_item:
                                clean_item['batch_id'] = batch_id
                                final_codes_for_batch.append(clean_item)

                    if final_codes_for_batch:
                        code_str = ", ".join([f"[{c['code']} ({c['confidence']}/5)]" for c in final_codes_for_batch])
                        log_msg = f"✅ Batch {batch_id} | 🪙{res['tokens']} | 🏷️ {code_str}"
                        
                        # 更新 Session
                        new_df = pd.DataFrame(final_codes_for_batch)
                        st.session_state.open_codes = pd.concat([st.session_state.open_codes, new_df], ignore_index=True)
                        
                        # 持久化
                        record_to_save = {
                            "batch_id": int(batch_id),
                            "source_file": batch_rows.iloc[0]['source_file'],
                            "batch_summary": batch_text_full[:50]+"...", 
                            "final_codes": final_codes_for_batch,
                            "model": st.session_state.selected_model
                        }
                        filename = get_current_filename(st.session_state.core_theme)
                        save_batch_record(record_to_save, filename)
                        
                        st.session_state.processed_batches.add(batch_id) 
                    else: 
                        log_msg = f"⚪ Batch {batch_id} | 🪙{res['tokens']} | 无有效编码"
                        st.session_state.processed_batches.add(batch_id) 
                else: 
                    log_msg = f"❌ API错误: {res['error']}"

                if log_msg: log_messages.append(log_msg)
                log_container.text_area("实时日志", value="\n".join(reversed(log_messages)), height=250)
                
                count += 1
                progress_bar.progress(count / total, text=f"进度: {count}/{total} (正在处理 Batch {batch_id})")
                
                if st.session_state.get('test_mode', False):
                    st.session_state.is_coding = False
                    st.session_state.test_mode = False
                    st.success("✅ 测试完成 (已处理1个组块)")
                    st.rerun()
            
            if st.session_state.is_coding:
                st.session_state.is_coding = False
                st.success("🎉 完成！")
                time.sleep(1); st.rerun()

# =======================================================================
# 5. 结果预览
# =======================================================================
if not st.session_state.open_codes.empty:
    with st.container(border=True):
        st.subheader("步骤 3: 结果预览与保存")
        
        cols = ['batch_id', 'source_file', 'code', 'quote', 'confidence', 'original_ids']
        for c in cols: 
            if c not in st.session_state.open_codes.columns: st.session_state.open_codes[c] = None
            
        edited = st.data_editor(
            st.session_state.open_codes, 
            column_order=cols,
            disabled=['source_file', 'quote', 'original_ids', 'batch_id'],
            num_rows="dynamic", key="editor", height=400
        )
        st.session_state.open_codes = edited
        
        st.markdown("#### 保存项目")
        meta_bg = "Custom" if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)" else f"纳入：{st.session_state.definition_logic}\n排除：{st.session_state.exclusion_logic}"
            
        excel_data = to_excel(
            df_atomic, 
            edited, 
            pd.DataFrame({"core_theme":[st.session_state.core_theme], "bg":[meta_bg]})
        )
        st.download_button("🚀 下载项目 (.xlsx)", data=excel_data, file_name=f"Project_{st.session_state.core_theme}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.page_link("pages/4_Axial_Coding.py", label="下一步 (轴心编码)", icon="➡️")
