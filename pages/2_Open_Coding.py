import streamlit as st 

import pandas as pd
import time
from openai import OpenAI
import json
import os
import glob
from io import BytesIO
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from paths import get_project_paths
from prompts import create_background_meta_prompt, create_final_coding_prompt, get_manual_prompt_template

if not st.session_state.get("authentication_status"):
    st.info("请先登录系统 🔒")
    st.switch_page("Home.py") # 强制跳转回登录页
    st.stop() # 停止运行下面的代码

OPEN_CODES_COLUMNS = [
    'code', 'quote', 'quote_highlight',
    'confidence', 'original_ids', 'evidence_span', 'batch_id','source_file'
]

DEFAULT_CUSTOM_PROMPT = get_manual_prompt_template() or """请在此输入自定义 Prompt，并使用 {batch_text} 作为文本占位符。"""


def init_opening_session_state():
    defaults = {
        #"active_project_selector": "default_project",
        "prompt_mode": "1. 智能向导 (全自动)",
        "custom_prompt": DEFAULT_CUSTOM_PROMPT,
        "custom_prompt_editor": DEFAULT_CUSTOM_PROMPT,
        "processed_batches": set(),
        "open_codes": pd.DataFrame(columns=OPEN_CODES_COLUMNS),
        "is_coding": False,
        "test_mode": False,
        "definition_logic": "",
        "exclusion_logic": "",
        "core_theme": "（请在此输入您的研究主题）",
        "selected_model": "qwen-plus",
        "openai_key": "",
        "gemini_key": "",
        "stop_requested": False,
        "is_processing": False,
        "temperature": 0.1,
        "total_token_usage": 0,
        "final_coding_data": None,
        "is_paused": False,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def hard_reset_opening_project():
    st.session_state.final_coding_data = None
    st.session_state.open_codes = pd.DataFrame(columns=OPEN_CODES_COLUMNS)
    st.session_state.processed_batches = set()
    st.session_state.core_theme = "（请重新输入主题）"
    st.session_state.definition_logic = ""
    st.session_state.exclusion_logic = ""
    st.session_state.custom_prompt = DEFAULT_CUSTOM_PROMPT
    st.session_state.custom_prompt_editor = DEFAULT_CUSTOM_PROMPT
    st.session_state.is_coding = False
    st.session_state.test_mode = False
    st.session_state.is_paused = False
    st.session_state.total_token_usage = 0


init_opening_session_state()

if st.session_state.get("open_codes") is None:
    st.session_state.open_codes = pd.DataFrame(columns=OPEN_CODES_COLUMNS)

current_user = st.session_state.get("username", "default_user")
USER_BASE_DIR = os.path.join("users_data", current_user)

existing_projects = []
if os.path.exists(USER_BASE_DIR):
    existing_projects = [
        d for d in os.listdir(USER_BASE_DIR)
        if os.path.isdir(os.path.join(USER_BASE_DIR, d))
    ]

# 只初始化一次：如果前面页面已经选过，就沿用
if "active_project_selector" not in st.session_state:
    st.session_state["active_project_selector"] = existing_projects[0] if existing_projects else None

# 只有当当前项目已经不存在时，才回退
elif st.session_state["active_project_selector"] not in existing_projects:
    st.session_state["active_project_selector"] = existing_projects[0] if existing_projects else None

# =========================
# 🧠 侧边栏：选择项目（唯一来源）
# =========================
with st.sidebar:
    if not existing_projects:
        st.error("❌ 没有可用项目")
        st.stop()

    st.selectbox(
        "项目管理",
        options=existing_projects,
        key="active_project_selector",
        on_change=hard_reset_opening_project
    )

    st.info("系统会自动将您的编码结果保存到云端文件夹中。")

# ✅ 全页统一使用这个项目
selected_project = st.session_state["active_project_selector"]

# =========================
# ✅ DIRS 必须在这里初始化（关键修复）
# =========================
DIRS = get_project_paths(current_user, selected_project)

# =======================================================================
# 0. 数据持久化与恢复模块
# =======================================================================

# =======================================================================
# ✅ 文件名函数（改为用 selected_project）
# =======================================================================
def get_current_filename(theme):
    safe_theme = "".join([c for c in theme if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_theme:
        safe_theme = "Untitled_Project"
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    return f"Opening_{selected_project}_{safe_theme}_{date_str}.jsonl"


# --- 数据转换与读写函数 ---
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_batch_record(record_dict, filename):
    filepath = os.path.join(DIRS["opening_autosave"], filename)
    record_dict['timestamp'] = datetime.datetime.now().isoformat()
    try:
        clean_dict = convert_numpy(record_dict)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"自动保存失败: {e}")

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
    processed_batches = set()

    for r in records:
        b_id = r.get('batch_id')
        if b_id is not None:
            processed_batches.add(int(b_id))

        codes_list = r.get('final_codes', [])
        for c in codes_list:
            flat_codes.append(c)

    df = pd.DataFrame(flat_codes)

    for col in OPEN_CODES_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df, processed_batches

# =======================================================================
# 1. 核心逻辑函数与页面配置 (保持不变)
# =======================================================================

def process_single_batch(batch_id, df_atomic, prompt_mode,
                         custom_prompt, core_theme,
                         definition_logic, exclusion_logic,
                         api_key, model_id, temperature):

    batch_rows = df_atomic[df_atomic['batch_id'] == batch_id]

    batch_text = "\n".join([
        f"[{r['global_id']}] {r['content']}"
        for _, r in batch_rows.iterrows()
    ])

    if prompt_mode == "3. 高级自定义 (完全手动)":
        prompt = custom_prompt.replace("{batch_text}", batch_text)
    else:
        prompt = create_final_coding_prompt(
            core_theme,
            definition_logic,
            exclusion_logic,
            batch_text
        )

    res = call_llm_api(api_key, model_id, prompt, temperature)

    return batch_id, res, batch_rows, batch_text


with st.sidebar:
    # =======================
    # 3️⃣ 恢复历史记录
    # =======================
    st.subheader("📥 恢复进度")

    jsonl_files = glob.glob(os.path.join(DIRS["opening_autosave"], "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)

    if jsonl_files:
        history_file = st.selectbox(
            "选择历史文件",
            [os.path.basename(f) for f in jsonl_files]
        )

        if st.button("🔄 恢复选中文件",
                     type="primary",
                     use_container_width=True):

            filepath = os.path.join(DIRS["opening_autosave"], history_file)
            loaded_df, processed_set = load_from_jsonl(filepath)

            if not loaded_df.empty:
                st.session_state.open_codes = loaded_df
                st.session_state.processed_batches = processed_set
                st.success(f"恢复 {len(loaded_df)} 条记录")
                time.sleep(0.5)
                st.rerun()
            else:
                st.warning("文件中无有效数据")
    else:
        st.caption("当前项目暂无历史备份")

    st.divider()

    # =======================
    # 4️⃣ 清空当前进度
    # =======================
    if st.button("🗑 清空当前进度",
                 use_container_width=True):
        hard_reset_opening_project()
        st.success("已清空进度")
        time.sleep(0.5)
        st.rerun()

    st.caption("⚠️ 请勿修改原始文件名或顺序，以免影响断点续传")

def get_api_key(provider_name: str) -> str:
    return st.secrets.get(provider_name, "")

def call_llm_api(api_key, model_id, prompt_text, temperature=0.1):
    try:
        if model_id.startswith("gpt-4o"):
            base_url = "https://api.openai.com/v1"
            client_key = api_key
        elif model_id.startswith("gemini"):
            base_url = "https://api.gemini.com/v1"
            client_key = api_key
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
        if "401" in error_str or "Incorrect API key" in error_str:
            return {"success": False, "error": "⚠️ API Key 无效 (401)：请检查密钥是否复制完整、是否有多余空格，或账户是否欠费。", "tokens": 0}
        else:
            return {"success": False, "error": f"API Exception: {error_str}", "tokens": 0}


def extract_json(text, start_char='[', end_char=']'):
    try:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        return [] if start_char == '[' else {}
    except:
        return [] if start_char == '[' else {}
def _normalize_id_list(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_value = [raw_value]
    if not isinstance(raw_value, list):
        return []

    cleaned = []
    seen = set()
    for item in raw_value:
        uid = str(item).strip()
        if not uid or uid in seen:
            continue
        seen.add(uid)
        cleaned.append(uid)
    return cleaned


def _get_valid_a_rows(id_list, atomic_lookup):
    valid_ids = []
    source_files = set()
    rows_by_id = {}

    for uid in id_list:
        if uid not in atomic_lookup.index:
            continue

        try:
            row = atomic_lookup.loc[uid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        except:
            continue

        if str(uid).startswith("Q-") or row['role_code'] == 'Q':
            continue

        valid_ids.append(uid)
        rows_by_id[uid] = row
        source_files.add(row['source_file'])

    return valid_ids, rows_by_id, source_files


def _build_quote_and_highlight(span_ids, original_ids, rows_by_id):
    quote_parts = []
    highlight_parts = []
    original_id_set = set(original_ids)

    for uid in span_ids:
        row = rows_by_id.get(uid)
        if row is None:
            continue

        text = str(row['content'])
        quote_parts.append(text)

        # 这里不用 **加粗**，因为 st.data_editor 和 pandas.to_excel 不会渲染局部富文本
        # 先用【】做稳定标记，人工复核最实用
        if uid in original_id_set:
            highlight_parts.append(f"【{text}】")
        else:
            highlight_parts.append(text)

    return "".join(quote_parts), "".join(highlight_parts)


def reconstruct_quote_and_validate(ai_item, atomic_lookup):
    """
    新逻辑:
    1. original_ids = 核心命中 ids
    2. evidence_span = 人工复核所需的最小自然语境范围
    3. quote 按 evidence_span 重组
    4. quote_highlight 也按 evidence_span 重组，但把 original_ids 对应内容做标记
    """
    raw_ids = _normalize_id_list(ai_item.get('ids', []))
    raw_evidence_span = _normalize_id_list(ai_item.get('evidence_span', raw_ids))

    valid_ids, id_rows, source_files_ids = _get_valid_a_rows(raw_ids, atomic_lookup)
    valid_span_ids, span_rows, source_files_span = _get_valid_a_rows(raw_evidence_span, atomic_lookup)

    if not valid_ids:
        return None

    # evidence_span 无效时，回退到 original_ids
    if not valid_span_ids:
        valid_span_ids = valid_ids
        span_rows = id_rows
        source_files_span = source_files_ids

    # 用 evidence_span 拼 quote；用 original_ids 做高亮标记
    quote_text, quote_highlight_text = _build_quote_and_highlight(
        valid_span_ids,
        valid_ids,
        span_rows
    )

    source_files = source_files_ids.union(source_files_span)

    return {
        "code": ai_item.get('code', 'Unnamed Code'),
        "quote": quote_text,
        "quote_highlight": quote_highlight_text,
        "original_ids": valid_ids,
        "evidence_span": valid_span_ids,
        "source_file": list(source_files)[0] if source_files else "Unknown",
        "confidence": ai_item.get('confidence', 3)
    }

def to_excel(df_raw, df_codes, df_meta):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if df_codes is not None:
            df_save = df_codes.copy()

            if 'original_ids' in df_save.columns:
                df_save['original_ids'] = df_save['original_ids'].astype(str)

            if 'evidence_span' in df_save.columns:
                df_save['evidence_span'] = df_save['evidence_span'].astype(str)

            if 'quote_highlight' in df_save.columns:
                df_save['quote_highlight'] = df_save['quote_highlight'].astype(str)

            df_save.to_excel(writer, index=False, sheet_name='open_codes')

        # if df_meta is not None:
        #     df_meta.to_excel(writer, index=False, sheet_name='project_meta')

    return output.getvalue()

# =======================================================================
# 3. 数据载入拦截器 (逻辑修正：确保载入后能看到恢复的数据)
# =======================================================================
if st.session_state.get('final_coding_data') is None:
    st.title(f"🚀 区域2: 开放性编码 ｜ 项目准备: {selected_project}")

    open_codes_df = st.session_state.get("open_codes")
    rec_count = len(open_codes_df) if open_codes_df is not None else 0

    if rec_count > 0:
        st.success(f"📈 内存中已载入历史进度：{rec_count} 条编码。请继续载入对应的原始数据以激活 AI。")

    st.info("💡 请载入 Step 1 生成的预处理文件（xlsx/csv）")

    t1, t2 = st.tabs(["📁 从云端目录导入", "📤 本地上传"])
    with t1:
        server_files = []
        if os.path.exists(DIRS["preprocessed"]):
            server_files = [f for f in os.listdir(DIRS["preprocessed"]) if f.endswith(('.xlsx', '.csv'))]

        if server_files:
            target_f = st.selectbox("选择文件:", ["-- 请选择 --"] + server_files, key="data_load_key")
            if st.button("✅ 确认载入数据并开始", type="primary"):
                if target_f != "-- 请选择 --":
                    p = os.path.join(DIRS["preprocessed"], target_f)
                    df = pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p)
                    st.session_state.final_coding_data = df
                    st.rerun()
        else:
            st.warning("1_preprocessed_data 文件夹为空。")

    with t2:
        up = st.file_uploader("手动上传原始数据", type=["xlsx", "csv"])
        if up:
            df_up = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            st.session_state.final_coding_data = df_up
            st.rerun()
    st.warning("无原始数据")
    st.stop()

# --- 数据就绪 ---
df_atomic = st.session_state.final_coding_data
atomic_lookup = df_atomic.set_index('global_id')

st.title("区域2: 开放性编码 Prompt生成与执行区 🛠️")

# =======================================================================
# 3. 配置区域
# =======================================================================
with st.container(border=True):
    st.subheader("步骤 1: 配置模式与规则")

    st.markdown("###### 🧠 模型选择")
    model_options = {
        "👑 Qwen-Max": "qwen-max",
        # "🌟 GPT-4o": "gpt-4o",
        # "🚀 Gemini": "gemini",
        "💎 GLM-4.6": "glm-4.6",
        "🔥 DeepSeek-V3.2": "deepseek-v3.2",
        "🔥 DeepSeek-R1": "deepseek-r1",
        "⚖️ Qwen-Plus": "qwen-plus",
    }
    model_ids = list(model_options.values())
    try:
        default_index = model_ids.index(st.session_state.selected_model)
    except ValueError:
        default_index = 0
    selected_model_name = st.selectbox("选择模型", options=model_options.keys(), index=default_index, label_visibility="collapsed")
    st.session_state.selected_model = model_options[selected_model_name]

    if st.session_state.selected_model.startswith("gpt-4o"):
        st.session_state.api_key = get_api_key("OPENAI_API_KEY")
    elif st.session_state.selected_model.startswith("gemini"):
        st.session_state.api_key = get_api_key("GEMINI_API_KEY")
    else:
        st.session_state.api_key = get_api_key("QWEN_API_KEY")

    st.divider()
    mode_options = ["1. 智能向导 (全自动)", "2. 外部辅助 (推荐，需用到网页端，适用最新大模型) ", "3. 高级自定义 (完全手动)"]
    selected_mode = st.radio("选择工作模式", mode_options, horizontal=True)
    st.session_state.prompt_mode = selected_mode


    st.markdown("#### 1. 核心研究主题")
    st.text_input("研究主题", key="core_theme", label_visibility="collapsed")
    core_theme_input = st.session_state.core_theme

    # --- 模式 A: 智能向导 ---
    if st.session_state.prompt_mode == "1. 智能向导 (全自动)":
        if st.button("🤖 一键生成判别标准", type="primary"):
            if not st.session_state.api_key:
                st.error("请输入 DashScope Key！")
                st.stop()
            elif not core_theme_input or "请在" in core_theme_input:
                st.error("请输入有效主题！")
                st.stop()

            with st.spinner("正在分析..."):
                meta_prompt = create_background_meta_prompt(st.session_state.core_theme)
                api_res = call_llm_api(st.session_state.api_key, st.session_state.selected_model, meta_prompt, temperature=0.3)

                if api_res["success"]:
                    st.session_state.total_token_usage += api_res["tokens"]
                    data = extract_json(api_res["text"], start_char='{', end_char='}')
                    if isinstance(data, dict):
                        st.session_state.definition_logic = data.get('definition_logic', '')
                        st.session_state.exclusion_logic = data.get('exclusion_logic', '')
                        st.success("标准生成成功！请在下方确认。")
                    else:
                        st.error(f"生成失败: {data}")
                else:
                    st.error(api_res["error"])

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

        current_prompt = st.session_state.get("custom_prompt_editor")
        if current_prompt is None or str(current_prompt).strip() == "":
            current_prompt = st.session_state.get("custom_prompt") or DEFAULT_CUSTOM_PROMPT

        edited_prompt = st.text_area(
            "完整 Prompt 编辑器",
            value=current_prompt,
            height=400
        )

        st.session_state.custom_prompt_editor = edited_prompt
        st.session_state.custom_prompt = edited_prompt


    # --- 公共区域：显示/编辑标准 ---
    st.divider()
    if st.session_state.prompt_mode in ["1. 智能向导 (全自动)", "2. 外部辅助 (推荐，需用到网页端，适用最新大模型) "]:
        col_def, col_exc = st.columns(2)
        with col_def:
            edited_def = st.text_area(
                "✅ 纳入标准 (Definition)",
                value=st.session_state.get("definition_logic", ""),
                height=100
            )
            st.session_state.definition_logic = edited_def

        with col_exc:
            edited_exc = st.text_area(
                "❌ 排除标准 (Exclusion)",
                value=st.session_state.get("exclusion_logic", ""),
                height=100
            )
            st.session_state.exclusion_logic = edited_exc

# =======================================================================
# 4. 执行区域（不卡UI版本）
# =======================================================================
with st.container(border=True):
    st.subheader("步骤 2: 批量编码执行")

    # ========= 是否允许运行（只控制按钮） =========
    can_run = False
    if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)":
        current_prompt = st.session_state.get("custom_prompt_editor") or DEFAULT_CUSTOM_PROMPT
        can_run = "{batch_text}" in current_prompt
    elif st.session_state.definition_logic and st.session_state.exclusion_logic:
        can_run = True

    # ========= 数据准备 =========
    if df_atomic is None:
        st.warning("⚠️ 请先加载数据。")
        st.stop()

    unique_batches = sorted(df_atomic['batch_id'].unique())
    pending_batches = [b for b in unique_batches if b not in st.session_state.processed_batches]

    st.markdown(
        f"**任务统计**: 总组块 `{len(unique_batches)}` | 已完成 `{len(st.session_state.processed_batches)}` | 待处理 `{len(pending_batches)}`"
    )

    col_act1, col_act2, col_act3 = st.columns([1, 1, 1])

    if col_act1.button(
        "▶️ 开始/继续",
        type="primary",
        disabled=(len(pending_batches) == 0 or not can_run)
    ):
        st.session_state.is_coding = True
        st.session_state.is_paused = False
        st.rerun()

    if col_act2.button("⏸️ 暂停"):
        st.session_state.is_paused = True

    if col_act3.button("🧪 测试1条"):
        st.session_state.is_coding = True
        st.session_state.test_mode = True
        st.session_state.is_paused = False
        st.rerun()

    if not can_run:
        st.info("⚠️ 请先完成上方提示词配置（纳入标准 & 排除标准 或 自定义Prompt）")

    # ============================================================
    # 🚀 执行引擎（只在 is_coding 时运行）
    # ============================================================
    if st.session_state.get("is_coding", False):

        if st.session_state.get("is_paused", False):
            if "executor" in st.session_state:
                st.session_state.executor.shutdown(wait=False)
                del st.session_state.executor
            for k in [
                "futures", "batch_groups", "submitted_index",
                "submitted_batches", "handled_batches", "run_batches"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.warning("⏸️ 已暂停")
            st.stop()

        # =========================
        # 初始化（只执行一次）
        # =========================
        if "executor" not in st.session_state:
            run_batches = pending_batches[:1] if st.session_state.get("test_mode", False) else pending_batches.copy()

            st.session_state.executor = ThreadPoolExecutor(max_workers=3)
            st.session_state.futures = []
            st.session_state.submitted_index = 0
            st.session_state.submitted_batches = set()
            st.session_state.handled_batches = set()
            st.session_state.run_batches = run_batches

            st.session_state.batch_groups = {
                b: df_atomic[df_atomic['batch_id'] == b]
                for b in run_batches
            }

            st.session_state.log_messages = []

        run_batches = st.session_state.get("run_batches", [])
        total = len(run_batches)
        completed_count = len(st.session_state.get("handled_batches", set()))
        progress_value = min(completed_count / total, 1.0) if total > 0 else 0.0

        # =========================
        # UI组件
        # =========================
        progress_bar = st.progress(
            progress_value,
            text=f"进度: {completed_count}/{total}"
        )

        log_container = st.empty()

        # =========================
        # 提交任务（保证每个 batch 只提交一次）
        # =========================
        submit_per_cycle = 2

        for _ in range(submit_per_cycle):
            if st.session_state.submitted_index < total:
                batch_id = run_batches[st.session_state.submitted_index]

                if batch_id in st.session_state.submitted_batches:
                    st.session_state.submitted_index += 1
                    continue

                current_prompt = st.session_state.get("custom_prompt_editor") or DEFAULT_CUSTOM_PROMPT

                future = st.session_state.executor.submit(
                    process_single_batch,
                    batch_id,
                    st.session_state.batch_groups[batch_id],
                    st.session_state.prompt_mode,
                    current_prompt,
                    st.session_state.core_theme,
                    st.session_state.definition_logic,
                    st.session_state.exclusion_logic,
                    st.session_state.api_key,
                    st.session_state.selected_model,
                    st.session_state.temperature
                )

                st.session_state.futures.append((batch_id, future))
                st.session_state.submitted_batches.add(batch_id)
                st.session_state.submitted_index += 1

        # =========================
        # 收集完成任务（保证每个 batch 只处理一次）
        # =========================
        done_list = []

        for batch_id, f in st.session_state.futures:
            if batch_id in st.session_state.handled_batches:
                continue
            if f.done():
                done_list.append((batch_id, f))

        for batch_id, f in done_list:

            if (batch_id, f) in st.session_state.futures:
                st.session_state.futures.remove((batch_id, f))

            if batch_id in st.session_state.handled_batches:
                continue

            try:
                batch_id, res, batch_rows, batch_text = f.result()
            except Exception as e:
                st.session_state.log_messages.append(f"❌ Batch {batch_id} 崩了: {e}")
                st.session_state.handled_batches.add(batch_id)
                continue

            if res["success"]:

                st.session_state.total_token_usage += res["tokens"]

                raw_codes = extract_json(res["text"])
                final_codes = []

                if isinstance(raw_codes, list):
                    for item in raw_codes:
                        clean = reconstruct_quote_and_validate(item, atomic_lookup)
                        if clean:
                            clean["batch_id"] = batch_id
                            final_codes.append(clean)

                if final_codes:
                    new_df = pd.DataFrame(final_codes)
                    st.session_state.open_codes = pd.concat(
                        [st.session_state.open_codes, new_df],
                        ignore_index=True
                    )

                    st.session_state.processed_batches.add(batch_id)

                    code_str = ", ".join([c['code'] for c in final_codes])
                    log_msg = f"✅ Batch {batch_id} | {code_str}"
                else:
                    st.session_state.processed_batches.add(batch_id)
                    log_msg = f"⚪ Batch {batch_id} | 无编码"

            else:
                log_msg = f"❌ {res['error']}"

            st.session_state.log_messages.append(log_msg)
            st.session_state.handled_batches.add(batch_id)

        # =========================
        # UI刷新
        # =========================
        completed_count = len(st.session_state.get("handled_batches", set()))
        progress_value = min(completed_count / total, 1.0) if total > 0 else 0.0

        progress_bar.progress(
            progress_value,
            text=f"进度: {completed_count}/{total}"
        )

        log_container.text_area(
            "实时日志",
            value="\n".join(reversed(st.session_state.log_messages)),
            height=250
        )

        # =========================
        # test模式
        # =========================
        if st.session_state.get("test_mode", False) and completed_count >= 1:
            if "executor" in st.session_state:
                st.session_state.executor.shutdown(wait=False)
                del st.session_state.executor
            for k in [
                "futures", "batch_groups", "submitted_index",
                "submitted_batches", "handled_batches", "run_batches"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.is_coding = False
            st.session_state.test_mode = False
            st.success("✅ 测试完成")
            st.rerun()

        # =========================
        # 是否继续
        # =========================
        if completed_count < total:
            time.sleep(0.2)
            st.rerun()
        else:
            st.success("🎉 全部完成")

            if "executor" in st.session_state:
                st.session_state.executor.shutdown(wait=False)
                del st.session_state.executor

            for k in [
                "futures", "batch_groups", "submitted_index",
                "submitted_batches", "handled_batches", "run_batches"
            ]:
                if k in st.session_state:
                    del st.session_state[k]

            st.session_state.is_coding = False

# =======================================================================
# 5. 结果预览
# =======================================================================
if st.session_state.open_codes is not None and not st.session_state.open_codes.empty:
    with st.container(border=True):
        st.subheader("步骤 3: 结果预览与保存")

        cols = ['quote_highlight', 'quote', 'code', 'confidence', 'original_ids', 'evidence_span', 'batch_id', 'source_file']

        for c in cols:
            if c not in st.session_state.open_codes.columns:
                st.session_state.open_codes[c] = None

        edited = st.data_editor(
            st.session_state.open_codes,
            column_order=cols,
            disabled=['source_file', 'quote', 'quote_highlight', 'original_ids', 'evidence_span', 'batch_id'],
            num_rows="dynamic",
            key="editor",
            height=400
        )
        st.session_state.open_codes = edited

        st.markdown("#### 保存项目")
        meta_bg = "Custom" if st.session_state.prompt_mode == "3. 高级自定义 (完全手动)" else f"纳入：{st.session_state.definition_logic}\n排除：{st.session_state.exclusion_logic}"

        excel_data = to_excel(
            df_atomic,
            edited,
            pd.DataFrame({"core_theme": [st.session_state.core_theme], "bg": [meta_bg]})
        )
        col1, col2 = st.columns([2, 1])
        st.caption("💡 保存用于系统内继续分析；下载用于导出到本地")
        with col1:
            if st.button("💾 保存到项目（用于后续分析）", type="primary", use_container_width=True):
                filename = f"{st.session_state.core_theme}_{time.strftime('%Y%m%d_%H%M')}_open_coding.xlsx"
                save_path = os.path.join(DIRS["opening_final"], filename)

                with open(save_path, "wb") as f:
                    f.write(excel_data)

                st.success("✅ 已保存到项目文件夹（Step 2）")
        with col2:
            st.download_button(
                "⬇️ 下载副本到本地",
                data=excel_data,
                file_name=f"Project_{st.session_state.core_theme}_{time.strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        st.page_link("pages/3_Visual_Analysis.py", label="下一步 (多位编码者对齐与相似编码合并)", icon="➡️")
