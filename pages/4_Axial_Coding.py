import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import json
import os
import glob
from datetime import datetime
from io import BytesIO
from prompts import create_axial_coding_prompt
from paths import get_project_paths

AXIAL_CODES_COLUMNS = ['code', 'category', 'confidence', 'reasoning', 'status']



# -----------------------------------------------------------------------
# 【必须加在所有页面 import 之后的第一段】
# -----------------------------------------------------------------------
# 1. 检查有没有登录？没登录直接踢回主页
if not st.session_state.get("authentication_status"):
    st.info("请先登录系统 🔒")
    st.switch_page("Home.py") # 强制跳转回登录页
    st.stop() # 停止运行下面的代码

def load_latest_dims_text_with_fallback(folder_path):
    try:
        # 1️⃣ 找所有目标文件
        files = glob.glob(os.path.join(folder_path, "aligned_merge_category_*.xlsx"))

        if not files:
            raise FileNotFoundError("没有找到分析结果文件")

        # 2️⃣ 按修改时间排序，取最新
        latest_file = max(files, key=os.path.getmtime)

        # 3️⃣ 读取 sheet2
        dims_df = pd.read_excel(latest_file, sheet_name="axial_dimensions")

        if not dims_df.empty and "dimension" in dims_df.columns:
            lines = []
            for _, row in dims_df.iterrows():
                dim = str(row.get("dimension", "")).strip()
                definition = str(row.get("definition", "")).strip()

                if dim:
                    if definition:
                        lines.append(f"{dim}：{definition}")
                    else:
                        lines.append(dim)

            if lines:
                return "\n".join(lines)

    except Exception as e:
        print(f"[WARN] 读取维度失败：{e}")

    # 4️⃣ fallback
    return """情绪识别：情绪识别的分类定义
情绪调节：情绪识别的分类定义
社会支持：情绪识别的分类定义"""

DEFAULT_DIMS_TEXT = load_latest_dims_text_with_fallback("analysis_final")

def get_api_key(provider_name: str) -> str:
    try:
        return st.secrets[provider_name]
    except Exception:
        return os.environ.get(provider_name, "")

# =======================================================================
# 0. 数据持久化与恢复模块 (适配多项目模式)
# =======================================================================

USER_ROOT = st.session_state.get('user_root_dir', '')

def init_axial_session_state():
    defaults = {
        "ax_proj_select": "default_project",
        "selected_model": "qwen-plus",
        "model_id": "qwen-plus",
        "api_key": "",
        "open_codes": None,
        "axial_codes_df": pd.DataFrame(columns=AXIAL_CODES_COLUMNS),
        "codes_to_review": [],
        "ai_suggestions": {},
        "all_unique_codes": [],
        "is_running_axial": False,
        "total_token_usage": 0,
        "dims_input_text": DEFAULT_DIMS_TEXT,
        "core_theme": "",
        "axial_mode": "🔸 半自动模式",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_axial_session_state()

def hard_reset_axial_project():
    st.session_state.open_codes = None
    st.session_state.axial_codes_df = pd.DataFrame(columns=AXIAL_CODES_COLUMNS)
    st.session_state.codes_to_review = []
    st.session_state.ai_suggestions = {}
    st.session_state.all_unique_codes = []
    st.session_state.is_running_axial = False
    st.session_state.total_token_usage = 0
    st.session_state.dims_input_text = DEFAULT_DIMS_TEXT
    st.session_state.core_theme = ""
    st.session_state.axial_mode = "🔸 半自动模式"

    for key in ["dims_input_area", "helper_dims_input"]:
        if key in st.session_state:
            del st.session_state[key]

# 项目选择
existing_projects = [d for d in os.listdir(USER_ROOT) if os.path.isdir(os.path.join(USER_ROOT, d))]

if not existing_projects:
    st.warning("请先创建项目")
    st.stop()

if st.session_state.get("active_project_selector") not in existing_projects:
    st.session_state["active_project_selector"] = existing_projects[0]
    
with st.sidebar:
    if not existing_projects:
        st.error("❌ 没有可用项目")
        st.stop()

    st.selectbox(
        "项目管理",
        options=existing_projects,
        key="active_project_selector",
        on_change=hard_reset_axial_project
    )
    st.info("系统会自动将您的编码结果保存到云端文件夹中。")

selected_project = st.session_state["active_project_selector"]
username = st.session_state.get("username")
DIRS = get_project_paths(username, selected_project)

# 定义路径：users_data/用户名/项目名/4_axial_coding
RECOVERY_DIR = DIRS["axial_autosave"]

def ensure_recovery_dir():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR, exist_ok=True)

def get_current_filename(topic, mode):
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_topic: safe_topic = "Untitled"
    safe_mode = "Auto" if "自动" in mode else "Semi" if "半自动" in mode else "Strict"
    date_str = datetime.now().strftime("%Y%m%d") 
    return f"Axial_{selected_project}_{safe_topic}_{safe_mode}_{date_str}.jsonl"

def save_record_to_jsonl(record_dict, filename):
    ensure_recovery_dir()
    filepath = os.path.join(RECOVERY_DIR, filename) # 这里会自动存入项目文件夹
    record_dict['timestamp'] = datetime.now().isoformat()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

def parse_dimension_names(dim_text):
    dims = []
    for line in str(dim_text).splitlines():
        line = str(line).strip()
        if not line:
            continue

        if "：" in line:
            name = line.split("：", 1)[0].strip()
        elif ":" in line:
            name = line.split(":", 1)[0].strip()
        else:
            name = line.strip()

        if name and name not in dims:
            dims.append(name)

    if "无对应维度" not in dims:
        dims.append("无对应维度")
    return dims


def normalize_text(s):
    return str(s).replace("\u3000", " ").replace("\n", " ").replace("\r", "").strip()

def load_from_jsonl(filepath):
    # 此处的 filepath 通常是用户从当前项目文件夹中选中的
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except: continue
    return pd.DataFrame(data) if data else pd.DataFrame()

# =======================================================================
# 1. 核心逻辑函数区
# =======================================================================

def call_qwen_api(api_key, model_id, messages, temperature=0.1):
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
            messages=messages,
        )
        usage = response.usage
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        content = response.choices[0].message.content
        if not content:
            return {"success": False, "error": "API 返回了空内容", "tokens": total_tokens}

        return {"success": True, "text": content, "tokens": total_tokens}
    except Exception as e:
        return {"success": False, "error": f"API Exception: {str(e)}", "tokens": 0}

def extract_json(text):
    try:
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = text[start_index : end_index + 1]
            return json.loads(json_str)
        else: return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def to_excel_axial(axial_mapping_df, original_df=None):
    """
    全量映射导出：将轴心编码规则映射回原始数据
    """
    output = BytesIO()
    
# === 安全合并版本 ===

    if original_df is not None and not original_df.empty and not axial_mapping_df.empty:

        if 'code' in original_df.columns and 'code' in axial_mapping_df.columns:

            mapping_rules = axial_mapping_df.drop_duplicates(subset=['code'], keep='last')

            cols_to_use = [
                c for c in mapping_rules.columns
                if c in ['code', 'category', 'confidence', 'reasoning', 'status']
            ]

            mapping_rules = mapping_rules[cols_to_use]

            # 🛡 删除可能冲突列
            conflict_cols = ['category', 'confidence', 'reasoning', 'status']
            original_df = original_df.drop(
                columns=[c for c in conflict_cols if c in original_df.columns]
            )

            merged_df = pd.merge(
                original_df,
                mapping_rules,
                on='code',
                how='left'
            )

            merged_df['category'] = merged_df['category'].fillna('待归类')

            final_df = merged_df
    else:
        final_df = axial_mapping_df

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Axial_Full_Data')
        if not axial_mapping_df.empty:
            axial_mapping_df.to_excel(writer, index=False, sheet_name='Coding_Rules_Only')
            
    processed_data = output.getvalue()
    return processed_data

def get_definition_prompt(domain, topic, raw_keywords):
    return f"""
你是一名资深的质性研究专家。
请基于【{domain}】领域，针对【{topic}】这一研究主题，为用户提供的维度关键词生成简短、精准的“操作性定义”。

【输入关键词】
{raw_keywords}

【任务要求】
1. **去重与精确化**：每个定义必须具有排他性，避免不同维度之间的定义重叠。
2. **语境结合**：定义必须紧扣“{topic}”的研究语境，而非通用的字典解释。
3. **格式**：直接输出列表，格式为“维度名: 定义内容”，无多余文字。

【输出示例】
(假设主题是远程办公效率)
技术障碍: 指员工在远程工作中遇到的网络延迟、软件崩溃或硬件故障等具体阻碍。
沟通断层: 指团队成员因缺乏非语言线索而导致的信息误解或反馈滞后。
    """

def generate_definitions(api_key, model_id, domain, topic, raw_keywords):
    prompt = get_definition_prompt(domain, topic, raw_keywords)
    messages = [{"role": "user", "content": prompt}]
    return call_qwen_api(api_key, model_id, messages, temperature=0.7)



def handle_axial_acceptance(code_name, category, confidence, reasoning=""):
    # 1. 更新 Session State
    if not st.session_state.axial_codes_df.empty:
        st.session_state.axial_codes_df = st.session_state.axial_codes_df[
            st.session_state.axial_codes_df['code'] != code_name
        ]

    record_dict = {
        'code': code_name, 
        'category': category, 
        'confidence': confidence, 
        'reasoning': reasoning, 
        'status': 'Accepted' if confidence > 0 else 'Manual'
    }
    
    new_record = pd.DataFrame([record_dict])
    st.session_state.axial_codes_df = pd.concat([st.session_state.axial_codes_df, new_record], ignore_index=True)
    
    if code_name in st.session_state.codes_to_review:
        st.session_state.codes_to_review.remove(code_name)
    
    if 'ai_suggestions' in st.session_state and code_name in st.session_state.ai_suggestions:
        del st.session_state.ai_suggestions[code_name]

    # 2. 自动保存到 JSONL
    current_topic = st.session_state.get('core_theme', 'Unspecified_Topic')
    current_mode = st.session_state.get('axial_mode', 'Manual')
    filename = get_current_filename(current_topic, current_mode)
    
    save_record_to_jsonl(record_dict, filename)

def clear_axial_results():
    st.session_state.axial_codes_df = pd.DataFrame(columns=['code', 'category', 'confidence', 'reasoning', 'status'])
    if 'all_unique_codes' in st.session_state:
        st.session_state.codes_to_review = st.session_state.all_unique_codes.copy()
    st.session_state.ai_suggestions = {}
    st.session_state.is_running_axial = False
    st.success("已清空结果，可以重新开始。")

def get_code_frequency(code_name):
    """获取编码在原始数据中的出现频率"""
    if st.session_state.open_codes is not None and 'code' in st.session_state.open_codes.columns:
        return len(st.session_state.open_codes[st.session_state.open_codes['code'] == code_name])
    return 1

# [NEW] 聚合引文功能
def get_aggregated_quotes(codes_df, code_name, limit=3):
    """
    提取某个编码对应的前 N 条不重复引文，拼接成字符串
    """
    if codes_df is None or codes_df.empty:
        return "无语境"
    
    # 筛选相关行
    related = codes_df[codes_df['code'] == code_name]
    if related.empty:
        return "无语境"
    
    # 获取不为空的 unique 引文
    valid_quotes = [
        str(q) for q in related['quote'].dropna().unique() 
        if str(q).strip() and str(q) not in ["无", "（无引用）", "nan"]
    ]
    
    if not valid_quotes:
        return "（无语境，仅基于编码分析）"
    
    # 截取前 N 条
    selected_quotes = valid_quotes[:limit]
    
    # 拼接
    if len(selected_quotes) == 1:
        return selected_quotes[0]
    else:
        return " || ".join([f"{i+1}. {q}" for i, q in enumerate(selected_quotes)])

# =======================================================================
# 2. Streamlit 页面布局
# =======================================================================
st.set_page_config(page_title="区域4: 轴心编码", layout="wide")

with st.sidebar:

    
    ensure_recovery_dir()#####
    jsonl_files = glob.glob(os.path.join(RECOVERY_DIR, "*.jsonl"))
    jsonl_files.sort(key=os.path.getmtime, reverse=True)
    
    if jsonl_files:
        st.subheader("📥 恢复进度")
        selected_file = st.selectbox("选择历史文件", [os.path.basename(f) for f in jsonl_files], index=0)
        
        if st.button("🔄 载入选中文件"):
            filepath = os.path.join(RECOVERY_DIR, selected_file)
            loaded_df = load_from_jsonl(filepath)
            
            if not loaded_df.empty:
                if 'code' in loaded_df.columns:
                    loaded_df = loaded_df.drop_duplicates(subset=['code'], keep='last')
                    st.session_state.axial_codes_df = loaded_df
                    
                    if st.session_state.get('all_unique_codes'):
                        completed_codes = loaded_df['code'].tolist()
                        remaining = [c for c in st.session_state.all_unique_codes if c not in completed_codes]
                        st.session_state.codes_to_review = remaining
                        
                    st.success(f"成功恢复 {len(loaded_df)} 条记录！")
                    st.rerun()
                else:
                    st.error("文件格式不正确，缺少 code 列")
            else:
                st.warning("该文件为空")
    else:
        st.caption("暂无历史存档")

    if st.button("🗑 清空当前进度",
                 use_container_width=True):
        hard_reset_axial_project()
        st.success("已清空进度")
        time.sleep(0.5)
        st.rerun()

st.title("🧠区域4: 轴心编码 ｜ 项目准备: {selected_project}")

if st.session_state.get("open_codes") is None or st.session_state.open_codes.empty:

    st.info("💡 请载入开放编码结果文件（xlsx/csv/json/jsonl）")

    t1, t2 = st.tabs(["📁 从云端目录导入", "📤 本地上传"])
    candidate_dirs = [
        DIRS["analysis_final"],
        DIRS["opening_final"],
    ]
    with t1:
        all_server_files = []

        for folder in candidate_dirs:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith((".xlsx", ".csv", ".json", ".jsonl")):
                        all_server_files.append((folder, f))

        if all_server_files:
            display_options = ["-- 请选择 --"] + [f"{os.path.basename(folder)} / {fname}" for folder, fname in all_server_files]

            selected_display = st.selectbox(
                "选择文件:",
                display_options,
                key="axial_data_load_key"
            )

            if st.button("✅ 确认载入数据并开始", type="primary", key="axial_load_server_btn"):
                if selected_display != "-- 请选择 --":
                    try:
                        selected_index = display_options.index(selected_display) - 1
                        folder, fname = all_server_files[selected_index]
                        p = os.path.join(folder, fname)

                        if p.endswith(".csv"):
                            df_loaded = pd.read_csv(p)
                        elif p.endswith(".jsonl"):
                            df_loaded = pd.read_json(p, lines=True)
                        elif p.endswith(".json"):
                            try:
                                df_loaded = pd.read_json(p)
                            except ValueError:
                                df_loaded = pd.read_json(p, lines=True)
                        else:
                            df_loaded = pd.read_excel(p, engine="openpyxl")

                        if 'code' not in df_loaded.columns:
                            st.error("错误：缺少 'code' 列")
                            st.stop()

                        if 'quote' not in df_loaded.columns:
                            df_loaded['quote'] = "（无引用）"

                        st.session_state.open_codes = df_loaded
                        st.success(f"✅ 已载入 {len(df_loaded)} 条开放编码")
                        st.rerun()

                    except Exception as e:
                        st.error(f"读取失败: {e}")
        else:
            st.warning("当前项目可用于轴心编码的数据文件夹为空。")

    with t2:
        uploaded_file = st.file_uploader(
            "📥 上传开放编码文件",
            type=["xlsx", "csv", "json", "jsonl"],
            key="axial_uploader"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_loaded = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.jsonl'):
                    df_loaded = pd.read_json(uploaded_file, lines=True)
                elif uploaded_file.name.endswith('.json'):
                    try:
                        df_loaded = pd.read_json(uploaded_file)
                    except ValueError:
                        uploaded_file.seek(0)
                        df_loaded = pd.read_json(uploaded_file, lines=True)
                else:
                    df_loaded = pd.read_excel(uploaded_file, engine='openpyxl')

                if 'code' not in df_loaded.columns:
                    st.error("错误：缺少 'code' 列")
                    st.stop()

                if 'quote' not in df_loaded.columns:
                    df_loaded['quote'] = "（无引用）"

                st.session_state.open_codes = df_loaded
                st.success(f"✅ 已载入 {len(df_loaded)} 条开放编码")
                st.rerun()

            except Exception as e:
                st.error(f"读取失败: {e}")
                st.stop()

    st.warning("无开放编码数据")
    st.stop()

# --- 数据就绪 ---
codes_df = st.session_state.open_codes

all_unique_codes = codes_df['code'].unique().tolist()
st.session_state.all_unique_codes = all_unique_codes

if not st.session_state.codes_to_review and st.session_state.axial_codes_df.empty:
     st.session_state.codes_to_review = all_unique_codes.copy()
codes_to_process = st.session_state.codes_to_review

config_col, results_col = st.columns([1, 2])

# --- 左侧：配置 ---
with config_col:
    with st.container(border=True):
        st.subheader("步骤 1: 配置与启动")
        
        model_options = {
            "👑 Qwen-Max": "qwen-max",
            "🔥 DeepSeek-V3": "deepseek-v3",
            "⚖️ Qwen-Plus": "qwen-plus",
            "🚀 DeepSeek-R1": "deepseek-r1",
            #"🌟 GPT-4o": "gpt-4o"
            ## "🚀 Gemini": "gemini",
        }
        model_keys = list(model_options.keys())
        current_key = next((k for k, v in model_options.items() if v == st.session_state.selected_model), model_keys[0])
        sel_label = st.selectbox("🧠 选择模型", options=model_keys, index=model_keys.index(current_key))
        st.session_state.selected_model = model_options[sel_label]
        st.session_state.model_id = st.session_state.selected_model

        if st.session_state.selected_model.startswith("gpt-4o"):
            st.session_state.api_key = get_api_key("OPENAI_API_KEY")
        elif st.session_state.selected_model.startswith("gemini"):
            st.session_state.api_key = get_api_key("GEMINI_API_KEY")
        else:
            st.session_state.api_key = get_api_key("QWEN_API_KEY")

        st.divider()
        st.markdown("#### 定义轴心维度")
        
        with st.expander("✨ AI 辅助生成定义 (推荐)", expanded=False):
            st.caption("为了让 AI 生成精准的定义，请补充以下背景信息：")
            
            col_ctx1, col_ctx2 = st.columns(2)
            input_domain = col_ctx1.text_input("1. 研究领域", placeholder="例如：发展心理学")
            input_topic = col_ctx2.text_input("2. 研究主题", placeholder="例如：青少年叛逆期冲突")
            if input_topic: st.session_state.core_theme = input_topic
            
            raw_dims_input = st.text_area("3. 维度关键词 (用换行分隔)", 
                                         value="", 
                                         height=100, 
                                         placeholder="例如：\n情绪爆发\n冷处理",
                                         key="helper_dims_input")
            
            col_h1, col_h2 = st.columns([1, 1])
            with col_h1:
                if st.button("🪄 生成并填充", type="primary"):
                    if not input_domain.strip() or not input_topic.strip() or not raw_dims_input.strip():
                        st.warning("请完整填写【研究领域】、【研究主题】和【维度关键词】，这决定了定义的准确性。")
                    elif not st.session_state.get('api_key'):
                        st.error("请先输入 API Key")
                    else:
                        with st.spinner("正在基于特定语境生成定义..."):
                            gen_res = generate_definitions(
                                st.session_state.api_key, 
                                st.session_state.model_id, 
                                input_domain, 
                                input_topic, 
                                raw_dims_input
                            )
                            if gen_res["success"]:
                                st.session_state['dims_input_area'] = gen_res["text"]
                                st.session_state.dims_input_text = gen_res["text"]
                                st.session_state.total_token_usage += gen_res["tokens"]
                                st.success(f"定义已生成！(消耗 {gen_res['tokens']} tokens)")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(gen_res["error"])
            
            with col_h2:
                if st.button("📋 查看 Prompt (网页端用)"):
                    d_val = input_domain if input_domain else "[研究领域]"
                    t_val = input_topic if input_topic else "[研究主题]"
                    k_val = raw_dims_input if raw_dims_input else "[维度关键词]"
                    prompt_text = get_definition_prompt(d_val, t_val, k_val)
                    st.code(prompt_text, language="markdown")

        dimensions_input = st.text_area(
            "维度列表 (格式：维度名: 定义)", 
            value=st.session_state.dims_input_text, 
            height=200,
            key="dims_input_area",
            help="AI 会根据这里的定义进行匹配。可以手动输入，也可以使用上方的辅助生成。"
        )
        st.session_state.dims_input_text = dimensions_input

        dimension_list = parse_dimension_names(dimensions_input)
        
        st.divider()
        st.markdown("#### 执行控制")
        mode = st.radio("模式", ["🔹 自动模式", "🔸 半自动模式", "🔺 严格模式"], index=1)
        st.session_state.axial_mode = mode 
        
        batch_size = st.number_input("每批发送条数", 1, 100, 10)

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🟢 继续/开始", type="primary"):
                if not st.session_state.get('api_key'): st.error("无 Key"); st.stop()
                st.session_state.is_running_axial = True
                st.rerun()
        with col_btn2:
            if st.button("⏸️ 暂停"):
                st.session_state.is_running_axial = False
                st.rerun()
        with col_btn3:
            if st.button("🗑️ 清空"):
                clear_axial_results()
                st.rerun()
        
        if st.button("🧪 测试运行 (3条)"):
             if not st.session_state.get('api_key'): st.error("无 Key"); st.stop()
             with st.spinner("测试中..."):
                 test_codes = codes_to_process[:3]
                 test_batch_data = []
                 for c in test_codes:
                     # [MODIFIED] 使用聚合引文
                     q = get_aggregated_quotes(codes_df, c)
                     test_batch_data.append({'code': c, 'quote': q})

                 messages = create_axial_coding_prompt(dimension_list, test_batch_data)
                 res = call_qwen_api(st.session_state.api_key, st.session_state.model_id, messages)

                 if res["success"]:
                     st.session_state.total_token_usage += res["tokens"]
                     st.info(f"测试运行成功 (消耗 {res['tokens']} tokens)")
                     parsed = extract_json(res["text"])
                     if parsed:
                         st.json(parsed)
                     else:
                         st.error("JSON 解析失败，原始返回如下：")
                         st.code(res["text"])
                 else: st.error(res["error"])

    with st.expander("📂 查看/修改 开放编码源数据"):
        edited_open_codes = st.data_editor(st.session_state.open_codes, num_rows="dynamic", key="open_codes_manager", height=300)
        st.session_state.open_codes = edited_open_codes

# --- 右侧：结果审查台 ---
with results_col:
    
    st.markdown("### 📊 进度看板")
    total_num = len(st.session_state.all_unique_codes)
    done_num = len(st.session_state.axial_codes_df)
    ready_num = len([c for c in st.session_state.codes_to_review if c in st.session_state.ai_suggestions])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总数", total_num)
    m2.metric("✅ 已完成", done_num)
    m3.metric("🤖 待审查", ready_num)
    m4.metric("💰 Token", st.session_state.total_token_usage) 
    
    if total_num > 0: 
        progress_val = min(done_num / total_num, 1.0)
        st.progress(progress_val)
    
    st.divider()

    if st.session_state.axial_mode == "🔹 自动模式":
        st.subheader(f"自动归类结果 (已归类: {len(st.session_state.axial_codes_df)})")
        if not st.session_state.axial_codes_df.empty:
            edited_df = st.data_editor(
                st.session_state.axial_codes_df,
                column_config={"category": st.column_config.SelectboxColumn("维度", options=dimension_list, required=True)},
                disabled=["code", "reasoning"], num_rows="dynamic", key="auto_editor", height=400
            )
            st.session_state.axial_codes_df = edited_df
        else:
            st.info("点击“🟢 开始”进行自动归类。")
    else:
        st.subheader(f"待审查 (剩余 {len(st.session_state.codes_to_review)} 条)")
        
        if mode == "🔸 半自动模式":
            ready_to_show = [
                c for c in st.session_state.codes_to_review
                if str(c) in [str(k) for k in st.session_state.ai_suggestions.keys()]
            ]
        else:
            ready_to_show = st.session_state.codes_to_review

        if ready_to_show:
            MAX_DISPLAY = 6 
            codes_batch_disp = ready_to_show[:MAX_DISPLAY]
            cols = st.columns(2)
            
            for i, code_name in enumerate(codes_batch_disp):
                # UI 上只显示第一条作为预览，但 AI 看到了聚合的
                quotes = codes_df[codes_df['code'] == code_name]['quote'].tolist()
                quote_preview = quotes[0] if quotes else "无语境"
                
                freq = get_code_frequency(code_name)
                
                suggestion = st.session_state.ai_suggestions.get(code_name, {})
                assigned_category = suggestion.get("category", "无对应维度")
                confidence = suggestion.get("confidence", 0) 
                
                is_ai = (mode == "🔸 半自动模式" and assigned_category in dimension_list)
                
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"### 🏷️ {code_name} `x{freq}`")
                        st.caption(f"引文: {quote_preview}")
                        st.divider()
                        
                        act_l, act_r = st.columns([1, 1])
                        with act_l:
                            if is_ai:
                                try:
                                    score_val = int(confidence)
                                except: score_val = 0
                                
                                score_val = max(0, min(5, score_val))
                                
                                full_s = score_val
                                empty_s = 5 - full_s
                                star_html = f"<span style='color: #FFC107; font-size: 1.2em;'>{'★' * full_s}</span><span style='color: #E0E0E0; font-size: 1.2em;'>{'★' * empty_s}</span>"
                                st.markdown(f"<span style='font-size:0.8em; color:gray'>AI 置信度:</span> {star_html} <span style='font-size:0.9em'>{score_val}/5</span>", unsafe_allow_html=True)

                                st.markdown(f"**{assigned_category}**") 
                                
                                st.button("✅ 接受", key=f"acc_{code_name}", type="primary",
                                          on_click=handle_axial_acceptance,
                                          args=(code_name, assigned_category, score_val, ""))
                            else: st.markdown("*(无建议)*")
                        
                        with act_r:
                            try: default_idx = dimension_list.index(assigned_category) if is_ai else 0
                            except: default_idx = len(dimension_list) - 1 
                            manual_cat = st.selectbox("人工归类", dimension_list, key=f"man_{code_name}", label_visibility="collapsed", index=default_idx)
                            st.button("⬇️ 确认", key=f"man_btn_{code_name}",
                                      on_click=handle_axial_acceptance,
                                      args=(code_name, manual_cat, 5, "人工")) 

            if len(ready_to_show) > MAX_DISPLAY:
                st.info("点击任意按钮加载下一批...")
            
            if mode == "🔸 半自动模式" and st.session_state.is_running_axial:
                 st.caption("🔄 后台正在持续生成建议中...")
                 
        elif not st.session_state.is_running_axial and mode == "🔸 半自动模式" and st.session_state.codes_to_review:
             st.info("暂无AI建议。请点击“🟢 继续/开始”让AI生成建议。")

    if not st.session_state.codes_to_review:
        st.success("🎉 所有待审查代码已处理完毕！")

    st.divider()
    st.subheader("步骤 3: 结果导出")
    if not st.session_state.axial_codes_df.empty:
        st.dataframe(st.session_state.axial_codes_df)

        excel_data = to_excel_axial(
            st.session_state.axial_codes_df,
            st.session_state.open_codes
        )

        cur_topic = st.session_state.get("core_theme", "").strip()
        safe_topic = "".join([c for c in cur_topic if c.isalnum() or c in (" ", "_", "-")]).strip()
        if not safe_topic:
            safe_topic = "Axial_Result"

        time_str = time.strftime("%Y%m%d_%H%M")

        col1, col2 = st.columns([2, 1])
        st.caption("💡 保存用于系统内继续分析；下载用于导出到本地")

        with col1:
            if st.button("💾 保存到项目（用于后续分析）", type="primary", use_container_width=True):
                filename = f"{safe_topic}_{time_str}_axial_coding.xlsx"
                save_path = os.path.join(DIRS["axial_final"], filename)

                with open(save_path, "wb") as f:
                    f.write(excel_data)

                st.success("✅ 已保存到项目文件夹（Step 4）")

        with col2:
            st.download_button(
                "⬇️ 下载副本到本地",
                data=excel_data,
                file_name=f"Project_{safe_topic}_{time_str}_axial_coding.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
# --- 核心处理逻辑 ---
if st.session_state.is_running_axial:
    pending_ai_codes = [c for c in st.session_state.codes_to_review if c not in st.session_state.ai_suggestions]
    
    if not pending_ai_codes and not st.session_state.codes_to_review:
        st.session_state.is_running_axial = False
        st.rerun()
    
    elif pending_ai_codes:
        if mode != "🔺 严格模式":
            batch_codes = pending_ai_codes[:batch_size]
            batch_data = []
            for c in batch_codes:
                # [MODIFIED] 使用聚合引文
                q = get_aggregated_quotes(codes_df, c)
                batch_data.append({'code': c, 'quote': q})
            
            with results_col:
                with st.spinner(f"🤖 正在后台分析 {len(batch_codes)} 条数据..."):
                    messages = create_axial_coding_prompt(dimension_list, batch_data)
                    res = call_qwen_api(st.session_state.api_key, st.session_state.model_id, messages)
                    if res["success"]:
                        st.session_state.total_token_usage += res["tokens"]
                        results = extract_json(res["text"])
                        if isinstance(results, list):
                            for item in results:
                                # 兼容不同字段名
                                c_name = str(
                                    item.get("CodeName")
                                    or item.get("code")
                                    or item.get("Code")
                                    or ""
                                ).strip()

                                category = str(
                                    item.get("AssignedCategory")
                                    or item.get("category")
                                    or item.get("Category")
                                    or "无对应维度"
                                ).strip()

                                confidence = item.get("Confidence")
                                if confidence is None:
                                    confidence = item.get("confidence")
                                if confidence is None:
                                    confidence = item.get("Score")
                                if confidence is None:
                                    confidence = 0

                                try:
                                    confidence = int(confidence)
                                except:
                                    confidence = 0

                                confidence = max(0, min(5, confidence))

                                # 关键：用 codes_to_review 里原始值做模糊对齐，避免空格/全半角/隐藏字符导致匹配失败
                                matched_code = None
                                for raw_code in st.session_state.codes_to_review:
                                    if str(raw_code).strip() == c_name:
                                        matched_code = raw_code
                                        break

                                if matched_code is None:
                                    # 再做一次宽松匹配
                                    norm_c_name = c_name.replace("\u3000", " ").replace("\n", " ").replace("\r", "").strip()
                                    for raw_code in st.session_state.codes_to_review:
                                        norm_raw = str(raw_code).replace("\u3000", " ").replace("\n", " ").replace("\r", "").strip()
                                        if norm_raw == norm_c_name:
                                            matched_code = raw_code
                                            break
                                matched_category = "无对应维度"
                                for dim in dimension_list:
                                    if normalize_text(dim) == normalize_text(category):
                                        matched_category = dim
                                        break

                                st.session_state.ai_suggestions[matched_code] = {
                                    "category": matched_category,
                                    "confidence": confidence,
                                    "reasoning": ""
                                }               

                                if mode == "🔹 自动模式":

                                    handle_axial_acceptance(
                                        matched_code,
                                        matched_category,
                                        confidence,
                                        ""
                                    )

                            st.rerun()
                        else:
                            st.error("⚠️ AI 返回数据格式错误，解析失败。请查看下方原始返回。")
                            with st.expander("🔍 调试：查看 AI 原始返回", expanded=True):
                                st.code(res["text"])
                            st.session_state.is_running_axial = False
                    else:
                        st.error(f"API Error: {res['error']}")
                        st.session_state.is_running_axial = False
        else:
            st.session_state.is_running_axial = False
            st.rerun()
