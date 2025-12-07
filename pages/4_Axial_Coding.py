import streamlit as st
import pandas as pd
import time
from openai import OpenAI
import json
import os
import glob
from datetime import datetime
from io import BytesIO

# =======================================================================
# 0. 数据持久化与恢复模块 (Data Persistence)
# =======================================================================

RECOVERY_DIR = "recovery_axial_coding"

def ensure_recovery_dir():
    if not os.path.exists(RECOVERY_DIR):
        os.makedirs(RECOVERY_DIR)

def get_current_filename(topic, mode):
    """
    生成文件名：主题_模式_日期.jsonl
    """
    safe_topic = "".join([c for c in topic if c.isalnum() or c in (' ', '_', '-')]).strip()
    if not safe_topic: safe_topic = "Untitled"
    
    safe_mode = "Auto" if "自动" in mode else "Semi" if "半自动" in mode else "Strict"
    date_str = datetime.now().strftime("%Y%m%d") 
    
    return f"{safe_topic}_{safe_mode}_{date_str}.jsonl"

def save_record_to_jsonl(record_dict, filename):
    ensure_recovery_dir()
    filepath = os.path.join(RECOVERY_DIR, filename)
    record_dict['timestamp'] = datetime.now().isoformat()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

def load_from_jsonl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
    if data:
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

# =======================================================================
# 1. 核心逻辑函数区
# =======================================================================

def call_qwen_api(api_key, model_id, messages, temperature=0.1):
    try:
        # 兼容多平台 API 调用
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
            messages=messages,
        )
        usage = response.usage
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0)
        else:
            total_tokens = 0
            
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
    
    if original_df is not None and not original_df.empty and not axial_mapping_df.empty:
        if 'code' in original_df.columns and 'code' in axial_mapping_df.columns:
            # 准备映射表 (取最新规则)
            mapping_rules = axial_mapping_df.drop_duplicates(subset=['code'], keep='last')
            cols_to_use = [c for c in mapping_rules.columns if c in ['code', 'category', 'confidence', 'reasoning', 'status']]
            mapping_rules = mapping_rules[cols_to_use]
            
            # Left Join
            merged_df = pd.merge(original_df, mapping_rules, on='code', how='left')
            merged_df['category'] = merged_df['category'].fillna('待归类')
            final_df = merged_df
        else:
            final_df = axial_mapping_df
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

def create_axial_coding_prompt(dimension_list, batch_data):
    """
    构建符合扎根理论逻辑的 Prompt
    batch_data: [{'code': '...', 'quote': '...'}] (quote 可能是拼接后的多条)
    """
    dims_display = list(dimension_list)
    if "无对应维度" not in dims_display:
        dims_display.append("无对应维度: 该编码无法归入上述任何维度，属于离群点或需要新维度。")
    
    dims_str = "\n".join([f"- {d}" for d in dims_display])
    
    system_content = f"""
你是一位执行“轴心编码（Axial Coding）”的质性研究助手。你的任务是将底层的“开放编码”归纳到核心维度中。

【一、编码手册 (Codebook)】
请严格基于以下维度的**操作性定义**进行分类，严禁仅凭维度名称猜测：
{dims_str}

【二、操作逻辑：不断比较法 (Constant Comparative Method)】
虽然你只需输出结果，但请在计算过程中严格执行以下步骤：
1. **情境还原**：仔细阅读引文（Quote）。若引文包含多条，请综合考虑其共性。若引文缺失或模糊，**下调置信度**。
2. **竞争性假设**：对于每条数据，不要只看它“像”什么，要反问它“为什么不是”其他维度。
3. **排他性判断**：如果一条数据同时符合两个维度的定义，选择**语义对应更直接**的那个。

【三、置信度评分量表 (1-5)】
5: **理论饱和**。编码与定义的关键词完全对应，且引文语境提供了强有力支撑。
4: **高度匹配**。逻辑通顺，无明显歧义。
3: **中度匹配**。符合核心定义，但缺乏语境细节，或存在多义性。
2: **证据不足**。仅有微弱联系，建议人工复核。
1: **无法判断**。信息缺失或完全不相关。

【四、输出格式】
仅输出 JSON 数组。不要解释，不要 Markdown。

[
    {{
        "CodeName": "...",
        "AssignedCategory": "...",
        "Confidence": 5
    }}
]
"""
    data_input_str = ""
    for item in batch_data:
        c = item.get('code', '未知')
        q = item.get('quote', '')
        if not q or q == "无" or q == "（无引用）":
            q_str = "（无语境，仅基于编码分析）"
        else:
            q_str = q
        data_input_str += f"- 编码: {c}\n  引文: {q_str}\n\n"

    user_content = f"请对以下 {len(batch_data)} 条数据进行编码归类，直接返回 JSON 数组：\n\n{data_input_str}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

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
    current_topic = st.session_state.get('research_topic_input', 'Unspecified_Topic')
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
    st.header("📂 进度管理")
    st.info("系统会自动将您的编码结果保存到 `recovery_axial_coding` 文件夹中。")
    
    ensure_recovery_dir()
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

st.title("区域4: 轴心编码 Prompt生成与执行区 🧠")

if 'open_codes' not in st.session_state: st.session_state.open_codes = None
if 'api_key' not in st.session_state: st.session_state.api_key = None
if 'openai_key' not in st.session_state: st.session_state.openai_key = "" 
if 'gemini_key' not in st.session_state: st.session_state.gemini_key = "" 
if 'selected_model' not in st.session_state: st.session_state.selected_model = 'qwen-plus' 
if 'axial_codes_df' not in st.session_state:
    st.session_state.axial_codes_df = pd.DataFrame(columns=['code', 'category', 'confidence', 'reasoning', 'status'])
if 'codes_to_review' not in st.session_state: st.session_state.codes_to_review = []
if 'ai_suggestions' not in st.session_state: st.session_state.ai_suggestions = {} 
if 'is_running_axial' not in st.session_state: st.session_state.is_running_axial = False
if 'total_token_usage' not in st.session_state: st.session_state.total_token_usage = 0
if 'dims_input_text' not in st.session_state: st.session_state.dims_input_text = "情绪识别\n情绪调节\n社会支持"
if 'research_topic_input' not in st.session_state: st.session_state.research_topic_input = "" 

# --- 数据加载 ---
codes_df = None
if st.session_state.open_codes is not None and not st.session_state.open_codes.empty:
    codes_df = st.session_state.open_codes

if codes_df is None:
    st.warning("⚠️ 请先在 Page 2 生成开放编码，或在此上传文件。")
    uploaded_file = st.file_uploader("📥 上传开放编码文件 (XLSX, CSV, JSON, JSONL)", type=["xlsx", "csv", "json", "jsonl"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): codes_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.jsonl'): codes_df = pd.read_json(uploaded_file, lines=True)
            elif uploaded_file.name.endswith('.json'):
                try: codes_df = pd.read_json(uploaded_file)
                except ValueError: uploaded_file.seek(0); codes_df = pd.read_json(uploaded_file, lines=True)
            else: codes_df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            if 'code' not in codes_df.columns: st.error("错误：缺少 'code' 列"); st.stop()
            if 'quote' not in codes_df.columns: codes_df['quote'] = "（无引用）"
            st.session_state.open_codes = codes_df
            st.success(f"✅ 加载成功: {len(codes_df)} 条"); st.rerun()
        except Exception as e: st.error(f"读取失败: {e}"); st.stop()

if codes_df is None: st.stop()

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
        
        api_key_input = st.text_input("🔑 DashScope Key", type="password", value=st.session_state.get('api_key', ''), label_visibility="visible")
        if api_key_input: st.session_state.api_key = api_key_input
        
        model_options = {"👑 Qwen-Max": "qwen-max", "🔥 DeepSeek-V3": "deepseek-v3", "⚖️ Qwen-Plus": "qwen-plus", "🚀 DeepSeek-R1": "deepseek-r1", "🌟 GPT-4o": "gpt-4o"}
        model_keys = list(model_options.keys())
        current_key = next((k for k, v in model_options.items() if v == st.session_state.selected_model), model_keys[0])
        sel_label = st.selectbox("🧠 选择模型", options=model_keys, index=model_keys.index(current_key))
        st.session_state.selected_model = model_options[sel_label]
        st.session_state.model_id = st.session_state.selected_model

        st.divider()
        st.markdown("#### 定义轴心维度")
        
        with st.expander("✨ AI 辅助生成定义 (推荐)", expanded=False):
            st.caption("为了让 AI 生成精准的定义，请补充以下背景信息：")
            
            col_ctx1, col_ctx2 = st.columns(2)
            input_domain = col_ctx1.text_input("1. 研究领域", placeholder="例如：发展心理学")
            input_topic = col_ctx2.text_input("2. 研究主题", placeholder="例如：青少年叛逆期冲突")
            if input_topic: st.session_state.research_topic_input = input_topic
            
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

        dimension_list = [line.split(":")[0].strip() for line in dimensions_input.splitlines() if line.strip()]
        if '无对应维度' not in dimension_list: dimension_list.append('无对应维度')
        
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
            ready_to_show = [c for c in st.session_state.codes_to_review if c in st.session_state.ai_suggestions]
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
        
        export_data = to_excel_axial(st.session_state.axial_codes_df, st.session_state.open_codes)
        
        # [MODIFIED] 动态文件名
        cur_topic = st.session_state.get('research_topic_input', 'Research')
        safe_topic = "".join([c for c in cur_topic if c.isalnum() or c in (' ', '_', '-')]).strip()
        if not safe_topic: safe_topic = "Axial_Result"
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"{safe_topic}_{date_str}.xlsx"
        
        st.download_button("💾 导出结果 (含原始行)", data=export_data, file_name=file_name)

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
                                c_name = item.get("CodeName")
                                category = item.get("AssignedCategory")
                                confidence = item.get("Confidence", 0) 
                                
                                st.session_state.ai_suggestions[c_name] = {
                                    "category": category,
                                    "confidence": confidence,
                                    "reasoning": ""
                                }
                                
                                if mode == "🔹 自动模式":
                                    handle_axial_acceptance(c_name, category, confidence, "")
                                    
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
