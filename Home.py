import streamlit as st
import yaml
import os
import time
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import pandas as pd

# =======================================================================
# 1. 页面配置 (必须是第一行)
# =======================================================================
st.set_page_config(
    page_title="扎根理论 AI 编码助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed", # 登录前默认收起侧边栏，更有沉浸感
)

def init_global_state():
    if (
        "axial_codes_df" not in st.session_state
        or not isinstance(st.session_state.axial_codes_df, pd.DataFrame)
    ):
        st.session_state.axial_codes_df = pd.DataFrame(
            columns=['code', 'category', 'confidence', 'reasoning', 'status']
        )

    if "all_unique_codes" not in st.session_state:
        st.session_state.all_unique_codes = []

    if "codes_to_review" not in st.session_state:
        st.session_state.codes_to_review = []

    if "ai_suggestions" not in st.session_state:
        st.session_state.ai_suggestions = {}

    if "total_token_usage" not in st.session_state:
        st.session_state.total_token_usage = 0

init_global_state()

# =======================================================================
# 2. 基础配置加载
# =======================================================================
try:
    with open('config.yaml', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("❌ 找不到 config.yaml 配置文件")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# =======================================================================
# 3. 核心逻辑分流
# =======================================================================

# 检查登录状态
if st.session_state.get("authentication_status"):
    # ------------------------------------------------------------------
    #  ✅ SCENE A: 登录成功 -> 进入【全功能科研模式】
    # ------------------------------------------------------------------
    
    # 1. 强制显示侧边栏 (CSS Hack)
    st.markdown("<style>[data-testid='stSidebar'] {display: block;}</style>", unsafe_allow_html=True)
    
    # 2. 用户数据与文件夹管理
    username = st.session_state.get("username")
    name = st.session_state.get("name")
    
    if username:
        user_root_dir = os.path.join("users_data", username)
        if not os.path.exists(user_root_dir):
            os.makedirs(user_root_dir, exist_ok=True)
            os.makedirs(os.path.join(user_root_dir, "recovery"), exist_ok=True)
        st.session_state['user_root_dir'] = user_root_dir

    # 3. 侧边栏内容
    with st.sidebar:
        st.write(f"👋 欢迎回来, **{name}**")
        st.caption(f"📂 工作区: {username}")
        authenticator.logout(location='sidebar')
        st.divider()

    # 4. 初始化 API Key
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""

    # =======================================================================
    #  🎨 (CRITICAL) 恢复你的莫兰迪美化 CSS
    # =======================================================================
    MORANDI_CSS = """
    <style>
    /* 1. 全局字体放大 */
    html, body, [class*="st-"], p, div, span, h1, h2, h3, h4, h5, h6 {
        font-size: 1.15rem !important; 
    }
    h1 { font-size: 2.5rem !important; }
    h2, [data-testid="stHeading"] { font-size: 2.0rem !important; }
    h3 { font-size: 1.7rem !important; }

    /* 2. 背景色恢复 (覆盖掉登录页的背景) */
    [data-testid="stAppViewContainer"] {
        background-image: none;
        background-color: #FDFBF5; 
    }
    [data-testid="stSidebar"] { background-color: #F8F5F0; }

    /* 3. 卡片式布局 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF; 
        border: 1px solid #EAE6DD; 
        border-radius: 10px;      
        padding: 30px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        margin-bottom: 20px; 
    }

    /* 4. 按钮美化 */
    .stButton > button[kind="primary"] {
        height: 4.0rem; width: 100%; font-size: 1.3rem !important; font-weight: bold;
        background-color: #B0C4DE; color: #FFFFFF; border: none; border-radius: 8px;
    }
    .stButton > button:not([kind="primary"]) {
        height: 3.0rem; width: 100%; font-size: 1.1rem !important; 
        border: 1px solid #B0C4DE; color: #B0C4DE; border-radius: 8px;
    }

    /* 5. Tabs 美化 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; padding: 0; }
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem; font-size: 1.2rem; border-radius: 5px 5px 0 0; background-color: #F0F0F0; border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; border-top: 3px solid #B0C4DE; color: black;
    }
    </style>
    """
    st.markdown(MORANDI_CSS, unsafe_allow_html=True)

    # ==============================================================================
    #  📝 恢复你的丰富内容：标题与简介
    # ==============================================================================
    st.title("🧠 扎根理论 AI 编码助手")
    st.caption("Grounded Theory AI Coding Assistant | Design for Research")

    with st.container(border=True):
        st.info("""
        **欢迎使用！** 这是一个基于大语言模型（LLM）的定性研究辅助工具，旨在帮助研究者高效、严谨地完成扎根理论（Grounded Theory）的编码工作。
        它结合了 **AI 的处理速度** 与 **研究者的专业判断**，通过人机协作完成从原始文本到理论构建的过程。
        """)

    # ==============================================================================
    #  📝 恢复你的丰富内容：研究工作流
    # ==============================================================================
    st.header("🗺️ 研究工作流")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("#### 1. 数据上传 (Upload)")
            st.markdown("**📍 Page 1**")
            st.markdown("""
            * 📥 上传 Word/Txt/Excel
            * ✂️ 智能分块 (Chunking)
            * 🧹 数据清洗
            """)

    with col2:
        with st.container(border=True):
            st.markdown("#### 2. 开放性编码 (Open)")
            st.markdown("**📍 Page 2**")
            st.markdown("""
            * 🔍 **原子化拆分**
            * 📝 初始概念生成
            * 💾 流式存档 (防丢失)
            """)

    with col3:
        with st.container(border=True):
            st.markdown("#### 3 清洗对齐")
            st.caption("📍 Page 3 ")
            st.markdown("""
            * 🤝 **队友分歧对齐**
            * 🧹 **同义词合并**
            * 🧱 **积木归类** (Axial)
            """)

    with col4:
        with st.container(border=True):
            st.markdown("#### 4. 轴心编码 (Axial)")
            st.markdown("**📍 Page 4**")
            st.markdown("""
            * 🔗 **聚类与范畴化**
            * ⚖️ 5星置信度评分
            * 📤 全量数据导出
            """)

    # ==============================================================================
    #  📝 恢复你的丰富内容：操作指南
    # ==============================================================================
    st.header("📘 操作指南 (User Guide)")

    tab_open, tab_visualanalyze, tab_axial, tab_tips = st.tabs(["🔍 开放性编码指南", "🧩 清洗与对齐","🔗 轴心编码指南", "💡 专家提示"])

    # --- 开放性编码指南 ---
    with tab_open:
        with st.container(border=True):
            st.subheader("如何进行：开放性编码 (Opening Coding)")
            st.markdown("""
            **目标**：将原始访谈/文本资料打散，提炼出最基础的“概念（Concepts）”。
            
            #### 🛠️ 核心步骤
            1.  **配置判别标准**：
                * 在左侧输入您的 **研究主题**（如：*青少年网络成瘾*）。
                * 点击 **"🤖 一键生成判别标准"**，AI 会自动生成“纳入标准”和“排除标准”。这能防止 AI 跑题。
            2.  **选择模式**：
                * 推荐使用 **"2.外部辅助（大模型网页端）"** 模式。
            3.  **执行编码**：
                * 点击 **"🚀 批量处理"**。程序会自动扫描未处理的行。
                * **智能跳过**：如果中途停止，下次无需重跑，系统会自动跳过已处理的数据。
            4.  **结果审查**：
                * AI 会将长难句拆解为多个短语（Code），并保留原文（Quote）。
            
            #### 🛡️ 数据安全机制
            * **实时保存**：每处理完一行，系统就会自动保存到 `recovery_opening_coding` 文件夹。
            * **断点续传**：如果网页崩溃，刷新后在侧边栏选择最新的 JSONL 文件，点击 **"🔄 载入"** 即可恢复现场。
            """)

    # --- 清洗与对齐 ---
    with tab_visualanalyze:
        with st.container(border=True):
            st.subheader("Step 3: 清洗与对齐 (净化数据)")
            st.markdown("""
            **一、连接“发散”与“收敛”的关键中间站：**
            1.  **队友对齐**：上传队友文件，AI 基于引文指纹自动匹配，解决编码分歧。
            2.  **同义合并**：AI 扫描“开心/高兴”等近义词，一键标准化命名。
            3.  **全量溯源**：底部清单完整保留【原始 vs 清洗后】对比，确保严谨。
                        
            **二、分类合并**：
            1.  **定维度**：输入领域主题，AI 辅助生成维度的学术定义。
            2.  **玩积木**：在 **"积木工作台"** 将开放编码拖拽归类，直观高效。
            3.  **全导出**：导出结果包含**引用频次**、**原始引文**及**置信度**。
            """)

    # --- 轴心编码指南 ---
    with tab_axial:
        with st.container(border=True):
            st.subheader("如何进行：轴心编码 (Axial Coding)")
            st.markdown("""
            **目标**：发现概念之间的联系，将散乱的编码归纳为更高层级的“范畴（Categories/Dimensions）”。
            
            #### 🛠️ 核心步骤
            1.  **加载数据**：系统会自动加载第二步生成的开放编码。
            2.  **定义核心维度**：
                * 使用 **"✨ AI 辅助生成定义"** 功能。
                * 输入您的**研究领域**和**主题**，AI 会为每个维度生成精准的学术定义（操作性定义）。
                * *为什么这很重要？* AI 需要根据定义（而不仅仅是名字）来判断归类。
            3.  **执行归类**：
                * 推荐 **"🔸 半自动模式"**。
                * AI 会给出建议（Category）和置信度（1-5 星 ★★★★★）。
                * 您点击 **"✅ 接受"** 或手动修改。
            4.  **全量导出**：
                * 导出时，系统会将“分类规则”映射回所有原始数据中，保留频率统计和所有引文。
            
            #### 🧠 隐式思维链
            * 即使不显示理由，AI 也在后台进行了**“不断比较（Constant Comparative）”**，对比不同维度的定义，确保分类准确。
            """)

    # --- 专家提示 ---
    with tab_tips:
        with st.container(border=True):
            st.subheader("💡 提高效率的专家建议")
            st.markdown("""
            1.  **不要随意修改文件名**：
                * 断点续传功能依赖于原始数据的行号（Index）。如果在 Excel 中删除了行或改变了顺序，旧的存档可能无法正确匹配。
                
            2.  **善用 "无对应维度"**：
                * 在轴心编码中，如果 AI 发现某个编码无法归入您给定的任何维度，它会归类为“无对应维度”。
                * 这通常意味着您发现了一个**新的范畴**，或者是数据中的噪音。
                
            3.  **关注置信度 (Confidence)**：
                * **5星 (★★★★★)**：AI 非常确定，通常可以直接接受。
                * **1-2星 (★)**：AI 拿不准，建议您仔细查看引文人工判断。
                
            4.  **Token 成本**：
                * 开放性编码最消耗 Token（因为要处理海量原文）。
                * 轴心编码相对便宜（只处理短语）。
                * 请留意右上角的 **Token 计数器**。
            """)

    st.divider()
    st.markdown("#### **请从左侧的““导航菜单””中选择一个步骤开始您的工作：**")

else:
    # ------------------------------------------------------------------
    #  ❌ SCENE B: 未登录 -> 中文登录/注册页
    # ------------------------------------------------------------------

    LOGIN_CSS = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f7f7fb;
        background-image:
            linear-gradient(90deg, #eceff5 1px, transparent 1px),
            linear-gradient(#eceff5 1px, transparent 1px);
        background-size: 36px 36px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    input[type="text"], input[type="password"], input[type="email"] {
        border-radius: 12px !important;
        padding: 12px !important;
        border: 1px solid #d9dce3 !important;
        background-color: #ffffff !important;
    }

    div.stButton > button {
        border-radius: 999px !important;
        height: 3rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        width: 100%;
    }

    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-bottom: 16px;
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 8px 18px;
    }

    .login-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #ebecef;
        border-radius: 20px;
        padding: 28px 28px 10px 28px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        backdrop-filter: blur(6px);
    }
    </style>
    """
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)

    c1, c_login, c2 = st.columns([1, 2, 1])

    with c_login:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-card">
                <h1 style='text-align:center; color:#2f3441; margin-bottom:6px;'>
                    🧠 扎根理论 AI 编码助手
                </h1>
                <p style='text-align:center; color:#7b8190; margin-bottom:24px;'>
                    智能扎根理论辅助系统
                </p>
            """,
            unsafe_allow_html=True
        )

        tab_login, tab_register = st.tabs(["登录", "注册"])

        # ---------------------------
        # 登录
        # ---------------------------
        with tab_login:
            st.markdown("##### 欢迎回来")
            st.caption("请输入账号和密码登录系统")

            authenticator.login(
                location='main',
                key='login_form_cn',
                fields={
                    'Form name': '用户登录',
                    'Username': '用户名',
                    'Password': '密码',
                    'Login': '立即登录',
                    # 如果你没开启 captcha，这个键留着也没关系
                    'Captcha': '验证码'
                }
            )

            if st.session_state.get("authentication_status") is False:
                st.error("❌ 用户名或密码不正确，请重新输入。")
            elif st.session_state.get("authentication_status") is None:
                st.info("👉 请输入用户名和密码。")

        # ---------------------------
        # 注册
        # ---------------------------
        with tab_register:
            st.markdown("##### 创建新账号")
            st.caption("注册时必须填写邮箱，便于找回账号与后续通知")

            try:
                # 说明：
                # 1) fields：中文化
                # 2) captcha=False：去掉验证码（你想保留的话改成 True）
                # 3) password_hint=False：去掉密码提示（更简洁）
                # 4) merge_username_email=False：用户名与邮箱分开
                #    如果你想“用户名直接用邮箱”，改成 True
                email, username_reg, name_reg = authenticator.register_user(
                    location='main',
                    key='register_form_cn',
                    captcha=False,
                    password_hint=False,
                    merge_username_email=False,
                    clear_on_submit=False,
                    fields={
                        'Form name': '新用户注册',
                        'First name': '姓名',
                        'Last name': '称呼/昵称（可选）',
                        'Email': '邮箱（必填）',
                        'Username': '用户名',
                        'Password': '密码',
                        'Repeat password': '确认密码',
                        'Password hint': '密码提示',
                        'Captcha': '验证码',
                        'Register': '立即注册'
                    }
                )

                # 注册成功
                if email and username_reg and name_reg:
                    # 二次保险：如果邮箱为空，就不给通过
                    if not str(email).strip():
                        st.error("❌ 注册失败：邮箱不能为空。")
                        st.stop()

                    # 保存配置
                    with open('config.yaml', 'w', encoding='utf-8') as file:
                        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

                    # 创建用户目录
                    user_folder = os.path.join("users_data", username_reg)
                    if not os.path.exists(user_folder):
                        os.makedirs(user_folder, exist_ok=True)
                        os.makedirs(os.path.join(user_folder, "recovery"), exist_ok=True)

                    st.success(f"🎉 注册成功，欢迎你，{name_reg}！")
                    st.info(f"你填写的邮箱是：{email}")
                    time.sleep(1.2)
                    st.rerun()

            except Exception as e:
                msg = str(e)

                # 对常见英文报错做中文翻译
                if "Email" in msg and ("empty" in msg.lower() or "required" in msg.lower()):
                    st.error("❌ 邮箱不能为空，请填写有效邮箱。")
                elif "Passwords do not match" in msg:
                    st.error("❌ 两次输入的密码不一致。")
                elif "already exists" in msg.lower():
                    st.error("❌ 用户名或邮箱已存在，请更换后重试。")
                elif "not valid" in msg.lower() and "email" in msg.lower():
                    st.error("❌ 邮箱格式不正确，请检查后重试。")
                elif "unpack" not in msg and "NoneType" not in msg:
                    st.error(f"注册失败：{msg}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<br><br><p style='text-align: center; color: #b8bdc9; font-size: 0.85rem;'>© 2025 Grounded Theory AI | Secure & Private</p>",
        unsafe_allow_html=True
    )
