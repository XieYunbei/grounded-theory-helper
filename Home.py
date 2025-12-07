import streamlit as st

# 1. 初始化 Session State (保留你原来的逻辑，防止报错)
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# 2. 页面基础配置
st.set_page_config(
    page_title="扎根理论 AI 编码助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================================================================
# --- (CRITICAL) 全局 ““大”” 美化 CSS (保留你的莫兰迪风格) ---
# =======================================================================
MORANDI_CSS = """
<style>
/* 1. (CRITICAL) 全局““大””字体 */
html, body, [class*="st-"], p, div, span, h1, h2, h3, h4, h5, h6 {
    font-size: 1.15rem !important; /* 将基础字体放大到 115% */
}
h1 {
    font-size: 2.5rem !important; /* 标题 H1 */
}
h2, [data-testid="stHeading"] {
    font-size: 2.0rem !important; /* 标题 H2 和 st.title */
}
h3 {
    font-size: 1.7rem !important; /* 标题 H3 和 st.subheader */
}

/* 2. 莫兰迪色系 - 整体背景 (米灰色) */
[data-testid="stAppViewContainer"] {
    background-color: #FDFBF5; 
}

/* 3. 莫兰迪色系 - 导航栏 (原侧边栏) */
[data-testid="stSidebar"] {
    background-color: #F8F5F0;
}

/* 4. 卡片式布局 (Streamlit 的 st.container(border=True) 会生成这个 Wrapper) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF; 
    border: 1px solid #EAE6DD; 
    border-radius: 10px;      
    padding: 30px; /* 增大卡片内部留白 */
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
    margin-bottom: 20px; /* 卡片间距 */
}

/* 5. (CRITICAL) ““大””按钮 (主要按钮) */
.stButton > button[kind="primary"] {
    height: 4.0rem; /* 显著增大按钮高度 */
    width: 100%;    
    font-size: 1.3rem !important; /* 增大按钮字体 */
    font-weight: bold;
    background-color: #B0C4DE; 
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}
.stButton > button[kind="primary"]:hover {
    background-color: #9CA8B8;
}

/* 6. (CRITICAL) ““大””按钮 (次要/普通按钮) */
.stButton > button:not([kind="primary"]) {
    height: 3.0rem; /* 普通按钮也增大 */
    width: 100%; /* 也让它占满容器宽度，更整洁 */
    font-size: 1.1rem !important; /* 增大字体 */
    border: 1px solid #B0C4DE; /* 统一使用主题色边框 */
    color: #B0C4DE;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
}
.stButton > button:not([kind="primary"]):hover {
    background-color: #F8F5F0;
    color: #9CA8B8;
    border-color: #9CA8B8;
}

/* 7. 针对 Tabs 的美化 (适配新内容) */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 3.5rem;
    font-size: 1.2rem;
    border-radius: 5px 5px 0 0;
    background-color: #F0F0F0;
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF;
    border-top: 3px solid #B0C4DE;
}

</style>
"""
st.markdown(MORANDI_CSS, unsafe_allow_html=True)

# ==============================================================================
# 3. 标题与简介 (内容部分)
# ==============================================================================
st.title("🧠 扎根理论 AI 编码助手")
st.caption("Grounded Theory AI Coding Assistant | Design for Research")

# 使用你的卡片样式包裹简介
with st.container(border=True):
    st.info("""
    **欢迎使用！** 这是一个基于大语言模型（LLM）的定性研究辅助工具，旨在帮助研究者高效、严谨地完成扎根理论（Grounded Theory）的编码工作。
    它结合了 **AI 的处理速度** 与 **研究者的专业判断**，通过人机协作完成从原始文本到理论构建的过程。
    """)

# ==============================================================================
# 4. 研究流程概览 (Workflow)
# ==============================================================================
st.header("🗺️ 研究工作流")

# 这里使用 st.container(border=True) 会自动应用你的“卡片式布局”CSS
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### 1. 数据上传 (Upload)")
        st.markdown("**📍 Page 1**")
        st.markdown("""
        * 📥 上传 Word/Txt/Excel
        * ✂️ 智能分块 (Chunking)
        * 🧹 数据清洗
        """)
        # 这里无需按钮，直接用侧边栏导航

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
        st.markdown("#### 3. 轴心编码 (Axial)")
        st.markdown("**📍 Page 3**")
        st.markdown("""
        * 🔗 **聚类与范畴化**
        * ⚖️ 5星置信度评分
        * 📤 全量数据导出
        """)

# ==============================================================================
# 5. 详细操作指南 (使用 Tabs)
# ==============================================================================
st.header("📘 操作指南 (User Guide)")

# 你的 CSS 会自动放大这些 Tab 的字体
tab_open, tab_axial, tab_tips = st.tabs(["🔍 开放性编码指南", "🔗 轴心编码指南", "💡 专家提示"])

# --- 开放性编码指南 ---
with tab_open:
    with st.container(border=True): # 再次包裹以应用白色背景卡片样式
        st.subheader("如何进行：开放性编码 (Opening Coding)")
        st.markdown("""
        **目标**：将原始访谈/文本资料打散，提炼出最基础的“概念（Concepts）”。
        
        #### 🛠️ 核心步骤
        1.  **配置判别标准**：
            * 在左侧输入您的 **研究主题**（如：*青少年网络成瘾*）。
            * 点击 **"🤖 一键生成判别标准"**，AI 会自动生成“纳入标准”和“排除标准”。这能防止 AI 跑题。
        2.  **选择模式**：
            * 推荐使用 **"1. 智能向导 (全自动)"** 模式。
        3.  **执行编码**：
            * 点击 **"🚀 批量处理"**。程序会自动扫描未处理的行。
            * **智能跳过**：如果中途停止，下次无需重跑，系统会自动跳过已处理的数据。
        4.  **结果审查**：
            * AI 会将长难句拆解为多个短语（Code），并保留原文（Quote）。
        
        #### 🛡️ 数据安全机制
        * **实时保存**：每处理完一行，系统就会自动保存到 `recovery_opening_coding` 文件夹。
        * **断点续传**：如果网页崩溃，刷新后在侧边栏选择最新的 JSONL 文件，点击 **"🔄 载入"** 即可恢复现场。
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
st.info("ℹ️ (界面已更新) 所有的设置（如API Key）和操作按钮都已移入相应的工作页面中，侧边栏仅用于页面导航和进度恢复。")