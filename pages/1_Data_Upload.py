# pages/1_Data_Upload.py (FIXED FOR EXCEL PROVENANCE)
import streamlit as st
import pandas as pd
import io
import docx 

# =======================================================================
# 辅助函数：智能分块读取 (保持不变)
# =======================================================================
CHUNK_SIZE = 800 

def read_docx_chunked(file, chunk_size):
    doc = docx.Document(file); chunks = []; current_chunk = "";
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
        if len(current_chunk) + len(text) < chunk_size:
            current_chunk += text + "\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = text + "\n"
    if current_chunk: chunks.append(current_chunk.strip())
    return pd.DataFrame(chunks, columns=["text_content"])

def read_txt_chunked(file, chunk_size):
    string_data = file.getvalue().decode("utf-8"); lines = string_data.splitlines(); chunks = []; current_chunk = "";
    for line in lines:
        text = line.strip();
        if not text: continue
        if len(current_chunk) + len(text) < chunk_size:
            current_chunk += text + "\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = text + "\n"
    if current_chunk: chunks.append(current_chunk.strip())
    return pd.DataFrame(chunks, columns=["text_content"])

# =======================================================================
# 页面逻辑
# =======================================================================
st.set_page_config(page_title="区域1: 数据上传", layout="wide")
st.title("区域1: 数据上传与合并 (智能分块版) 📂")

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

with st.container(border=True):
    st.subheader("步骤 1: 批量上传访谈文件")
    st.info(f"💡 **Token节约策略已激活**：Word/Txt文件会被合并为 **800字左右的大块**。\n注意：\n若导入excel，请包含'被试编号', 'Participant_ID',或'ID'作为识别被试编号的列\n若导入其他文件类型，将以文件名作为文件来源标记")
    
    col_size, col_upload = st.columns([1, 3])
    with col_size:
        user_chunk_size = st.number_input("合并阈值 (字符数)", min_value=100, max_value=3000, value=800, step=100)

    with col_upload:
        uploaded_files = st.file_uploader(
            "拖拽上传或点击选择 (支持多选)", 
            type=["csv", "txt", "xlsx", "docx"],
            accept_multiple_files=True
        )

if uploaded_files:
    try:
        all_dfs = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            temp_df = pd.DataFrame()
            source_column = 'source_file' # 默认使用文件名

            if uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # Word/TXT 文件使用分块逻辑
                if uploaded_file.type == "text/plain":
                    temp_df = read_txt_chunked(uploaded_file, chunk_size=user_chunk_size)
                else:
                    temp_df = read_docx_chunked(uploaded_file, chunk_size=user_chunk_size)
            
            elif uploaded_file.type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                # CSV/Excel 文件处理 (新逻辑)
                if uploaded_file.type == "text/csv":
                    temp_df = pd.read_csv(uploaded_file)
                else:
                    temp_df = pd.read_excel(uploaded_file)
                
                # 识别被试编号列 (新逻辑)
                id_cols = ['被试编号', 'Participant_ID', 'ID']
                
                found_id_col = next((col for col in id_cols if col in temp_df.columns), None)
                if found_id_col:
                    source_column = found_id_col
                    st.caption(f"✅ 识别到溯源字段: `{source_column}`。")
                else:
                    st.caption(f"⚠️ 未找到溯源字段。将使用文件名 `{file_name}`。")
            
            # 2. 整合数据 (确保有 text_content 列)
            if not temp_df.empty:
                # 优先寻找 text_content，如果 CSV/Excel 没有，则用第一个非 ID 列替代
                text_col = 'text_content'
                if text_col not in temp_df.columns:
                    non_id_cols = [col for col in temp_df.columns if col not in id_cols and col != source_column]
                    if non_id_cols:
                        temp_df.rename(columns={non_id_cols[0]: text_col}, inplace=True)
                        st.caption(f"⚠️ 自动将 `{non_id_cols[0]}` 列识别为文本内容。")
                    else:
                        st.warning(f"文件 {file_name} 无法识别文本内容，已跳过。")
                        continue
                
                temp_df['source_file'] = temp_df[source_column] if source_column != 'source_file' else file_name
                temp_df = temp_df[[text_col, 'source_file']].rename(columns={text_col: 'text_content'})
                all_dfs.append(temp_df)
                st.caption(f"✅ 已加载: `{file_name}` -> 共 **{len(temp_df)}** 条数据。")

        # 3. 最终合并
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            st.divider()
            st.success(f"🎉 处理完成！共合并为 **{len(final_df)}** 条数据。")
            
            with st.container(border=True):
                st.subheader("合并后数据预览")
                cols = ['source_file', 'text_content']
                st.dataframe(final_df[cols], height=400)
            
            st.session_state.raw_data = final_df
            st.button("确认无误，前往步骤2进行编码", type="primary", on_click=lambda: st.switch_page("pages/2_Open_Coding.py"))
        else:
            st.error("未能从上传的文件中提取到有效数据。")

    except Exception as e:
        st.error(f"处理过程中发生错误: {e}")

# (可选) 数据清洗按钮
if st.session_state.raw_data is not None:
    with st.expander("数据清洗工具"):
        if st.button("删除所有空行"):
            old_len = len(st.session_state.raw_data)
            df_cleaned = st.session_state.raw_data.dropna(subset=['text_content'])
            df_cleaned = df_cleaned[df_cleaned['text_content'].str.strip() != ""]
            new_len = len(df_cleaned)
            st.session_state.raw_data = df_cleaned.reset_index(drop=True)
            st.success(f"已删除 {old_len - new_len} 条空数据。")
            st.rerun()
