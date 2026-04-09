import json
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


def build_meaning_unit_prompt(core_theme, definition_logic, exclusion_logic, batch_text):
    return f"""你是严谨的扎根理论研究助手。你的任务是：在研究主题约束下，从已原子化切分的访谈文本中归并独立意义单元。

规则：
1. 只处理 A- 开头的行；Q- 开头的行仅用于理解语境，绝不能进入输出。
2. 只保留符合纳入标准且不属于排除标准的内容。
3. 意义单元的边界是“最小但完整的主题相关意思”。
4. 默认先判断相邻 A 行能否共同构成同一个主题相关的完整意思；只有当出现新的独立意思时，才拆分为新单元。
5. 在不混淆不同独立意义的前提下，优先使用更少的单元表示文本。
6. 同一 A 行若包含多个彼此独立的主题相关意思，必须拆成多个单元。
7. 相邻 A 行若存在补充说明、因果链条、做法与结果、判断与依据、感受与原因、转折限定、递进深化、举例展开，通常应合并。
8. 若出现新的独立动作、判断、感受、策略，或并列成分各自可独立编码，通常应拆开。
9. 重点检查是否遗漏本应合并的相邻 A 行，不要默认一行一个单元。
10. 只输出 JSON 数组，不输出解释、摘要或引文。

核心焦点：{core_theme}
纳入标准：{definition_logic}
排除标准：{exclusion_logic}

输出格式：
[
  {{
    "unit_id": "U001",
    "ids": ["A-01-003", "A-01-004"],
  }}
]

待处理文段：
{batch_text}
"""

def create_final_coding_prompt1(core_theme, definition_logic, exclusion_logic, batch_text):
    return f"""
你是严谨的扎根理论研究专家。你正在处理“已经完成独立意义划分”的访谈数据。每一行文本都带有唯一 ID，并且每一行已经是一个可独立编码的意义单元。

你的任务是：对提供的[待处理文段]进行开放性编码。

一、核心焦点
{core_theme}

二、判别标准
- 纳入标准：{definition_logic}
- 排除标准：{exclusion_logic}

三、身份协议（必须严格执行）
- 每一行文本都带有唯一 ID，例如 [A-01-001]。
- 如果出现 [Q-...] 开头的行，它们仅作为背景信息，严禁编码。
- 只有 [A-...] 开头的行可以被编码。

四、关键前提（必须严格执行）
1. 当前输入中的每一行已经是一个独立意义单元。
2. 你不得再跨行合并、重组或打包多个 ID。
3. 你只能基于单个 ID 对应的那一行文本进行编码。
4. 同一个 ID 若确实包含多个彼此独立、且都符合纳入标准的意义，可以生成多条不同的 code。
5. 不得因为语气词、重复表达、口头禅、补充说明而机械拆出多条 code。
6. 不得根据猜测、脑补、常识推演补足文本中没有明确表达的意思。

五、编码原则
1. code 必须贴地、具体、低抽象。
2. 优先使用简短的动作短语、状态短语、感受短语。
3. 不得使用理论化、概括化、总结腔标签。
4. 不得把原文整句改写成长句式标签。
5. 同一意义不要用近义词重复编码。
6. 若某行不符合纳入标准，则不编码。

六、confidence 评分标准
1 = 高度不稳，原文证据不足
2 = 较不稳，存在明显歧义
3 = 基本可判定，但仍有歧义
4 = 较稳，原文支持较明确
5 = 非常稳，原文支持直接明确

七、输出格式
只输出一个 JSON 数组，每个对象必须包含：
- code
- ids
- confidence

注意：
- ids 必须是输入中单个独立意义单元的 ID 列表，但每条编码只能对应一个 ID
- 也就是说，每个对象的 ids 必须形如 ["A-01-005"]
- 同一个 ID 可以出现多次，对应多条不同 code
- 严禁把多个不同 ID 放进同一个编码对象

示例：
[
  {{
    "code": "担心做错",
    "ids": ["A-01-005"],
    "confidence": 5
  }},
  {{
    "code": "反复确认要求",
    "ids": ["A-01-005"],
    "confidence": 4
  }},
  {{
    "code": "先看别人做法",
    "ids": ["A-01-006"],
    "confidence": 5
  }}
]

[待处理文段]:
{batch_text}

提醒：严格按照规定 JSON 输出，不输出任何其他内容。
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
    * 在确定 ids 后，进一步判断为了让该编码可被人工理解，最少还需要哪些相邻 A 行，并将其写入 evidence_span
3.穷尽性审计：
    * 重新核对：将你生成的初始代码列表与[待处理文段]进行对比。
    * 检查遗漏：检查原始文段中是否还有任何符合纳入标准的、但未被编码的并列词、转折句或对立概念（例如：既要A又要B）。
    * 补充：如果发现遗漏，请立即补充完整。
4.提炼与命名：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行原则一（语义纯化）和原则四（贴地性原则），生成最终 Code。
5.零引文：不要返回原文 Quote，仅返回 IDs。
6.进行置信度confidence评分：进行五点评分，1分为非常不确定，2分为比较确定，3分为有点确定，4分为比较确定，5分为非常确定。
7.格式化：生成JSON。

六、输出格式
只输出一个JSON数组，每个对象必须包含 code 、ids、evidence_span 和 confidence。
多条编码示例:
[
  {{
    "code": "(第一个编码标签)",
    "ids": ["A-01-005", "A-01-006"],
    "evidence_span": ["A-01-004", "A-01-005", "A-01-006"],
    "confidence": 5
  }},
  {{
    "code": "(第二个编码标签)",
    "ids": ["A-01-006"],
    "evidence_span": ["A-01-005", "A-01-006"],
    "confidence": 4
  }}
]
零条编码示例: []

[待处理文段]:
{batch_text}
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
    * 在确定 ids 后，进一步判断为了让该编码可被人工理解，最少还需要哪些相邻 A 行，并将其写入 evidence_span
3.穷尽性审计：
    * 重新核对：将你生成的初始代码列表与[待处理文段]进行对比。
    * 检查遗漏：检查原始文段中是否还有任何符合纳入标准的、但未被编码的并列词、转折句或对立概念（例如：既要A又要B）。
    * 补充：如果发现遗漏，请立即补充完整。
4.提炼与命名：对所有代码执行剥离外壳，保留内核，并进行净化提炼。对每个意义单元，执行原则一（语义纯化）和原则四（贴地性原则），生成最终 Code。
5.零引文：不要返回原文 Quote，仅返回 IDs。
6.进行置信度confidence评分：进行五点评分，1分为非常不确定，2分为比较确定，3分为有点确定，4分为比较确定，5分为非常确定。
7.格式化：生成JSON。

六、输出格式
只输出一个JSON数组，每个对象必须包含 code 、ids、evidence_span 和 confidence。
多条编码示例:
[
  {{
    "code": "(第一个编码标签)",
    "ids": ["A-01-005", "A-01-006"],
    "evidence_span": ["A-01-004", "A-01-005", "A-01-006"],
    "confidence": 5
  }},
  {{
    "code": "(第二个编码标签)",
    "ids": ["A-01-006"],
    "evidence_span": ["A-01-005", "A-01-006"],
    "confidence": 4
  }}
]
零条编码示例: []

[待处理文段]:
{batch_text}

提醒：严格遵守判别标准与编码步骤，按照规定JSON格式输出！不输出其他内容！
"""

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
1. 情境还原：仔细阅读引文（Quote）。若引文包含多条，请综合考虑其共性。若引文缺失或模糊，**下调置信度**。
2. 竞争性假设：对于每条数据，不要只看它“像”什么，要反问它“为什么不是”其他维度。
3. 排他性判断：如果一条数据同时符合两个维度的定义，选择**语义对应更直接**的那个。

【三、置信度评分量表 (1-5)】
5: 理论饱和。编码与定义的关键词完全对应，且引文语境提供了强有力支撑。
4: 高度匹配。逻辑通顺，无明显歧义。
3: 中度匹配。符合核心定义，但缺乏语境细节，或存在多义性。
2: 证据不足。仅有微弱联系，建议人工复核。
1: 无法判断。信息缺失或完全不相关。

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

def generate_ai_dimension_meta(cluster_payload, api_key, research_domain="", research_topic=""):
    if not cluster_payload:
        return {}

    research_domain = (research_domain or "").strip()
    research_topic = (research_topic or "").strip()

    prompt = f"""
你正在帮助研究者进行中文质性研究的“轴心编码”。

研究领域：{research_domain if research_domain else "未提供"}
研究主题：{research_topic if research_topic else "未提供"}

下面给你若干组已经聚好的开放编码，请你结合上述研究领域与研究主题，为每一组生成：

1. 一个简短、像研究维度的中文名称（2~8个字）
2. 一句非常短的定义（8~20个字）

请注意：
- 这不是普通的语义聚类命名
- 必须尽量贴合“研究主题”进行轴心归类
- 维度名称应具有研究解释性，而不是仅仅概括字面意思

请严格返回 JSON 数组，不要解释，不要 markdown，不要代码块。
格式如下：
[
  {{
    "cluster_id": "0",
    "header": "初始蜜月期",
    "definition": "初入组织时的积极体验"
  }}
]

要求：
- 名称必须像研究维度，不要用“其他”“杂项”“维度1”
- 定义必须是一句话，简洁、能直接显示在看板列头下
- 用词尽量概括，不要照抄编码
- 必须覆盖每个 cluster_id

聚类数据如下：
{json.dumps(cluster_payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        res = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = res.choices[0].message.content.strip()
        parsed = parse_ai_json_content(content)

        meta = {}
        for item in parsed:
            cid = str(item.get("cluster_id", "")).strip()
            if cid:
                meta[cid] = {
                    "header": str(item.get("header", "")).strip(),
                    "definition": str(item.get("definition", "")).strip()
                }
        return meta
    except Exception:
        return {}
    
    
