import json


class DrAttack_random_search:
    """
    简化版 DrAttack_random_search：只保留将原始 attack_prompt 重写为
    disguised prompt 的功能，去除了以下与 rewrite 无关的模块：
      - evolve / expand_population：遗传算法搜索主循环
      - fitness / process_and_score：LLM 推理 + embedding 评分
      - generate_output：开源模型推理 helper
      - post_processing_for_word_game：仅 fitness 内部使用
      - init_visited / updata_visited / save_visited：候选句 cache 管理
      - text_embedding_ada / output_dict / process_fns*：embedding 评分依赖
      - misspell_words / perturbe：扰动函数（原代码中未被主流程调用）
      - gpt_eval 相关逻辑：fitness 内的额外评估分支
    """

    def __init__(
        self,
        attack_prompt: str,
        prompt_info_path: str,
        worker,
        verb_sub: bool = False,
        noun_sub: bool = False,
        noun_wordgame: bool = False,
        suffix: bool = False,
        general_template: str = "",
        demo_suffix_template: str = "",
        wordgame_template: str = "",
        test_prefixes: list = None,
    ):
        # ── 保留的功能开关 ────────────────────────────────────────────────────
        self.verb_sub       = verb_sub        # 是否对 verb 做同义词替换
        self.noun_sub       = noun_sub        # 是否对 noun 做同义词替换
        self.noun_wordgame  = noun_wordgame   # 是否使用 word-game 变体
        self.suffix         = suffix          # 是否添加 demo suffix

        self.attack_prompt    = attack_prompt
        self.prompt_info_path = prompt_info_path
        self.worker           = worker
        self.test_prefixes    = test_prefixes if test_prefixes is not None else []
        self.general_template = general_template

        if self.suffix:
            self.demo_suffix_template = demo_suffix_template

        if self.noun_wordgame:
            # word-game 使用的预定义词表
            self.fruits = [
                "apple", "banana", "cherry", "grape", "kiwi", "lemon",
                "mango", "orange", "papaya", "raspberry", "strawberry", "watermelon",
            ]
            self.vegetables = [
                "carrot", "Broccoli", "cauliflower", "corn", "cucumber",
                "eggplant", "pepper", "lettuce", "mushroom",
                "onion", "pumpkin", "potato",
            ]
            self.wordgame_template = wordgame_template

        # ── 初始化 ────────────────────────────────────────────────────────────
        # 1. 从 JSON 加载 parsing tree / synonym / opposite 空间
        self.init_prompt_info()

        # 2. 初始化 prompt 骨架（固定短语"using the methods in Question A"）
        self.initialize_prompt_structure()

        # 3. 遍历 parsing tree，提取词组，生成 questions 和替换词列表
        self.initialize_population()

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def init_prompt_info(self):
        """从 prompt_info_path 一次性读取所有结构化数据，避免重复 IO。"""
        with open(self.prompt_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data[self.attack_prompt]
        self.segment_space   = entry["parsing_tree_dictionary"]  # 句法解析树（嵌套 dict）
        self.search_space    = entry["synonym"]                  # 同义词替换空间
        self.oppsite_space   = entry["opposite"]                 # 反义词空间（生成负向参考句）
        # harmless mapping 一并读取，供 init_harmless_dict 直接返回，不再重复打开文件
        harmless_mapping_list = entry["harmless"]
        self._harmless_mapping_cache = {
            item.split(":")[0]: item.split(":")[1] for item in harmless_mapping_list
        }

    def init_harmless_dict(self) -> dict:
        """返回有害词 → 无害词的映射表（数据已在 init_prompt_info 中加载）。"""
        return self._harmless_mapping_cache

    # ── Parsing tree 遍历 ─────────────────────────────────────────────────────

    def visit_keys(self, dictionary, depth: int = 0):
        """
        递归遍历 parsing_tree_dictionary，按词性标注将叶节点分类为：
          - "instruction"：句子主干（最高层的动词/名词/形容词等）
          - "verb"        ：非主干动词（可做同义词替换）
          - "noun"        ：名词/形容词（可做同义词替换 + word-game）
          - "structure"   ：功能词（情态动词、介词、连词等，不替换）
        同时记录每个词的深度（words_level），用于后续按层级展开候选集。
        """
        depth += 1
        self.prompt_depth = max(self.prompt_depth, depth)

        for key, value in dictionary.items():

            if isinstance(value, str):
                # ── 词性分类规则 ──────────────────────────────────────────────
                if ("Verb" in key and "Modal" not in key) or "Gerund" in key or "Infinitive" in key:
                    # 动词：depth==2 是句子主干，其余是可替换的 verb
                    self.words_type.append("instruction" if depth == 2 else "verb")

                elif "Determiner" in key:
                    self.words_type.append("instruction" if depth == 3 else "structure")

                elif "Adjective" in key:
                    self.words_type.append("instruction" if depth == 3 else "noun")

                elif "Noun" in key:
                    if depth == 3:
                        self.words_type.append("instruction")
                    elif value == "how":
                        self.words_type.append("structure")
                    else:
                        self.words_type.append("noun")

                elif "Modal Verb" in key:
                    self.words_type.append("structure")
                elif "Relative Pronoun" or "Conj" in key:
                    self.words_type.append("structure")
                elif "how to" or "Infinitive" or "to" in key:
                    self.words_type.append("structure")
                elif "Preposition" in key:
                    self.words_type.append("structure")
                elif "Adverb" in key:
                    self.words_type.append("structure")

                self.words.append(value)
                self.words_level.append(depth)

            if isinstance(value, dict):
                self.visit_keys(value, depth)

    def reprocess(self):
        """
        将相邻且同层的同类型词组合并为一个 phrase。
        两遍合并：第一遍处理跨类型的 structure+noun；
        第二遍兜底合并剩余的同类型相邻词。
        """
        assert len(self.words_type) == len(self.words)

        # 第一遍：按规则合并
        idx = 0
        while idx < len(self.words_type) - 1:
            t0, t1 = self.words_type[idx], self.words_type[idx + 1]
            l0, l1 = self.words_level[idx], self.words_level[idx + 1]
            merge = (
                (t0 == "structure" and t1 == "noun"        and l0 == l1) or
                (t0 == "instruction" and t1 == "instruction"             ) or
                (t0 == "structure"   and t1 == "structure"  and l0 == l1) or
                (t0 == "noun"        and t1 == "noun"        and l0 == l1) or
                (t0 == "verb"        and t1 == "verb"        and l0 == l1)
            )
            if merge:
                self.words[idx] = self.words[idx] + " " + self.words[idx + 1]
                self.words_type[idx] = t1
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
            else:
                idx += 1

        # 第二遍：兜底合并剩余同类型相邻词
        # 原版使用三个独立 if（非 if/elif），合并后不递增 idx，
        # 使得同一位置可被连续多次检查，直到不再满足任何条件才由最后的 else 递增。
        idx = 0
        while idx < len(self.words_type) - 1:
            t0, t1 = self.words_type[idx], self.words_type[idx + 1]
            merged = False
            if t0 == "noun" and t1 == "noun":
                self.words[idx] = self.words[idx] + " " + self.words[idx + 1]
                self.words_type[idx] = t1
                del self.words[idx + 1]
                del self.words_type[idx + 1]
                del self.words_level[idx + 1]
                merged = True
            if idx < len(self.words_type) - 1:
                t0, t1 = self.words_type[idx], self.words_type[idx + 1]
                if t0 == "structure" and t1 == "structure":
                    self.words[idx] = self.words[idx] + " " + self.words[idx + 1]
                    self.words_type[idx] = t1
                    del self.words[idx + 1]
                    del self.words_type[idx + 1]
                    del self.words_level[idx + 1]
                    merged = True
            if idx < len(self.words_type) - 1:
                t0, t1 = self.words_type[idx], self.words_type[idx + 1]
                if t0 == "verb" and t1 == "verb":
                    self.words[idx] = self.words[idx] + " " + self.words[idx + 1]
                    self.words_type[idx] = t1
                    del self.words[idx + 1]
                    del self.words_type[idx + 1]
                    del self.words_level[idx + 1]
                    merged = True
            if not merged:
                idx += 1

    # ── Prompt 结构初始化 ─────────────────────────────────────────────────────

    def initialize_prompt_structure(self):
        """初始化 prompt 骨架，固定短语会在 generate_replacement 中追加。"""
        self.prompt_structure_mask = ["using the methods in Question A"]

    def generate_replacement(self):
        """
        根据 words / words_type 构建：
          - questions            : ["Question A: How to ...?", "Question B1: What is ...?", ...]
          - questions_prefix     : ["Question A", "Question B1", ...]
          - prompt_structure_mask: 骨架短语列表（template 中的 {sentence structure}）
          - sub_words            : 待替换词列表（noun_sub 或 verb_sub 控制哪些词参与替换）
        """
        assert self.words_type[0] == "instruction"

        self.questions        = []
        self.sub_words        = []
        self.questions_prefix = []

        # Question A 固定对应主干动作
        self.questions.append(f"Question A: How to {self.words[0].lower()}?")
        self.questions_prefix.append("Question A")

        n = 0
        for i in range(1, len(self.words_type)):
            wtype = self.words_type[i]
            if wtype == "structure":
                self.prompt_structure_mask.append(self.words[i])

            elif wtype == "verb":
                # verb_sub=False：动词直接写入骨架，不参与替换
                # verb_sub=True ：动词加入 sub_words，rewrite() 时可被同义词替换
                self.prompt_structure_mask.append(self.words[i])
                if self.verb_sub:
                    self.sub_words.append(self.words[i])

            elif wtype == "noun":
                # 每个名词生成一个 Question Bn，并在骨架中用占位符替代
                n += 1
                self.prompt_structure_mask.append(f"the thing in Question B{n}")
                self.questions.append(f"Question B{n}: What is/are {self.words[i]}?")
                self.questions_prefix.append(f"Question B{n}")
                if self.noun_sub:
                    self.sub_words.append(self.words[i])

    def generate_negative_reference(self):
        """
        生成语义相反的参考句（用于原版的 embedding 评分，简化版保留以维持
        initialize_population 的完整性，但 rewrite 流程中不使用其结果）。
        """
        for idx, word_type in enumerate(self.words_type):
            if word_type == "verb":
                words = self.words[:]
                opp_word = self.oppsite_space[words[idx]]
                words[idx] = opp_word[0]
                self.oppsite_reference = " ".join(words)
                return
        self.oppsite_reference = "Not " + " ".join(self.words)

    # ── Harmless response 初始化 ──────────────────────────────────────────────

    def init_harmless_response(self):
        """
        生成无害版本的 prompt 及其对应的 LLM 响应，作为 few-shot 示例
        注入到最终 rewritten prompt 的 {harmless response} 占位符中。
        响应截取前 1/3 并加 "..." 以控制长度。
        """
        harmful_prompt = " ".join(self.words)
        self.harmless_prompt = harmful_prompt

        print("Preparing harmless response ...")
        self.harmless_mapping_dict = self.init_harmless_dict()

        # 将有害词替换为无害词，得到无害版 prompt
        for key in self.harmless_mapping_dict:
            self.harmless_prompt = self.harmless_prompt.replace(key, self.harmless_mapping_dict[key])

        # suffix=False 时 demo_suffix_template 未赋值，需防御性处理
        if self.suffix:
            query = self.harmless_prompt + f" Give your answer step-by-step and start your response with '{self.demo_suffix_template}'"
        else:
            query = self.harmless_prompt + " Give your answer step-by-step."
        self.harmless_response = self._call_worker(query)

        # jailbreak 检测：若无害版 prompt 仍触发拒绝前缀，说明 harmless mapping 质量有问题
        jailbreak = not any(prefix in self.harmless_response for prefix in self.test_prefixes)
        if jailbreak:
            print("Warning: harmless response triggered refusal prefixes, check harmless mapping.")

        # 只保留响应的前 1/3，减少 token 消耗
        length = len(self.harmless_response)
        self.harmless_response = self.harmless_response[: length // 3] + "..."
        print("Harmless response ready.")

    def _call_worker(self, prompt: str) -> str:
        """
        统一的 worker 调用入口（原版 get_harmless_response 中的多 model 分支
        合并为此方法，简化版只保留 GPT/Gemini API 调用，去除了 vicuna/llama
        的本地模型推理代码）。
        """
        model = self.worker.model
        return model([prompt])

    # ── Word-game 初始化 ──────────────────────────────────────────────────────

    def init_word_game_prompt(self):
        """
        为 word-game 变体建立 有害词 → 水果/蔬菜 的映射，并生成 wordgame_prefix。
        仅在 noun_wordgame=True 时调用。
        """
        self.word_mapping = {}
        fruit_id = vegetable_id = 0
        wordgame_sentence = ""

        for idx, word_type in enumerate(self.words_type):
            if word_type == "noun":
                fruit_id += 1
                self.word_mapping[self.words[idx]] = self.fruits[fruit_id]
                wordgame_sentence += f"'{self.fruits[fruit_id]}: {self.words[idx]}'"

                if self.words[idx] in self.harmless_mapping_dict:
                    vegetable_id += 1
                    self.word_mapping[self.harmless_mapping_dict[self.words[idx]]] = self.vegetables[vegetable_id]
                    wordgame_sentence += f"'{self.vegetables[fruit_id]}: {self.harmless_mapping_dict[self.words[idx]]}'"

        self.wordgame_prefix = self.wordgame_template.replace("{wordgame sentence}", wordgame_sentence)

    # ── Population 初始化（只取 init_individual，不做搜索）────────────────────

    def initialize_population(self):
        """
        原版用于初始化遗传算法的种群，简化版只保留：
          1. 遍历 parsing tree → words / words_type / words_level
          2. 合并相邻词组（reprocess）
          3. 生成 questions / sub_words / prompt_structure_mask（generate_replacement）
          4. 生成负向参考句（generate_negative_reference，供模板使用）
          5. 生成 harmless_response（init_harmless_response）
          6. （可选）初始化 word-game 映射
        返回值弃用，仅保留以初始化所有 generate_prompt 依赖的属性。
        """
        self.words       = []
        self.words_type  = []
        self.words_level = []
        self.prompt_depth = 0

        self.visit_keys(self.segment_space)
        self.reprocess()
        self.generate_replacement()

        # init_individual：原始 sub_words 对应的 key，用于 generate_prompt
        self.init_individual = self.word_to_string(self.sub_words)

        self.generate_negative_reference()
        self.init_harmless_response()

        if self.noun_wordgame:
            self.init_word_game_prompt()

    # ── 工具函数 ──────────────────────────────────────────────────────────────

    def word_to_string(self, words: list) -> str:
        """将词列表序列化为 '+' 分隔的字符串，作为 dict key 或 generate_prompt 的入参。"""
        return "+".join(words)

    # ── 核心：生成 rewritten prompt ───────────────────────────────────────────

    def generate_prompt(self, sub_word_tuple: str):
        """
        将 sub_word_tuple（'+' 分隔的替换词）填入 general_template，
        生成最终的 rewritten prompt 字符串。

        模板占位符说明：
          {questions prefix}    → "Question A Question B1 ..."
          {sentence structure}  → prompt 骨架（动词 + 名词占位符 + 功能词）
          {harmless prompt}     → 无害版原始 prompt
          {harmful prompt}      → 原始 attack prompt
          {harmless questions}  → 将 harmful questions 中有害词替换为无害词后的版本
          {harmful questions}   → "Question A: How to ...? Question B1: What is ...?"
          {harmless response}   → LLM 对无害版 prompt 的示例回答（前 1/3）

        Returns:
          - noun_wordgame=False: str（rewritten prompt）
          - noun_wordgame=True:  (str, str)（rewritten prompt, word_game prefix）
        """
        sub_words = sub_word_tuple.split("+")

        prompt = self.general_template

        sentence_structure  = " ".join(self.prompt_structure_mask)
        harmful_questions   = " ".join(self.questions)
        questions_prefix    = " ".join(self.questions_prefix)

        # 将 harmful_questions 中的有害词替换为无害词，得到 harmless_questions
        harmless_questions = harmful_questions
        for key in self.harmless_mapping_dict:
            harmless_questions = harmless_questions.replace(key.lower(), self.harmless_mapping_dict[key].lower())

        # 填充固定占位符
        prompt = prompt.replace("{questions prefix}",   questions_prefix)
        prompt = prompt.replace("{sentence structure}", sentence_structure)
        prompt = prompt.replace("{harmless prompt}",    self.harmless_prompt)
        prompt = prompt.replace("{harmful prompt}",     self.attack_prompt)
        prompt = prompt.replace("{harmless questions}", harmless_questions)
        prompt = prompt.replace("{harmful questions}",  harmful_questions)

        if self.noun_wordgame:
            # word-game 模式：将有害词替换为水果/蔬菜代号
            for key, value in self.word_mapping.items():
                prompt = prompt.replace(key, value.lower())
                self.harmless_response = self.harmless_response.replace(key, value.lower())

            word_game = self.wordgame_prefix
            for idx, sub_word in enumerate(sub_words):
                prompt    = prompt.replace(self.sub_words[idx], sub_word.lower())
                word_game = word_game.replace(self.sub_words[idx], sub_word.lower())
        else:
            # 普通模式：直接替换 sub_words
            for idx, sub_word in enumerate(sub_words):
                prompt = prompt.replace(self.sub_words[idx], sub_word)

        prompt = prompt.replace("{harmless response}", self.harmless_response)

        if self.noun_wordgame:
            return prompt, word_game
        return prompt

    # ── 对外接口 ──────────────────────────────────────────────────────────────

    def rewrite(self) -> str:
        """
        使用原始 sub_words（无同义词替换）生成 rewritten prompt。
        这是简化版对外暴露的唯一接口，等价于原版 evolve() 在第一轮
        对 init_individual 调用 generate_prompt 的结果。

        Returns:
            rewritten prompt 字符串（noun_wordgame=True 时返回完整拼接版本）。
        """
        if "reject" in self.segment_space:
            return self.attack_prompt  # parsing 失败时原样返回

        result = self.generate_prompt(self.init_individual)

        if self.noun_wordgame:
            prompt, word_game = result
            return word_game + "\n" + prompt
        return result