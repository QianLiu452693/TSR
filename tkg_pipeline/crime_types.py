"""二级犯罪类型编码词典：刑法分则 + 街面/社区犯罪实务分类。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# 街面/社区犯罪二级编码（4 位）：章 2 位 + 类 2 位
TYPE_SPECS: list[dict] = [
    {
        "code": "0101",
        "name": "盗窃罪",
        "chapter": "侵犯财产",
        "aliases": ["盗窃", "入户盗窃", "扒窃", "携带凶器盗窃", "多次盗窃"],
        "keywords": ["盗窃", "窃取", "入户", "扒窃", "赃物", "财物", "秘密"],
        "elements": ["非法占有", "秘密窃取", "财物"],
    },
    {
        "code": "0102",
        "name": "抢劫罪",
        "chapter": "侵犯财产",
        "aliases": ["抢劫", "入户抢劫", "持械抢劫"],
        "keywords": ["抢劫", "暴力", "胁迫", "财物", "人身"],
        "elements": ["非法占有", "暴力胁迫", "财物", "人身"],
    },
    {
        "code": "0103",
        "name": "抢夺罪",
        "chapter": "侵犯财产",
        "aliases": ["抢夺", "飞车抢夺"],
        "keywords": ["抢夺", "夺取", "公然", "财物"],
        "elements": ["非法占有", "公然夺取", "财物"],
    },
    {
        "code": "0104",
        "name": "诈骗罪",
        "chapter": "侵犯财产",
        "aliases": ["诈骗", "街头诈骗"],
        "keywords": ["诈骗", "虚构", "隐瞒", "财物", "欺骗"],
        "elements": ["非法占有", "虚构事实", "财物"],
    },
    {
        "code": "0105",
        "name": "敲诈勒索罪",
        "chapter": "侵犯财产",
        "aliases": ["敲诈勒索", "敲诈"],
        "keywords": ["敲诈", "勒索", "威胁", "要挟", "财物"],
        "elements": ["非法占有", "威胁要挟", "财物"],
    },
    {
        "code": "0106",
        "name": "故意毁坏财物罪",
        "chapter": "侵犯财产",
        "aliases": ["故意毁坏财物", "毁坏财物"],
        "keywords": ["毁坏", "财物", "故意"],
        "elements": ["故意", "毁坏财物", "财物"],
    },
    {
        "code": "0107",
        "name": "掩饰、隐瞒犯罪所得罪",
        "chapter": "侵犯财产",
        "aliases": [
            "掩饰、隐瞒犯罪所得",
            "掩饰隐瞒犯罪所得",
            "隐瞒犯罪所得",
            "掩饰、隐瞒犯罪所得、犯罪所得收益",
            "窝藏、转移、收购、销售赃物",
            "窝藏转移收购销售赃物",
        ],
        "keywords": ["掩饰", "隐瞒", "犯罪所得", "赃物", "窝藏", "收购"],
        "elements": ["明知", "犯罪所得", "财物"],
    },
    {
        "code": "0201",
        "name": "故意伤害罪",
        "chapter": "侵犯人身",
        "aliases": ["故意伤害", "伤害"],
        "keywords": ["故意伤害", "伤害", "轻伤", "重伤", "身体"],
        "elements": ["故意", "伤害", "身体"],
    },
    {
        "code": "0202",
        "name": "故意杀人罪",
        "chapter": "侵犯人身",
        "aliases": ["故意杀人", "杀人"],
        "keywords": ["故意杀人", "杀人", "死亡", "生命"],
        "elements": ["故意", "非法剥夺", "生命"],
    },
    {
        "code": "0203",
        "name": "强奸罪",
        "chapter": "侵犯人身",
        "aliases": ["强奸"],
        "keywords": ["强奸", "暴力", "胁迫", "违背"],
        "elements": ["暴力胁迫", "性自主", "违背意愿"],
    },
    {
        "code": "0204",
        "name": "强制猥亵、侮辱罪",
        "chapter": "侵犯人身",
        "aliases": ["强制猥亵、侮辱", "强制猥亵", "强制侮辱", "猥亵妇女", "强制猥亵、侮辱妇女"],
        "keywords": ["强制猥亵", "猥亵", "侮辱", "暴力"],
        "elements": ["暴力胁迫", "猥亵", "侮辱"],
    },
    {
        "code": "0205",
        "name": "非法拘禁罪",
        "chapter": "侵犯人身",
        "aliases": ["非法拘禁", "拘禁"],
        "keywords": ["非法拘禁", "拘禁", "人身自由"],
        "elements": ["非法", "剥夺", "人身自由"],
    },
    {
        "code": "0206",
        "name": "绑架罪",
        "chapter": "侵犯人身",
        "aliases": ["绑架"],
        "keywords": ["绑架", "人质", "勒索"],
        "elements": ["劫持", "人质", "勒索"],
    },
    {
        "code": "0301",
        "name": "寻衅滋事罪",
        "chapter": "扰乱公共秩序",
        "aliases": ["寻衅滋事"],
        "keywords": ["寻衅滋事", "随意殴打", "辱骂", "公共秩序"],
        "elements": ["随意殴打", "情节恶劣", "公共秩序"],
    },
    {
        "code": "0302",
        "name": "聚众斗殴罪",
        "chapter": "扰乱公共秩序",
        "aliases": ["聚众斗殴", "斗殴"],
        "keywords": ["聚众斗殴", "斗殴", "聚众"],
        "elements": ["聚众", "斗殴", "公共秩序"],
    },
    {
        "code": "0303",
        "name": "妨害公务罪",
        "chapter": "扰乱公共秩序",
        "aliases": ["妨害公务"],
        "keywords": ["妨害公务", "阻碍", "公务", "执法"],
        "elements": ["暴力威胁", "阻碍", "国家机关公务"],
    },
    {
        "code": "0304",
        "name": "聚众扰乱公共场所秩序、交通秩序罪",
        "chapter": "扰乱公共秩序",
        "aliases": ["聚众扰乱公共场所秩序、交通秩序", "聚众扰乱公共场所秩序", "聚众扰乱交通秩序"],
        "keywords": ["聚众", "扰乱", "公共场所", "交通秩序"],
        "elements": ["聚众", "扰乱", "公共场所"],
    },
    {
        "code": "0305",
        "name": "聚众扰乱社会秩序罪",
        "chapter": "扰乱公共秩序",
        "aliases": ["聚众扰乱社会秩序"],
        "keywords": ["聚众", "扰乱", "社会秩序"],
        "elements": ["聚众", "扰乱", "社会秩序"],
    },
]

# 这些罪名一旦作为案件主罪名出现，视为与街面/社区犯罪无关，整案剔除。
ALWAYS_OUT = [
    "猥亵儿童",
    "合同诈骗",
    "信用卡诈骗",
    "贷款诈骗",
    "保险诈骗",
    "票据诈骗",
    "金融凭证诈骗",
    "集资诈骗",
    "危险驾驶",
    "交通肇事",
    "贩卖毒品",
    "运输毒品",
    "制造毒品",
    "走私毒品",
    "非法持有毒品",
    "容留他人吸毒",
    "非法种植毒品",
    "窝藏毒品",
    "引诱他人吸毒",
    "介绍卖淫",
    "容留卖淫",
    "组织卖淫",
    "强迫卖淫",
    "协助组织卖淫",
    "开设赌场",
    "赌博",
    "职务侵占",
    "贪污",
    "受贿",
    "行贿",
    "挪用公款",
    "挪用资金",
    "滥用职权",
    "玩忽职守",
    "徇私枉法",
    "非法经营",
    "虚开增值税",
    "逃税",
    "污染环境",
    "滥伐林木",
    "盗伐林木",
    "非法采矿",
    "非法捕捞",
    "非法狩猎",
    "盗掘古文化遗址",
    "盗掘古墓葬",
    "倒卖文物",
    "掩饰、隐瞒犯罪所得、犯罪所得收益",  # kept in whitelist via alias; not here
]

# 上面最后一项是白名单罪名，不能放进黑名单。立刻删掉误写入的条目。
ALWAYS_OUT = [x for x in ALWAYS_OUT if "犯罪所得" not in x]

ALWAYS_OUT.extend(
    [
        "伪造、变造、买卖国家机关公文、证件、印章",
        "伪造国家机关",
        "买卖国家机关证件",
        "伪造公司",
        "生产、销售有毒、有害食品",
        "销售有毒、有害食品",
        "生产、销售伪劣产品",
        "销售伪劣产品",
        "非法采伐",
        "毁坏国家重点保护植物",
        "拒不支付劳动报酬",
        "拒不执行判决、裁定",
        "帮助信息网络犯罪活动",
        "侵犯公民个人信息",
        "非法获取计算机信息系统数据",
        "破坏计算机信息系统",
        "非法行医",
        "医疗事故",
        "重大责任事故",
        "危险驾驶",
        "以危险方法危害公共安全",
        "放火",
        "爆炸",
        "投放危险物质",
        "失火",
        "非法持有枪支",
        "非法制造枪支",
        "走私",
        "组织他人偷越国（边）境",
        "偷越国（边）境",
        "聚众淫乱",
        "盗窃、侮辱、故意毁坏尸体",
        "侮辱尸体",
        "招摇撞骗",
        "妨害信用卡管理",
        "信用卡诈骗",
        "非法吸收公众存款",
        "组织、领导传销活动",
        "假冒注册商标",
        "销售假冒注册商标的商品",
        "侵犯著作权",
        "故意毁坏文物",
        "过失致人死亡",
        "过失致人重伤",
        "过失损坏",
        "虐待",
        "遗弃",
        "重婚",
        "破坏军婚",
        "暴力干涉婚姻自由",
        "拐卖妇女、儿童",
        "收买被拐卖的妇女、儿童",
        "非法侵入住宅",
    ]
)

GENERIC_CASE_TYPES = {
    "刑事",
    "刑事案件",
    "刑事案",
    "刑事判决",
    "刑事判决书",
    "刑事一审",
    "刑事一审判决",
    "刑事一审判决书",
    "刑事附带民事",
    "刑事附带民事判决",
    "刑事附带民事诉讼",
    "公诉",
    "公诉案件",
    "判决",
    "判决书",
    "案",
}

CHARGE_WITH_FAN = re.compile(r"犯([\u4e00-\u9fff、，,]{1,40}?)罪")
CHARGE_ANY = re.compile(r"([\u4e00-\u9fff、，,]{2,40}?)罪")
CIVIL_HINT = re.compile(r"纠纷|合同|民事|行政|赔偿|离婚|继承|劳动争议|商标|专利")


@dataclass(frozen=True)
class TypeHit:
    code: str
    name: str
    score: float
    method: str


def _norm_phrase(text: str) -> str:
    text = text.strip()
    text = text.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    text = re.sub(r"\s+", "", text)
    return text


def _strip_suffix(text: str) -> str:
    text = _norm_phrase(text)
    text = re.sub(r"(刑事附带民事判决书|刑事附带民事|刑事一审判决书|刑事判决书|一审判决书|判决书|刑事判决|刑事一审)$", "", text)
    return text


class CrimeTypeDictionary:
    def __init__(self) -> None:
        self.by_code = {item["code"]: item for item in TYPE_SPECS}
        self.alias_to_code: dict[str, str] = {}
        for item in TYPE_SPECS:
            names = [item["name"], item["name"].replace("罪", "")] + item["aliases"]
            for name in names:
                key = _norm_phrase(name).replace("、", "").replace("，", "").replace(",", "")
                key = key.replace("罪", "")
                if key:
                    self.alias_to_code[key] = item["code"]
        self._white_keys = sorted(self.alias_to_code, key=len, reverse=True)
        self._black_keys = sorted(
            {_norm_phrase(x).replace("、", "").replace("，", "").replace(",", "").replace("罪", "") for x in ALWAYS_OUT},
            key=len,
            reverse=True,
        )
        self._vocab, self._idf, self._type_vecs = self._build_tfidf()

    def _build_tfidf(self):
        vocab: list[str] = []
        seen = set()
        for item in TYPE_SPECS:
            for token in item["keywords"] + item["elements"] + item["aliases"] + [item["name"]]:
                token = token.replace("、", "")
                if len(token) >= 2 and token not in seen:
                    seen.add(token)
                    vocab.append(token)
        vocab.sort(key=len, reverse=True)
        docs = []
        for item in TYPE_SPECS:
            blob = " ".join([item["name"], item["chapter"], *item["aliases"], *item["keywords"], *item["elements"]])
            docs.append(blob)
        df = np.zeros(len(vocab), dtype=np.float64)
        raw = []
        for blob in docs:
            vec = np.zeros(len(vocab), dtype=np.float64)
            for i, token in enumerate(vocab):
                if token in blob:
                    vec[i] = 1.0
                    df[i] += 1.0
            raw.append(vec)
        n = float(len(docs))
        idf = np.log((n + 1.0) / (df + 1.0)) + 1.0
        vecs = {}
        for item, vec in zip(TYPE_SPECS, raw):
            weighted = vec * idf
            vecs[item["code"]] = weighted
        return vocab, idf, vecs

    def entity_id(self, code: str) -> str:
        return f"C{code}"

    def match_phrase(self, phrase: str) -> TypeHit | None:
        if not phrase:
            return None
        compact = _norm_phrase(phrase).replace("、", "").replace("，", "").replace(",", "").replace("罪", "")
        if not compact:
            return None
        black = next((b for b in self._black_keys if b and b in compact), "")
        white = next((w for w in self._white_keys if w and w in compact), "")
        if black and (not white or len(black) > len(white)):
            return TypeHit("OUT", "非街面社区犯罪", 1.0, "blacklist")
        if white:
            code = self.alias_to_code[white]
            return TypeHit(code, self.by_code[code]["name"], 0.97, "dictionary")
        if compact in self.alias_to_code:
            code = self.alias_to_code[compact]
            return TypeHit(code, self.by_code[code]["name"], 1.0, "dictionary")
        return None

    def extract_charge(self, case_type: str, details: str) -> str:
        ct = _strip_suffix(case_type or "")
        if ct in GENERIC_CASE_TYPES or ct.startswith("刑事"):
            ct_charge = ""
        else:
            ct_charge = ct
        if ct_charge:
            found = CHARGE_ANY.search(ct_charge)
            if found:
                return found.group(1)
            return ct_charge
        text = details or ""
        found = CHARGE_WITH_FAN.search(text) or CHARGE_ANY.search(text)
        if found:
            return found.group(1)
        return ""

    def vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        if not text:
            return vec
        for i, token in enumerate(self._vocab):
            if token in text:
                vec[i] = 1.0
        return vec * self._idf

    def cosine_match(self, text: str) -> TypeHit | None:
        query = self.vectorize(text)
        qn = float(np.linalg.norm(query))
        if qn == 0.0:
            return None
        best_code = ""
        best_score = 0.0
        for code, vec in self._type_vecs.items():
            vn = float(np.linalg.norm(vec))
            if vn == 0.0:
                continue
            score = float(np.dot(query, vec) / (qn * vn))
            if score > best_score:
                best_score = score
                best_code = code
        if not best_code:
            return None
        return TypeHit(best_code, self.by_code[best_code]["name"], best_score, "tfidf")

    def classify(self, case_type: str, details: str) -> TypeHit | None:
        case_type = "" if case_type is None else str(case_type)
        details = "" if details is None else str(details)
        if CIVIL_HINT.search(case_type) and "罪" not in case_type:
            return TypeHit("OUT", "非刑事案件", 1.0, "civil")
        charge = self.extract_charge(case_type, details)
        hit = self.match_phrase(charge)
        if hit is not None:
            return hit
        # 主罪名未命中时，再用案由原文做一次词典包含匹配
        hit = self.match_phrase(case_type)
        if hit is not None:
            return hit
        semantic = self.cosine_match(f"{case_type} {details[:400]}")
        return semantic

    def jaccard_pairs(self, similar: float = 0.34, derived: float = 0.18) -> list[dict]:
        pairs = []
        items = TYPE_SPECS
        for i in range(len(items)):
            a = set(items[i]["elements"])
            for j in range(i + 1, len(items)):
                b = set(items[j]["elements"])
                union = a | b
                if not union:
                    continue
                score = len(a & b) / len(union)
                if score >= similar:
                    relation = "类型相似"
                elif score >= derived and items[i]["chapter"] == items[j]["chapter"]:
                    relation = "类型衍生"
                elif score <= 0.02:
                    relation = "类型互斥"
                else:
                    continue
                pairs.append(
                    {
                        "code_a": items[i]["code"],
                        "code_b": items[j]["code"],
                        "relation_type": relation,
                        "confidence": round(float(score if relation != "类型互斥" else 1.0 - score), 4),
                    }
                )
        return pairs
