from functools import partial
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class Reward:
    def __init__(self, property, reward, weight=1.0, preprocess=None):
        self.property = property
        self.reward = reward
        self.weight = weight
        self.preprocess = preprocess

    def __call__(self, input):
        if self.preprocess:
            input = self.preprocess(input)
        property = self.property(input)
        reward = self.weight * self.reward(property)
        return reward, property


def identity(x):
    return x


def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_config.json"
MODEL = "qwen/qwen3-14b"

REWARD_COMMENTS = {}

STRICT_DRUG_LIKENESS_PROMPT = """/no_think You are a conservative medicinal chemistry reviewer triaging early small-molecule candidates.

Task:
Evaluate each molecule using a penalty-first developability framework. Start from the assumption that a molecule must earn a high score by avoiding major liabilities. Penalize clear red flags such as reactive or unstable groups, toxicophores, structural alerts, poor physicochemical profile, low oral developability, high synthetic complexity, and likely safety concerns.
Assign a high score only when the molecule has no major developability concerns and looks suitable for further optimization.
Do not assume a target, indication, dose, formulation, or clinical benefit unless it is evident from the molecule identity.

Scoring:
- Return `drug_likeness_score` on a {min_score}-{max_score} scale.
- Higher is better: {max_score} means very drug-like/promising with no major red flags; {min_score} means unsuitable or strongly problematic.
- Use the full range and be conservative when liabilities are present.
- Judge the molecules by general medicinal chemistry criteria, not by any dataset labels or expected classes.

Comment:
- Add a concise English comment for every molecule.
- Mention the main liabilities driving penalties and any important redeeming features.
- Keep each comment to one sentence.

Input representation: {representation}
Molecules:
{molecule_batch}

Return strictly valid JSON with exactly this schema:
{{
  "results": [
    {{
      "molecule_id": "M0001",
      "drug_likeness_score": <number between {min_score} and {max_score}>,
      "comment": "<short English medicinal-chemistry comment>"
    }}
  ]
}}
Return one result for every input molecule. Do not include markdown or extra keys.
"""


_client = None
_llm_config = None


def load_llm_config(config_path=None):
    path = Path(config_path or os.getenv("FREEDPP_LLM_CONFIG", DEFAULT_LLM_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"LLM config not found: {path}. Copy configs/llm_config.example.json "
            "to configs/llm_config.json and fill api_base/api_key."
        )
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    for key in ["api_base", "api_key"]:
        if not config.get(key):
            raise ValueError(f"Missing `{key}` in {path}")
    return config


def get_llm_config():
    global _llm_config
    if _llm_config is None:
        _llm_config = load_llm_config()
    return _llm_config


def get_client():
    global _client
    if _client is None:
        config = get_llm_config()
        _client = OpenAI(api_key=config["api_key"], base_url=config["api_base"])
    return _client


def _normalize_llm_content(content):
    if isinstance(content, str):
        return content.strip()
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("value")
                if value:
                    parts.append(str(value))
        return "".join(parts).strip()
    return str(content).strip()


def _is_transient_llm_error(error):
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    return (
        status_code in {408, 409, 429, 500, 502, 503, 504}
        or "timeout" in message
        or "timed out" in message
        or "connection" in message
        or "rate limit" in message
    )


def call_llm(prompt: str, model: str | None = None):
    config = get_llm_config()
    last_content = ""
    attempts = int(config.get("retries", 2)) + 1
    for attempt in range(attempts):
        try:
            min_interval = 1.15
            if min_interval > 0:
                time.sleep(min_interval)
            response = get_client().chat.completions.create(
                model=model or config.get("model", MODEL),
                messages=[{"role": "user", "content": prompt}],
                temperature=float(config.get("temperature", 0.1)),
                n=1,
                max_tokens=int(config.get("max_tokens", 1500)),
                extra_headers=config.get("extra_headers", {}),
            )
        except Exception as error:
            if attempt == attempts - 1 or not _is_transient_llm_error(error):
                raise
            delay = 2 ** attempt
            print(
                f"[call_llm] transient error, retrying in {delay}s "
                f"({attempt + 1}/{attempts - 1}): {error}"
            )
            time.sleep(delay)
            continue
        last_content = _normalize_llm_content(response.choices[0].message.content)
        if last_content:
            return last_content
    return last_content


def safe_parse_json(text):
    text = _normalize_llm_content(text)
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [text]
    if fenced:
        candidates.insert(0, fenced.group(1))

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        candidates.append(match.group())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def parse_score_comment(response: str, *, fallback_comment: str):
    data = safe_parse_json(response)
    if data is None:
        raise ValueError(f"Failed to parse score JSON: {response}")

    results = data.get("results") or []
    if not results:
        raise ValueError(f"Missing results in LLM response: {response}")

    item = results[0]
    raw_score = item.get("drug_likeness_score")
    if raw_score is None:
        raise ValueError(f"Missing drug_likeness_score in LLM response: {response}")

    score = max(0.0, min(1.0, float(raw_score)))
    comment = str(item.get("comment") or fallback_comment).strip()
    return score, comment


def build_drug_likeness_prompt(smiles: str) -> str:
    molecule_batch = [
        {
            "molecule_id": "M0001",
            "smiles": smiles,
        }
    ]
    return STRICT_DRUG_LIKENESS_PROMPT.format(
        min_score=0.0,
        max_score=1.0,
        representation="smiles",
        molecule_batch=json.dumps(molecule_batch, ensure_ascii=False),
    )


def drug_likeness_reward(mol):
    try:
        smi = Chem.MolToSmiles(mol)
        prompt = build_drug_likeness_prompt(smi)
        response = call_llm(prompt)
        try:
            score, comment = parse_score_comment(
                response,
                fallback_comment="No drug-likeness explanation was returned by the LLM.",
            )
        except ValueError:
            retry_prompt = prompt + "\n\nYour previous answer was not valid JSON. Return only the JSON object with the required schema."
            response = call_llm(retry_prompt)
            score, comment = parse_score_comment(
                response,
                fallback_comment="No drug-likeness explanation was returned by the LLM.",
            )
        REWARD_COMMENTS[smi] = comment
        return score
    except Exception as e:
        try:
            smi = Chem.MolToSmiles(mol)
            REWARD_COMMENTS[smi] = f"LLM drug-likeness scoring failed: {e}"
        except Exception:
            pass
        print(f"[drug_likeness_reward] fallback due to error: {e}")
        return 0.0
