# -*- coding: utf-8 -*-
"""领域视图配置（劳动法 / 后续电商合规等）。

设计原则：领域是配置，不是代码分支。
  - lawrag 的 Milvus 集合已有 law_name 标量字段，lawfaq 已有 domain 字段；
  - 领域视图只需要把「领域 -> 法名列表 / FAQ domain」映射出来，检索时拼 expr；
  - 不新建 collection、不新增 Milvus 字段、不重建索引。
"""

import json
import os

from base.config import config
from base.logger import logger

_PROFILE_PATH = os.path.join(config.DATA_DIR, "domain_profiles.json")
_cache = None


def load_domain_profiles(path=None):
    """读取领域配置；默认路径结果做进程内缓存。"""
    global _cache
    src = path or _PROFILE_PATH
    if path is None and _cache is not None:
        return _cache
    try:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"领域配置不存在: {src}")
        data = {}
    except Exception as e:
        logger.error(f"领域配置读取失败: {e}")
        data = {}
    if path is None:
        _cache = data
    return data


def get_domain(name):
    """按领域名取配置；不存在返回 None。"""
    if not name:
        return None
    return load_domain_profiles().get(name)


def law_names_for(name, include_supplemental=True):
    """领域对应的法名列表（去重保序）；用于 Milvus 的 law_name in [...]。"""
    profile = get_domain(name)
    if not profile:
        return []
    names = list(profile.get("laws") or [])
    if include_supplemental:
        names += list(profile.get("supplemental_laws") or [])
    return list(dict.fromkeys(names))


def faq_domain_for(name):
    """领域对应的 FAQ domain；没有配置返回 None。"""
    return (get_domain(name) or {}).get("faq_domain")


def list_domains():
    """给 /api/domains 用的摘要。"""
    out = []
    for key, profile in load_domain_profiles().items():
        out.append({
            "name": key,
            "label": profile.get("label", key),
            "law_count": len(law_names_for(key)),
            "faq_domain": profile.get("faq_domain"),
        })
    return out
