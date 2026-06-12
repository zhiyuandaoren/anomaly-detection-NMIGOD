#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测算法默认参数收集工具
============================================================
扫描所有算法的 detector.py，提取可配置参数的默认值，生成对比表格。

用法:
    python tools/collect_params.py                   # 交互模式
    python tools/collect_params.py --no-interactive   # 自动扫描全部
    python tools/collect_params.py --output params.csv # 指定输出文件
"""

import os
import sys
import importlib
import argparse
from pathlib import Path

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 框架基础设施属性（所有算法共有，非算法参数）
INFRA_ATTRS = {
    'df_raw', 'df_processed', 'feature_columns', 'target_column',
    'exclude_columns', 'anomaly_values', 'y_true', 'scores',
    'results_df', 'best_threshold', 'output_folder', 'model',
    'device',  # PyTorch 设备
}

# 算法内部状态属性（中间计算结果，非配置参数）
INTERNAL_ATTRS = {
    # GCOD
    'K_binary', 'E_matrix', 'extents', 'GCD', 'M_matrix', 'GOD',
    'U_size', 'L_size', 'M_size', 'n_jobs',
    # DASOD
    'formal_context', 'attr_names', 'object_granular_concepts',
    'attribute_granular_concepts', 'granular_concepts', 'core_concepts',
    'GCD_matrix', 'M_matrix', 'N_matrix', 'GOD_scores', 'CCOF_scores',
    'GPDF_scores',
    # NMIGOD
    'dataset_configs',
    # KNN
    'preprocessor',
    # ADFNR
    'numerical_mask',
    # NIEOD
    'feature_indices',
}

# 已知算法的参数描述（用于增强表格可读性）
PARAM_DESCRIPTIONS = {
    # ADFNR
    'epsilon': '模糊邻域半径 ε，控制邻居判定宽松度',
    # DASOD
    'K': '离散化粒度（等宽区间数）',
    'lambda_ratio': '核心概念选择比例 λ',
    # GCOD
    'n_jobs': '并行核心数（None=自动使用全部）',
    # KNN
    'k': 'K 近邻数量',
    # NIEOD
    'lambda_param': '邻域半径调节参数 λ',
}


def find_algorithm_modules():
    """扫描项目根目录，找到所有包含 detector.py 的算法子目录"""
    algo_modules = {}
    for item in sorted(PROJECT_ROOT.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith('.') or item.name.startswith('_'):
            continue
        detector_path = item / 'detector.py'
        if detector_path.exists():
            algo_modules[item.name] = str(detector_path)
    return algo_modules


def extract_params(module_path, algo_name):
    """导入算法模块并提取可配置参数及其默认值"""
    module_name = f"_param_scan_{algo_name.replace('-', '_').lower()}"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"  [WARN] 导入 {algo_name} 失败: {e}")
        return {}

    if not hasattr(module, 'AnomalyDetectionFramework'):
        print(f"  [WARN] {algo_name} 中未找到 AnomalyDetectionFramework 类")
        return {}

    cls = module.AnomalyDetectionFramework

    # 尝试实例化（无参数，捕获 TypeError）
    try:
        instance = cls()
    except TypeError as e:
        # 可能需要参数，尝试用默认值
        try:
            instance = cls.__new__(cls)
            init_params = {}
            for name in cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]:
                default_idx = name in cls.__init__.__defaults__ or False
                # 尝试用 None 实例化
            instance = cls()
        except Exception:
            # 检查 __init__ 签名
            import inspect
            sig = inspect.signature(cls.__init__)
            default_kwargs = {}
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                if param.default is not inspect.Parameter.empty:
                    default_kwargs[name] = param.default
            try:
                instance = cls(**default_kwargs)
            except Exception:
                print(f"  [WARN] 无法实例化 {algo_name}")
                # 回退到仅从源码分析
                return _extract_from_source(module_path)

    params = {}
    for attr_name in dir(instance):
        if attr_name.startswith('_'):
            continue
        if attr_name in INFRA_ATTRS:
            continue
        if attr_name in INTERNAL_ATTRS:
            continue
        if callable(getattr(instance, attr_name)):
            continue

        value = getattr(instance, attr_name)
        # 跳过复杂对象（ndarray, DataFrame, list 内容等）
        if isinstance(value, (list, dict, set, tuple)):
            if len(value) > 0:
                # 如果列表非空，可能是配置
                if all(isinstance(v, (int, float, str, bool)) for v in value):
                    params[attr_name] = str(value)
                continue
            continue
        if hasattr(value, '__module__') and value.__module__ not in ('builtins', ''):
            continue
        if isinstance(value, (int, float, str, bool)):
            params[attr_name] = value

    # 清理临时模块
    sys.modules.pop(module_name, None)
    del instance

    return params


def _extract_from_source(filepath):
    """从源码中提取参数（回退方案）"""
    import re
    params = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 匹配 self.xxx = 默认值 的模式（在 __init__ 方法内）
    init_match = re.search(r'def __init__\(.*?\).*?:(.*?)(?=\n    def |\n    def |\Z)',
                           content, re.DOTALL)
    if init_match:
        init_body = init_match.group(1)
        # 匹配 self.attr = value
        pattern = r'self\.(\w+)\s*=\s*([^#\n]+?)(?:\s*#.*)?$'
        for match in re.finditer(pattern, init_body, re.MULTILINE):
            name = match.group(1)
            value_str = match.group(2).strip()
            if name in INFRA_ATTRS or name in INTERNAL_ATTRS:
                continue
            # 尝试转换值
            try:
                value = eval(value_str)
                if isinstance(value, (int, float, str, bool)):
                    params[name] = value
                else:
                    params[name] = value_str
            except Exception:
                params[name] = value_str
    return params


def build_table(all_params):
    """
    构建对比表格。
    行 = 参数名, 列 = 算法名, 单元格 = 默认值。
    """
    # 收集所有参数名
    all_param_names = set()
    for algo, params in all_params.items():
        all_param_names.update(params.keys())
    all_param_names = sorted(all_param_names)

    # 构建数据
    data = {}
    for param in all_param_names:
        row = {'参数': param}
        if param in PARAM_DESCRIPTIONS:
            row['说明'] = PARAM_DESCRIPTIONS[param]
        else:
            row['说明'] = ''
        for algo in sorted(all_params.keys()):
            row[algo] = all_params[algo].get(param, '—')
        data[param] = row

    return data


def print_table(data, algos):
    """打印格式化表格到终端"""
    col_widths = {'参数': 18, '说明': 40}
    for algo in algos:
        col_widths[algo] = max(len(algo), 12)

    # 表头
    header = f"{'参数':<18s} | {'说明':<40s}"
    for algo in algos:
        header += f" | {algo:>12s}"
    print(header)
    print("-" * len(header))

    for param, row in data.items():
        desc = row['说明'] if row['说明'] else '—'
        line = f"{param:<18s} | {desc:<40s}"
        for algo in algos:
            val = row[algo]
            if isinstance(val, float):
                line += f" | {val:>12.4f}"
            else:
                line += f" | {str(val):>12s}"
        print(line)


def interactive_main():
    """交互式模式"""
    print("\n" + "=" * 60)
    print("  异常检测算法参数收集工具")
    print("=" * 60)

    # 扫描算法
    algo_modules = find_algorithm_modules()
    if not algo_modules:
        print("未找到任何算法的 detector.py 文件。")
        return

    print(f"\n扫描到 {len(algo_modules)} 个算法:\n")
    for i, name in enumerate(sorted(algo_modules.keys()), 1):
        print(f"  [{i}] {name}")

    print(f"\n  [A] 全部扫描")
    print(f"  [Q] 退出")

    choice = input("\n请选择 (数字/字母, 逗号分隔, 默认A): ").strip().upper()

    if not choice or choice == 'A':
        selected = dict(algo_modules)
    elif choice == 'Q':
        print("已取消。")
        return
    else:
        selected = {}
        parts = [p.strip() for p in choice.split(',')]
        dir_list = sorted(algo_modules.keys())
        for p in parts:
            if p.isdigit():
                idx = int(p) - 1
                if 0 <= idx < len(dir_list):
                    name = dir_list[idx]
                    selected[name] = algo_modules[name]
            elif p in algo_modules:
                selected[p] = algo_modules[p]

    if not selected:
        print("未选择任何算法。")
        return

    # 提取参数
    print("\n正在提取算法参数...")
    all_params = {}
    for algo_name, module_path in sorted(selected.items()):
        print(f"  [{algo_name}] 分析中...")
        params = extract_params(module_path, algo_name)
        all_params[algo_name] = params
        if params:
            for k, v in params.items():
                print(f"    {k} = {v}")
        else:
            print(f"    (无特殊参数)")

    # 询问输出
    default_output = str(PROJECT_ROOT / 'algorithm_params.csv')
    out_choice = input(f"\n输出CSV文件路径 (回车使用默认: {default_output}): ").strip()
    output_path = out_choice if out_choice else default_output

    # 构建表格
    data = build_table(all_params)
    algos = sorted(all_params.keys())

    # 保存 CSV
    if data:
        rows = []
        for param, row in data.items():
            r = {'参数': param, '说明': row['说明']}
            for algo in algos:
                r[algo] = row[algo]
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n参数表格已保存至: {output_path}")
    else:
        print("\n未收集到任何参数。")
        return

    # 打印表格
    print(f"\n{'=' * 60}")
    print(f"  算法参数对比 (共 {len(algos)} 个算法)")
    print(f"{'=' * 60}")
    print_table(data, algos)


def main():
    parser = argparse.ArgumentParser(description='异常检测算法参数收集工具')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出CSV文件路径 (默认: algorithm_params.csv)')
    parser.add_argument('--no-interactive', '-n', action='store_true',
                        help='非交互模式, 自动扫描所有算法')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='强制使用交互式模式')
    args = parser.parse_args()

    # 检查是否有命令行参数
    has_cli_args = args.output is not None or args.no_interactive

    if args.interactive or not has_cli_args:
        interactive_main()
        return

    # 非交互模式
    algo_modules = find_algorithm_modules()
    if not algo_modules:
        print("未找到任何算法的 detector.py 文件。")
        return

    print(f"扫描到 {len(algo_modules)} 个算法, 正在提取参数...\n")

    all_params = {}
    for algo_name, module_path in sorted(algo_modules.items()):
        print(f"  [{algo_name}]")
        params = extract_params(module_path, algo_name)
        all_params[algo_name] = params

    data = build_table(all_params)
    algos = sorted(all_params.keys())

    output_path = args.output or str(PROJECT_ROOT / 'algorithm_params.csv')

    if data:
        rows = []
        for param, row in data.items():
            r = {'参数': param, '说明': row['说明']}
            for algo in algos:
                r[algo] = row[algo]
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n参数表格已保存至: {output_path}")

    print(f"\n{'=' * 60}")
    print(f"  算法参数对比 (共 {len(algos)} 个算法)")
    print(f"{'=' * 60}")
    print_table(data, algos)


if __name__ == '__main__':
    main()
