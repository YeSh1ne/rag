import json
import re
from pathlib import Path

def _extract_page_map(markdown: str) -> dict:
    """从 markdown 中提取行号到页码的映射"""
    page_map = {}
    current_page = 1
    for i, line in enumerate(markdown.split("\n")):
        # 识别 <span id="page-X-Y">
        if '<span id="page-' in line:
            match = re.search(r'page-(\d+)-', line)
            if match:
                current_page = int(match.group(1)) + 1  # 0-based → 1-based
        # 识别 {N}---
        elif re.search(r'\{(\d+)\}-+', line):
            match = re.search(r'\{(\d+)\}-+', line)
            if match:
                current_page = int(match.group(1)) + 1  # {1}--- → 第2页
        page_map[i] = current_page
    return page_map

def get_chunk_page_from_first_line(chunk_text: str) -> int:
    """检查 chunk 第一行是否有分页标记"""
    first_line = chunk_text.split("\n")[0].strip()
    
    # 识别 <span id="page-X-Y">
    match = re.search(r'<span id="page-(\d+)-', first_line)
    if match:
        return int(match.group(1)) + 1
    
    # 识别 {N}---
    match = re.search(r'\{(\d+)\}-+', first_line)
    if match:
        return int(match.group(1)) + 1
    
    return None

def get_chunk_page_from_anywhere(chunk_text: str) -> int:
    """检查 chunk 任何位置是否有分页标记"""
    # 识别 <span id="page-X-Y">
    match = re.search(r'<span id="page-(\d+)-', chunk_text)
    if match:
        return int(match.group(1)) + 1
    
    # 识别 {N}---
    match = re.search(r'\{(\d+)\}-+', chunk_text)
    if match:
        return int(match.group(1)) + 1
    
    return None

def get_chunk_start_line(markdown: str, chunk_text: str) -> int:
    """找到 chunk 文本在 markdown 中的起始行号"""
    lines = markdown.split("\n")
    
    # 取 chunk 的第一行有效内容（跳过空行、分页标记、分隔符）
    for line in chunk_text.split("\n"):
        first_line = line.strip()
        if first_line and not first_line.startswith("<span") and not first_line.startswith("{") and "---" not in first_line:
            break
    
    # 在 markdown 中找这一行
    for i, md_line in enumerate(lines):
        if first_line[:50] in md_line:
            return i
    
    return 0

def fix_page_map(output_dir: str):
    """批量修复 JSON 文件中的 page_map 和 chunks 的 page 字段"""
    json_files = sorted(Path(output_dir).glob("*.json"))
    print(f"📁 共 {len(json_files)} 个文件待修复")
    
    for idx, json_path in enumerate(json_files, 1):
        print(f"[{idx}/{len(json_files)}] 修复: {json_path.name}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 重新提取 page_map
        new_page_map = _extract_page_map(data["markdown"])
        data["page_map"] = new_page_map
        
        # 修复每个 chunk 的 page 字段
        fixed_count = 0
        prev_page = 1  # 跟踪前一个 chunk 的页码
        
        for chunk in data["chunks"]:
            # 方法：用 chunk 起始行匹配页码
            start_line = get_chunk_start_line(data["markdown"], chunk["text"])
            current_page = data["page_map"].get(start_line, 1)
            
            # 确保页码不会回退（非递减）
            if current_page < prev_page:
                current_page = prev_page
            
            if chunk["page"] != current_page:
                chunk["page"] = current_page
                fixed_count += 1
            
            prev_page = current_page
        
        # 保存
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        chunk_pages = [c["page"] for c in data["chunks"]]
        print(f"   ✅ 完成，修复了 {fixed_count} 个 chunk，页码范围: {min(chunk_pages)}-{max(chunk_pages)}")

if __name__ == "__main__":
    fix_page_map("./parsed_output_256")
    print("\n🎉 所有 page_map 和 chunks.page 修复完成！")