"""
parsePaper_GPU.py - PDF 解析与分片（GPU 加速版）

功能：
- ✅ 自动检测并使用 GPU
- ✅ 支持按 token/字符分片
- ✅ 性能统计（耗时、速度）
- ✅ 跳过已处理文件
- ✅ 完整错误处理
- ✅ 多实验对比支持
"""

import os
import json
import re
import time
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
import tiktoken
import torch

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    计算文本的 token 数量
    
    Args:
        text: 待计算的文本
        encoding_name: 编码器名称，cl100k_base 适用于 BGE-M3、GPT 等模型
    
    Returns:
        token 数量
    """
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def split_into_chunks_by_tokens(markdown: str, page_map: dict, paper_id: str, 
                                 chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """
    按 token 数量分片（适合对比实验）
    
    Args:
        markdown: Markdown 格式的文本
        page_map: 行号到页码的映射 {line_number: page_number}
        paper_id: 论文 ID（文件名）
        chunk_size: 每个 chunk 的最大 token 数
        chunk_overlap: 重叠的 token 数
    
    Returns:
        分片列表
    """
    lines = markdown.split("\n")
    chunks = []
    current_chunk_lines = []
    current_chunk_tokens = 0
    chunk_start_page = 1
    chunk_idx = 0
    
    for i, line in enumerate(lines):
        # ✅ 先获取页码（page_map 已包含标记行的正确页码）
        page_num = page_map.get(i, 1)
        
        # ✅ 再跳过分页标记行（此时 page_num 已是最新正确值）
        stripped_line = line.strip()
        if (stripped_line.startswith("<!-- Page") or 
            '<span id="page-' in stripped_line or 
            re.match(r'^\{\d+\}-+$', stripped_line)):
            continue
        
        line_tokens = count_tokens(line, "cl100k_base")
        
        # 如果当前 chunk 已满，保存并开始新 chunk
        if current_chunk_tokens + line_tokens > chunk_size and current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunk_id = f"{paper_id}_chunk_{chunk_idx:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": chunk_start_page,
                    "paper_id": paper_id,
                    "token_count": current_chunk_tokens,
                    "char_count": len(chunk_text)
                })
                chunk_idx += 1
            
            # 重叠策略：保留最后几行作为下一个 chunk 的开头
            if chunk_overlap > 0:
                overlap_lines = []
                overlap_tokens = 0
                for prev_line in reversed(current_chunk_lines):
                    prev_tokens = count_tokens(prev_line, "cl100k_base")
                    if overlap_tokens + prev_tokens > chunk_overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_tokens += prev_tokens
                current_chunk_lines = overlap_lines
                current_chunk_tokens = overlap_tokens
            else:
                current_chunk_lines = []
                current_chunk_tokens = 0
            
            chunk_start_page = page_num
        
        current_chunk_lines.append(line)
        current_chunk_tokens += line_tokens
    
    # 处理最后一个 chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunk_id = f"{paper_id}_chunk_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": chunk_start_page,
                "paper_id": paper_id,
                "token_count": current_chunk_tokens,
                "char_count": len(chunk_text)
            })
    
    return chunks

def split_into_chunks(markdown: str, page_map: dict, paper_id: str, 
                      chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    将 Markdown 文本按字符数分片（兼容旧版本）
    
    Args:
        markdown: Markdown 格式的文本
        page_map: 行号到页码的映射 {line_number: page_number}
        paper_id: 论文 ID（文件名）
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 重叠的字符数
    
    Returns:
        分片列表
    """
    lines = markdown.split("\n")
    chunks = []
    current_chunk_lines = []
    current_chunk_len = 0
    chunk_start_page = 1
    chunk_idx = 0
    
    for i, line in enumerate(lines):
        page_num = page_map.get(i, 1)
        
        # 遇到分页标记，重新开始一个 chunk（可选策略）
        if (line.strip().startswith("<!-- Page") or 
            '<span id="page-' in line or 
            re.search(r'\{\d+\}-+', line)):
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines).strip()
                if chunk_text:
                    chunk_id = f"{paper_id}_chunk_{chunk_idx:03d}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "page": chunk_start_page,
                        "paper_id": paper_id,
                        "char_count": len(chunk_text)
                    })
                    chunk_idx += 1
                current_chunk_lines = []
                current_chunk_len = 0
            continue
        
        # 如果当前 chunk 已满，保存并开始新 chunk
        line_len = len(line)
        if current_chunk_len + line_len > chunk_size and current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines).strip()
            if chunk_text:
                chunk_id = f"{paper_id}_chunk_{chunk_idx:03d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": chunk_start_page,
                    "paper_id": paper_id,
                    "char_count": len(chunk_text)
                })
                chunk_idx += 1
            
            # 重叠策略：保留最后几行作为下一个 chunk 的开头
            if chunk_overlap > 0:
                overlap_lines = []
                overlap_len = 0
                for prev_line in reversed(current_chunk_lines):
                    if overlap_len + len(prev_line) > chunk_overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_len += len(prev_line)
                current_chunk_lines = overlap_lines
                current_chunk_len = overlap_len
            else:
                current_chunk_lines = []
                current_chunk_len = 0
            
            chunk_start_page = page_num
        
        current_chunk_lines.append(line)
        current_chunk_len += line_len
    
    # 处理最后一个 chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunk_id = f"{paper_id}_chunk_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page": chunk_start_page,
                "paper_id": paper_id,
                "char_count": len(chunk_text)
            })
    
    return chunks

def _extract_page_map(markdown: str) -> dict:
    page_map = {}
    current_page = 1
    lines = markdown.split("\n")
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 格式1: <!-- Page X --> (1-indexed)
        if stripped.startswith("<!-- Page"):
            try:
                new_page = int(stripped.replace("<!-- Page", "").replace("-->", "").strip())
                current_page = max(current_page, new_page)
            except ValueError:
                pass
        
        # 格式2: <span id="page-X-Y"></span> (X is 0-indexed)
        elif '<span id="page-' in stripped:
            match = re.search(r'page-(\d+)-\d+', stripped)
            if match:
                new_page = int(match.group(1)) + 1
                current_page = max(current_page, new_page)
        
        # 格式3: {N}--- (N is 0-indexed)
        elif re.match(r'^\{(\d+)\}-+$', stripped):
            match = re.search(r'\{(\d+)\}-+', stripped)
            if match:
                new_page = int(match.group(1)) + 1
                current_page = max(current_page, new_page)
        
        page_map[i] = current_page
    
    # ✅ 添加诊断：检查 page_map 是否真的检测到了多页
    unique_pages = set(page_map.values())
    if len(unique_pages) <= 1:
        print(f"   ⚠️  警告: page_map 仅检测到 {unique_pages}，可能分页标记格式未被识别")
        # 打印前几行帮助排查
        for line in lines[:20]:
            if 'page' in line.lower() or '<!--' in line or '{' in line:
                print(f"      疑似标记行: {line.strip()[:80]}")
    
    return page_map

def parse_pdf_to_markdown(pdf_dir: str, markdown_output_dir: str,
                         batch_size: int = 4, skip_existing: bool = True):
    """
    阶段1：解析PDF为Markdown（只做一次）
    
    Args:
        pdf_dir: PDF文件所在目录
        markdown_output_dir: 原始Markdown输出目录
        batch_size: GPU批处理大小
        skip_existing: 跳过已处理文件
    
    Returns:
        解析成功的文件列表
    """
    print("="*60)
    print("📄 阶段1：PDF → Markdown（只执行一次）")
    print("="*60)
    
    # GPU检测
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        device = 'cpu'
        print("⚠️  CPU 模式")
        batch_size = 1
    
    # 加载Marker模型
    print("\n📦 加载Marker模型...")
    start_time = time.time()
    config = {
        "output_format": "markdown",
        "paginate_output": True,
        "batch_size": batch_size,
        "force_ocr": False,
        "ocr_all_pages": False,
        "languages": ["en"],
        "table_output_format": "markdown",
        "strip_existing_ocr": False,
        "disable_image_extraction": True,
    }
    model_dict = create_model_dict()
    converter = PdfConverter(
        config=ConfigParser(config).generate_config_dict(),
        artifact_dict=model_dict,
    )
    print(f"✅ 模型加载完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 解析流程
    os.makedirs(markdown_output_dir, exist_ok=True)
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    print(f"\n📄 共 {len(pdf_files)} 个PDF待处理")
    print(f"   输出目录: {markdown_output_dir}")
    print(f"   跳过已处理: {skip_existing}\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] 📄 处理: {pdf_path.name}")
        
        # 检查是否已处理
        md_path = Path(markdown_output_dir) / f"{pdf_path.stem}.json"
        if skip_existing and md_path.exists():
            print(f"   ⏭️  已存在，跳过")
            skip_count += 1
            continue
        
        file_start_time = time.time()
        
        try:
            # 执行PDF转换
            rendered = converter(str(pdf_path))
            markdown_text = rendered.markdown
            metadata = rendered.metadata
            
            # 提取页码映射
            page_map = _extract_page_map(markdown_text)
            
            # 保存原始Markdown（不分片）
            result = {
                "paper_id": pdf_path.stem,
                "title": metadata.get("title", pdf_path.stem),
                "markdown": markdown_text,
                "page_map": page_map,
                "source_file": pdf_path.name,
                "device": device,
                "processing_time": time.time() - file_start_time
            }
            
            with open(md_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            print(f"   ✅ 完成！耗时: {time.time() - file_start_time:.2f}秒")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            fail_count += 1
            continue
    
    # 总结
    print("\n" + "="*60)
    print("🎉 阶段1完成！")
    print("="*60)
    print(f"   处理成功: {success_count} 篇")
    print(f"   跳过: {skip_count} 篇")
    print(f"   失败: {fail_count} 篇")
    print("="*60)
    
    return success_count
def chunk_markdown_by_size(markdown_dir: str, output_dir: str,
                           chunk_size: int = 512, chunk_overlap: int = 50,
                           use_token_split: bool = True, skip_existing: bool = True,
                           regenerate_page_map: bool = True):  # ✅ 新增参数
    """
    阶段2：将已解析的Markdown按chunk_size分片
    
    Args:
        ...（原有参数）
        regenerate_page_map: 是否重新生成 page_map（修复页码错误的关键）
    """
    print("="*60)
    print(f"📦 阶段2：Markdown → Chunks (chunk_size={chunk_size})")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    md_files = sorted(Path(markdown_dir).glob("*.json"))
    print(f"\n📄 共 {len(md_files)} 个Markdown待分片")
    print(f"   输入目录: {markdown_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   重新生成 page_map: {regenerate_page_map}")  # ✅ 新增提示
    print(f"   分片方式: {'token' if use_token_split else 'char'}")
    print(f"   Chunk Size: {chunk_size}")
    print(f"   Chunk Overlap: {chunk_overlap}\n")
    
    results = []
    success_count = 0
    skip_count = 0
    
    for idx, md_path in enumerate(md_files, 1):
        print(f"[{idx}/{len(md_files)}] 📄 处理: {md_path.stem}")
        
        # 检查是否已处理
        out_path = Path(output_dir) / md_path.name
        if skip_existing and out_path.exists():
            print(f"   ⏭️  已存在，跳过")
            skip_count += 1
            continue
        
        file_start_time = time.time()
        
        try:
            # 读取原始Markdown
            with open(md_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 执行分片
            paper_id = data["paper_id"]
            markdown_text = data["markdown"]
            
            # ✅ 核心修复：重新生成 page_map，不使用 JSON 中的旧数据
            if regenerate_page_map:
                page_map = _extract_page_map(markdown_text)
                unique_pages = len(set(page_map.values()))
                print(f"   🔧 重新生成 page_map: 检测到 {unique_pages} 页")
            else:
                page_map = data["page_map"]  # 保留旧逻辑（可选）
                print(f"   📌 使用 JSON 中的旧 page_map")
            
            if use_token_split:
                chunks = split_into_chunks_by_tokens(markdown_text, page_map, paper_id,
                                                     chunk_size, chunk_overlap)
            else:
                chunks = split_into_chunks(markdown_text, page_map, paper_id,
                                          chunk_size, chunk_overlap)
            
            # 构建结果
            result = {
                "paper_id": paper_id,
                "title": data.get("title", paper_id),
                "markdown": markdown_text,
                "page_map": page_map,  # ✅ 保存新生成的 page_map
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chunk_size_config": chunk_size,
                "chunk_overlap_config": chunk_overlap,
                "split_method": "token" if use_token_split else "char",
                "source_file": data["source_file"],
                "processing_time": time.time() - file_start_time
            }
            
            # 保存分片结果
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            results.append(result)
            success_count += 1
            
            # 打印统计
            if use_token_split:
                total_tokens = sum(c.get('token_count', 0) for c in chunks)
                print(f"   ✅ 完成！Chunks: {len(chunks)}, Tokens: {total_tokens}, 耗时: {time.time() - file_start_time:.2f}秒")
            else:
                print(f"   ✅ 完成！Chunks: {len(chunks)}, 耗时: {time.time() - file_start_time:.2f}秒")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            continue
    
    # 总结
    print("\n" + "="*60)
    print("🎉 阶段2完成！")
    print("="*60)
    print(f"   处理成功: {success_count} 篇")
    print(f"   跳过: {skip_count} 篇")
    print("="*60)
    
    return results
    
if __name__ == "__main__":
    # ========== GPU检测 ==========
    print("\n" + "="*60)
    print("🔍 系统检测")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size = 8 if gpu_memory > 16 else 4
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.2f} GB)")
    else:
        print("⚠️  CPU 模式")
        batch_size = 1
    
    # ========== 配置路径 ==========
    PDF_DIR = r'E:\桌面\深圳大学项目\论文'
    MARKDOWN_OUTPUT_DIR = "./parsed_markdown_raw"  # 原始Markdown保存目录
    
    # ========== 阶段1：解析PDF（只执行一次）==========
    print("\n" + "="*60)
    print("🚀 启动阶段1：PDF → Markdown")
    print("="*60)
    
    parse_pdf_to_markdown(
        pdf_dir=PDF_DIR,
        markdown_output_dir=MARKDOWN_OUTPUT_DIR,
        batch_size=batch_size,
        skip_existing=True
    )
    
    # ========== 阶段2：按不同chunk_size分片 ==========
    print("\n" + "="*60)
    print("🚀 启动阶段2：Markdown → Chunks（不同chunk_size）")
    print("="*60)
    
    # 实验1: chunk_size=256
    print("\n" + "="*60)
    print("🧪 实验1: chunk_size=256 (token)")
    print("="*60)
    papers_256 = chunk_markdown_by_size(
        markdown_dir=MARKDOWN_OUTPUT_DIR,
        output_dir="./parsed_output_256",
        chunk_size=256,
        chunk_overlap=50,
        use_token_split=True,
        skip_existing=True
    )
    
    # 实验2: chunk_size=512
    print("\n" + "="*60)
    print("🧪 实验2: chunk_size=512 (token)")
    print("="*60)
    papers_512 = chunk_markdown_by_size(
        markdown_dir=MARKDOWN_OUTPUT_DIR,
        output_dir="./parsed_output_512",
        chunk_size=512,
        chunk_overlap=50,
        use_token_split=True,
        skip_existing=True
    )
    
    # 实验3: chunk_size=1024
    print("\n" + "="*60)
    print("🧪 实验3: chunk_size=1024 (token)")
    print("="*60)
    papers_1024 = chunk_markdown_by_size(
        markdown_dir=MARKDOWN_OUTPUT_DIR,
        output_dir="./parsed_output_1024",
        chunk_size=1024,
        chunk_overlap=100,
        use_token_split=True,
        skip_existing=True
    )
    
    # ========== 最终总结 ==========
    print("\n" + "="*60)
    print("📊 实验总结")
    print("="*60)
    print(f"   ✅ 实验1 (256 token): {len(papers_256)} 篇论文")
    print(f"   ✅ 实验2 (512 token): {len(papers_512)} 篇论文")
    print(f"   ✅ 实验3 (1024 token): {len(papers_1024)} 篇论文")
    print("="*60)
    
    # 打印分片示例
    if papers_256:
        print(f"\n📝 分片示例（第一篇论文，256 token）:")
        for i, chunk in enumerate(papers_256[0]["chunks"][:3]):
            print(f"\n--- [{chunk['chunk_id']}] (Page {chunk['page']}, {chunk['token_count']} tokens) ---")
            print(chunk["text"][:150] + "...")

