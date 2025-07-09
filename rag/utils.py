import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
from bs4 import BeautifulSoup
import re

# 分词器，用于本地统计chunk的token长度来进行划分
enc = tiktoken.get_encoding("cl100k_base")

class ReadFiles:
    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith('.md'):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        all_tokens = enc.encode(text)
        all_tokens_num = len(all_tokens)

        chunk_list = []
        current_pos = 0

        effective_cover_content = min(cover_content, max_token_len - 1)
        if effective_cover_content < 0: effective_cover_content = 0

        while current_pos < all_tokens_num:
            chunk_start = max(0, current_pos - effective_cover_content)
            chunk_end = max_token_len + chunk_start
            if chunk_end > all_tokens_num:
                chunk_end = all_tokens_num
            if all_tokens_num - chunk_start < max_token_len and chunk_start > 0:
                # 剩余长度不足最大长度
                chunk_start = max(0, all_tokens_num - max_token_len)
                chunk_end = all_tokens_num

            current_chunk_tokens = all_tokens[chunk_start:chunk_end]
            chunk_list.append(enc.decode(current_chunk_tokens))

            # 更新下一个chunk的起始位置
            current_pos += (max_token_len - effective_cover_content)
            if current_pos >= all_tokens_num:
                break
        return chunk_list

    @classmethod
    def read_file_content(cls, file_path: str):
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text)
            return text

    @classmethod
    def read_text(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class Documents:
    def __init__(self, path: str = ''):
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content