# Система HR с AI, геймификацией и workflow - полная версия
import streamlit as st
import os
import json
import sqlite3
import time
import psutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import subprocess
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shutil
from openai import OpenAI
import faiss
from rank_bm25 import BM25Okapi
from io import BytesIO
import PyPDF2
from docx import Document
import pdfplumber
import openpyxl
import tempfile

# SciBox API конфигурация
SCIBOX_CONFIG = {
    'api_key': "sk-LRwqBFBToIkqBPogfcTxlw",
    'base_url': "https://llm.t1v.scibox.tech/v1",
    'llm_model': "Qwen2.5-72B-Instruct-AWQ",
    'embedding_model': "bge-m3"
}

# Создание клиента OpenAI для SciBox
client = OpenAI(
    api_key=SCIBOX_CONFIG['api_key'],
    base_url=SCIBOX_CONFIG['base_url']
)

# Добавляем клиента в конфигурацию
SCIBOX_CONFIG['client'] = client

# Streamlit конфигурация
st.set_page_config(
    page_title="AI HR система с геймификацией | MilRAG",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пути для workflow
JOB_RESUME_DIR = Path("./job_resume")
MANAGER_REVIEW_DIR = Path("./manager_review")
HR_FINAL_DIR = Path("./hr_final")
HR_DATABASE_PATH = "hr_shared_database.db"

# Создание директорий для workflow
for dir_path in [JOB_RESUME_DIR, MANAGER_REVIEW_DIR, HR_FINAL_DIR]:
    dir_path.mkdir(exist_ok=True)
    for subdir in ["pending", "approved", "rejected"]:
        (dir_path / subdir).mkdir(exist_ok=True)

# Инициализация session state
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

if 'scanned_skills' not in st.session_state:
    st.session_state.scanned_skills = {}

if 'skills_edit_mode' not in st.session_state:
    st.session_state.skills_edit_mode = False

if 'current_user_role' not in st.session_state:
    st.session_state.current_user_role = "employee"

@dataclass
class SkillDetection:
    skill_name: str
    evidence_type: str  # "process", "package", "file", "command"
    confidence: float   # 0.0 - 1.0
    last_detected: datetime
    total_time_minutes: int = 0
    frequency: int = 1
    description: str = ""
    category: str = ""
    experience_level: str = "Начинающий"  # "Начинающий", "Средний", "Продвинутый"
    icon: str = "🔧"

@dataclass
class UserProfile:
    username: str
    session_start: datetime
    skills_detected: Dict[str, SkillDetection]
    total_xp: int = 0
    level: int = 1
    badges: Set[str] = None
    daily_streak: int = 0
    last_activity: datetime = None
    additional_info: str = ""
    contact_info: Dict[str, str] = None
    career_goals: str = ""
    current_projects: List[str] = None

    def __post_init__(self):
        if self.badges is None:
            self.badges = set()
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.contact_info is None:
            self.contact_info = {}
        if self.current_projects is None:
            self.current_projects = []


class AICareerConsultant:
    """AI консультант для карьерных рекомендаций"""

    def __init__(self):
        self.client = OpenAI(
            api_key=SCIBOX_CONFIG['api_key'],
            base_url=SCIBOX_CONFIG['base_url']
        )

    def generate_career_recommendations(self, profile_data: Dict) -> str:
        """Генерация карьерных рекомендаций через AI"""
        try:
            # Формируем промпт с данными профиля
            profile_info = profile_data.get('profile_info', {})
            skills = profile_data.get('skills', {})
            user_profile = profile_data.get('user_profile', {})

            skills_list = []
            for skill_name, skill_data in skills.items():
                level = skill_data.get('experience_level', 'Начинающий')
                confidence = skill_data.get('confidence', 0)
                skills_list.append(f"{skill_name} ({level}, {confidence:.2f})")

            prompt = f"""
Проанализируй профиль сотрудника и дай карьерные рекомендации:

ПРОФИЛЬ СОТРУДНИКА:
Имя: {profile_info.get('user_name', 'Не указано')}
Желаемая должность: {profile_info.get('position', 'Не указано')}
Локация: {profile_info.get('location', 'Не указано')}
Уровень: {user_profile.get('level', 1)}
Общий XP: {user_profile.get('total_xp', 0)}

НАВЫКИ:
{chr(10).join(skills_list[:10])}

ЗАДАЧА: 
Дай конкретные рекомендации по карьерному развитию, подходящие позиции, пробелы в навыках.
Ответ должен быть практическим и структурированным.
"""

            response = self.client.chat.completions.create(
                model=SCIBOX_CONFIG['llm_model'],
                messages=[
                    {"role": "system", "content": "Ты - опытный HR консультант и карьерный коуч."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Ошибка генерации рекомендаций: {e}"


class HRDatabaseManager:
    def __init__(self, db_path: str = HR_DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация базы данных HR"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    position TEXT,
                    department TEXT,
                    email TEXT,
                    phone TEXT,
                    skills TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    requirements TEXT,
                    department TEXT,
                    source_file TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS scanned_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    skills_detected TEXT,
                    total_xp INTEGER DEFAULT 0,
                    level INTEGER DEFAULT 1,
                    badges TEXT,
                    scan_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def get_all_employees(self) -> List[Dict]:
        """Получение всех сотрудников"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM employees")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_positions(self) -> List[Dict]:
        """Получение всех позиций"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM positions")
            return [dict(row) for row in cursor.fetchall()]

    def get_scanned_profiles(self) -> List[Dict]:
        """Получение всех сканированных профилей"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM scanned_profiles")
            profiles = []
            for row in cursor.fetchall():
                profile = dict(row)
                # Парсим JSON поля
                try:
                    profile['skills_detected'] = json.loads(profile['skills_detected'] or '{}')
                    profile['badges'] = json.loads(profile['badges'] or '[]')
                    profile['scan_data'] = json.loads(profile['scan_data'] or '{}')
                except:
                    pass
                profiles.append(profile)
            return profiles

    def add_scanned_profile(self, profile_data: Dict) -> int:
        """Добавление сканированного профиля"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO scanned_profiles (username, skills_detected, total_xp, level, badges, scan_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile_data.get('username', ''),
                json.dumps(profile_data.get('skills_detected', {}), ensure_ascii=False),
                profile_data.get('total_xp', 0),
                profile_data.get('level', 1),
                json.dumps(profile_data.get('badges', []), ensure_ascii=False),
                json.dumps(profile_data.get('scan_data', {}), ensure_ascii=False)
            ))
            conn.commit()
            return cursor.lastrowid

    def add_position(self, position_data: Dict) -> int:
        """Добавление позиции"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO positions (title, description, requirements, department, source_file)
                VALUES (?, ?, ?, ?, ?)
            """, (
                position_data.get('title', ''),
                position_data.get('description', ''),
                position_data.get('requirements', ''),
                position_data.get('department', ''),
                position_data.get('source_file', '')
            ))
            conn.commit()
            return cursor.lastrowid


class HRSystemAdvanced:
    """Расширенная HR система с AI и поиском"""

    def __init__(self):
        self.db = HRDatabaseManager()
        self.client = OpenAI(
            api_key=SCIBOX_CONFIG['api_key'],
            base_url=SCIBOX_CONFIG['base_url']
        )
        self.faiss_index = None
        self.documents = []
        self.bm25 = None
        self._init_search_indexes()

    def _init_search_indexes(self):
        """Инициализация поисковых индексов"""
        try:
            # Получаем все данные из базы
            employees = self.db.get_all_employees()
            positions = self.db.get_all_positions()
            profiles = self.db.get_scanned_profiles()

            # Подготавливаем документы для индексации
            documents = []

            # Индексируем сотрудников
            for emp in employees:
                doc_text = f"Сотрудник: {emp.get('name', '')} {emp.get('position', '')} {emp.get('department', '')} {emp.get('skills', '')}"
                documents.append({
                    'text': doc_text,
                    'metadata': {'type': 'employee', 'data': emp}
                })

            # Индексируем позиции
            for pos in positions:
                doc_text = f"Позиция: {pos.get('title', '')} {pos.get('description', '')} {pos.get('requirements', '')}"
                documents.append({
                    'text': doc_text,
                    'metadata': {'type': 'position', 'data': pos}
                })

            # Индексируем профили
            for profile in profiles:
                skills_text = " ".join([skill for skill in profile.get('skills_detected', {}).keys()])
                doc_text = f"Профиль: {profile.get('username', '')} навыки: {skills_text}"
                documents.append({
                    'text': doc_text,
                    'metadata': {'type': 'scanned_profile', 'data': profile}
                })

            self.documents = documents

            # Инициализация BM25
            if documents:
                corpus = [doc['text'] for doc in documents]
                self.bm25 = BM25Okapi([doc.split() for doc in corpus])

        except Exception as e:
            st.warning(f"Ошибка инициализации поиска: {e}")

    def parse_document(self, uploaded_file) -> Tuple[str, Dict]:
        """Парсинг документа через AI"""
        try:
            content = ""

            # Извлечение текста в зависимости от типа файла
            if uploaded_file.type == "application/pdf":
                content = self._extract_pdf_content(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = self._extract_docx_content(uploaded_file)
            elif uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                content = self._extract_excel_content(uploaded_file)
            elif uploaded_file.type == "text/csv":
                content = self._extract_csv_content(uploaded_file)
            elif uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/json":
                content = str(uploaded_file.read(), "utf-8")

            # AI структурирование данных
            structured_data = self._structure_with_ai(content, uploaded_file.name)

            return content, structured_data

        except Exception as e:
            raise Exception(f"Ошибка парсинга документа: {e}")

    def _extract_pdf_content(self, uploaded_file) -> str:
        """Извлечение текста из PDF"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()

                with pdfplumber.open(tmp_file.name) as pdf:
                    content = ""
                    for page in pdf.pages:
                        content += page.extract_text() + "\n"

                os.unlink(tmp_file.name)
                return content
        except Exception as e:
            return f"Ошибка извлечения PDF: {e}"

    def _extract_docx_content(self, uploaded_file) -> str:
        """Извлечение текста из DOCX"""
        try:
            doc = Document(uploaded_file)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        except Exception as e:
            return f"Ошибка извлечения DOCX: {e}"

    def _extract_excel_content(self, uploaded_file) -> str:
        """Извлечение данных из Excel"""
        try:
            df = pd.read_excel(uploaded_file)
            return df.to_string()
        except Exception as e:
            return f"Ошибка извлечения Excel: {e}"

    def _extract_csv_content(self, uploaded_file) -> str:
        """Извлечение данных из CSV"""
        try:
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        except Exception as e:
            return f"Ошибка извлечения CSV: {e}"

    def _structure_with_ai(self, content: str, filename: str) -> Dict:
        """Структурирование данных через AI"""
        try:
            prompt = f"""
Проанализируй следующий документ и извлеки структурированную информацию:

ФАЙЛ: {filename}
СОДЕРЖИМОЕ:
{content[:2000]}

Определи тип документа и извлеки:
1. Если это резюме - имя, навыки, опыт, контакты
2. Если это описание вакансии - название, требования, описание
3. Если это профиль сотрудника - личные данные, навыки, достижения

Верни результат в JSON формате с полями:
- type: тип документа (resume/vacancy/employee_profile/other)
- extracted_data: основные данные
- skills: список навыков
- metadata: дополнительная информация
"""

            response = self.client.chat.completions.create(
                model=SCIBOX_CONFIG['llm_model'],
                messages=[
                    {"role": "system", "content": "Ты - эксперт по анализу HR документов."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )

            # Парсим JSON ответ
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            # Возвращаем базовую структуру при ошибке
            return {
                'type': 'other',
                'extracted_data': {'content': content[:500]},
                'skills': [],
                'metadata': {'filename': filename, 'error': str(e)}
            }

    def add_document_to_database(self, structured_data: Dict, filename: str) -> str:
        """Добавление документа в базу данных"""
        try:
            doc_type = structured_data.get('type', 'other')

            if doc_type == 'resume' or doc_type == 'scanned_profile':
                # Добавляем как профиль сотрудника
                return self._add_employee_profile(structured_data, filename)
            elif doc_type == 'vacancy':
                # Добавляем как позицию
                return self._add_position(structured_data, filename)
            else:
                # Добавляем как общий документ
                return self._add_general_document(structured_data, filename)

        except Exception as e:
            return f"Ошибка добавления в базу: {e}"

    def _add_employee_profile(self, data: Dict, filename: str) -> str:
        """Добавление профиля сотрудника"""
        try:
            extracted = data.get('extracted_data', {})

            profile_data = {
                'username': extracted.get('name', data.get('username', 'Unknown')),
                'skills_detected': data.get('skills_detected', {}),
                'total_xp': data.get('total_xp', 0),
                'level': data.get('level', 1),
                'badges': data.get('badges', []),
                'scan_data': data
            }

            profile_id = self.db.add_scanned_profile(profile_data)
            self._update_search_indexes()

            return f"Профиль сотрудника добавлен (ID: {profile_id})"

        except Exception as e:
            return f"Ошибка добавления профиля: {e}"

    def _add_position(self, data: Dict, filename: str) -> str:
        """Добавление позиции"""
        try:
            extracted = data.get('extracted_data', {})

            position_data = {
                'title': extracted.get('title', 'Unknown Position'),
                'description': extracted.get('description', ''),
                'requirements': str(data.get('skills', [])),
                'department': extracted.get('department', ''),
                'source_file': filename
            }

            position_id = self.db.add_position(position_data)
            self._update_search_indexes()

            return f"Позиция добавлена (ID: {position_id})"

        except Exception as e:
            return f"Ошибка добавления позиции: {e}"

    def _add_general_document(self, data: Dict, filename: str) -> str:
        """Добавление общего документа"""
        return f"Документ {filename} обработан и сохранен"

    def _update_search_indexes(self):
        """Обновление поисковых индексов"""
        self._init_search_indexes()

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Гибридный поиск BM25 + семантический"""
        try:
            if not self.bm25 or not self.documents:
                return []

            # BM25 поиск
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            # Формируем результаты
            results = []
            for i, (doc, score) in enumerate(zip(self.documents, bm25_scores)):
                if score > 0:
                    results.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': float(score)
                    })

            # Сортируем по релевантности
            results.sort(key=lambda x: x['score'], reverse=True)

            return results[:k]

        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            return []

    def smart_query_with_context(self, query: str, k: int = 5) -> str:
        """Умный запрос с контекстом через RAG"""
        try:
            # Поиск релевантных документов
            search_results = self.hybrid_search(query, k)

            if not search_results:
                return "Не найдено релевантных данных для ответа на ваш запрос."

            # Формируем контекст
            context_parts = []
            for result in search_results:
                context_parts.append(f"[{result['metadata']['type']}] {result['text']}")

            context = "\n\n".join(context_parts)

            # Формируем промпт
            prompt = f"""
Контекст из HR базы данных:
{context}

Вопрос пользователя: {query}

Ответь на вопрос пользователя, используя только информацию из контекста выше. 
Если в контексте нет нужной информации, честно сообщи об этом.
Структурируй ответ и приведи конкретные примеры из найденных данных.
"""

            response = self.client.chat.completions.create(
                model=SCIBOX_CONFIG['llm_model'],
                messages=[
                    {"role": "system", "content": "Ты - AI ассистент для работы с HR данными. Отвечай только на основе предоставленного контекста."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Ошибка обработки запроса: {e}"


class SmartAutoProfiler:
    """Smart Auto-Profiler с детекцией навыков"""

    # Словарь для распознавания процессов
    PROCESS_SKILLS = {
        "pycharm": {"skill": "Python Development", "confidence": 0.95, "category": "Development", "icon": "🐍"},
        "pycharm64": {"skill": "Python Development", "confidence": 0.95, "category": "Development", "icon": "🐍"},
        "code": {"skill": "Code Editing", "confidence": 0.8, "category": "Development", "icon": "💻"},
        "docker": {"skill": "DevOps & Containerization", "confidence": 0.9, "category": "DevOps", "icon": "🐳"},
        "docker-compose": {"skill": "Container Orchestration", "confidence": 0.9, "category": "DevOps", "icon": "🐙"},
        "jupyter": {"skill": "Data Science & Jupyter", "confidence": 0.9, "category": "Data Science", "icon": "📊"},
        "jupyter-lab": {"skill": "Advanced Data Science", "confidence": 0.95, "category": "Data Science", "icon": "🔬"},
        "mysql": {"skill": "MySQL Database", "confidence": 0.85, "category": "Database", "icon": "🗄️"},
        "postgres": {"skill": "PostgreSQL Database", "confidence": 0.85, "category": "Database", "icon": "🐘"},
        "git": {"skill": "Version Control", "confidence": 0.9, "category": "Development", "icon": "🌿"},
        "node": {"skill": "Node.js Development", "confidence": 0.8, "category": "Development", "icon": "🟢"},
        "npm": {"skill": "Package Management", "confidence": 0.7, "category": "Development", "icon": "📦"},
        "python": {"skill": "Python Programming", "confidence": 0.85, "category": "Development", "icon": "🐍"},
        "tensorflow": {"skill": "Deep Learning", "confidence": 0.95, "category": "AI/ML", "icon": "🧠"},
        "streamlit": {"skill": "Web App Development", "confidence": 0.9, "category": "Development", "icon": "⚡"},
        "fastapi": {"skill": "API Development", "confidence": 0.9, "category": "Development", "icon": "🚀"},
        "nginx": {"skill": "Web Server Administration", "confidence": 0.8, "category": "DevOps", "icon": "🌐"},
        "ansible": {"skill": "Configuration Management", "confidence": 0.9, "category": "DevOps", "icon": "⚙️"},
        "kubectl": {"skill": "Kubernetes", "confidence": 0.9, "category": "DevOps", "icon": "☸️"},
        "terraform": {"skill": "Infrastructure as Code", "confidence": 0.9, "category": "DevOps", "icon": "🏗️"}
    }

    # Словарь для распознавания пакетов
    PACKAGE_SKILLS = {
        "streamlit": {"skill": "Streamlit Web Apps", "confidence": 0.9, "category": "Development", "icon": "⚡"},
        "fastapi": {"skill": "FastAPI Development", "confidence": 0.9, "category": "Development", "icon": "🚀"},
        "django": {"skill": "Django Web Framework", "confidence": 0.9, "category": "Development", "icon": "🌱"},
        "flask": {"skill": "Flask Microframework", "confidence": 0.8, "category": "Development", "icon": "🌪️"},
        "pandas": {"skill": "Data Analysis with Pandas", "confidence": 0.9, "category": "Data Science", "icon": "🐼"},
        "numpy": {"skill": "Scientific Computing", "confidence": 0.8, "category": "Data Science", "icon": "🔢"},
        "tensorflow": {"skill": "TensorFlow Deep Learning", "confidence": 0.95, "category": "AI/ML", "icon": "🧠"},
        "pytorch": {"skill": "PyTorch Deep Learning", "confidence": 0.95, "category": "AI/ML", "icon": "🔥"},
        "scikit-learn": {"skill": "Machine Learning", "confidence": 0.9, "category": "AI/ML", "icon": "🤖"},
        "opencv": {"skill": "Computer Vision", "confidence": 0.9, "category": "AI/ML", "icon": "👁️"},
        "selenium": {"skill": "Web Automation", "confidence": 0.8, "category": "Testing", "icon": "🤖"},
        "requests": {"skill": "HTTP Requests & APIs", "confidence": 0.7, "category": "Development", "icon": "🌐"},
        "beautifulsoup4": {"skill": "Web Scraping", "confidence": 0.8, "category": "Development", "icon": "🕷️"},
        "chromadb": {"skill": "Vector Database", "confidence": 0.9, "category": "AI/ML", "icon": "🔍"},
        "langchain": {"skill": "LLM Development", "confidence": 0.9, "category": "AI/ML", "icon": "🔗"},
        "transformers": {"skill": "NLP & Transformers", "confidence": 0.9, "category": "AI/ML", "icon": "🤗"},
        "openai": {"skill": "OpenAI API Integration", "confidence": 0.8, "category": "AI/ML", "icon": "🧠"},
        "plotly": {"skill": "Data Visualization", "confidence": 0.8, "category": "Data Science", "icon": "📊"},
        "matplotlib": {"skill": "Plotting & Visualization", "confidence": 0.8, "category": "Data Science", "icon": "📈"},
        "seaborn": {"skill": "Statistical Visualization", "confidence": 0.8, "category": "Data Science", "icon": "📉"},
        "jupyter": {"skill": "Jupyter Notebooks", "confidence": 0.8, "category": "Data Science", "icon": "📔"},
        "pytest": {"skill": "Python Testing", "confidence": 0.8, "category": "Testing", "icon": "🧪"},
        "redis": {"skill": "Redis Caching", "confidence": 0.8, "category": "Database", "icon": "🔴"},
        "celery": {"skill": "Task Queue Processing", "confidence": 0.8, "category": "Development", "icon": "🌿"}
    }

    # Словарь для файловых паттернов
    FILE_PATTERNS = {
        ".py": {"skill": "Python Development", "confidence": 0.7, "category": "Development", "icon": "🐍"},
        ".js": {"skill": "JavaScript Development", "confidence": 0.7, "category": "Development", "icon": "📜"},
        ".java": {"skill": "Java Development", "confidence": 0.7, "category": "Development", "icon": "☕"},
        ".sql": {"skill": "Database Development", "confidence": 0.8, "category": "Database", "icon": "🗄️"},
        ".dockerfile": {"skill": "Docker Containerization", "confidence": 0.8, "category": "DevOps", "icon": "🐳"},
        "docker-compose.yml": {"skill": "Docker Compose", "confidence": 0.9, "category": "DevOps", "icon": "🐙"},
        ".tf": {"skill": "Terraform IaC", "confidence": 0.9, "category": "DevOps", "icon": "🏗️"},
        ".yaml": {"skill": "YAML Configuration", "confidence": 0.6, "category": "DevOps", "icon": "⚙️"},
        ".yml": {"skill": "YAML Configuration", "confidence": 0.6, "category": "DevOps", "icon": "⚙️"},
        "requirements.txt": {"skill": "Python Dependency Management", "confidence": 0.7, "category": "Development", "icon": "📋"},
        "package.json": {"skill": "Node.js Development", "confidence": 0.8, "category": "Development", "icon": "📦"},
        ".ipynb": {"skill": "Jupyter Notebook Development", "confidence": 0.8, "category": "Data Science", "icon": "📓"}
    }

    # XP награды
    XP_REWARDS = {
        "new_skill_detected": 100,
        "skill_usage_hour": 15,
        "package_installation": 30,
        "project_creation": 150,
        "daily_coding_streak": 50,
        "weekly_consistency": 200,
        "badge_earned": 300,
        "level_up": 500
    }

    # Система достижений
    BADGES = {
        "FirstSteps": {
            "requirement": lambda p: len(p.skills_detected) >= 1,
            "icon": "🎯",
            "desc": "Первые шаги"
        },
        "SkillCollector": {
            "requirement": lambda p: len(p.skills_detected) >= 5,
            "icon": "🎒",
            "desc": "Собрал 5 навыков"
        },
        "Polyglot": {
            "requirement": lambda p: len(p.skills_detected) >= 10,
            "icon": "🌍",
            "desc": "Освоил 10 навыков"
        },
        "Expert": {
            "requirement": lambda p: len(p.skills_detected) >= 15,
            "icon": "👨‍🔬",
            "desc": "Эксперт - 15 навыков"
        },
        "PythonMaster": {
            "requirement": lambda p: any("Python" in s.skill_name for s in p.skills_detected.values()),
            "icon": "🐍",
            "desc": "Мастер Python"
        },
        "MLEnthusiast": {
            "requirement": lambda p: sum(1 for s in p.skills_detected.values() if "ML" in s.skill_name or "Machine Learning" in s.skill_name) >= 2,
            "icon": "🤖",
            "desc": "Энтузиаст ML"
        },
        "DevOpsEngineer": {
            "requirement": lambda p: sum(1 for s in p.skills_detected.values() if "DevOps" in s.skill_name or "Docker" in s.skill_name) >= 3,
            "icon": "⚙️",
            "desc": "DevOps инженер"
        },
        "DataScientist": {
            "requirement": lambda p: sum(1 for s in p.skills_detected.values() if "Data" in s.skill_name) >= 3,
            "icon": "📊",
            "desc": "Data Scientist"
        },
        "CodeWarrior": {
            "requirement": lambda p: p.total_xp >= 1000,
            "icon": "⚔️",
            "desc": "Воин кода - 1000 XP"
        },
        "DedicationMaster": {
            "requirement": lambda p: p.daily_streak >= 7,
            "icon": "🔥",
            "desc": "Мастер упорства - 7 дней"
        },
        "Level5Hero": {
            "requirement": lambda p: p.level >= 5,
            "icon": "🏆",
            "desc": "Герой 5-го уровня"
        },
        "AIPioneer": {
            "requirement": lambda p: any("AI" in s.skill_name or "LLM" in s.skill_name or "OpenAI" in s.skill_name for s in p.skills_detected.values()),
            "icon": "🚀",
            "desc": "Пионер ИИ"
        }
    }

    def __init__(self):
        self.profile = st.session_state.get('user_profile')

    def scan_active_processes(self) -> List[SkillDetection]:
        """Сканирование активных процессов"""
        detected_skills = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'create_time', 'cmdline']):
                try:
                    proc_name = proc.info['name'].lower()
                    cmdline = " ".join(proc.info.get('cmdline', [])).lower()

                    # Ищем соответствия в процессах
                    for pattern, skill_info in self.PROCESS_SKILLS.items():
                        if pattern in proc_name or pattern in cmdline:
                            create_time = datetime.fromtimestamp(proc.info['create_time'])
                            runtime_minutes = max(1, int((datetime.now() - create_time).total_seconds() / 60))

                            skill = SkillDetection(
                                skill_name=skill_info['skill'],
                                evidence_type='process',
                                confidence=skill_info['confidence'],
                                last_detected=datetime.now(),
                                total_time_minutes=runtime_minutes,
                                category=skill_info['category'],
                                icon=skill_info['icon']
                            )
                            detected_skills.append(skill)
                            break

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            st.error(f"Ошибка сканирования процессов: {e}")

        return detected_skills

    def scan_installed_packages(self, venv_path: str) -> List[SkillDetection]:
        """Сканирование установленных пакетов"""
        detected_skills = []

        try:
            # Проверяем разные возможные пути
            possible_paths = [
                Path(venv_path) / "lib" / "python3.11" / "site-packages",
                Path(venv_path) / "lib" / "python3.10" / "site-packages",
                Path(venv_path) / "lib" / "python3.9" / "site-packages",
                Path(venv_path) / "site-packages"
            ]

            site_packages_path = None
            for path in possible_paths:
                if path.exists():
                    site_packages_path = path
                    break

            if site_packages_path and site_packages_path.exists():
                for item in site_packages_path.iterdir():
                    if item.is_dir():
                        package_name = item.name.lower().split('-')[0].replace('_', '-')

                        if package_name in self.PACKAGE_SKILLS:
                            skill_info = self.PACKAGE_SKILLS[package_name]
                            skill = SkillDetection(
                                skill_name=skill_info['skill'],
                                evidence_type='package',
                                confidence=skill_info['confidence'],
                                last_detected=datetime.now(),
                                category=skill_info['category'],
                                icon=skill_info['icon']
                            )
                            detected_skills.append(skill)

            # Дополнительная проверка через pip list
            try:
                pip_result = subprocess.run([f"{venv_path}/bin/pip", "list", "--format=freeze"],
                                          capture_output=True, text=True, timeout=10)
                if pip_result.returncode == 0:
                    for line in pip_result.stdout.split('\n'):
                        if '==' in line:
                            package_name = line.split('==')[0].lower()
                            if package_name in self.PACKAGE_SKILLS:
                                skill_info = self.PACKAGE_SKILLS[package_name]
                                skill = SkillDetection(
                                    skill_name=skill_info['skill'],
                                    evidence_type='package',
                                    confidence=skill_info['confidence'],
                                    last_detected=datetime.now(),
                                    category=skill_info['category'],
                                    icon=skill_info['icon']
                                )
                                detected_skills.append(skill)
            except:
                pass  # Игнорируем ошибки pip

        except Exception as e:
            st.warning(f"Ошибка сканирования пакетов: {e}")

        return detected_skills

    def scan_project_files(self, project_dirs: List[str]) -> List[SkillDetection]:
        """Сканирование файлов проектов"""
        detected_skills = []
        file_counts = defaultdict(int)

        try:
            for project_dir in project_dirs:
                project_path = Path(project_dir)
                if project_path.exists():
                    # Сканируем файлы рекурсивно
                    for file_path in project_path.rglob("*"):
                        if file_path.is_file():
                            # Проверяем расширения файлов
                            suffix = file_path.suffix.lower()
                            if suffix in self.FILE_PATTERNS:
                                file_counts[suffix] += 1

                            # Проверяем конкретные имена файлов
                            file_name = file_path.name.lower()
                            for pattern, skill_info in self.FILE_PATTERNS.items():
                                if pattern in file_name:
                                    file_counts[pattern] += 1

            # Создаем навыки на основе найденных файлов
            for pattern, count in file_counts.items():
                if pattern in self.FILE_PATTERNS and count > 0:
                    skill_info = self.FILE_PATTERNS[pattern]
                    # Увеличиваем confidence в зависимости от количества файлов
                    confidence = min(0.95, skill_info['confidence'] + (count * 0.05))

                    skill = SkillDetection(
                        skill_name=skill_info['skill'],
                        evidence_type='file',
                        confidence=confidence,
                        last_detected=datetime.now(),
                        frequency=count,
                        category=skill_info['category'],
                        icon=skill_info['icon'],
                        description=f"Найдено {count} файлов типа {pattern}"
                    )
                    detected_skills.append(skill)

        except Exception as e:
            st.error(f"Ошибка сканирования файлов: {e}")

        return detected_skills

    def update_profile_xp(self, skills: List[SkillDetection]):
        """Обновление XP профиля"""
        if not self.profile:
            return

        # Считаем новые навыки
        new_skills = []
        for skill in skills:
            if skill.skill_name not in self.profile.skills_detected:
                new_skills.append(skill)
                self.profile.skills_detected[skill.skill_name] = skill

        # Начисляем XP за новые навыки
        xp_gained = 0
        for skill in new_skills:
            xp_gained += self.XP_REWARDS["new_skill_detected"]

        # Начисляем XP за активность
        for skill in skills:
            if skill.evidence_type == 'process' and skill.total_time_minutes > 30:
                hours = skill.total_time_minutes // 60
                xp_gained += hours * self.XP_REWARDS["skill_usage_hour"]

        self.profile.total_xp += xp_gained

        # Проверяем повышение уровня
        old_level = self.profile.level
        self.profile.level = (self.profile.total_xp // 200) + 1

        if self.profile.level > old_level:
            xp_gained += self.XP_REWARDS["level_up"]
            self.profile.total_xp += self.XP_REWARDS["level_up"]

        # Проверяем новые достижения
        self.check_new_badges()

        # Обновляем последнюю активность
        self.profile.last_activity = datetime.now()

        return xp_gained, new_skills

    def check_new_badges(self):
        """Проверка новых достижений"""
        if not self.profile:
            return

        new_badges = []
        for badge_name, badge_info in self.BADGES.items():
            if badge_name not in self.profile.badges:
                if badge_info["requirement"](self.profile):
                    self.profile.badges.add(badge_name)
                    new_badges.append(badge_name)
                    self.profile.total_xp += self.XP_REWARDS["badge_earned"]

        return new_badges

    def save_profile_to_resume_folder(self, profile_data: Dict, skills_data: Dict) -> str:
        """Сохранение профиля в папку резюме для рабочего процесса"""
        try:
            # Создаем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{profile_data.get('username', 'user').replace(' ', '_')}_{timestamp}.json"
            filepath = JOB_RESUME_DIR / filename

            # Готовим данные для экспорта
            export_data = {
                "profile_info": profile_data,
                "skills": {
                    skill_name: {
                        "skill_name": skill.skill_name,
                        "category": skill.category,
                        "experience_level": skill.experience_level,
                        "description": skill.description,
                        "confidence": skill.confidence,
                        "frequency": skill.frequency,
                        "evidence_type": skill.evidence_type,
                        "icon": getattr(skill, 'icon', '🔧'),
                        "last_detected": skill.last_detected.isoformat()
                    }
                    for skill_name, skill in skills_data.items()
                },
                "user_profile": {
                    "total_xp": self.profile.total_xp,
                    "level": self.profile.level,
                    "badges": list(self.profile.badges) if self.profile.badges else [],
                    "daily_streak": self.profile.daily_streak,
                    "career_goals": getattr(self.profile, 'career_goals', ''),
                    "current_projects": getattr(self.profile, 'current_projects', [])
                },
                "export_timestamp": datetime.now().isoformat(),
                "status": "ready_for_manager_review",
                "workflow_stage": "employee_submitted"
            }

            # Сохраняем файл
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            return str(filepath)

        except Exception as e:
            raise Exception(f"Ошибка сохранения профиля: {e}")


def render_header():
    """Рендер заголовка приложения"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            🚀 AI HR система с геймификацией
        </h1>
        <h3 style="color: white; text-align: center; margin: 10px 0 0 0; font-weight: 300;">
            AI для роста, геймификация для мотивации от MilRAG
        </h3>
    </div>
    """, unsafe_allow_html=True)


def render_role_selector():
    """Селектор ролей пользователя"""
    st.sidebar.markdown("### 👤 Выберите роль")

    role = st.sidebar.selectbox(
        "Роль:",
        ["employee", "manager", "hr"],
        format_func=lambda x: {"employee": "👨‍💻 Сотрудник",
                              "manager": "👔 Руководитель",
                              "hr": "🏢 HR специалист"}[x],
        key="role_selector"
    )

    if role != st.session_state.current_user_role:
        st.session_state.current_user_role = role
        st.rerun()

    return role


def render_sidebar():
    """Рендер боковой панели"""
    st.sidebar.markdown("### ⚙️ Настройки")

    username = st.sidebar.text_input("👤 Имя пользователя:", "")

    venv_path = st.sidebar.text_input("🐍 Путь к venv:", "/home/karfel/GitHub/venv",
                                    help="Путь к Python окружению")

    project_paths_text = st.sidebar.text_area("📁 Пути к проектам:",
                                            "/home/karfel/GitHub/AI Challenge Sber\n/home/karfel/GitHub/1AX5X5",
                                            help="Пути к проектам (по одному на строку)")

    project_paths = [path.strip() for path in project_paths_text.split('\n') if path.strip()]

    return {
        "username": username,
        "venv_path": venv_path,
        "project_paths": project_paths
    }


def render_gamification_dashboard():
    """Дашборд геймификации"""
    if not st.session_state.user_profile:
        return

    profile = st.session_state.user_profile

    st.markdown("### 🎮 Игровой профиль")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Уровень", profile.level)
    with col2:
        st.metric("Общий XP", profile.total_xp)
    with col3:
        st.metric("Навыки", len(profile.skills_detected))
    with col4:
        st.metric("Достижения", len(profile.badges))

    # Прогресс до следующего уровня
    current_level_xp = (profile.level - 1) * 200
    next_level_xp = profile.level * 200
    progress = min(100, ((profile.total_xp - current_level_xp) / (next_level_xp - current_level_xp)) * 100)

    st.markdown(f"**Прогресс до уровня {profile.level + 1}:**")
    st.progress(progress / 100)
    st.caption(f"{profile.total_xp - current_level_xp}/{next_level_xp - current_level_xp} XP")

    # Достижения
    if profile.badges:
        st.markdown("### 🏆 Достижения")
        badge_cols = st.columns(min(len(profile.badges), 6))

        profiler = SmartAutoProfiler()
        for i, badge_name in enumerate(list(profile.badges)[:6]):
            with badge_cols[i]:
                badge_info = profiler.BADGES.get(badge_name, {"icon": "🏆", "desc": badge_name})
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 10px; 
                           background: linear-gradient(45deg, #f3ec78, #af4261);">
                    <div style="font-size: 2em">{badge_info['icon']}</div>
                    <div style="font-size: 0.8em; color: white; font-weight: bold">{badge_info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)


def render_skills_editor():
    """Редактор навыков"""
    if not st.session_state.get('skills_edit_mode', False):
        return False

    st.markdown("### ✏️ Редактор навыков")
    st.markdown("*Редактирование обнаруженных навыков*")

    skills = st.session_state.scanned_skills.copy()

    if not skills:
        st.warning("📭 Навыки не найдены. Запустите сканирование.")
        return False

    # Группируем навыки по категориям
    skills_by_category = defaultdict(list)
    for skill_name, skill_data in skills.items():
        skills_by_category[skill_data.category].append((skill_name, skill_data))

    edited_skills = {}

    # Создаем табы для категорий
    categories = list(skills_by_category.keys())
    if categories:
        tabs = st.tabs([f"{skill_data.icon if hasattr(skill_data, 'icon') else '📂'} {cat}" for cat in categories])

        for i, category in enumerate(categories):
            with tabs[i]:
                st.markdown(f"#### {category}")

                for skill_name, skill_data in skills_by_category[category]:
                    with st.expander(f"{getattr(skill_data, 'icon', '🔧')} {skill_name}", expanded=True):
                        col1, col2 = st.columns(2)

                        with col1:
                            new_name = st.text_input("Название навыка:",
                                                   value=skill_data.skill_name,
                                                   key=f"name_{skill_name}")

                            new_experience = st.selectbox("Уровень опыта:",
                                                        ["Начинающий", "Средний", "Продвинутый"],
                                                        index=["Начинающий", "Средний", "Продвинутый"].index(skill_data.experience_level),
                                                        key=f"exp_{skill_name}")

                            new_category = st.selectbox("Категория:",
                                                       ["Development", "Data Science", "AI/ML", "DevOps", "Database", "Testing", "Прочее"],
                                                       index=["Development", "Data Science", "AI/ML", "DevOps", "Database", "Testing", "Прочее"].index(skill_data.category) if skill_data.category in ["Development", "Data Science", "AI/ML", "DevOps", "Database", "Testing", "Прочее"] else 6,
                                                       key=f"cat_{skill_name}")

                        with col2:
                            new_description = st.text_area("Описание:",
                                                         value=skill_data.description,
                                                         key=f"desc_{skill_name}")

                            include_skill = st.checkbox("Включить в профиль",
                                                       value=True,
                                                       key=f"include_{skill_name}")

                        # Сохраняем отредактированный навык
                        if include_skill:
                            edited_skill = SkillDetection(
                                skill_name=new_name,
                                evidence_type=skill_data.evidence_type,
                                confidence=skill_data.confidence,
                                last_detected=skill_data.last_detected,
                                total_time_minutes=skill_data.total_time_minutes,
                                frequency=skill_data.frequency,
                                description=new_description,
                                category=new_category,
                                experience_level=new_experience,
                                icon=getattr(skill_data, 'icon', '🔧')
                            )
                            edited_skills[new_name] = edited_skill

    st.markdown("---")

    # Кнопки управления
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Сохранить изменения", type="primary"):
            st.session_state.scanned_skills = edited_skills
            st.session_state.skills_edit_mode = False
            st.success("✅ Навыки обновлены!")
            st.rerun()

    with col2:
        if st.button("❌ Отменить"):
            st.session_state.skills_edit_mode = False
            st.rerun()

    with col3:
        if st.button("🔄 Сбросить"):
            # Перезапуск сканирования
            st.session_state.skills_edit_mode = False
            st.rerun()

    return True


def render_hr_interface():
    """Интерфейс для HR специалиста с возможностью загрузки файлов и AI обработкой"""
    st.markdown("## 🏢 HR Панель")
    st.markdown("*Финальная обработка и добавление в базу данных*")

    # Инициализация системы HR
    if 'hr_system' not in st.session_state:
        st.session_state.hr_system = HRSystemAdvanced()

    if 'ai_consultant' not in st.session_state:
        st.session_state.ai_consultant = AICareerConsultant()

    hrsystem = st.session_state.hr_system
    ai_consultant = st.session_state.ai_consultant

    # Получаем данные из базы
    employees = hrsystem.db.get_all_employees()
    positions = hrsystem.db.get_all_positions()
    profiles = hrsystem.db.get_scanned_profiles()

    # Создаем табы
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Обработка резюме",
        "📊 База данных HR",
        "🤖 AI Консультант",
        "📁 Импорт данных",
        "📈 Аналитика"
    ])

    with tab1:
        # Обработка одобренных руководителем резюме
        st.markdown("### 📋 Резюме, одобренные руководителем")

        # ОТЛАДКА: показываем все папки
        st.markdown("#### 🔍 Отладочная информация:")
        manager_review_dir = MANAGER_REVIEW_DIR
        st.write(f"**Базовая папка руководителя:** {manager_review_dir}")
        st.write(f"**Существует:** {manager_review_dir.exists()}")

        if manager_review_dir.exists():
            subdirs = [d for d in manager_review_dir.iterdir() if d.is_dir()]
            st.write(f"**Подпапки:** {[d.name for d in subdirs]}")

            # Показываем содержимое всех подпапок
            for subdir in subdirs:
                files = list(subdir.glob("*.json"))
                st.write(f"**{subdir.name}:** {len(files)} файлов")
                if files:
                    for f in files:
                        st.write(f"  - {f.name}")

        st.markdown("---")

        approved_files = []

        # Ищем одобренные файлы в разных местах
        possible_locations = [
            MANAGER_REVIEW_DIR / "approved",
            MANAGER_REVIEW_DIR,  # Если файлы сохранились в корневой папке
            JOB_RESUME_DIR  # На случай если workflow поломался
        ]

        for location in possible_locations:
            st.write(f"**Проверяем:** {location}")
            if location.exists():
                for file_path in location.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Ищем файлы с правильным workflow_stage
                        workflow_stage = data.get('workflow_stage', '')
                        status = data.get('status', '')

                        st.write(f"  - {file_path.name}: stage='{workflow_stage}', status='{status}'")

                        # Принимаем файлы, одобренные руководителем
                        if (workflow_stage == 'manager_approved' or
                                status == 'ready_for_hr_review' or
                                'manager_approved' in str(data)):
                            approved_files.append({
                                'filename': file_path.name,
                                'filepath': str(file_path),
                                'data': data,
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                                'location': str(location)
                            })

                    except Exception as e:
                        st.error(f"Ошибка чтения файла {file_path}: {e}")

        if not approved_files:
            st.warning("📁 Нет резюме, одобренных руководителем")
            st.markdown("*Резюме появятся здесь после одобрения руководителем*")

            # Показываем все файлы для отладки
            all_files = []
            for location in possible_locations:
                if location.exists():
                    all_files.extend(list(location.glob("*.json")))

            if all_files:
                st.markdown("#### 🔍 Все найденные файлы:")
                for f in all_files:
                    try:
                        with open(f, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                        st.write(
                            f"**{f.name}:** workflow_stage='{data.get('workflow_stage', 'none')}', status='{data.get('status', 'none')}'")
                    except:
                        st.write(f"**{f.name}:** ошибка чтения")
        else:
            st.success(f"📄 Найдено {len(approved_files)} резюме для финальной обработки")

            # Обрабатываем каждое одобренное резюме
            for i, resume_file in enumerate(approved_files):
                data = resume_file['data']
                profile_info = data.get('profile_info', {})
                skills = data.get('skills', {})
                user_profile = data.get('user_profile', {})

                with st.expander(f"📄 {profile_info.get('user_name', 'Неизвестно')} - {resume_file['filename']}",
                                 expanded=i < 2):
                    # Показываем откуда файл
                    st.info(f"📂 Источник: {resume_file['location']}")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 👤 Информация о сотруднике")
                        st.write(f"**Имя:** {profile_info.get('user_name', 'Не указано')}")
                        st.write(f"**Желаемая должность:** {profile_info.get('position', 'Не указано')}")
                        st.write(f"**Email:** {profile_info.get('email', 'Не указано')}")
                        st.write(f"**Телефон:** {profile_info.get('phone', 'Не указано')}")
                        st.write(f"**Локация:** {profile_info.get('location', 'Не указано')}")

                        if data.get('manager_notes'):
                            st.markdown("**Комментарии руководителя:**")
                            st.info(data['manager_notes'])

                        # AI анализ профиля
                        if st.button(f"🤖 AI анализ для {profile_info.get('user_name', 'сотрудника')}",
                                     key=f"ai_rec_{i}"):
                            with st.spinner("Генерируем рекомендации..."):
                                full_profile_data = {
                                    'profile_info': profile_info,
                                    'skills': skills,
                                    'user_profile': user_profile
                                }
                                recommendations = ai_consultant.generate_career_recommendations(full_profile_data)
                                st.markdown("**🎯 AI Рекомендации:**")
                                st.write(recommendations)

                        # Топ навыки
                        st.markdown("### 🏆 Топ навыки")
                        if skills:
                            top_skills = sorted(skills.items(),
                                                key=lambda x: x[1].get('confidence', 0),
                                                reverse=True)[:8]

                            for skill_name, skill_data in top_skills:
                                icon = skill_data.get('icon', '🔧')
                                level = skill_data.get('experience_level', 'Начинающий')
                                confidence = skill_data.get('confidence', 0)

                                st.write(f"**{icon} {skill_name}** ({level})")
                                st.caption(f"Confidence: {confidence:.2f}")
                        else:
                            st.write("Навыки не найдены")

                    with col2:
                        # Геймификация
                        st.markdown("### 🎮 Игровой профиль")
                        st.metric("Уровень", user_profile.get('level', 1))
                        st.metric("Общий XP", user_profile.get('total_xp', 0))
                        st.metric("Streak", user_profile.get('daily_streak', 0))

                        badges = user_profile.get('badges', [])[:3]
                        if badges:
                            st.write("**🏆 Достижения:**")
                            for badge in badges:
                                st.write(f"• {badge}")

                    # HR форма обработки
                    st.markdown("---")
                    st.markdown("### 🏢 Финальная обработка HR")

                    hr_form_key = f"hr_form_{i}"
                    with st.form(hr_form_key):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            department = st.text_input("Отдел:",
                                                       value=profile_info.get('department', ''),
                                                       key=f"dept_{i}")
                            salary_range = st.text_input("Зарплатная вилка:",
                                                         placeholder="80000-120000",
                                                         key=f"salary_{i}")

                        with col_b:
                            career_track = st.selectbox("Карьерный трек:",
                                                        ["Technical", "Management", "Consulting", "Research"],
                                                        key=f"track_{i}")
                            priority = st.selectbox("Приоритет:",
                                                    ["High", "Medium", "Low"],
                                                    key=f"priority_{i}")

                        hr_notes = st.text_area("Комментарии HR:",
                                                placeholder="Заметки HR специалиста...",
                                                key=f"hr_notes_{i}")

                        col_approve, col_reject = st.columns(2)

                        with col_approve:
                            approve_btn = st.form_submit_button("✅ Добавить в базу данных", type="primary")

                        with col_reject:
                            reject_btn = st.form_submit_button("❌ Отклонить")

                        # Обработка действий
                        if approve_btn:
                            try:
                                # Подготавливаем данные профиля для HR системы
                                hr_profile_data = {
                                    'type': 'scanned_profile',
                                    'username': profile_info.get('user_name', 'Unknown'),
                                    'skills_detected': skills,
                                    'total_xp': user_profile.get('total_xp', 0),
                                    'level': user_profile.get('level', 1),
                                    'badges': user_profile.get('badges', []),
                                    'scan_data': {
                                        'profile_info': profile_info,
                                        'department': department,
                                        'salary_range': salary_range,
                                        'career_track': career_track,
                                        'priority': priority,
                                        'hr_notes': hr_notes,
                                        'manager_notes': data.get('manager_notes', ''),
                                        'processed_at': datetime.now().isoformat()
                                    }
                                }

                                # Добавляем в HR систему через AI обработку
                                result = hrsystem.add_document_to_database(hr_profile_data, resume_file['filename'])

                                # Перемещаем файл в финальную папку
                                HR_FINAL_DIR.mkdir(exist_ok=True)
                                final_filepath = HR_FINAL_DIR / resume_file['filename']

                                data['workflow_stage'] = 'hr_approved'
                                data['hr_notes'] = hr_notes
                                data['hr_approved_by'] = "HR Specialist"
                                data['hr_approved_at'] = datetime.now().isoformat()

                                with open(final_filepath, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, ensure_ascii=False, indent=2)

                                # Удаляем исходный файл
                                os.remove(resume_file['filepath'])

                                st.success(
                                    f"✅ Профиль {profile_info.get('user_name', 'сотрудника')} добавлен в базу данных!")
                                st.success(result)
                                st.balloons()
                                time.sleep(2)
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Ошибка обработки: {e}")

                        elif reject_btn:
                            try:
                                # Отклонение - возвращаем руководителю
                                data['workflow_stage'] = 'hr_rejected'
                                data['hr_notes'] = hr_notes
                                data['status'] = 'returned_to_manager'

                                rejected_filepath = MANAGER_REVIEW_DIR / "hr_rejected" / resume_file['filename']
                                rejected_filepath.parent.mkdir(exist_ok=True)

                                with open(rejected_filepath, 'w', encoding='utf-8') as f:
                                    json.dump(data, f, ensure_ascii=False, indent=2)

                                os.remove(resume_file['filepath'])

                                st.warning(f"❌ Профиль {profile_info.get('user_name', 'сотрудника')} отклонен")
                                time.sleep(2)
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Ошибка обработки: {e}")

    # Остальные табы остаются без изменений...
    with tab2:
        # База данных HR (код без изменений)
        st.header("📊 База данных HR")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Сотрудники", len(employees))
        with col2:
            st.metric("Позиции", len(positions))
        with col3:
            st.metric("Профили", len(profiles))
        with col4:
            total_xp = sum(profile.get('total_xp', 0) for profile in profiles)
            st.metric("Общий XP", total_xp)

        # Подтабы для разных типов данных
        subtab1, subtab2, subtab3 = st.tabs(["👥 Сотрудники", "💼 Позиции", "📋 Профили"])

        with subtab1:
            if employees:
                employees_df = pd.DataFrame(employees)
                st.dataframe(employees_df, use_container_width=True)
            else:
                st.info("📁 Нет данных о сотрудниках")

        with subtab2:
            if positions:
                positions_df = pd.DataFrame(positions)
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("📁 Нет данных о позициях")

        with subtab3:
            if profiles:
                # Отображение профилей с геймификацией
                profiles_display = []
                for profile in profiles:
                    profiles_display.append({
                        'Имя': profile.get('username', ''),
                        'Уровень': profile.get('level', 1),
                        'XP': profile.get('total_xp', 0),
                        'Навыки': len(profile.get('skills_detected', {})),
                        'Достижения': len(profile.get('badges', [])),
                        'Обновлено': profile.get('updated_at', '')[:19] if profile.get('updated_at') else ''
                    })

                profiles_df = pd.DataFrame(profiles_display)
                st.dataframe(profiles_df, use_container_width=True)
            else:
                st.info("📁 Нет сканированных профилей")

    with tab3:
        # AI Консультант
        st.header("🤖 AI Консультант")
        st.markdown("*RAG-система для работы с HR данными*")

        # Поиск с AI
        search_query = st.text_input("💬 Задайте вопрос:",
                                   placeholder="Найдите Python разработчика с опытом > 3 лет")

        col1, col2 = st.columns([3, 1])

        with col1:
            search_type = st.radio("Тип поиска:", ["RAG-поиск с AI", "Обычный поиск"], index=0)

        with col2:
            k_results = st.selectbox("Результатов:", [3, 5, 10, 15], index=1)

        if search_query and st.button("🔍 Найти", type="primary"):
            with st.spinner("Ищем через AI..."):
                if search_type == "RAG-поиск с AI":
                    # Используем AI для умного поиска
                    response = hrsystem.smart_query_with_context(search_query, k=k_results)
                    st.success("🎯 Результаты AI поиска:")
                    st.markdown("---")
                    st.write(response)
                else:
                    # Обычный гибридный поиск
                    search_results = hrsystem.hybrid_search(search_query, k=k_results)

                    if search_results:
                        st.success(f"🔍 Найдено {len(search_results)} результатов!")

                        for i, result in enumerate(search_results, 1):
                            metadata = result['metadata']
                            with st.expander(f"{i}. {metadata['type'].title()} - Score: {result['score']:.3f}"):
                                st.write(result['text'])
                    else:
                        st.warning("Ничего не найдено")

        # Показ детального поиска
        with st.expander("🔍 Детальные результаты поиска", expanded=False):
            if search_query:
                search_results = hrsystem.hybrid_search(search_query, k=k_results)

                for i, result in enumerate(search_results, 1):
                    metadata = result['metadata']
                    st.write(f"**{i}. Score: {result['score']:.3f}**")
                    st.write(f"**Тип:** {metadata['type']}")

                    if metadata['type'] == 'employee':
                        emp_data = metadata['data']
                        st.write(f"**Имя:** {emp_data.get('name', '')}")
                    elif metadata['type'] == 'position':
                        pos_data = metadata['data']
                        st.write(f"**Позиция:** {pos_data.get('title', '')}")

                    st.text(result['text'][:200] + "...")
                    st.markdown("---")

    with tab4:
        # Импорт данных (как в первом коде)
        st.header("📁 Импорт данных")
        st.markdown("*Загрузите файлы для автоматической обработки через AI*")

        # Загрузка файлов различных форматов
        st.subheader("📄 Загрузка документов")
        uploaded_files = st.file_uploader(
            "Выберите файлы для загрузки:",
            type=['pdf', 'docx', 'xlsx', 'csv', 'txt', 'json'],
            accept_multiple_files=True,
            help="Поддерживаемые форматы: PDF, DOCX, XLSX, CSV, TXT, JSON"
        )

        if uploaded_files and st.button("📤 Обработать файлы", type="primary"):
            results = []
            progress_bar = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Обрабатываем {uploaded_file.name}..."):
                    # Парсим документ через AI
                    content, structured_data = hrsystem.parse_document(uploaded_file)

                    # Добавляем в базу данных
                    result = hrsystem.add_document_to_database(structured_data, uploaded_file.name)
                    results.append(f"{uploaded_file.name}: {result}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

            # Показываем результаты
            st.success("✅ Обработка завершена!")
            with st.sidebar:
                st.header("📋 Результаты обработки")
                for result in results:
                    st.success(result)
                st.rerun()

        st.markdown("---")

        # Импорт Smart Auto-Profiler профилей
        st.subheader("🤖 Импорт Smart Auto-Profiler профилей")
        uploaded_profile = st.file_uploader(
            "Загрузить JSON профиль:",
            type=['json'],
            help="JSON файлы, созданные Smart Auto-Profiler"
        )

        if uploaded_profile:
            try:
                profile_data = json.load(uploaded_profile)

                # Показываем предварительный просмотр
                st.json(profile_data)

                if st.button("➕ Добавить профиль в HR базу", type="primary"):
                    # Форматируем данные для HR системы
                    hr_profile_data = {
                        'type': 'scanned_profile',
                        'username': profile_data.get('username', 'Unknown'),
                        'skills_detected': profile_data.get('skills_detected', {}),
                        'total_xp': profile_data.get('total_xp', 0),
                        'level': profile_data.get('level', 1),
                        'badges': profile_data.get('badges', []),
                        'scan_data': profile_data
                    }

                    # Добавляем через AI систему
                    result = hrsystem.add_document_to_database(hr_profile_data, uploaded_profile.name)
                    st.success(result)
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Ошибка обработки файла: {e}")

    with tab5:
        # Аналитика
        st.header("📈 HR Аналитика")

        if not employees and not positions and not profiles:
            st.info("📊 Нет данных для аналитики. Загрузите файлы в раздел 'Импорт данных'")
            return

        # XP аналитика
        if profiles:
            xp_data = [profile.get('total_xp', 0) for profile in profiles]
            if xp_data:
                xp_df = pd.DataFrame({
                    'Имя': [p.get('username', '') for p in profiles],
                    'XP': xp_data
                })
                st.bar_chart(xp_df.set_index('Имя'))
        else:
            st.info("📊 Нет данных профилей для построения графика XP")


def render_manager_interface():
    """Интерфейс для руководителя"""
    st.markdown("## 👔 Панель руководителя")
    st.markdown("*Проверка и валидация резюме сотрудников*")

    # Сканируем папку с резюме сотрудников
    resume_files = []
    if JOB_RESUME_DIR.exists():
        for file_path in JOB_RESUME_DIR.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if data.get('status') == 'ready_for_manager_review':
                    resume_files.append({
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'data': data,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime)
                    })
            except Exception as e:
                st.error(f"Ошибка чтения файла {file_path}: {e}")

    if not resume_files:
        st.info("📁 Нет резюме для проверки руководителем")
        st.markdown("*Резюме появятся здесь после отправки сотрудниками*")
        return

    st.success(f"📄 Найдено {len(resume_files)} резюме для проверки")

    # Обрабатываем каждое резюме
    for i, resume_file in enumerate(resume_files):
        data = resume_file['data']
        profile_info = data.get('profile_info', {})
        skills = data.get('skills', {})
        user_profile = data.get('user_profile', {})

        with st.expander(f"📄 {profile_info.get('user_name', 'Неизвестно')} - {resume_file['filename']}",
                         expanded=i < 2):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("### 👤 Информация о сотруднике")
                st.write(f"**Имя:** {profile_info.get('user_name', 'Не указано')}")
                st.write(f"**Желаемая должность:** {profile_info.get('position', 'Не указано')}")
                st.write(f"**Email:** {profile_info.get('email', 'Не указано')}")
                st.write(f"**Телефон:** {profile_info.get('phone', 'Не указано')}")
                st.write(f"**Локация:** {profile_info.get('location', 'Не указано')}")

                if profile_info.get('summary'):
                    st.markdown("**Краткое описание:**")
                    st.write(profile_info['summary'])

                if profile_info.get('career_goals'):
                    st.markdown("**Карьерные цели:**")
                    st.write(profile_info['career_goals'])

                # Навыки по категориям - заменяем вложенные expander'ы на заголовки
                st.markdown("### 💼 Навыки сотрудника")

                skills_by_category = defaultdict(list)
                for skill_name, skill_data in skills.items():
                    skills_by_category[skill_data.get('category', 'Прочее')].append((skill_name, skill_data))

                # Создаем табы для категорий навыков
                if skills_by_category:
                    categories = list(skills_by_category.keys())
                    if len(categories) > 1:
                        skill_tabs = st.tabs([f"📂 {category}" for category in categories])

                        for idx, (category, category_skills) in enumerate(skills_by_category.items()):
                            with skill_tabs[idx]:
                                st.markdown(f"**{len(category_skills)} навыков в категории**")

                                for skill_name, skill_data in category_skills:
                                    icon = skill_data.get('icon', '🔧')
                                    level = skill_data.get('experience_level', 'Начинающий')
                                    confidence = skill_data.get('confidence', 0)

                                    # Контейнер для каждого навыка
                                    with st.container():
                                        st.markdown(f"**{icon} {skill_name}** ({level}) - Confidence: {confidence:.2f}")

                                        if skill_data.get('description'):
                                            st.caption(f"Описание: {skill_data['description']}")

                                        st.caption(
                                            f"Источник: {skill_data.get('evidence_type', 'unknown')} | Частота: {skill_data.get('frequency', 0)}")
                                        st.markdown("---")
                    else:
                        # Если только одна категория, показываем без табов
                        category, category_skills = list(skills_by_category.items())[0]
                        st.markdown(f"#### 📂 {category} ({len(category_skills)} навыков)")

                        for skill_name, skill_data in category_skills:
                            icon = skill_data.get('icon', '🔧')
                            level = skill_data.get('experience_level', 'Начинающий')
                            confidence = skill_data.get('confidence', 0)

                            with st.container():
                                st.markdown(f"**{icon} {skill_name}** ({level}) - Confidence: {confidence:.2f}")

                                if skill_data.get('description'):
                                    st.caption(f"Описание: {skill_data['description']}")

                                st.caption(
                                    f"Источник: {skill_data.get('evidence_type', 'unknown')} | Частота: {skill_data.get('frequency', 0)}")
                                st.markdown("---")
                else:
                    st.info("Навыки не найдены")

            with col2:
                # Геймификация
                st.markdown("### 🎮 Игровой профиль")
                st.metric("Уровень", user_profile.get('level', 1))
                st.metric("Общий XP", user_profile.get('total_xp', 0))
                st.metric("Streak", user_profile.get('daily_streak', 0))

                badges = user_profile.get('badges', [])
                if badges:
                    st.write("**🏆 Достижения:**")
                    for badge in badges[:5]:
                        st.write(f"• {badge}")

                # Статистика навыков
                st.markdown("### 📊 Статистика")
                st.write(f"**Всего навыков:** {len(skills)}")

                categories = {}
                for skill_data in skills.values():
                    cat = skill_data.get('category', 'Прочее')
                    categories[cat] = categories.get(cat, 0) + 1

                for cat, count in categories.items():
                    st.write(f"• {cat}: {count}")

            # Форма для проверки руководителем
            st.markdown("---")
            st.markdown("### 👔 Проверка руководителем")

            form_key = f"manager_form_{i}"
            with st.form(form_key):
                manager_notes = st.text_area(
                    "Комментарии руководителя:",
                    placeholder="Ваши комментарии по профилю сотрудника...",
                    key=f"notes_{i}"
                )

                col_approve, col_reject = st.columns(2)

                with col_approve:
                    approve_btn = st.form_submit_button("✅ Одобрить и отправить в HR", type="primary")

                with col_reject:
                    reject_btn = st.form_submit_button("❌ Отклонить и вернуть сотруднику")

                # Обработка действий
                if approve_btn:
                    try:
                        # Создаем папки если не существуют
                        (MANAGER_REVIEW_DIR / "approved").mkdir(parents=True, exist_ok=True)

                        # Обновляем данные профиля
                        data['workflow_stage'] = 'manager_approved'
                        data['manager_notes'] = manager_notes
                        data['manager_approved_by'] = "Manager"  # Можно добавить авторизацию
                        data['manager_approved_at'] = datetime.now().isoformat()

                        # Перемещаем файл в папку manager_review
                        new_filepath = MANAGER_REVIEW_DIR / "approved" / resume_file['filename']

                        with open(new_filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        # Удаляем исходный файл
                        os.remove(resume_file['filepath'])

                        st.success(
                            f"✅ Резюме {profile_info.get('user_name', 'сотрудника')} одобрено и отправлено в HR!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Ошибка обработки: {e}")

                elif reject_btn:
                    try:
                        # Создаем папки если не существуют
                        (MANAGER_REVIEW_DIR / "rejected").mkdir(parents=True, exist_ok=True)

                        # Обновляем данные
                        data['workflow_stage'] = 'manager_rejected'
                        data['manager_notes'] = manager_notes
                        data['status'] = 'returned_to_employee'

                        # Перемещаем в папку отклоненных
                        rejected_filepath = MANAGER_REVIEW_DIR / "rejected" / resume_file['filename']

                        with open(rejected_filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)

                        # Удаляем исходный файл
                        os.remove(resume_file['filepath'])

                        st.warning(f"❌ Резюме {profile_info.get('user_name', 'сотрудника')} отклонено")
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Ошибка обработки: {e}")


def render_employee_interface(user_settings):
    """Интерфейс для сотрудника с Smart Auto-Profiler"""

    render_header()

    profiler = SmartAutoProfiler()

    # Создание или загрузка профиля пользователя
    if not st.session_state.user_profile and user_settings["username"]:
        st.session_state.user_profile = UserProfile(
            username=user_settings["username"],
            session_start=datetime.now(),
            skills_detected={}
        )

    # Проверяем, идет ли редактирование навыков
    if render_skills_editor():
        return

    # Основной интерфейс сотрудника
    st.markdown("### 🔍 Smart Auto-Profiler")
    st.markdown("*Автоматическое обнаружение навыков и создание профиля*")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Запустить полное сканирование", type="primary", use_container_width=True):
            with st.spinner("Сканируем ваши навыки..."):
                all_detected_skills = []

                # Сканирование процессов
                process_skills = profiler.scan_active_processes()
                all_detected_skills.extend(process_skills)
                st.success(f"🔍 Найдено {len(process_skills)} навыков из активных процессов")

                # Сканирование пакетов
                if user_settings["venv_path"]:
                    package_skills = profiler.scan_installed_packages(user_settings["venv_path"])
                    all_detected_skills.extend(package_skills)
                    st.success(f"📦 Найдено {len(package_skills)} навыков из установленных пакетов")

                # Сканирование файлов проектов
                if user_settings["project_paths"]:
                    file_skills = profiler.scan_project_files(user_settings["project_paths"])
                    all_detected_skills.extend(file_skills)
                    st.success(f"📂 Найдено {len(file_skills)} навыков из файлов проектов")

                if all_detected_skills:
                    # Объединяем и обновляем навыки
                    skills_dict = {}
                    for skill in all_detected_skills:
                        if skill.skill_name not in skills_dict:
                            skills_dict[skill.skill_name] = skill
                        else:
                            # Объединяем данные (берем максимальную уверенность)
                            existing = skills_dict[skill.skill_name]
                            if skill.confidence > existing.confidence:
                                skills_dict[skill.skill_name] = skill

                    st.session_state.scanned_skills = skills_dict

                    # Обновляем XP профиля
                    if st.session_state.user_profile:
                        xp_gained, new_skills = profiler.update_profile_xp(list(skills_dict.values()))
                        if xp_gained > 0:
                            st.success(f"🎉 Получено {xp_gained} XP! Обнаружено {len(new_skills)} новых навыков!")

                    st.balloons()
                else:
                    st.warning("😔 Навыки не обнаружены. Проверьте настройки путей.")

    # Показ результатов сканирования
    if st.session_state.scanned_skills:
        st.markdown("---")
        st.markdown("### 🎯 Обнаруженные навыки")

        # Статистика
        skills = st.session_state.scanned_skills
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Всего навыков", len(skills))
        with col2:
            categories = len(set(skill.category for skill in skills.values()))
            st.metric("Категории", categories)
        with col3:
            avg_confidence = sum(skill.confidence for skill in skills.values()) / len(skills)
            st.metric("Ср. уверенность", f"{avg_confidence:.2f}")
        with col4:
            high_conf_skills = sum(1 for skill in skills.values() if skill.confidence > 0.8)
            st.metric("Высокая увер.", high_conf_skills)

        # Группировка по категориям
        skills_by_category = defaultdict(list)
        for skill_name, skill in skills.items():
            skills_by_category[skill.category].append((skill_name, skill))

        # Создаем табы для категорий
        if skills_by_category:
            categories = list(skills_by_category.keys())
            tabs = st.tabs([f"{categories[i]}" for i in range(len(categories))])

            for i, category in enumerate(categories):
                with tabs[i]:
                    st.markdown(f"### 📂 {category}")
                    category_skills = skills_by_category[category]

                    for skill_name, skill in sorted(category_skills, key=lambda x: x[1].confidence, reverse=True):
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.markdown(f"**{getattr(skill, 'icon', '🔧')} {skill.skill_name}**")
                            if skill.description:
                                st.caption(skill.description)

                        with col2:
                            st.metric("Уверенность", f"{skill.confidence:.2f}")
                            st.caption(f"Уровень: {skill.experience_level}")

                        with col3:
                            st.metric("Источник", skill.evidence_type.title())
                            if skill.frequency > 0:
                                st.caption(f"Частота: {skill.frequency}")
                            elif skill.total_time_minutes > 0:
                                st.caption(f"Время: {skill.total_time_minutes}м")

        # Кнопки управления
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✏️ Редактировать навыки"):
                st.session_state.skills_edit_mode = True
                st.rerun()

        with col2:
            if st.button("📄 Создать резюме-профиль"):
                st.session_state.show_profile_creation = True
                st.rerun()

        with col3:
            if st.button("🗑️ Очистить"):
                st.session_state.scanned_skills = {}
                st.session_state.user_profile = None
                st.rerun()

    # Геймификация
    if st.session_state.user_profile:
        render_gamification_dashboard()

        # История сканирований
        if st.session_state.scan_history:
            with st.expander("📈 История сканирований"):
                history_df = pd.DataFrame(st.session_state.scan_history)
                st.dataframe(history_df)

    # Создание профиля-резюме
    if st.session_state.get('show_profile_creation', False):
        render_profile_creation_interface()


def render_profile_creation_interface():
    """Интерфейс создания профиля-резюме"""
    st.markdown("---")
    st.markdown("### 📝 Создание профиля-резюме")
    st.markdown("*Заполните дополнительную информацию для создания полного профиля*")

    with st.form("profile_creation_form"):
        col1, col2 = st.columns(2)

        with col1:
            user_name = st.text_input("👤 Полное имя:",
                                      value=st.session_state.user_profile.username if st.session_state.user_profile else "")
            email = st.text_input("📧 Email:")
            phone = st.text_input("📱 Телефон:")
            location = st.text_input("🌍 Локация:", value="Москва")

        with col2:
            position = st.text_input("💼 Желаемая должность:")
            department = st.text_input("🏢 Предпочитаемый отдел:")
            linkedin = st.text_input("🔗 LinkedIn профиль:")
            github = st.text_input("💻 GitHub профиль:")

        summary = st.text_area("📋 Краткое описание (summary):",
                               placeholder="Краткое описание ваших профессиональных навыков и опыта...")

        career_goals = st.text_area("🎯 Карьерные цели:",
                                    placeholder="Ваши карьерные цели и планы развития...")

        additional_skills = st.text_area("➕ Дополнительные навыки:",
                                         placeholder="Навыки, которые не были автоматически обнаружены...")

        col_submit, col_cancel = st.columns(2)

        with col_submit:
            submit_btn = st.form_submit_button("🚀 Создать и отправить профиль", type="primary")

        with col_cancel:
            cancel_btn = st.form_submit_button("❌ Отмена")

        if submit_btn:
            try:
                # Подготавливаем данные профиля
                profile_data = {
                    "user_name": user_name,
                    "email": email,
                    "phone": phone,
                    "location": location,
                    "position": position,
                    "department": department,
                    "linkedin": linkedin,
                    "github": github,
                    "summary": summary,
                    "career_goals": career_goals,
                    "additional_skills": additional_skills,
                    "created_at": datetime.now().isoformat()
                }

                # Подготавливаем навыки
                skills_data = st.session_state.scanned_skills

                # Добавляем дополнительные навыки
                if additional_skills:
                    additional_skills_list = [s.strip() for s in additional_skills.split(',') if s.strip()]
                    for skill_name in additional_skills_list:
                        if skill_name not in skills_data:
                            skills_data[skill_name] = SkillDetection(
                                skill_name=skill_name,
                                evidence_type="manual",
                                confidence=0.8,
                                last_detected=datetime.now(),
                                category="Manual",
                                experience_level="Средний",
                                description="Добавлено вручную"
                            )

                # Создаем Smart Auto-Profiler и сохраняем
                profiler = SmartAutoProfiler()
                file_path = profiler.save_profile_to_resume_folder(profile_data, skills_data)

                st.success("🎉 Профиль создан и отправлен на рассмотрение руководителю!")
                st.info(f"📁 Файл сохранен: {file_path}")
                st.balloons()

                # Сброс состояния
                st.session_state.show_profile_creation = False
                time.sleep(2)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Ошибка создания профиля: {e}")

        if cancel_btn:
            st.session_state.show_profile_creation = False
            st.rerun()


def main():
    """Основная функция приложения"""

    # Рендер заголовка и селектора ролей
    render_header()

    # Получаем роль пользователя
    user_role = render_role_selector()

    # Рендер соответствующего интерфейса в зависимости от роли
    if user_role == "employee":
        user_settings = render_sidebar()
        render_employee_interface(user_settings)
    elif user_role == "manager":
        render_manager_interface()
    elif user_role == "hr":
        render_hr_interface()


if __name__ == "__main__":
    main()
