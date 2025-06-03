import os
import uuid
import json
import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfigDict, SafetySettingDict, HarmCategory, HarmBlockThreshold
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import zlib
import traceback # Import traceback for detailed error logging

# Configurar variáveis de ambiente
load_dotenv()

# API KEY do Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API Key do Gemini não encontrada. Defina a variável de ambiente GEMINI_API_KEY.")

genai.configure(api_key=api_key)

# Configurações do modelo
generation_config: GenerationConfigDict = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048, # Mantém um limite razoável para a resposta final
}

# Configurações de segurança
safety_settings: list[SafetySettingDict] = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
]

# --- INÍCIO: Modificação do Modelo ---
# Use gemini-1.5-flash que geralmente é mais eficiente
MODEL_NAME = "gemini-1.5-flash-latest" # Ou "gemini-1.5-flash" se preferir uma versão específica
# --- FIM: Modificação do Modelo ---

# Inicializa modelo Gemini principal
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(f"Modelo Gemini principal ({MODEL_NAME}) inicializado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar o modelo Gemini ({MODEL_NAME}): {e}")
    raise Exception(f"Falha ao inicializar o modelo Gemini. Erro: {e}")

# Inicializa modelo de embeddings (mantém o mesmo)
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
CHAT_LOGS_DIR = os.path.join(BASE_DIR, "chat_logs")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(CHAT_LOGS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Flask App
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(BASE_DIR, 'flask_session')
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# --- INÍCIO: Variável Global Removida ---
# Variáveis globais para cache de embeddings não são mais necessárias aqui,
# a lógica de carregamento/criação está encapsulada nas funções.
# --- FIM: Variável Global Removida ---

# Middleware para garantir session_id
@app.before_request
def create_user_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        print(f"Nova sessão criada: {session['session_id']}")

# Utilitários
def get_session_filepath(session_id: str) -> str:
    return os.path.join(CHAT_LOGS_DIR, f"{session_id}.json")

def load_chat_history(session_id: str) -> list:
    filepath = get_session_filepath(session_id)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if not content:
                    return []
                # Adiciona tratamento para arquivo vazio ou inválido
                if content.strip() == "":
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Aviso: Arquivo de histórico corrompido ou vazio para session_id {session_id}. Resetando.")
            # Opcional: renomear/deletar arquivo corrompido
            try:
                os.rename(filepath, filepath + ".corrupted")
            except OSError:
                pass # Ignora se não conseguir renomear
            return []
        except Exception as e:
            print(f"Erro ao carregar histórico de chat ({session_id}): {e}")
            return []
    return []

def save_chat_message(session_id: str, role: str, message: str):
    # Garante que o histórico é sempre uma lista válida antes de adicionar
    history = load_chat_history(session_id)
    if not isinstance(history, list):
        print(f"Aviso: Histórico inválido detectado para session_id {session_id}. Resetando.")
        history = []

    history.append({"role": role, "message": message, "timestamp": datetime.datetime.now().isoformat()})
    filepath = get_session_filepath(session_id)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar mensagem no chat ({session_id}): {e}")

# Funções de processamento de artigos binários (sem alterações significativas)
def process_article_to_binary(text_content: str, article_title: str = "") -> bytes:
    sections = [s.strip() for s in text_content.split('\n\n') if s.strip()]
    article_data = {
        'version': 1,
        'title': article_title if article_title else text_content[:50].replace('\n', ' ').strip() + "...",
        'created_at': datetime.datetime.now().isoformat(),
        'sections': []
    }
    if not sections: # Lida com conteúdo vazio
        print("Aviso: Conteúdo vazio fornecido para process_article_to_binary.")
        return pickle.dumps(article_data)

    embeddings = embedding_model.encode(sections) # Gera embeddings em lote

    for i, section in enumerate(sections):
        embedding = embeddings[i].tolist()
        compressed_section = zlib.compress(section.encode('utf-8'))
        article_data['sections'].append({
            'content': compressed_section, # Armazena como bytes diretamente
            'original_size': len(section.encode('utf-8')), # Calcula tamanho em bytes
            'compressed_size': len(compressed_section),
            'embedding': embedding
        })
    return pickle.dumps(article_data)

def save_binary_article(binary_data: bytes, filename: str) -> str:
    if not filename.endswith('.bin'):
        filename += '.bin'
    filepath = os.path.join(TRAINING_DATA_DIR, filename)
    with open(filepath, 'wb') as f:
        f.write(binary_data)
    return filepath

def load_binary_article(filepath: str) -> dict:
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
        print(f"Erro ao carregar artigo binário {filepath}: {e}")
        # Retorna uma estrutura vazia ou lança exceção, dependendo do tratamento desejado
        return {'version': 0, 'title': 'Erro ao Carregar', 'sections': []}
    except Exception as e:
        print(f"Erro inesperado ao carregar artigo binário {filepath}: {e}")
        return {'version': 0, 'title': 'Erro Inesperado', 'sections': []}


# --- INÍCIO: Função de Busca Otimizada ---
def get_relevant_sections_from_binary(query: str, k: int = 3) -> list:
    """
    Busca as seções mais relevantes em todos os artigos binários.
    Retorna uma lista de dicionários contendo as seções descompactadas e metadados.
    Utiliza um índice FAISS único para todas as seções.
    """
    binary_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.bin')]
    if not binary_files:
        print("Nenhum arquivo .bin encontrado para busca.")
        return []

    all_embeddings = []
    all_sections_metadata = [] # Armazena metadados para mapear índices FAISS de volta

    print("Carregando e indexando seções dos arquivos .bin...")
    for bin_file in binary_files:
        filepath = os.path.join(TRAINING_DATA_DIR, bin_file)
        article = load_binary_article(filepath)

        # Verifica se o artigo foi carregado corretamente e tem seções
        if not article or 'sections' not in article or not article['sections']:
            print(f"Aviso: Artigo {bin_file} vazio ou inválido, pulando.")
            continue

        for i, section in enumerate(article['sections']):
             # Garante que 'embedding' existe e é uma lista ou array numpy
            if 'embedding' in section and isinstance(section['embedding'], (list, np.ndarray)):
                all_embeddings.append(section['embedding'])
                all_sections_metadata.append({
                    'file': bin_file,
                    'article_title': article.get('title', 'Sem Título'),
                    'section_index': i,
                    'compressed_content': section['content'],
                    'original_size': section.get('original_size', 0) # Usa .get com default
                })
            else:
                print(f"Aviso: Seção {i} do artigo {bin_file} não possui embedding válido, pulando.")


    if not all_embeddings:
        print("Nenhuma seção com embedding válido encontrada nos arquivos .bin.")
        return []

    # Converte para arrays numpy para FAISS
    try:
        embeddings_array = np.array(all_embeddings, dtype='float32')
        if embeddings_array.ndim != 2: # Verifica se o array é 2D
             raise ValueError(f"Array de embeddings com dimensão inesperada: {embeddings_array.ndim}")
    except ValueError as e:
         print(f"Erro ao converter embeddings para array numpy: {e}")
         # Tenta depurar: imprime os primeiros embeddings problemáticos
         for i, emb in enumerate(all_embeddings):
             if not isinstance(emb, (list, np.ndarray)) or np.array(emb).ndim != 1:
                 print(f"Embedding problemático no índice {i}: type={type(emb)}, value={emb}")
                 break
         return []


    # Cria índice FAISS
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    print(f"Índice FAISS criado com {index.ntotal} vetores de dimensão {dimension}.")

    # Busca as seções mais relevantes
    query_embedding = embedding_model.encode(query).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k) # k é o número de vizinhos a buscar

    # Processa os resultados
    relevant_sections = []
    if indices.size == 0 or indices[0][0] == -1: # Verifica se a busca retornou resultados válidos
        print("Nenhuma seção relevante encontrada para a query.")
        return []

    print(f"Índices encontrados: {indices[0]}")
    for idx in indices[0]:
        if idx == -1: # FAISS retorna -1 se não encontrar k vizinhos
            continue
        if idx < len(all_sections_metadata): # Verifica se o índice é válido
            section_data = all_sections_metadata[idx]
            try:
                # Descompacta o conteúdo
                decompressed = zlib.decompress(section_data['compressed_content']).decode('utf-8')
                relevant_sections.append({
                    'article': section_data['article_title'],
                    'file': section_data['file'],
                    'section_index': section_data['section_index'],
                    'content': decompressed, # Conteúdo descompactado
                    'original_size': section_data['original_size']
                })
            except zlib.error as e:
                print(f"Erro ao descomprimir seção {section_data['section_index']} do arquivo {section_data['file']}: {e}")
            except UnicodeDecodeError as e:
                print(f"Erro ao decodificar seção {section_data['section_index']} do arquivo {section_data['file']} (assumindo UTF-8): {e}")
            except Exception as e:
                print(f"Erro inesperado ao processar seção {section_data['section_index']} do arquivo {section_data['file']}: {e}")
        else:
             print(f"Aviso: Índice FAISS {idx} fora dos limites dos metadados (total: {len(all_sections_metadata)}).")


    print(f"Retornando {len(relevant_sections)} seções relevantes.")
    return relevant_sections
# --- FIM: Função de Busca Otimizada ---


# --- INÍCIO: Nova Função de Sumarização de Contexto ---
def summarize_context(question: str, relevant_sections: list, max_summary_tokens: int = 250) -> tuple[str, int]:
    """
    Usa o modelo Gemini para sumarizar as seções relevantes com foco na pergunta.
    Retorna o resumo e a contagem de tokens usados na sumarização.
    """
    if not relevant_sections:
        return "Nenhum contexto relevante encontrado.", 0

    # Concatena o conteúdo das seções relevantes
    context_text = ""
    for i, section in enumerate(relevant_sections, 1):
        context_text += f"\n--- Trecho {i} (Artigo: {section['article']}) ---\n"
        context_text += section['content'] + "\n"

    # Cria o prompt para a sumarização
    summarization_prompt = [
        f"Pergunta do Usuário: \"{question}\"",
        "\nContexto Extraído da Base de Conhecimento:",
        context_text,
        f"\nCom base no contexto acima, resuma APENAS as informações ESSENCIAIS para responder à pergunta do usuário de forma concisa.",
        f"Seja direto e objetivo. O resumo NÃO deve exceder aproximadamente {max_summary_tokens // 4} palavras (para economizar tokens)."
        "Se o contexto não contiver informação relevante para a pergunta, responda 'O contexto fornecido não contém informações relevantes para esta pergunta.'."
    ]

    try:
        # Usa uma configuração de geração talvez mais restrita para sumarização
        summary_generation_config = GenerationConfigDict({
            "temperature": 0.2, # Menos criativo para sumarização
            "max_output_tokens": max_summary_tokens, # Limita o tamanho do resumo
        })

        # Chama o modelo para gerar o resumo
        # Nota: Usamos o mesmo 'model' aqui, mas poderia ser um modelo diferente/mais barato se disponível/necessário
        summary_response = model.generate_content(
            contents=summarization_prompt,
            generation_config=summary_generation_config,
            # safety_settings=safety_settings # Reutiliza as config de segurança
        )

        summary_text = summary_response.text

        # Calcula os tokens usados na *geração* do resumo (aproximado)
        # A contagem exata do *prompt* de sumarização não é trivial sem chamar count_tokens antes
        # Focamos nos tokens *gastos* na geração do resumo
        summary_completion_tokens = model.count_tokens(summary_text).total_tokens

        print(f"Resumo gerado com {summary_completion_tokens} tokens.")
        return summary_text, summary_completion_tokens

    except Exception as e:
        print(f"Erro ao gerar resumo do contexto: {e}")
        print(traceback.format_exc())
        return "Erro ao processar o contexto relevante.", 0

# Add this new helper function
def _get_value_from_path(data, path_str):
    """Helper to navigate a nested dict/list structure using a dot-separated path."""
    keys = path_str.split('.')
    current = data
    for key in keys:
        if current is None: # If any part of the path leads to None, stop
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list):
            try:
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None # Index out of bounds
            except (ValueError, IndexError):
                return None # Key is not a valid int or index out of bounds
        else:
            return None # Path segment not found or not traversable
    return current

def extract_content_from_response(response_content: bytes, content_type_header: str, extraction_key: str = None) -> str:
    """
    Extracts text content from an API response.
    Handles JSON and plain text.
    For JSON, uses extraction_key to find the text.
    If extraction_key is empty for JSON, the whole JSON is stringified.
    """
    text_to_process = ""
    content_type = content_type_header.lower() if content_type_header else ""

    try:
        if 'application/json' in content_type:
            json_data = json.loads(response_content.decode('utf-8'))
            if not extraction_key:
                text_to_process = json.dumps(json_data, ensure_ascii=False, indent=2)
            else:
                extracted_value = _get_value_from_path(json_data, extraction_key)
                if isinstance(extracted_value, str):
                    text_to_process = extracted_value
                elif isinstance(extracted_value, list):
                    # Join if list of strings, otherwise serialize the list
                    if all(isinstance(item, str) for item in extracted_value):
                        text_to_process = "\n".join(extracted_value)
                    else:
                        text_to_process = json.dumps(extracted_value, ensure_ascii=False, indent=2)
                elif extracted_value is not None:
                    # Fallback to string conversion for other types (numbers, booleans)
                    text_to_process = str(extracted_value)
                else:
                    # Key not found or led to None
                    raise ValueError(f"Extraction key '{extraction_key}' not found or resulted in no data in the JSON response.")
        else: # Assume plain text or other text-based format
            text_to_process = response_content.decode('utf-8', errors='replace')
    except json.JSONDecodeError as e:
        # If JSON parsing fails but content_type suggested JSON, or if no specific type and parsing was attempted.
        error_msg = f"Failed to decode JSON: {e}. "
        # Fallback to trying to decode as plain text if it makes sense.
        try:
            text_to_process = response_content.decode('utf-8', errors='replace')
            error_msg += "Interpreting as plain text."
            print(f"Warning: {error_msg}")
        except UnicodeDecodeError as ue:
            raise ValueError(f"Failed to decode JSON and also failed to decode as UTF-8 text: {ue}")

    if not text_to_process.strip():
        raise ValueError("Extracted content is empty or whitespace after processing.")

    return text_to_process
# --- FIM: Nova Função de Sumarização de Contexto ---


# --- Rotas ---

@app.route('/')
def index():
    session_id = session.get('session_id', 'N/A') # Garante que sempre haja um session_id (mesmo que 'N/A')
    # Garante que a chave existe antes de tentar carregar
    if 'session_id' in session:
         history = load_chat_history(session['session_id'])
    else:
         history = [] # Define como lista vazia se não houver session_id ainda
    return render_template('index.html', chat_history=history)


@app.route('/chat', methods=['POST'])
def chat_api():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'ID de sessão não encontrado. Recarregue a página.'}), 500

    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Requisição inválida. Corpo JSON esperado com a chave "question".'}), 400

        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Pergunta não pode ser vazia.'}), 400

        save_chat_message(session_id, "user", question)

        # --- INÍCIO: Modificação - Busca e Sumarização ---
        # 1. Busca seções relevantes (como antes, talvez com k menor se desejar, ex: k=2)
        relevant_sections = get_relevant_sections_from_binary(question, k=3) # Mantendo k=3 por enquanto

        # 2. Sumariza as seções encontradas com foco na pergunta
        summarized_context, summary_tokens_used = summarize_context(question, relevant_sections)
        # --- FIM: Modificação - Busca e Sumarização ---

        history = load_chat_history(session_id)

        # --- INÍCIO: Otimização do Prompt Principal ---
        prompt_parts = [
            # Instrução mais concisa
            "Você é um assistente de suporte técnico. Responda perguntas baseado no CONTEXTO e no HISTÓRICO fornecidos.",
            "\n--- CONTEXTO RELEVANTE (Resumo da Base de Conhecimento) ---",
            summarized_context, # Usa o resumo!
            "\n--- FIM DO CONTEXTO ---",
             # Instrução de uso do contexto mais direta
            "Use o contexto acima para formular sua resposta. Se o contexto não ajudar, use seu conhecimento geral."
        ]

        # Adicionar histórico recente (LIMITE REDUZIDO)
        history_limit = 2  # Reduzido de 3 para 2 (ou até 1) para economizar tokens
        relevant_history = history[-(history_limit * 2):-1] # Exclui a última mensagem do usuário (que é a pergunta atual)

        if relevant_history:
             prompt_parts.append("\n--- HISTÓRICO DA CONVERSA ---")
             for entry in relevant_history:
                 role = entry.get("role")
                 message = entry.get("message")
                 if role and message:
                     prefix = "Usuário:" if role == "user" else "Assistente:"
                     prompt_parts.append(f"{prefix} {message}")
             prompt_parts.append("--- FIM DO HISTÓRICO ---")


        # Adiciona a pergunta atual do usuário no final
        prompt_parts.append(f"\nUsuário: {question}")
        prompt_parts.append("\nAssistente:") # Pede a resposta do assistente

        # Junta as partes do prompt
        final_prompt = "\n".join(prompt_parts)
        # --- FIM: Otimização do Prompt Principal ---

        # Contagem de tokens ANTES de enviar (APENAS PARA DEBUG)
        # Note que count_tokens também custa uma chamada API leve
        prompt_token_count = -1 # Inicializa como -1
        try:
            prompt_token_count = model.count_tokens(final_prompt).total_tokens
            print(f"Tokens no prompt FINAL (estimado): {prompt_token_count}")
        except Exception as e:
            print(f"Erro ao contar tokens do prompt final: {e}")
            # Não interrompe a execução, apenas loga o erro


        # Gerar resposta com o prompt otimizado
        # Usando a configuração de geração padrão definida globalmente
        response = model.generate_content(
            final_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )
        answer = response.text

        # Contagem de tokens da resposta (APENAS PARA DEBUG)
        completion_token_count = -1 # Inicializa como -1
        try:
             # Verifica se a resposta tem candidatos antes de contar tokens
             if response.candidates and response.candidates[0].content.parts:
                 completion_token_count = model.count_tokens(response.candidates[0].content).total_tokens
                 print(f"Tokens na resposta (estimado): {completion_token_count}")
             else:
                 print("Resposta não contém 'candidates' ou 'parts' válidos para contagem de tokens.")
        except Exception as e:
             print(f"Erro ao contar tokens da resposta: {e}")


        save_chat_message(session_id, "assistant", answer)

        # Retorna incluindo os tokens de sumarização (se houver) para clareza
        total_prompt_tokens = prompt_token_count # Tokens do prompt principal
        total_completion_tokens = completion_token_count # Tokens da resposta final
        # Opcional: Adicionar tokens da *geração* do resumo ao custo total,
        # já que foi uma chamada de API necessária para construir o prompt final.
        # total_cost_tokens = total_prompt_tokens + total_completion_tokens + summary_tokens_used

        return jsonify({
            'answer': answer,
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            # 'summary_generation_tokens': summary_tokens_used, # Opcional: informar custo da sumarização
            'total_tokens': (total_prompt_tokens + total_completion_tokens) if total_prompt_tokens != -1 and total_completion_tokens != -1 else -1,
            'sources': [s['article'] for s in relevant_sections] if relevant_sections else [] # Ainda informa as fontes originais
        })

    except Exception as e:
        print(f"Erro crítico no endpoint /chat: {e}\n{traceback.format_exc()}")
        # Garante que a resposta de erro seja sempre um JSON válido
        error_response = {
            'answer': "Desculpe, ocorreu um erro inesperado ao processar sua pergunta. Tente novamente mais tarde.",
            'prompt_tokens': -1,
            'completion_tokens': -1,
            'total_tokens': -1,
            'sources': []
        }
        # Tenta salvar uma mensagem de erro no histórico
        try:
            save_chat_message(session_id, "assistant", f"Erro Interno: {e}")
        except Exception as save_err:
             print(f"Erro adicional ao tentar salvar mensagem de erro no histórico: {save_err}")

        return jsonify(error_response), 500


@app.route('/train', methods=['GET', 'POST'])
def train():
    session_id = session.get('session_id', 'N/A')

    if request.method == 'POST':
        training_type = request.form.get('training_type', 'text') # Default to text

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if training_type == 'text':
                training_text = request.form.get('training_text')
                article_title = request.form.get('article_title', '').strip()

                if not training_text or not training_text.strip():
                    flash("The training text field cannot be empty.", 'warning')
                    return redirect(url_for('train'))

                safe_title_part = "".join(c if c.isalnum() else "_" for c in article_title[:20]) if article_title else "article"
                filename = f"{safe_title_part}_{timestamp}.bin"
                display_title = article_title if article_title else 'New Article'

                binary_data = process_article_to_binary(training_text, article_title)
                filepath = save_binary_article(binary_data, filename)
                flash(f"Article '{display_title}' saved as '{filename}' and optimized successfully.", 'success')

            elif training_type == 'api':
                api_title = request.form.get('api_title', '').strip()
                api_url = request.form.get('api_url', '').strip()
                api_method = request.form.get('api_method', 'GET').upper()
                api_headers_str = request.form.get('api_headers', '{}')
                api_body_str = request.form.get('api_body', None) # None if empty
                api_content_key = request.form.get('api_content_key', '').strip()

                if not api_title:
                    flash("API Data Source Title cannot be empty.", 'warning')
                    return redirect(url_for('train'))
                if not api_url:
                    flash("API URL cannot be empty.", 'warning')
                    return redirect(url_for('train'))

                headers = {}
                try:
                    if api_headers_str:
                        headers = json.loads(api_headers_str)
                except json.JSONDecodeError:
                    flash("Invalid JSON format for API Headers.", 'danger')
                    return redirect(url_for('train'))

                body_data = None
                if api_method == 'POST' and api_body_str:
                    try:
                        body_data = json.loads(api_body_str) # For sending JSON body
                    except json.JSONDecodeError:
                        # If not JSON, treat as raw string data for POST
                        body_data = api_body_str


                print(f"Fetching API: URL={api_url}, Method={api_method}, Headers={headers}, Body={body_data}, Key={api_content_key}")

                api_response_content = ""
                try:
                    if api_method == 'GET':
                        response = requests.get(api_url, headers=headers, timeout=30)
                    elif api_method == 'POST':
                        # Decide content type for POST, default to json if body_data is dict
                        if isinstance(body_data, dict) and 'Content-Type' not in headers:
                             headers['Content-Type'] = 'application/json'

                        if headers.get('Content-Type') == 'application/json' and isinstance(body_data, dict):
                            response = requests.post(api_url, json=body_data, headers=headers, timeout=30)
                        else: # Send as data (form-encoded or raw string)
                            response = requests.post(api_url, data=body_data, headers=headers, timeout=30)
                    else:
                        flash(f"Unsupported API method: {api_method}", 'danger')
                        return redirect(url_for('train'))

                    response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

                    # Extract content
                    api_response_content = extract_content_from_response(
                        response.content,
                        response.headers.get('Content-Type', ''),
                        api_content_key
                    )

                except requests.exceptions.Timeout:
                    flash(f"API request timed out: {api_url}", 'danger')
                    return redirect(url_for('train'))
                except requests.exceptions.HTTPError as e:
                    flash(f"API request failed with status {e.response.status_code}: {e.response.text[:200]}", 'danger')
                    return redirect(url_for('train'))
                except requests.exceptions.RequestException as e:
                    flash(f"API request failed: {e}", 'danger')
                    return redirect(url_for('train'))
                except ValueError as e: # Catch errors from extract_content_from_response
                    flash(f"Error processing API response: {e}", 'danger')
                    return redirect(url_for('train'))


                if not api_response_content or not api_response_content.strip():
                    flash(f"No content extracted from API response or content was empty. URL: {api_url}", 'warning')
                    return redirect(url_for('train'))

                # Process and save the extracted API content
                safe_api_title_part = "".join(c if c.isalnum() else "_" for c in api_title[:20])
                filename = f"api_{safe_api_title_part}_{timestamp}.bin"

                binary_data = process_article_to_binary(api_response_content, api_title)
                filepath = save_binary_article(binary_data, filename)
                flash(f"Data from API '{api_title}' saved as '{filename}' and optimized successfully.", 'success')

            else:
                flash(f"Unknown training type: {training_type}", 'danger')

        except Exception as e:
            print(f"Error during training process: {e}\n{traceback.format_exc()}")
            flash(f"Internal error during training: {e}", "danger")

        return redirect(url_for('train'))

    # Method GET: List articles (unchanged logic for listing)
    articles_info = []
    try:
        files = sorted([f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.bin')])
        for f_name in files: # Renamed f to f_name to avoid conflict with previous f=open(...)
            filepath = os.path.join(TRAINING_DATA_DIR, f_name)
            article = load_binary_article(filepath)
            if article and article.get('version', 0) > 0:
                articles_info.append({
                    'filename': f_name,
                    'title': article.get('title', 'Sem título'),
                    'created_at': article.get('created_at', 'Data desconhecida'),
                    'sections': len(article.get('sections', [])),
                    'total_size_kb': round(sum(s.get('original_size', 0) for s in article.get('sections', [])) / 1024, 1) if article.get('sections') else 0
                })
            else:
                articles_info.append({
                    'filename': f_name, 'title': 'Artigo Inválido ou Corrompido',
                    'created_at': '-', 'sections': 0, 'total_size_kb': 0
                })
    except FileNotFoundError:
        print("Training directory not found when listing files.")
        flash("Training directory not found.", "warning")
    except Exception as e:
        print(f"Error listing training files: {e}\n{traceback.format_exc()}")
        flash(f"Error listing existing articles: {e}", "error")

    return render_template('train.html', articles=articles_info)

# --- INÍCIO: Remoção da Criação Inicial do Índice ---
# A função `get_relevant_sections_from_binary` agora carrega e indexa
# os dados dinamicamente a cada chamada. Isso simplifica o fluxo,
# especialmente ao adicionar novos arquivos, mas pode ser menos performático
# com um número MUITO grande de arquivos .bin.
# Se a busca ficar lenta, reintroduza a lógica de criar e salvar/carregar
# um índice FAISS global único para todas as seções.
# create_embeddings_index() # REMOVIDO
# --- FIM: Remoção da Criação Inicial do Índice ---

# Execução
if __name__ == '__main__':
    if not api_key:
        print("ERRO FATAL: API Key do Gemini não configurada.")
    else:
        port = int(os.getenv("PORT", 5000))  # Render define a variável PORT
        print(f"Iniciando Flask app em host 0.0.0.0 porta {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
