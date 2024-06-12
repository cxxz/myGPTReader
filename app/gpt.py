
import os
import logging
import tarfile
import re
import requests
import hashlib
import random
import uuid
import openai
from openai import OpenAI
from pathlib import Path
from llama_index.core import ServiceContext, GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Document
from llama_index.legacy.llm_predictor.base import LLMPredictor
from llama_index.readers.web import RssReader
# from llama_index.readers.schema.base import Document
from langchain_community.chat_models import ChatOpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from app.fetch_web_post import get_urls, get_youtube_transcript, get_webpage_md_by_jina, scrape_website_by_phantomjscloud, download_audio_from_youtube
from app.prompt import get_prompt_template
from app.util import get_language_code, get_youtube_video_id, get_arxiv_id

import whisperx
from whisperx.utils import get_writer

from pdftext.extraction import plain_text_output

from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.models import load_all_models
import traceback
    
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HF_TOKEN = os.environ.get('HF_TOKEN')
assert HF_TOKEN.startswith("hf_")
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SPEECH_KEY = os.environ.get('SPEECH_KEY')
SPEECH_REGION = os.environ.get('SPEECH_REGION')
openai.api_key = OPENAI_API_KEY

index_cache_web_dir = Path(os.path.expanduser('~/workspace/content_data/cache_web/'))
index_cache_file_dir = Path(os.path.expanduser('~/workspace/content_data/file/'))
tex_cache_file_dir = Path(os.path.expanduser('~/workspace/content_data/tex/'))
index_cache_voice_dir = Path(os.path.expanduser('~/workspace/content_data/voice/'))
transcribed_cache_file_dir = Path(os.path.expanduser('~/workspace/content_data/file/'))

if not index_cache_web_dir.is_dir():
    index_cache_web_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_voice_dir.is_dir():
    index_cache_voice_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_file_dir.is_dir():
    index_cache_file_dir.mkdir(parents=True, exist_ok=True)

if not tex_cache_file_dir.is_dir():
    tex_cache_file_dir.mkdir(parents=True, exist_ok=True)

if not transcribed_cache_file_dir.is_dir():
    transcribed_cache_file_dir.mkdir(parents=True, exist_ok=True)

whisperx_device = "cuda"
whisperx_bs = 32
whisperx_args = {"max_line_width":None, "max_line_count":None, "highlight_words":False}

whisper_model = whisperx.load_model("/local/openai/faster-whisper-large-v3", whisperx_device)

client = OpenAI()

llm_predictor = LLMPredictor(llm=ChatOpenAI(
    temperature=0, model_name="gpt-3.5-turbo"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
web_storage_context = StorageContext.from_defaults()
file_storage_context = StorageContext.from_defaults()

all_marker_models = load_all_models()

def marker_pdf_to_md(pdf_file, output_filepath):
    md_file = output_filepath.split("/")[-1]
    out_folder = output_filepath.replace(md_file, "")

    if markdown_exists(out_folder, md_file):
        return output_filepath.replace(".md","/") + md_file

    try:
        # Skip trying to convert files that don't have a lot of embedded text
        # This can indicate that they were scanned, and not OCRed properly
        # Usually these files are not recent/high-quality

        full_text, images, out_metadata = convert_single_pdf(pdf_file, all_marker_models)
        if len(full_text.strip()) > 0:
            sub_folder = save_markdown(out_folder, md_file, full_text, images, out_metadata)
            return f"{sub_folder}/{md_file}"
        else:
            print(f"Empty file: {pdf_file}.  Could not convert.")
            return None
    except Exception as e:
        print(f"Error converting {pdf_file}: {e}")
        print(traceback.format_exc())
        return None

def get_content_PDFText(pdf_file):
    return plain_text_output(pdf_file, workers=8)

def transcribe_audio(audio_file, format):
    filename = audio_file.split("/")[-1].replace(".mp3","").replace(".m4a","")
    output_format = format.replace(".","")

    logging.info(f"Loading {audio_file}")
    audio = whisperx.load_audio(audio_file)
    logging.info(f"Start transcribing with batch size {whisperx_bs}")
    result = whisper_model.transcribe(audio, batch_size=whisperx_bs)
    lang = result['language']
    logging.info(f"Transcription complete. Language detected: {lang}")
    
    if output_format == "srt":
        writer = get_writer("srt", transcribed_cache_file_dir)
        writer(result, f"{filename}", whisperx_args)
        text_file = f"{transcribed_cache_file_dir}/{filename}.srt"
        logging.info(f"srt file written to {text_file}")
        return text_file
    
    logging.info("Start word aligning...")
    model_a, metadata = whisperx.load_align_model(language_code=lang, device=whisperx_device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, whisperx_device, return_char_alignments=False)
    logging.info("Alignment done.")

    logging.info("Start diarization...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=whisperx_device)
    diarize_segments = diarize_model(audio)
    logging.info("Diarization done.")

    logging.info("Start assigning words to speaker...")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    logging.info("Transcription fully complete.")

    if output_format == "vtt":
        result['language'] = lang
        writer = get_writer("vtt", transcribed_cache_file_dir)
        writer(result, f"{filename}", whisperx_args)
        text_file = f"{transcribed_cache_file_dir}/{filename}.vtt"
        logging.info(f"srt file written to {text_file}")
        return text_file
    else:
        logging.error(f"{output_format} not supported")
        return None

def get_unique_md5(urls):
    urls_str = ''.join(sorted(urls))
    hashed_str = hashlib.md5(urls_str.encode('utf-8')).hexdigest()
    return hashed_str

def format_dialog_messages(messages):
    return "\n".join(messages)

def get_official_transcript_from_youtube(video_id):
    if video_id is None:
        return None
    # print(f"CONG TEST video_id: {video_id}")
    transcript = get_youtube_transcript(video_id)
    # print(f"CONG TEST transcript: {transcript}")
    if transcript is None:
        return None
    # print(f"CONG TEST doc: {transcript}")
    return transcript

def remove_prompt_from_text(text):
    return text.replace('chatGPT:', '').strip()

def get_filename_from_media(index_name, prefix = "media", format = ".txt"):
    md5_index = index_name
    # md5_index = hashlib.md5(index_name).hexdigest()
    file_name = f"{index_cache_file_dir}/{prefix}_{md5_index}{format}"
    if os.path.exists(file_name):
        logging.info(f"CONG TEST {file_name} exists for {md5_index}")
        return True, file_name
    return False, file_name

def get_filename_from_url(url, prefix = "page", format = ".txt"):
    if prefix == "youtube":
        video_id = get_youtube_video_id(url)
        md5_url = video_id
    elif prefix == "arxiv":
        arxiv_id = get_arxiv_id(url)
        md5_url = arxiv_id
        format = ".tex"
    elif prefix == "page":
        format = ".md"
        md5_url = hashlib.md5(url.encode()).hexdigest()
    else:
        md5_url = hashlib.md5(url.encode()).hexdigest()
    file_name = f"{index_cache_file_dir}/{prefix}_{md5_url}{format}"
    if os.path.exists(file_name):
        logging.info(f"CONG TEST {file_name} exists for {url}")
        return True, file_name
    return False, file_name
    
def write_text_to_file(text, file_name):
    logging.info(f"CONG TEST write_text_to_file: {file_name}")
    with open(file_name, "w") as f:
        f.write(text)
    return file_name

def get_documents_from_urls(urls, format = ".txt"):
    logging.info(f"CONG TEST getting {format} documents from urls: {urls}")
    docs = {}
    if len(urls['page_urls']) > 0:
        for url in urls['page_urls']:
            file_exist, file_name = get_filename_from_url(url, "page", format)
            docs[url] = file_name if file_exist else write_text_to_file(get_webpage_md_by_jina(url), file_name)
    if len(urls['arxiv_urls']) > 0:
        for url in urls['arxiv_urls']:
            file_exist, file_name = get_filename_from_url(url, "arxiv", format)
            axiv_id = get_arxiv_id(url)
            docs[url] = file_name if file_exist else write_text_to_file(get_arxiv_tex(axiv_id), file_name)
    if len(urls['rss_urls']) > 0:
        rss_documents = RssReader().load_data(urls['rss_urls'])
        for i, url in enumerate(urls['rss_urls']):
            file_exist, file_name = get_filename_from_url(url, "rss", format)
            docs[url] = file_name if file_exist else write_text_to_file(rss_documents[i], file_name)
    if len(urls['phantomjscloud_urls']) > 0:
        for url in urls['phantomjscloud_urls']:
            file_exist, file_name = get_filename_from_url(url, "phantomjscloud", format)
            docs[url] = file_name if file_exist else write_text_to_file(scrape_website_by_phantomjscloud(url), file_name)
    if len(urls['youtube_urls']) > 0:
        for url in urls['youtube_urls']:
            video_id = get_youtube_video_id(url)
            # print(f"CONG TEST video_id: {video_id}")
            file_exist, file_name = get_filename_from_url(url, "youtube", format)
            if file_exist:
                logging.info(f"CONG TEST {file_name} already exists!")
                docs[url] = file_name 
            elif format == ".txt":
                docs[url] = write_text_to_file(get_official_transcript_from_youtube(video_id), file_name)
            elif format == ".m4a" or format == ".mp3":
                docs[url] = download_audio_from_youtube(url, file_name)
            elif format == ".srt" or format == ".vtt":
                audio_file = file_name.replace(format, ".m4a")
                if not os.path.exists(audio_file):
                    logging.info(f"CONG TEST downloading {url} audio to {audio_file}")
                    audio_file = download_audio_from_youtube(url, audio_file)
                file_name = transcribe_audio(audio_file, format)
                if file_name:
                    docs[url] = file_name

            logging.info(f"CONG TEST youtube doc: {docs.keys()}")

    logging.info(f'Cong TEST got {len(docs)} documents from urls')
    return docs

def get_index_from_web_cache(name):
    try:
        index = load_index_from_storage(web_storage_context, index_id=name)
    except Exception as e:
        logging.error(e)
        return None
    return index

def get_index_from_file_cache(name):
    try:
        index = load_index_from_storage(file_storage_context, index_id=name)
    except Exception as e:
        logging.error(e)
        return None
    return index

def get_index_name_from_file(file: str):
    file_md5_with_extension = str(Path(file).relative_to(index_cache_file_dir).name)
    file_md5 = file_md5_with_extension.split('.')[0]
    return file_md5

def get_answer_from_chatGPT(messages):
    dialog_messages = format_dialog_messages(messages)
    logging.info('=====> Use chatGPT to answer!')
    logging.info(dialog_messages)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": dialog_messages}]
    )
    logging.info(completion.usage)
    total_tokens = completion.usage.total_tokens
    return completion.choices[0].message.content, total_tokens, None

def get_docs_from_web(messages, urls):
    # dialog_messages = format_dialog_messages(messages)
    # lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    latest_msg = messages[-1]
    # logging.info(f"CONG TEST recent_msg: {latest_msg}")
    format = ".txt"
    if '.m4a' in latest_msg:
        format = ".m4a"
    elif '.mp3' in latest_msg:
        format = ".mp3"
    elif '.srt' in latest_msg:
        format = ".srt"
    elif '.vtt' in latest_msg:
        format = ".vtt"
    combained_urls = get_urls(urls)
    logging.info(combained_urls)
    documents = get_documents_from_urls(combained_urls, format = format)
    logging.info(documents)
    return documents, 0, 0


def get_answer_from_llama_web(messages, urls):
    dialog_messages = format_dialog_messages(messages)
    lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    combained_urls = get_urls(urls)
    logging.info(combained_urls)
    index_file_name = get_unique_md5(urls)
    index = get_index_from_web_cache(index_file_name)
    if index is None:
        logging.info(f"=====> Build index from web!")
        documents = get_documents_from_urls(combained_urls)
        logging.info(documents)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id(index_file_name)
        index.storage_context.persist()
        logging.info(
            f"=====> Save index to disk path: {index_cache_web_dir / index_file_name}")
    prompt = get_prompt_template(lang_code)
    logging.info('=====> Use llama web with chatGPT to answer!')
    logging.info('=====> dialog_messages')
    logging.info(dialog_messages)
    logging.info('=====> text_qa_template')
    logging.info(prompt.prompt)
    answer = index.as_query_engine(text_qa_template=prompt).query(dialog_messages)
    total_llm_model_tokens = llm_predictor.last_token_usage
    total_embedding_model_tokens = service_context.embed_model.last_token_usage
    return answer, total_llm_model_tokens, total_embedding_model_tokens

def get_content_from_media(messages, media_file):
    # logging.info(f"CONG TEST getting content {media_file}")
    dialog_messages = format_dialog_messages(messages)
    latest_msg = messages[-1]

    prefix = None
    format = None
    media_file_str = str(media_file)

    if media_file_str.endswith(".m4a") or media_file_str.endswith(".mp3"):
        prefix = "audio"
        format = ".vtt" if ".vtt" in latest_msg else ".srt"
    elif media_file_str.endswith(".pdf"):
        prefix = "pdf"
        format = ".md" if ".md" in latest_msg else ".txt"

    # logging.info(f"CONG TEST latest_msg {latest_msg}")
    index_name = get_index_name_from_file(media_file)
    # logging.info(f"CONG TEST index_name {index_name}")
    file_exist, file_name = get_filename_from_media(index_name, prefix, format)
    # logging.info(f"CONG TEST file_name {file_name}")

    content = {}
    content_file = None
    # logging.info(f"CONG TEST Getting content from {media_file_str}")

    if file_exist:
        content[media_file_str] = file_name
        return content, 0, 0
    
    if prefix == "audio":
        logging.info(f"Transcribing {media_file_str} to {format}")
        content_file = transcribe_audio(media_file_str, format)
    elif prefix == "pdf":
        logging.info(f"Extracting pdf text from {media_file_str} to {format}")
        if format == ".md":
            content_file = marker_pdf_to_md(media_file_str, file_name)
            logging.info(f"CONG TEST file written to {content_file}")
        else:
            pdf_text = get_content_PDFText(media_file_str)
            content_file = write_text_to_file(pdf_text, file_name)
    else:
        msg = f"Uploaded file: {media_file_str}\nindex name: {index_name}\nNot written to:{file_name}"
        content_file = write_text_to_file(msg, file_name)
        logging.info.info(msg)

    if content_file is not None:
        content[media_file_str] = content_file
    return content, 0, 0

def get_answer_from_llama_file(messages, file):
    dialog_messages = format_dialog_messages(messages)
    lang_code = get_language_code(remove_prompt_from_text(messages[-1]))
    index_name = get_index_name_from_file(file)
    index = get_index_from_file_cache(index_name)
    if index is None:
        logging.info(f"=====> Build index from file!")
        documents = SimpleDirectoryReader(input_files=[file]).load_data()
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id(index_name)
        index.storage_context.persist()
        logging.info(
            f"=====> Save index to disk path: {index_cache_file_dir / index_name}")
    prompt = get_prompt_template(lang_code)
    logging.info('=====> Use llama file with chatGPT to answer!')
    logging.info('=====> dialog_messages')
    logging.info(dialog_messages)
    logging.info('=====> text_qa_template')
    logging.info(prompt)
    answer = answer = index.as_query_engine(text_qa_template=prompt).query(dialog_messages)
    total_llm_model_tokens = llm_predictor.last_token_usage
    total_embedding_model_tokens = service_context.embed_model.last_token_usage
    return answer, total_llm_model_tokens, total_embedding_model_tokens

def get_text_from_whisper(voice_file_path):
    with open(voice_file_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript.text

lang_code_voice_map = {
    'zh': ['zh-CN-XiaoxiaoNeural', 'zh-CN-XiaohanNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural'],
    'en': ['en-US-JennyNeural', 'en-US-RogerNeural', 'en-IN-NeerjaNeural', 'en-IN-PrabhatNeural', 'en-AU-AnnetteNeural', 'en-AU-CarlyNeural', 'en-GB-AbbiNeural', 'en-GB-AlfieNeural'],
    'ja': ['ja-JP-AoiNeural', 'ja-JP-DaichiNeural'],
    'de': ['de-DE-AmalaNeural', 'de-DE-BerndNeural'],
}

def convert_to_ssml(text, voice_name=None):
    try:
        logging.info("=====> Convert text to ssml!")
        logging.info(text)
        text = remove_prompt_from_text(text)
        lang_code = get_language_code(text)
        if voice_name is None:
            voice_name = random.choice(lang_code_voice_map[lang_code])
    except Exception as e:
        logging.warning(f"Error: {e}. Using default voice.")
        voice_name = random.choice(lang_code_voice_map['zh'])
    ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">'
    ssml += f'<voice name="{voice_name}">{text}</voice>'
    ssml += '</speak>'

    return ssml

def get_voice_file_from_text(text, voice_name=None):
    speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.set_speech_synthesis_output_format(
        SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    speech_config.speech_synthesis_language = "en"
    file_name = f"{index_cache_voice_dir}{uuid.uuid4()}.mp3"
    file_config = AudioOutputConfig(filename=file_name)
    synthesizer = SpeechSynthesizer(
        speech_config=speech_config, audio_config=file_config)
    ssml = convert_to_ssml(text, voice_name)
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == ResultReason.SynthesizingAudioCompleted:
        logging.info("Speech synthesized for text [{}], and the audio was saved to [{}]".format(
            text, file_name))
    elif result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        logging.info("Speech synthesis canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == CancellationReason.Error:
            logging.error("Error details: {}".format(
                cancellation_details.error_details))
    return file_name

def clean_tex(text):
    # remove all comments from the tex file
    # remove any line that starts with %
    text = '\n'.join(line for line in text.split('\n') if not line.strip().startswith("%"))
    # remove any text between \begin{comment} and \end{comment}
    while "\\begin{comment}" in text:
        start = text.index("\\begin{comment}")
        end = text.index("\\end{comment}") + len("\\end{comment}")
        text = text[:start] + text[end:]
    return text

def get_arxiv_tex(arxiv_id):
    logging.info(f"Retrieving Latex source file for arxiv id: {arxiv_id}")

    url = f"https://arxiv.org/src/{arxiv_id}"
    tex_folder = tex_cache_file_dir / arxiv_id

    if os.path.exists(tex_folder):
        logging.info(f"Tex folder already exists: {tex_folder}")
    else:
        response = requests.get(url, verify=False)  # Setting verify=False skips SSL certificate verification

        # Open a file with the desired name in binary write mode
        with open(f'tmp.tar.gz', 'wb') as file:
            file.write(response.content)  # Write the content of the response to the file

        # untar tmp.tar.gz to a folder named <arxiv_id>
        with tarfile.open("tmp.tar.gz", "r:gz") as tar:
            tar.extractall(tex_folder)

    # list all .tex file in tex_folder recursively
    tex_files = []
    for root, dirs, files in os.walk(tex_folder):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    # print(tex_files)

    # create a dict of tex_files and their content
    tex_content = {}
    for file in tex_files:
        with open(file, 'r') as f:
            tex_content[file] = clean_tex(f.read())
    # print(tex_content)

    # iterate over tex_content and find the file that contains the abstract
    main_tex = None
    for file, content in tex_content.items():
        if "begin{abstract}" in content and "begin{document}" in content:
            print(f"Found main tex: {file}")
            main_tex = file
            break

    if not main_tex:
        logging.info(f"Cannot find main tex file in {tex_folder}")
        return None
    
    main_content = tex_content[main_tex]
    # identity the location of first "\title" in main_content
    title_start = main_content.find("\\title")
    doc_start = main_content.find("\\begin{document}")
    if title_start == -1:
        title_start = doc_start
    main_end = main_content.find("\\end{document}") + len("\\end{document}")

    main_start = min(title_start, doc_start)
    main_content = main_content[main_start:main_end]

    def replace_input_sections_inline(content):
        lines_to_replace = False
        lines = [line for line in content.split('\n')]
        for i, line in enumerate(lines):
            if not line.strip():
                line = ""
            if line.startswith("\\input{"):
                lines_to_replace = True
                # get the tex_file in input{tex_file}
                tex_file = line.split("{")[1].split("}")[0]
                if not tex_file.endswith(".tex"):
                    tex_file += ".tex"
                tex_path = os.path.join("/".join(main_tex.split("/")[:-1]), tex_file)
                print(tex_path)
                if tex_path not in tex_content:
                    raise Exception(f"File {tex_path} not found in tex_content")
                else:
                    lines[i] = tex_content[tex_path]
                    # print(line)

        return lines_to_replace, '\n'.join(lines)
    
    has_input_lines, main_content = replace_input_sections_inline(main_content)
    while has_input_lines:
        has_input_lines, main_content = replace_input_sections_inline(main_content)

    main_content = re.sub(r'\n\n\s*\n', '\n', main_content)

    logging.info(f"CONG TEST len(main_content): {len(main_content)}")
    return main_content