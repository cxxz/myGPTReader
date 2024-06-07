
import os
import logging
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
from langchain.chat_models import ChatOpenAI
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason, CancellationReason, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from app.fetch_web_post import get_urls, get_youtube_transcript, scrape_website, scrape_website_by_phantomjscloud, download_audio_from_youtube
from app.prompt import get_prompt_template
from app.util import get_language_code, get_youtube_video_id

import whisperx
from whisperx.utils import get_writer

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
index_cache_voice_dir = Path(os.path.expanduser('~/workspace/content_data/voice/'))
transcribed_cache_file_dir = Path(os.path.expanduser('~/workspace/content_data/transcribed/'))

if not index_cache_web_dir.is_dir():
    index_cache_web_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_voice_dir.is_dir():
    index_cache_voice_dir.mkdir(parents=True, exist_ok=True)

if not index_cache_file_dir.is_dir():
    index_cache_file_dir.mkdir(parents=True, exist_ok=True)

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

def get_file_from_url(url, prefix = "page", format = ".txt"):
    if prefix == "youtube":
        video_id = get_youtube_video_id(url)
        md5_url = video_id
    else:
        md5_url = hashlib.md5(url.encode()).hexdigest()
    file_name = f"{index_cache_file_dir}/{prefix}_{md5_url}{format}"
    if os.path.exists(file_name):
        logging.info(f"CONG TEST {file_name} exists for {url}")
        return True, file_name
    return False, file_name
    
def write_url_doc(text, file_name):
    logging.info(f"CONG TEST write_url_doc: {file_name}")
    with open(file_name, "w") as f:
        f.write(text)
    return file_name

def get_documents_from_urls(urls, format = ".txt"):
    logging.info(f"CONG TEST getting {format} documents from urls: {urls}")
    docs = {}
    if len(urls['page_urls']) > 0:
        for url in urls['page_urls']:
            file_exist, file_name = get_file_from_url(url, "page", format)
            docs[url] = file_name if file_exist else write_url_doc(scrape_website(url), file_name)
    if len(urls['rss_urls']) > 0:
        rss_documents = RssReader().load_data(urls['rss_urls'])
        for i, url in enumerate(urls['rss_urls']):
            file_exist, file_name = get_file_from_url(url, "rss", format)
            docs[url] = file_name if file_exist else write_url_doc(rss_documents[i], file_name)
    if len(urls['phantomjscloud_urls']) > 0:
        for url in urls['phantomjscloud_urls']:
            file_exist, file_name = get_file_from_url(url, "phantomjscloud", format)
            docs[url] = file_name if file_exist else write_url_doc(scrape_website_by_phantomjscloud(url), file_name)
    if len(urls['youtube_urls']) > 0:
        for url in urls['youtube_urls']:
            video_id = get_youtube_video_id(url)
            # print(f"CONG TEST video_id: {video_id}")
            file_exist, file_name = get_file_from_url(url, "youtube", format)
            if file_exist:
                logging.info(f"CONG TEST {file_name} already exists!")
                docs[url] = file_name 
            elif format == ".txt":
                docs[url] = write_url_doc(get_official_transcript_from_youtube(video_id), file_name)
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
