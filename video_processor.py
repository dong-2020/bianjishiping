import cv2
import numpy as np
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, shared_memory, Queue, Process
import os
import time
from collections import OrderedDict, deque
import psutil
import asyncio
from queue import Queue as ThreadQueue
from threading import Event, Lock
import logging
from datetime import datetime
import json
from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa
import soundfile as sf
import subprocess

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('VideoProcessor')

class VideoProcessingError(Exception):
    """视频处理相关的自定义异常"""
    pass

class ConfigManager:
    """配置管理类"""
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return self._default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return self._default_config()
            
    def _default_config(self):
        """默认配置"""
        return {
            'max_prefetch': 10,
            'cache_size': 30,
            'chunk_size': 120,
            'align_boundary': 64,
            'hardware_acceleration': True,
            'output_format': 'mp4',
            'output_quality': 'high',
            'processing_options': {
                'denoise': True,
                'stabilize': False,
                'enhance': True
            }
        }
        
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
            
    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)
        
    def set(self, key, value):
        """设置配置项"""
        self.config[key] = value
        self.save_config()

class OutputManager:
    """输出管理类"""
    def __init__(self, base_dir='output'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.base_dir / self.current_session
        self.session_dir.mkdir(exist_ok=True)
        
    def get_output_path(self, filename):
        """获取输出文件路径"""
        return str(self.session_dir / filename)
        
    def save_frame(self, frame, frame_number):
        """保存单帧"""
        output_path = self.get_output_path(f'frame_{frame_number:06d}.jpg')
        cv2.imwrite(output_path, frame)
        return output_path
        
    def save_metadata(self, metadata):
        """保存元数据"""
        metadata_path = self.get_output_path('metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    def cleanup_old_sessions(self, max_sessions=10):
        """清理旧的会话目录"""
        sessions = sorted(list(self.base_dir.glob('*')))
        if len(sessions) > max_sessions:
            for session in sessions[:-max_sessions]:
                try:
                    if session.is_dir():
                        for file in session.glob('*'):
                            file.unlink()
                        session.rmdir()
                except Exception as e:
                    logger.error(f"清理旧会话失败: {str(e)}")

class MemoryPool:
    def __init__(self, max_size=1024*1024*512):  # 512MB
        self.max_size = max_size
        self.current_size = 0
        self.pools = {}
        self.last_access = {}
        
    def get_buffer(self, shape, dtype):
        """获取或创建内存缓冲区"""
        key = (shape, dtype)
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        if key in self.pools:
            self.last_access[key] = time.time()
            return self.pools[key]
            
        # 如果需要清理内存
        while self.current_size + size > self.max_size and self.pools:
            self._cleanup_oldest()
            
        # 创建新的缓冲区
        buffer = np.empty(shape, dtype)
        self.pools[key] = buffer
        self.last_access[key] = time.time()
        self.current_size += size
        return buffer
        
    def _cleanup_oldest(self):
        """清理最旧的缓冲区"""
        if not self.pools:
            return
            
        oldest_key = min(self.last_access.items(), key=lambda x: x[1])[0]
        size = np.prod(oldest_key[0]) * np.dtype(oldest_key[1]).itemsize
        del self.pools[oldest_key]
        del self.last_access[oldest_key]
        self.current_size -= size

class LRUCache(OrderedDict):
    def __init__(self, maxsize=128):
        super().__init__()
        self.maxsize = maxsize
        
    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None
        
    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)

class AsyncFrameLoader:
    def __init__(self, max_prefetch=10):
        self.prefetch_queue = ThreadQueue(maxsize=max_prefetch)
        self.stop_event = Event()
        self.frame_ready_event = Event()
        self.lock = Lock()
        self.current_frame = None
        self.loader_thread = None
        
    def start(self, video_capture, frame_processor):
        """启动异步加载器"""
        self.stop_event.clear()
        self.loader_thread = threading.Thread(
            target=self._load_frames,
            args=(video_capture, frame_processor)
        )
        self.loader_thread.daemon = True
        self.loader_thread.start()
        
    def stop(self):
        """停止异步加载器"""
        self.stop_event.set()
        if self.loader_thread:
            self.loader_thread.join()
        # 清空队列
        while not self.prefetch_queue.empty():
            self.prefetch_queue.get()
            
    def _load_frames(self, video_capture, frame_processor):
        """异步加载帧"""
        while not self.stop_event.is_set():
            if self.prefetch_queue.qsize() < self.prefetch_queue.maxsize:
                ret, frame = video_capture.read()
                if not ret:
                    break
                    
                # 预处理帧
                processed_frame = frame_processor(frame)
                
                # 将处理后的帧放入队列
                self.prefetch_queue.put((ret, processed_frame))
                self.frame_ready_event.set()
            else:
                time.sleep(0.001)  # 避免CPU过度使用
                
    def get_frame(self):
        """获取下一帧"""
        if self.prefetch_queue.empty():
            self.frame_ready_event.wait(timeout=1.0)
            self.frame_ready_event.clear()
            
        try:
            return self.prefetch_queue.get_nowait()
        except ThreadQueue.Empty:
            return False, None

class AsyncProcessor:
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending = {}
        self.lock = Lock()
        
    async def process_async(self, frame, frame_id, process_func):
        """异步处理帧"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.executor,
            process_func,
            frame
        )
        
        with self.lock:
            self.pending[frame_id] = future
            
        try:
            result = await future
            return result
        finally:
            with self.lock:
                self.pending.pop(frame_id, None)
                
    def cancel_all(self):
        """取消所有待处理的任务"""
        with self.lock:
            for future in self.pending.values():
                future.cancel()
            self.pending.clear()

class VideoProcessor:
    def __init__(self, config_path='config.json'):
        # 配置管理
        self.config = ConfigManager(config_path)
        self.output_manager = OutputManager()
        
        # 异步处理器初始化
        self.async_processor = AsyncProcessor()
        self.frame_loader = AsyncFrameLoader(
            max_prefetch=self.config.get('max_prefetch', 10)
        )
        self.processing_queue = deque(maxlen=30)
        self.frame_count = 0
        
        # 获取系统内存信息
        mem = psutil.virtual_memory()
        
        # 设置缓存大小为可用内存的10%
        cache_size = int(mem.available * 0.1)
        self.memory_pool = MemoryPool(max_size=cache_size)
        
        # 创建LRU缓存
        self.frame_cache = LRUCache(maxsize=self.config.get('cache_size', 30))
        self.effect_cache = LRUCache(maxsize=10)
        
        # 获取CPU核心数，预留一个核心给系统
        self.num_cores = max(1, cpu_count() - 1)
        
        # 创建进程池和共享内存
        self.process_pool = Pool(processes=self.num_cores)
        self.shared_mem = None
        
        # 创建线程池用于I/O操作
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self._cached_gradients = {}
        self.cap = None
        
        # 检测并启用硬件加速支持
        if self.config.get('hardware_acceleration', True):
            self._setup_hardware_acceleration()
        
        # 性能监控
        self.perf_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'errors': [],
            'warnings': []
        }
        
        # 视频处理预设
        self.resolution_presets = {
            'high': {
                'max_width': 3840,  # 4K
                'quality_factor': 0.95
            },
            'medium': {
                'max_width': 1920,  # 1080p
                'quality_factor': 0.85
            },
            'low': {
                'max_width': 1280,  # 720p
                'quality_factor': 0.75
            }
        }
        
        self.fps_presets = {
            'high': 60,
            'medium': 30,
            'low': 24
        }
        
        self.encoder_presets = {
            'fast': {
                'preset': 'veryfast',
                'tune': 'zerolatency'
            },
            'balanced': {
                'preset': 'medium',
                'tune': 'film'
            },
            'quality': {
                'preset': 'slow',
                'tune': 'film'
            }
        }
        
        self.keyframe_settings = {
            'high': {
                'min_interval': 10,
                'max_interval': 250,
                'scene_threshold': 0.3
            },
            'medium': {
                'min_interval': 30,
                'max_interval': 300,
                'scene_threshold': 0.4
            },
            'low': {
                'min_interval': 50,
                'max_interval': 350,
                'scene_threshold': 0.5
            }
        }
        
        self.bitrate_presets = {
            'high': {
                'min_bitrate': '4000k',
                'max_bitrate': '8000k',
                'buffer_size': '6000k'
            },
            'medium': {
                'min_bitrate': '2000k',
                'max_bitrate': '4000k',
                'buffer_size': '3000k'
            },
            'low': {
                'min_bitrate': '800k',
                'max_bitrate': '2000k',
                'buffer_size': '1500k'
            }
        }
        
        self.compression_params = {
            'high': {
                'crf': 18,
                'qmin': 10,
                'qmax': 51,
                'aq-mode': 3
            },
            'medium': {
                'crf': 23,
                'qmin': 15,
                'qmax': 51,
                'aq-mode': 2
            },
            'low': {
                'crf': 28,
                'qmin': 20,
                'qmax': 51,
                'aq-mode': 1
            }
        }
        
        # 高级场景检测和编码配置
        self.scene_detection_config = {
            'high': {
                'motion_threshold': 0.15,
                'color_threshold': 0.25,
                'edge_threshold': 0.3,
                'temporal_window': 5
            },
            'medium': {
                'motion_threshold': 0.2,
                'color_threshold': 0.3,
                'edge_threshold': 0.4,
                'temporal_window': 3
            },
            'low': {
                'motion_threshold': 0.25,
                'color_threshold': 0.35,
                'edge_threshold': 0.45,
                'temporal_window': 2
            }
        }
        
        self.advanced_encoding = {
            'high': {
                'me_method': 'umh',        # 精确运动估计
                'subq': 7,                 # 子像素运动估计质量
                'trellis': 2,              # 基于速率失真优化的量化
                'refs': 16,                # 参考帧数量
                'b_strategy': 2,           # B帧策略
                'mixed_refs': True,        # 混合参考帧
                'adaptive_quantization': 3  # 自适应量化模式
            },
            'medium': {
                'me_method': 'hex',
                'subq': 5,
                'trellis': 1,
                'refs': 8,
                'b_strategy': 1,
                'mixed_refs': True,
                'adaptive_quantization': 2
            },
            'low': {
                'me_method': 'dia',
                'subq': 3,
                'trellis': 0,
                'refs': 4,
                'b_strategy': 1,
                'mixed_refs': False,
                'adaptive_quantization': 1
            }
        }
        
        # 初始化场景分析缓存
        self.scene_cache = []
        self.motion_history = []
        
        # 添加音频处理选项
        self.audio_options = {
            'background_music': {
                'enabled': False,
                'path': '',
                'volume_ratio': 0.3
            }
        }
        
        logger.info("VideoProcessor initialized successfully")
        
    def detect_scene_change(self, frame1, frame2, threshold):
        """检测场景变化"""
        try:
            if frame1 is None or frame2 is None:
                return False
                
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 计算直方图
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            
            # 计算直方图差异
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 判断是否为场景变化
            return abs(1 - diff) > threshold
            
        except Exception as e:
            self.logger.error(f"场景变化检测失败: {str(e)}")
            return False
            
    def calculate_content_complexity(self, frame):
        """计算内容复杂度"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算梯度
            dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            
            # 计算梯度幅值
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # 计算复杂度分数
            complexity = np.mean(magnitude)
            
            # 归一化到0-1范围
            complexity = min(1.0, complexity / 128.0)
            
            return complexity
            
        except Exception as e:
            self.logger.error(f"内容复杂度计算失败: {str(e)}")
            return 0.5
            
    def get_adaptive_bitrate(self, frame_size, complexity, quality='medium'):
        """获取自适应比特率"""
        try:
            preset = self.bitrate_presets.get(quality, self.bitrate_presets['medium'])
            
            # 解析比特率值
            min_bitrate = int(preset['min_bitrate'].replace('k', '000'))
            max_bitrate = int(preset['max_bitrate'].replace('k', '000'))
            
            # 根据复杂度和分辨率调整比特率
            base_bitrate = min_bitrate + (max_bitrate - min_bitrate) * complexity
            
            # 根据分辨率调整
            pixels = frame_size[0] * frame_size[1]
            resolution_factor = min(1.0, pixels / (1920 * 1080))
            
            # 计算最终比特率
            final_bitrate = int(base_bitrate * resolution_factor)
            
            # 确保在预设范围内
            final_bitrate = max(min_bitrate, min(final_bitrate, max_bitrate))
            
            return f"{final_bitrate}k"
            
        except Exception as e:
            self.logger.error(f"比特率计算失败: {str(e)}")
            return preset['min_bitrate']
            
    def get_compression_params(self, complexity, quality='medium'):
        """获取压缩参数"""
        try:
            preset = self.compression_params.get(quality, self.compression_params['medium'])
            
            # 根据内容复杂度调整CRF值
            crf = preset['crf']
            if complexity > 0.7:
                crf = max(preset['crf'] - 2, 1)  # 复杂内容使用更低的CRF
            elif complexity < 0.3:
                crf = min(preset['crf'] + 2, 51)  # 简单内容可以使用更高的CRF
                
            return {
                'crf': crf,
                'qmin': preset['qmin'],
                'qmax': preset['qmax'],
                'aq-mode': preset['aq-mode']
            }
            
        except Exception as e:
            self.logger.error(f"压缩参数计算失败: {str(e)}")
            return self.compression_params['medium']
            
    def process_video(self, input_path, output_path=None, params=None):
        """处理视频的主函数"""
        try:
            if not os.path.exists(input_path):
                raise VideoProcessingError(f"输入文件不存在: {input_path}")
                
            # 合并选项
            processing_options = self.config.get('processing_options', {})
            if params:
                processing_options.update(params)
                
            # 处理音频选项
            if params and 'audio_options' in params:
                self.audio_options.update(params['audio_options'])
                
            # 设置输出路径
            if output_path is None:
                output_path = self.output_manager.get_output_path(
                    f'processed_{os.path.basename(input_path)}'
                )
                
            # 开始处理
            logger.info(f"开始处理视频: {input_path}")
            start_time = time.time()
            
            # 打开视频
            self.open_video(input_path)
            
            # 获取视频信息
            metadata = self._get_video_metadata()
            
            # 启动异步处理
            self.start_async_processing()
            
            # 提取原始音频
            temp_audio_path = os.path.join(os.path.dirname(input_path), 'temp_audio.mp3')
            self.extract_audio(input_path, temp_audio_path)
            
            # 处理音频
            if self.audio_options['background_music']['enabled']:
                audio_processor = AudioProcessor(temp_audio_path)
                audio_processor.add_background_music(
                    self.audio_options['background_music']['path'],
                    self.audio_options['background_music']['volume_ratio']
                )
                audio_processor.save_audio(temp_audio_path)
                
            # 处理视频帧
            self._process_video_frames(output_path, processing_options)
            
            # 合并处理后的音频和视频
            final_output = self.merge_audio_video(output_path, temp_audio_path)
            
            # 保存元数据
            metadata['processing_time'] = time.time() - start_time
            metadata['processing_options'] = processing_options
            metadata['performance_stats'] = self.get_performance_stats()
            self.output_manager.save_metadata(metadata)
            
            # 清理临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
            logger.info(f"视频处理完成: {final_output}")
            return final_output
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}", exc_info=True)
            self.perf_stats['errors'].append(str(e))
            raise VideoProcessingError(f"视频处理失败: {str(e)}")
            
        finally:
            self.stop_async_processing()
            self.close_video()
            
    def _get_video_metadata(self):
        """获取视频元数据"""
        if self.cap is None:
            return {}
            
        return {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'processing_date': datetime.now().isoformat()
        }
        
    def add_warning(self, message):
        """添加警告信息"""
        logger.warning(message)
        self.perf_stats['warnings'].append(message)
        
    def add_error(self, message):
        """添加错误信息"""
        logger.error(message)
        self.perf_stats['errors'].append(message)

    def start_async_processing(self):
        """启动异步处理"""
        if self.cap is not None:
            self.frame_loader.start(self.cap, self._preprocess_frame)
            
    def stop_async_processing(self):
        """停止异步处理"""
        self.frame_loader.stop()
        self.async_processor.cancel_all()
        
    def _preprocess_frame(self, frame):
        """预处理帧"""
        # 对齐内存
        frame = self._align_array(frame)
        # 进行基础的预处理
        return frame
        
    async def get_next_frame_async(self):
        """异步获取下一帧"""
        ret, frame = self.frame_loader.get_frame()
        if ret:
            self.perf_stats['prefetch_hits'] += 1
        else:
            self.perf_stats['prefetch_misses'] += 1
        return ret, frame
        
    def __del__(self):
        """清理资源"""
        self.stop_async_processing()
        self.close_video()
        if hasattr(self, 'process_pool'):
            self.process_pool.close()
            self.process_pool.join()
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'shared_mem') and self.shared_mem is not None:
            self.shared_mem.close()
            self.shared_mem.unlink()
        self._cached_gradients.clear()
        self.frame_cache.clear()
        self.effect_cache.clear()

    def _align_array(self, array):
        """确保数组内存对齐"""
        if array.ctypes.data % 64 != 0:
            return np.ascontiguousarray(array, align=64)
        return array
        
    def _create_shared_memory(self, shape, dtype):
        """创建共享内存区域"""
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        if self.shared_mem is not None:
            self.shared_mem.close()
            self.shared_mem.unlink()
        self.shared_mem = shared_memory.SharedMemory(create=True, size=size)
        return np.ndarray(shape, dtype=dtype, buffer=self.shared_mem.buf)
        
    def _get_cached_frame(self, frame_number):
        """获取缓存的帧"""
        return self.frame_cache.get(frame_number)
        
    def _cache_frame(self, frame_number, frame):
        """缓存处理后的帧"""
        self.frame_cache.put(frame_number, frame)
        
    def _get_cached_effect(self, effect_key):
        """获取缓存的效果"""
        return self.effect_cache.get(effect_key)
        
    def _cache_effect(self, effect_key, effect):
        """缓存处理后的效果"""
        self.effect_cache.put(effect_key, effect)
        
    def process_frame(self, frame, params=None):
        """优化的帧处理函数"""
        start_time = time.time()
        
        # 对齐内存
        frame = self._align_array(frame)
        
        # 获取质量设置
        quality = params.get('quality', 'medium') if params else 'medium'
        
        # 应用分辨率优化
        frame = self.optimize_resolution(frame, quality)
        
        # 检查效果缓存
        effect_key = str(params)
        cached_effect = self._get_cached_effect(effect_key)
        if cached_effect is not None:
            self.perf_stats['cache_hits'] += 1
            result = cached_effect.copy()
        else:
            self.perf_stats['cache_misses'] += 1
            
            try:
                # 创建共享内存数组
                shared_frame = self._create_shared_memory(frame.shape, frame.dtype)
                np.copyto(shared_frame, frame)
                
                # 分割帧
                chunks = self._split_frame(shared_frame)
                chunk_params = [(chunk, params) for chunk in chunks]
                
                # 使用进程池并行处理
                processed_chunks = self.process_pool.map(self._process_chunk, chunk_params)
                
                # 合并结果
                result = self._merge_chunks(processed_chunks)
                
                # 缓存结果
                self._cache_effect(effect_key, result)
                
            except Exception as e:
                print(f"并行处理失败: {str(e)}, 回退到CPU处理")
                result = self._process_frame_cpu(frame, params)
                
        # 记录处理时间
        process_time = time.time() - start_time
        self.perf_stats['processing_times'].append(process_time)
        
        return result
        
    def _optimize_memory_usage(self):
        """优化内存使用"""
        # 清理过期缓存
        current_memory = psutil.Process().memory_info().rss
        target_memory = psutil.virtual_memory().available * 0.8
        
        if current_memory > target_memory:
            # 清理帧缓存
            while len(self.frame_cache) > 10:
                self.frame_cache.popitem(last=False)
                
            # 清理效果缓存
            while len(self.effect_cache) > 5:
                self.effect_cache.popitem(last=False)
                
            # 清理梯度缓存
            self._cached_gradients.clear()
            
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = {
            'cache_hit_rate': self.perf_stats['cache_hits'] / 
                            (self.perf_stats['cache_hits'] + self.perf_stats['cache_misses']) 
                            if (self.perf_stats['cache_hits'] + self.perf_stats['cache_misses']) > 0 else 0,
            'avg_processing_time': np.mean(self.perf_stats['processing_times']) 
                                 if self.perf_stats['processing_times'] else 0,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cache_size': len(self.frame_cache)
        }
        return stats
        
    def _setup_hardware_acceleration(self):
        """设置硬件加速"""
        # 检测并启用 Metal 支持
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            # 设置 OpenCL 设备为 GPU
            cv2.ocl.Device.getDefault().setPreferableTarget(cv2.ocl.Device.TARGET_GPU)
            print("Metal acceleration enabled")
            
        # 检查可用的硬件加速器
        self.available_backends = []
        if cv2.videoio_registry.hasBackend(cv2.CAP_VIDEOTOOLBOX):
            self.available_backends.append(cv2.CAP_VIDEOTOOLBOX)
            print("VideoToolbox acceleration available")
            
    def read_frame(self):
        """优化的帧读取"""
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            return False, None
            
        # 确保帧格式适合 GPU 处理
        if frame.strides[0] % 4 != 0:
            frame = np.ascontiguousarray(frame)
            
        return True, frame
        
    def close_video(self):
        """关闭视频并清理资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.ocl.finish()  # 确保所有 GPU 操作完成
        
    @lru_cache(maxsize=32)
    def _create_corner_mask(self, height, width, radius):
        """缓存圆角蒙版计算结果"""
        mask = np.zeros((height, width), np.uint8)
        cv2.rectangle(mask, (radius, radius), (width-radius, height-radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (width-radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, height-radius), radius, 255, -1)
        cv2.circle(mask, (width-radius, height-radius), radius, 255, -1)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255

    def apply_corner_radius(self, image, radius):
        """优化后的圆角效果应用"""
        if radius <= 0:
            return image
            
        h, w = image.shape[:2]
        mask = self._create_corner_mask(h, w, radius)
        return cv2.multiply(image.astype(float), mask).astype(np.uint8)
    
    def _get_gradient_cache_key(self, shape, direction):
        """生成渐变缓存的键"""
        return f"{shape}_{direction}"
    
    def _create_gradient(self, shape, direction):
        """创建并缓存渐变效果"""
        cache_key = self._get_gradient_cache_key(shape, direction)
        if cache_key not in self._cached_gradients:
            h, w = shape[:2]
            if direction == "horizontal":
                gradient = np.linspace(0.7, 1.3, w)
                gradient = np.tile(gradient, (h, 1))
            else:  # vertical
                gradient = np.linspace(0.7, 1.3, h)
                gradient = np.tile(gradient.reshape(-1, 1), (1, w))
            gradient = np.dstack([gradient] * 3)
            self._cached_gradients[cache_key] = gradient
        return self._cached_gradients[cache_key]

    def apply_background_effect(self, image, effect):
        """优化后的背景特效应用"""
        if effect == "镜像":
            return cv2.flip(image, 1)
        elif effect == "左右渐变":
            gradient = self._create_gradient(image.shape, "horizontal")
            return cv2.multiply(image.astype(float), gradient).astype(np.uint8)
        elif effect == "上下渐变":
            gradient = self._create_gradient(image.shape, "vertical")
            return cv2.multiply(image.astype(float), gradient).astype(np.uint8)
        return image

    def adjust_brightness_contrast(self, img, brightness=0):
        """优化后的亮度调整"""
        if brightness == 0:
            return img

        # 使用查找表(LUT)优化亮度调整
        if brightness > 0:
            lut = np.array([min(255, int(i * (1 + brightness/255))) for i in range(256)], dtype=np.uint8)
        else:
            lut = np.array([max(0, int(i * (1 + brightness/255))) for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, lut)

    def adjust_saturation(self, img, saturation):
        """优化后的饱和度调整"""
        if saturation == 100:
            return img
            
        # 使用并行处理优化饱和度调整
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        h, s, v = cv2.split(img_hsv)
        s = np.clip(s * (saturation/100), 0, 255)
        img_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    def adjust_hue(self, image, hue):
        """优化后的色调调整"""
        if hue == 0:
            return image
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        h = (h + hue) % 180
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _apply_effect_gpu(self, gpu_frame, effect):
        """GPU 加速的效果处理"""
        if effect == "镜像":
            return cv2.flip(gpu_frame, 1)
        elif effect == "左右渐变":
            gradient = cv2.UMat(self._create_gradient(gpu_frame.get().shape, "horizontal"))
            return cv2.multiply(gpu_frame, gradient)
        elif effect == "上下渐变":
            gradient = cv2.UMat(self._create_gradient(gpu_frame.get().shape, "vertical"))
            return cv2.multiply(gpu_frame, gradient)
        return gpu_frame
        
    def _adjust_brightness_gpu(self, gpu_frame, brightness):
        """GPU 加速的亮度调整"""
        if brightness == 0:
            return gpu_frame
        
        scale = 1 + brightness/255
        return cv2.multiply(gpu_frame, scale)
        
    def _adjust_saturation_gpu(self, gpu_frame, saturation):
        """GPU 加速的饱和度调整"""
        if saturation == 100:
            return gpu_frame
            
        hsv = cv2.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation/100)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    def _process_chunk_cpu(self, chunk, params):
        """在CPU上处理数据块"""
        result = chunk.copy()
        
        if params.get('effect') != "无":
            result = self.apply_background_effect(result, params.get('effect'))
            
        if params.get('brightness') != 0:
            result = self.adjust_brightness_contrast(result, params.get('brightness'))
            
        if params.get('saturation') != 100:
            result = self.adjust_saturation(result, params.get('saturation'))
            
        return result
        
    def _process_frame_cpu(self, frame, params):
        """CPU 处理帧"""
        # 使用线程池并行处理不同的效果
        futures = []
        
        # 并行处理背景效果
        if params.get('effect') != "无":
            futures.append(self.thread_pool.submit(
                self.apply_background_effect, frame.copy(), params.get('effect')))
            
        # 并行处理亮度
        if params.get('brightness') != 0:
            futures.append(self.thread_pool.submit(
                self.adjust_brightness_contrast, frame.copy(), params.get('brightness')))
            
        # 并行处理饱和度
        if params.get('saturation') != 100:
            futures.append(self.thread_pool.submit(
                self.adjust_saturation, frame.copy(), params.get('saturation')))
            
        # 等待所有效果处理完成
        processed_frames = [f.result() for f in futures]
        
        # 合并处理结果
        result = frame
        for processed in processed_frames:
            result = cv2.addWeighted(result, 0.5, processed, 0.5, 0)
            
        return result

    def open_video(self, path):
        """打开视频文件并设置硬件加速"""
        if self.cap is not None:
            self.cap.release()
            
        # 创建带硬件加速的VideoCapture
        if cv2.CAP_VIDEOTOOLBOX in self.available_backends:
            self.cap = cv2.VideoCapture(path, cv2.CAP_VIDEOTOOLBOX)
        else:
            self.cap = cv2.VideoCapture(path)
            
        if not self.cap.isOpened():
            return False
            
        # 配置视频捕获参数
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)
        self.cap.set(cv2.CAP_PROP_FORMAT, -1)
        
        # 获取视频信息
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 优化块大小
        self._optimize_chunk_size(self.frame_height)
        
        return True
        
    def _optimize_chunk_size(self, frame_height):
        """优化块大小"""
        # 根据图像高度和CPU核心数优化块大小
        optimal_chunks = self.num_cores * 2  # 每个核心处理2个块
        self.chunk_size = max(16, frame_height // optimal_chunks)
        self.chunk_size = min(self.chunk_size, 240)  # 限制最大块大小
        
    def _split_frame(self, frame):
        """将帧分割成多个块以并行处理"""
        height = frame.shape[0]
        chunks = []
        for i in range(0, height, self.chunk_size):
            end = min(i + self.chunk_size, height)
            chunks.append(frame[i:end])
        return chunks
        
    def _merge_chunks(self, chunks):
        """合并处理后的块"""
        return np.vstack(chunks)
        
    def _process_chunk(self, chunk_data):
        """处理单个数据块"""
        chunk, params = chunk_data
        try:
            # 在GPU上处理数据块
            gpu_chunk = cv2.UMat(chunk)
            
            # 应用效果
            if params.get('effect') != "无":
                gpu_chunk = self._apply_effect_gpu(gpu_chunk, params.get('effect'))
                
            # 调整亮度
            if params.get('brightness') != 0:
                gpu_chunk = self._adjust_brightness_gpu(gpu_chunk, params.get('brightness'))
                
            # 调整饱和度
            if params.get('saturation') != 100:
                gpu_chunk = self._adjust_saturation_gpu(gpu_chunk, params.get('saturation'))
                
            # 返回处理后的数据块
            return gpu_chunk.get()
            
        except cv2.error:
            # 如果GPU处理失败，回退到CPU处理
            return self._process_chunk_cpu(chunk, params)

    def _process_video_frames(self, output_path, processing_options):
        """处理视频帧"""
        # 获取视频信息
        fps = self.fps
        frame_width = self.frame_width
        frame_height = self.frame_height
        
        # 获取质量设置
        quality = processing_options.get('quality', 'medium')
        
        # 优化帧率
        output_fps = self.optimize_fps(fps, quality)
        
        # 获取编码预设
        encoder_preset = self.get_encoder_preset(quality)
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            output_fps,
            (frame_width, frame_height)
        )
        
        # 处理视频帧
        frame_count = 0
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
                
            # 处理帧
            processed_frame = self.process_frame(frame, processing_options)
            
            # 保存帧
            out.write(processed_frame)
            frame_count += 1
            
            # 保存单帧
            if frame_count % 100 == 0:
                self.output_manager.save_frame(processed_frame, frame_count)
                
        out.release()

    def optimize_resolution(self, frame, quality='medium'):
        """智能分辨率调整"""
        try:
            preset = self.resolution_presets.get(quality, self.resolution_presets['medium'])
            current_height, current_width = frame.shape[:2]
            
            # 计算目标尺寸
            if current_width > preset['max_width']:
                scale_factor = preset['max_width'] / current_width
                target_width = preset['max_width']
                target_height = int(current_height * scale_factor)
                
                # 确保高度是2的倍数（编码要求）
                target_height = target_height - (target_height % 2)
                
                # 使用LANCZOS重采样进行缩放
                optimized_frame = cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LANCZOS4
                )
                
                self.logger.info(
                    f"分辨率优化: {current_width}x{current_height} -> {target_width}x{target_height}"
                )
                return optimized_frame
                
            return frame
            
        except Exception as e:
            self.logger.error(f"分辨率优化失败: {str(e)}")
            return frame
            
    def optimize_fps(self, input_fps, quality='medium'):
        """基础帧率优化"""
        try:
            target_fps = self.fps_presets.get(quality, self.fps_presets['medium'])
            
            # 如果输入帧率低于目标帧率，保持原帧率
            if input_fps < target_fps:
                return input_fps
                
            # 如果输入帧率高于目标帧率，降低到目标帧率
            return target_fps
            
        except Exception as e:
            self.logger.error(f"帧率优化失败: {str(e)}")
            return input_fps
            
    def get_encoder_preset(self, quality='balanced'):
        """获取编码预设"""
        try:
            return self.encoder_presets.get(quality, self.encoder_presets['balanced'])
        except Exception as e:
            self.logger.error(f"获取编码预设失败: {str(e)}")
            return self.encoder_presets['balanced']

    def analyze_motion(self, frame1, frame2):
        """分析运动强度"""
        try:
            if frame1 is None or frame2 is None:
                return 0.0
                
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # 计算运动强度
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_score = np.mean(magnitude)
            
            # 归一化
            motion_score = min(1.0, motion_score / 30.0)
            
            return motion_score
            
        except Exception as e:
            self.logger.error(f"运动分析失败: {str(e)}")
            return 0.0
            
    def analyze_edges(self, frame):
        """分析边缘复杂度"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用Canny边缘检测
            edges = cv2.Canny(gray, 100, 200)
            
            # 计算边缘密度
            edge_density = np.mean(edges > 0)
            
            return edge_density
            
        except Exception as e:
            self.logger.error(f"边缘分析失败: {str(e)}")
            return 0.0
            
    def analyze_color_distribution(self, frame):
        """分析颜色分布"""
        try:
            # 计算每个通道的直方图
            color_scores = []
            for channel in cv2.split(frame):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                # 计算直方图的标准差作为颜色复杂度指标
                std_dev = np.std(hist)
                color_scores.append(std_dev)
                
            # 综合三个通道的分数
            color_complexity = np.mean(color_scores) / 1000.0  # 归一化
            return min(1.0, color_complexity)
            
        except Exception as e:
            self.logger.error(f"颜色分析失败: {str(e)}")
            return 0.0
            
    def advanced_scene_detection(self, frame, prev_frame, quality='medium'):
        """高级场景检测"""
        try:
            if prev_frame is None:
                return False
                
            config = self.scene_detection_config[quality]
            
            # 分析各个特征
            motion_score = self.analyze_motion(prev_frame, frame)
            edge_score = self.analyze_edges(frame)
            color_score = self.analyze_color_distribution(frame)
            
            # 更新运动历史
            self.motion_history.append(motion_score)
            if len(self.motion_history) > config['temporal_window']:
                self.motion_history.pop(0)
                
            # 计算运动变化
            if len(self.motion_history) >= 2:
                motion_change = abs(motion_score - np.mean(self.motion_history[:-1]))
            else:
                motion_change = 0
                
            # 综合判断场景变化
            is_scene_change = (
                motion_change > config['motion_threshold'] or
                edge_score > config['edge_threshold'] or
                color_score > config['color_threshold']
            )
            
            return is_scene_change
            
        except Exception as e:
            self.logger.error(f"高级场景检测失败: {str(e)}")
            return False
            
    def get_advanced_encoding_params(self, frame_complexity, quality='medium'):
        """获取高级编码参数"""
        try:
            preset = self.advanced_encoding.get(quality, self.advanced_encoding['medium'])
            
            # 根据内容复杂度调整参数
            params = preset.copy()
            
            if frame_complexity > 0.7:
                # 复杂场景使用更精确的运动估计
                params['subq'] = min(params['subq'] + 1, 7)
                params['refs'] = min(params['refs'] + 2, 16)
            elif frame_complexity < 0.3:
                # 简单场景可以使用更快的设置
                params['subq'] = max(params['subq'] - 1, 1)
                params['refs'] = max(params['refs'] - 2, 1)
                
            return params
            
        except Exception as e:
            self.logger.error(f"获取高级编码参数失败: {str(e)}")
            return self.advanced_encoding['medium']

    def extract_audio(self, video_path, audio_path):
        """从视频中提取音频"""
        try:
            command = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'libmp3lame',
                '-y', audio_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            self.logger.info(f"音频提取成功: {audio_path}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"音频提取失败: {str(e)}")
            raise VideoProcessingError(f"音频提取失败: {str(e)}")
            
    def merge_audio_video(self, video_path, audio_path):
        """合并音频和视频"""
        try:
            output_path = os.path.splitext(video_path)[0] + '_with_audio.mp4'
            command = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y', output_path
            ]
            subprocess.run(command, check=True, capture_output=True)
            self.logger.info(f"音视频合并成功: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"音视频合并失败: {str(e)}")
            raise VideoProcessingError(f"音视频合并失败: {str(e)}")
            
class AudioProcessor:
    """音频处理类"""
    def __init__(self, input_file=None):
        self.audio = None
        self.sample_rate = None
        self.logger = logging.getLogger('AudioProcessor')
        if input_file:
            self.load_audio(input_file)
            
    def load_audio(self, file_path):
        """加载音频文件"""
        try:
            # 支持多种格式
            self.audio = AudioSegment.from_file(file_path)
            self.logger.info(f"成功加载音频文件: {file_path}")
        except Exception as e:
            self.logger.error(f"加载音频文件失败: {str(e)}")
            raise AudioProcessingError(f"加载音频文件失败: {str(e)}")
            
    def save_audio(self, output_path, format='mp3', quality='high'):
        """保存音频文件"""
        try:
            # 设置输出质量
            quality_params = {
                'high': {'mp3': {'bitrate': '320k'}, 'wav': {'bitrate': '32'}, 'aac': {'bitrate': '320k'}},
                'medium': {'mp3': {'bitrate': '192k'}, 'wav': {'bitrate': '24'}, 'aac': {'bitrate': '192k'}},
                'low': {'mp3': {'bitrate': '128k'}, 'wav': {'bitrate': '16'}, 'aac': {'bitrate': '128k'}}
            }
            
            params = quality_params.get(quality, quality_params['medium'])
            format_params = params.get(format, {})
            
            self.audio.export(output_path, format=format, **format_params)
            self.logger.info(f"成功保存音频文件: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存音频文件失败: {str(e)}")
            raise AudioProcessingError(f"保存音频文件失败: {str(e)}")
            
    def adjust_volume(self, factor=1.0):
        """调整音量"""
        try:
            if not 0.0 <= factor <= 5.0:
                raise ValueError("音量调节因子必须在0.0到5.0之间")
                
            self.audio = self.audio + (20 * np.log10(factor))
            self.logger.info(f"音量调整完成，因子: {factor}")
            return True
            
        except Exception as e:
            self.logger.error(f"音量调整失败: {str(e)}")
            raise AudioProcessingError(f"音量调整失败: {str(e)}")
            
    def normalize_volume(self, target_dbfs=-20):
        """音量标准化"""
        try:
            self.audio = normalize(self.audio, target_dbfs)
            self.logger.info(f"音量标准化完成，目标dBFS: {target_dbfs}")
            return True
            
        except Exception as e:
            self.logger.error(f"音量标准化失败: {str(e)}")
            raise AudioProcessingError(f"音量标准化失败: {str(e)}")
            
    def compress_dynamic_range(self, threshold=-20, ratio=2.0):
        """动态范围压缩"""
        try:
            self.audio = compress_dynamic_range(self.audio, threshold=threshold, ratio=ratio)
            self.logger.info(f"动态范围压缩完成，阈值: {threshold}, 比率: {ratio}")
            return True
            
        except Exception as e:
            self.logger.error(f"动态范围压缩失败: {str(e)}")
            raise AudioProcessingError(f"动态范围压缩失败: {str(e)}")
            
    def fade_in(self, duration_ms=1000):
        """淡入效果"""
        try:
            self.audio = self.audio.fade_in(duration_ms)
            self.logger.info(f"添加淡入效果完成，持续时间: {duration_ms}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"添加淡入效果失败: {str(e)}")
            raise AudioProcessingError(f"添加淡入效果失败: {str(e)}")
            
    def fade_out(self, duration_ms=1000):
        """淡出效果"""
        try:
            self.audio = self.audio.fade_out(duration_ms)
            self.logger.info(f"添加淡出效果完成，持续时间: {duration_ms}ms")
            return True
            
        except Exception as e:
            self.logger.error(f"添加淡出效果失败: {str(e)}")
            raise AudioProcessingError(f"添加淡出效果失败: {str(e)}")
            
    def reduce_noise(self, level='medium'):
        """降噪处理"""
        try:
            # 将音频转换为numpy数组
            samples = np.array(self.audio.get_array_of_samples())
            
            # 根据级别设置参数
            noise_params = {
                'low': {'order': 2, 'cutoff': 0.1},
                'medium': {'order': 3, 'cutoff': 0.2},
                'high': {'order': 4, 'cutoff': 0.3}
            }
            
            params = noise_params.get(level, noise_params['medium'])
            
            # 应用巴特沃斯滤波器
            b, a = butter(params['order'], params['cutoff'], btype='low')
            filtered_samples = filtfilt(b, a, samples)
            
            # 转换回AudioSegment
            self.audio = self.audio._spawn(filtered_samples.astype(np.int16))
            
            self.logger.info(f"降噪处理完成，级别: {level}")
            return True
            
        except Exception as e:
            self.logger.error(f"降噪处理失败: {str(e)}")
            raise AudioProcessingError(f"降噪处理失败: {str(e)}")
            
    def detect_noise_type(self):
        """识别环境噪声类型"""
        try:
            # 获取音频特征
            samples = np.array(self.audio.get_array_of_samples())
            
            # 计算基本特征
            rms = np.sqrt(np.mean(samples**2))
            zero_crossings = np.sum(np.diff(np.signbit(samples)))
            spectral_centroid = np.mean(np.abs(np.fft.fft(samples)))
            
            # 基于特征判断噪声类型
            noise_types = {
                'background': {'rms': 1000, 'zc': 1000, 'sc': 1000},
                'white': {'rms': 5000, 'zc': 5000, 'sc': 5000},
                'impulse': {'rms': 10000, 'zc': 10000, 'sc': 10000}
            }
            
            # 简单的噪声类型判断
            detected_type = 'unknown'
            min_distance = float('inf')
            
            for noise_type, features in noise_types.items():
                distance = (
                    abs(rms - features['rms']) +
                    abs(zero_crossings - features['zc']) +
                    abs(spectral_centroid - features['sc'])
                )
                if distance < min_distance:
                    min_distance = distance
                    detected_type = noise_type
                    
            self.logger.info(f"噪声类型识别完成: {detected_type}")
            return detected_type
            
        except Exception as e:
            self.logger.error(f"噪声类型识别失败: {str(e)}")
            raise AudioProcessingError(f"噪声类型识别失败: {str(e)}")
            
    def batch_process(self, input_dir, output_dir, format='mp3', quality='high'):
        """批量处理音频文件"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取所有支持的音频文件
            supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac']
            audio_files = [
                f for f in os.listdir(input_dir)
                if os.path.splitext(f)[1].lower() in supported_formats
            ]
            
            processed_files = []
            for audio_file in audio_files:
                try:
                    # 处理单个文件
                    input_path = os.path.join(input_dir, audio_file)
                    output_path = os.path.join(
                        output_dir,
                        f"{os.path.splitext(audio_file)[0]}.{format}"
                    )
                    
                    # 加载并处理音频
                    self.load_audio(input_path)
                    self.normalize_volume()
                    self.reduce_noise()
                    self.save_audio(output_path, format=format, quality=quality)
                    
                    processed_files.append({
                        'input': input_path,
                        'output': output_path,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    processed_files.append({
                        'input': input_path,
                        'error': str(e),
                        'status': 'failed'
                    })
                    
            self.logger.info(f"批量处理完成，共处理{len(processed_files)}个文件")
            return processed_files
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            raise AudioProcessingError(f"批量处理失败: {str(e)}")
            
    def get_audio_info(self):
        """获取音频信息"""
        try:
            return {
                'duration': len(self.audio),
                'channels': self.audio.channels,
                'sample_width': self.audio.sample_width,
                'frame_rate': self.audio.frame_rate,
                'frame_count': int(len(self.audio) * self.audio.frame_rate / 1000)
            }
        except Exception as e:
            self.logger.error(f"获取音频信息失败: {str(e)}")
            raise AudioProcessingError(f"获取音频信息失败: {str(e)}")

    def add_background_music(self, music_path, volume_ratio=0.3):
        """添加背景音乐，可调节主音频和背景音乐的比例
        
        Args:
            music_path (str): 背景音乐文件路径
            volume_ratio (float): 背景音乐音量比例，范围0.0-1.0
            
        Returns:
            bool: 操作是否成功
            
        Raises:
            AudioProcessingError: 音频处理失败时抛出
        """
        try:
            if not 0.0 <= volume_ratio <= 1.0:
                raise ValueError("音量比例必须在0.0到1.0之间")
                
            # 加载背景音乐
            background_music = AudioSegment.from_file(music_path)
            
            # 确保背景音乐长度匹配主音频
            if len(background_music) < len(self.audio):
                # 如果背景音乐较短，循环播放直到匹配主音频长度
                repeat_times = len(self.audio) // len(background_music) + 1
                background_music = background_music * repeat_times
                
            # 裁剪背景音乐到主音频长度
            background_music = background_music[:len(self.audio)]
            
            # 调整背景音乐音量
            background_music = background_music - (20 * np.log10(1/volume_ratio))
            
            # 混合音频
            self.audio = self.audio.overlay(background_music)
            
            self.logger.info(f"成功添加背景音乐，音量比例: {volume_ratio}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加背景音乐失败: {str(e)}")
            raise AudioProcessingError(f"添加背景音乐失败: {str(e)}")
            
class AudioProcessingError(Exception):
    """音频处理相关的自定义异常"""
    pass
