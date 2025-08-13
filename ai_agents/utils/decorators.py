"""
Decoradores útiles para el framework AI Agents.
"""

import time
import logging
import functools
from typing import Any, Callable, Optional, Type, Union
import asyncio

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorador para reintentar funciones que fallan.
    
    Args:
        max_attempts: Número máximo de intentos
        delay: Delay inicial entre intentos (segundos)
        backoff: Factor de multiplicación del delay
        exceptions: Excepciones que activan el retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"Función {func.__name__} falló después de {max_attempts} intentos")
                        raise
                    
                    logger.warning(f"Intento {attempt + 1} falló para {func.__name__}: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        logger.error(f"Función {func.__name__} falló después de {max_attempts} intentos")
                        raise
                    
                    logger.warning(f"Intento {attempt + 1} falló para {func.__name__}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        
        # Retornar wrapper apropiado según si la función es async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def measure_time(include_args: bool = False):
    """
    Decorador para medir tiempo de ejecución.
    
    Args:
        include_args: Si incluir argumentos en el log
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_msg = f"{func.__name__} ejecutada en {execution_time:.4f}s"
                if include_args:
                    log_msg += f" con args={args}, kwargs={kwargs}"
                
                logger.info(log_msg)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} falló después de {execution_time:.4f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log_msg = f"{func.__name__} ejecutada en {execution_time:.4f}s"
                if include_args:
                    log_msg += f" con args={args}, kwargs={kwargs}"
                
                logger.info(log_msg)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} falló después de {execution_time:.4f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_calls(level: str = "DEBUG", include_result: bool = False):
    """
    Decorador para loggear llamadas a funciones.
    
    Args:
        level: Nivel de logging
        include_result: Si incluir el resultado en el log
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_level = getattr(logging, level.upper(), logging.DEBUG)
            
            logger.log(log_level, f"Llamando {func.__name__} con args={args}, kwargs={kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                
                if include_result:
                    logger.log(log_level, f"{func.__name__} retornó: {result}")
                else:
                    logger.log(log_level, f"{func.__name__} completada exitosamente")
                
                return result
                
            except Exception as e:
                logger.log(log_level, f"{func.__name__} falló: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_level = getattr(logging, level.upper(), logging.DEBUG)
            
            logger.log(log_level, f"Llamando {func.__name__} con args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    logger.log(log_level, f"{func.__name__} retornó: {result}")
                else:
                    logger.log(log_level, f"{func.__name__} completada exitosamente")
                
                return result
                
            except Exception as e:
                logger.log(log_level, f"{func.__name__} falló: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_result(ttl: Optional[float] = None):
    """
    Decorador simple para cachear resultados.
    
    Args:
        ttl: Time to live en segundos (None = sin expiración)
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Crear key del cache
            key = str(args) + str(sorted(kwargs.items()))
            
            # Verificar si está en cache y no expirado
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit para {func.__name__}")
                    return result
                else:
                    del cache[key]
            
            # Ejecutar función y cachear resultado
            logger.debug(f"Cache miss para {func.__name__}")
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        return wrapper
    
    return decorator


def validate_types(**type_checks):
    """
    Decorador para validar tipos de argumentos.
    
    Args:
        **type_checks: Diccionario con nombre_arg: tipo esperado
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtener nombres de argumentos
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validar tipos
            for arg_name, expected_type in type_checks.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    if value is not None and not isinstance(value, expected_type):
                        raise TypeError(
                            f"{func.__name__}: argumento '{arg_name}' debe ser {expected_type.__name__}, "
                            f"recibido {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator
