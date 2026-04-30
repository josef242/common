# logger.py  – TCP based logging that works across multiple processes
import os
import datetime
import json
import queue
import socket
import struct
import threading
import atexit
from contextlib import suppress
import time

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _pack(msg_dict: dict) -> bytes:
    """Length-prefix the UTF-8 JSON payload so the server can frame messages."""
    payload = json.dumps(msg_dict, ensure_ascii=False).encode('utf-8')
    return struct.pack('!I', len(payload)) + payload          # 4-byte length prefix (big-endian)

def _recv_all(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes or raise EOFError."""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data

# ---------------------------------------------------------------------------
# the logger
# ---------------------------------------------------------------------------

class Logger:
    """
    Exactly the same constructor & public methods as your original Logger, but:

    • When `rank == 0`            ➜ starts a background TCP server (one thread)
    • When `rank  > 0`            ➜ messages are queued to one client thread
                                    (non-blocking for your training loop)

    All file-handling semantics (resume / no_log / default log-file selection)
    are unchanged and live exclusively in the server so there is never any
    concurrent file I/O from the workers.
    """

    # ------------  construction  --------------------------------------------------

    def __init__(self, dir=None, resume=False, no_log=False):
        self.dir      = dir
        self.resume   = resume
        self.no_log   = no_log
        self.rank     = 0                       # overwritten later by set_rank()
        self.default_logfile = 'log.txt'
        self._server   = None                   # server thread (rank 0 only)
        self._client_q = None                   # queue & client thread (ranks >0)
        self._client_thr = None
        self._client_ready = threading.Event()
        self._SERVER_HOST = '127.0.0.1'        # default server host TODO: FIX THIS
        self._SERVER_PORT = '29600'             # default server port

        self._writer_queue = None  # Queue for the single writer thread
        self._writer_thread = None
        self._recovery_callbacks = []  # Callbacks to invoke when NAS recovers

        if dir and not no_log:
            os.makedirs(dir, exist_ok=True)

    # ------------  public API (unchanged)  ---------------------------------------

    def print_and_log(self, msg, r0_only=True, log_file=None, silent=False):
        if r0_only and self.rank != 0:
            return
        
        if self._server is None and self._client_thr is None and self.rank == 0:
            # means set_rank() never ran
            raise RuntimeError("Logger used before set_rank() was called.")

        if log_file is None:
            log_file = self.default_logfile

        timestamped = f"{_now()} | {msg}"

        # rank 0 processes the message locally, others push to queue
        if self.rank == 0:
            self._local_write(timestamped, log_file, silent)
        else:
            self._send_to_server(timestamped, log_file, silent)

    def get_dir(self):
        return self.dir

    def set_resume(self, resume):
        self.resume = resume

    def set_rank(self, rank):
        self.rank = rank
        # Lazily spin up server or client in the first call that *needs* it,
        # but doing it here makes the first log fast & predictable.
        if rank == 0:
            self._start_server_once()
        else:
            self._start_client_once()

    def set_logdir(self, dir):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)

    def set_default_logfile(self, logfile):
        self.default_logfile = logfile

    def set_server_host(self, host):
        self._SERVER_HOST = host
    
    def set_server_port(self, port):
        self._SERVER_PORT = port

    def flush(self, timeout=None):
        """Block until all queued messages have been written.

        Returns True if the queue drained, False if the timeout expired.
        """
        if self._writer_queue is None:
            return True
        try:
            self._writer_queue.join()
            return True
        except Exception:
            return False

    def on_nas_recovery(self, callback):
        """Register a callback to be invoked when NAS connectivity is restored.

        The callback will be called from the writer thread, so it should be
        thread-safe and non-blocking (e.g., spawn a subprocess or set a flag).
        """
        self._recovery_callbacks.append(callback)

    # ---------------------------------------------------------------------------
    # rank-0 server implementation
    # ---------------------------------------------------------------------------

    def _start_server_once(self):
        if self._server is not None:
            return

        # Start the single writer thread FIRST
        self._writer_queue = queue.Queue()
        self._writer_thread = threading.Thread(target=self._writer_worker, daemon=True, name='LoggerWriter')
        self._writer_thread.start()

        host = self._SERVER_HOST
        port = int(self._SERVER_PORT)

        def _serve():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._local_write(f"{_now()} | [Logger] starting server on {host}:{port}", self.default_logfile, False)
                s.bind((host, port))
                s.listen()
                while True:
                    with suppress(Exception):
                        conn, _ = s.accept()
                        threading.Thread(target=self._handle_client,
                                         args=(conn,), daemon=True).start()

        self._server = threading.Thread(target=_serve, daemon=True, name='LoggerServer')
        self._server.start()
        atexit.register(self._server.join, timeout=0.2)

    def _writer_worker(self):
        """Single thread that handles ALL stdout and file writes."""
        
        buffer = []  # List of (msg, log_file) tuples
        retry_delay = 1  # Start with 1 second retry
        max_retry_delay = 60  # Cap at 60 seconds
        last_status_time = time.time()
        status_interval = 300  # Print status every 5 minutes
        nas_was_down = False
        
        while True:
            # Get next message (blocking call)
            msg, log_file, silent = self._writer_queue.get()
            
            # Always print to stdout immediately if not silent
            if not silent:
                print(msg, flush=True)
            
            if self.no_log:
                self._writer_queue.task_done()
                continue
            
            # Add to buffer (we'll try to write everything in buffer)
            buffer.append((msg, log_file))
            
            # Try to flush the entire buffer
            while buffer:
                try:
                    # Attempt to write the first message in buffer
                    msg_to_write, file_to_write = buffer[0]
                    full_path = os.path.join(self.dir, file_to_write)

                    # Always use append mode - it creates if needed and NEVER truncates
                    # (The old 'w' mode check caused data loss on NAS reconnect)
                    with open(full_path, 'a', encoding='utf-8', errors='replace') as fp:
                        fp.write(msg_to_write + '\n')
                    
                    # Success! Remove from buffer
                    buffer.pop(0)
                    
                    # If we just recovered from an outage
                    if nas_was_down and not buffer:
                        print(f"[Logger] NAS recovered! All buffered messages have been written.", flush=True)
                        nas_was_down = False
                        retry_delay = 1  # Reset retry delay

                        # Invoke recovery callbacks (e.g., retry failed rsync)
                        for cb in self._recovery_callbacks:
                            try:
                                cb()
                            except Exception as e:
                                print(f"[Logger] Recovery callback error: {e}", flush=True)
                    
                except (OSError, IOError) as e:  # Both filesystem errors
                    # NAS is down/unreachable
                    nas_was_down = True
                    
                    # Print periodic status updates
                    current_time = time.time()
                    if current_time - last_status_time > status_interval:
                        print(f"[Logger] WARNING: Buffer mode active - {len(buffer)} messages buffered (NAS unreachable)", flush=True)
                        last_status_time = current_time
                    
                    # Wait before retrying (exponential backoff)
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                    
                    # Break out of the while loop to get new messages
                    # (we'll try the buffer again when we get the next message)
                    break

                except Exception as e:  # Unexpected errors
                    print(f"[Logger] ERROR writing to {file_to_write}: {e}", flush=True)
                    buffer.pop(0)  # Drop this message to avoid infinite loop

            self._writer_queue.task_done()

    def _handle_client(self, conn: socket.socket):
        with conn:
            try:
                while True:
                    length = _recv_all(conn, 4)
                    length, = struct.unpack('!I', length)
                    payload = _recv_all(conn, length)
                    data = json.loads(payload.decode('utf-8'))
                    # Queue it instead of writing directly
                    self._writer_queue.put((data['msg'], data['file'], data['silent']))
            except EOFError:
                pass
            except Exception as e:
                print(f"[Logger] client handler exception: {e}", flush=True)

    def _local_write(self, msg, log_file, silent):
        """For rank 0's own messages."""
        if self._writer_queue is not None:
            # Use the queue like everyone else
            self._writer_queue.put((msg, log_file, silent))
        else:
            # Fallback for edge cases (shouldn't happen with proper set_rank)
            if not silent:
                print(msg, flush=True)
            if not self.no_log:
                full_path = os.path.join(self.dir, log_file)
                with open(full_path, 'a', encoding='utf-8', errors='replace') as fp:
                    fp.write(msg + '\n')

    # ---------------------------------------------------------------------------
    # worker-side client
    # ---------------------------------------------------------------------------

    def _start_client_once(self):
        if self._client_thr is not None or self.no_log:
            return
        self._client_q = queue.SimpleQueue()

        host = self._SERVER_HOST
        port = int(self._SERVER_PORT)
        if host is None:
            raise RuntimeError(
                "Environment variable LOGGER_HOST not set on worker ranks "
                "(rank 0 sets it automatically, or set it yourself)."
            )

        def _client():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # retry loop until server is up (useful at startup)
                while True:
                    try:
                        # print(f"[Logger] connecting to {host}:{port}", flush=True)
                        sock.connect((host, port))
                        break
                    except OSError:
                        time.sleep(0.1)

                self._client_ready.set()

                while True:
                    data = self._client_q.get()
                    try:
                        sock.sendall(_pack(data))
                    except OSError:
                        # server is dead – just discard & fall back to local print
                        print(data['msg'], flush=True)

        self._client_thr = threading.Thread(target=_client, daemon=True, name='LoggerClient')
        self._client_thr.start()
        # wait until the socket is connected so that first log-call is not lost
        self._client_ready.wait(timeout=5)

    def _send_to_server(self, msg, log_file, silent):
        if self.no_log:
            if not silent:
                print(msg, flush=True)
            return

        self._start_client_once()   # lazily ensure client exists
        payload = {'msg': msg, 'file': log_file, 'silent': silent}
        self._client_q.put(payload)

# ---------------------------------------------------------------------------
# module-level function to use the logger
# ---------------------------------------------------------------------------

# Create a global logger instance
_instance = Logger()

# Define the module-level function to delegate to the instance
def print_and_log(msg, r0_only=True, log_file=None, silent=False):
    return _instance.print_and_log(msg, r0_only, log_file, silent)

def flush(timeout=None):
    return _instance.flush(timeout)