import os
import time
import threading
import psutil

class MemoryTracker:
    def __init__(self, interval=0.1):
        """
        Initializing the class instance for tracking memory usage at fixed time intervals using RSS information to minimize the overhead from using other third-party api's

        Args:
            interval (float): How often to sample memory (in seconds).
        """
        self.interval = interval
        self.keep_measuring = True
        self.process = psutil.Process(os.getpid())
        
        self.peak_rss = 0
        self.total_rss = 0
        self.sample_count = 0


    def _measure(self) -> None:
        """
        Function to call iteratively at each time interval corresponding to the value self.interval.
        You can cancel the measurement by setting the self.keep_measuring to False.
        """
        while self.keep_measuring:
            rss = self.process.memory_info().rss
            
            if rss > self.peak_rss:
                self.peak_rss = rss
                
            self.total_rss += rss
            self.sample_count += 1
            
            time.sleep(self.interval)

    def start(self) -> None:
        """
        Function to call for starting the memory tracker. The tracker is spawned on a separate thread.
        """
        self.peak_rss = 0
        self.total_rss = 0
        self.sample_count = 0
        self.keep_measuring = True
        
        self.thread = threading.Thread(target=self._measure, daemon=True)
        self.thread.start()


    def stop(self) -> tuple[float | int, float | int]:
        """
        Function to stop the memory tracker, by resetting the keep_measuring flag and closing the separate threads.
        It then takes all logged RSS values and converts them to MB for better interpretation. 

        Returns:
            tuple[float | int, float | int]: A tuple of float or ints representing the (average_rss, peak_rss)
        """
        self.keep_measuring = False
        self.thread.join()
        
        if self.sample_count == 0:
            return 0.0, 0.0
            
        mb_divisor = 1024 * 1024
        avg_rss = (self.total_rss / self.sample_count) / mb_divisor
        peak_rss = self.peak_rss / mb_divisor
        
        return avg_rss, peak_rss