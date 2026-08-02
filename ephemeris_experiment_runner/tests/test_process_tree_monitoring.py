from __future__ import annotations

import subprocess
import sys
import time
import unittest

from ephemeris_experiments.progress import process_tree_metrics


class ProcessTreeMonitoringTests(unittest.TestCase):
    def test_process_tree_metrics_include_cpu_child(self):
        code = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "\"import time; end=time.time()+3; x=0\\nwhile time.time()<end:\\n x+=1\"])\n"
            "time.sleep(3.5)\n"
            "child.wait()\n"
        )
        process = subprocess.Popen([sys.executable, "-c", code])
        try:
            time.sleep(0.5)
            samples = []
            for _ in range(6):
                samples.append(process_tree_metrics(process.pid))
                time.sleep(0.25)
            self.assertTrue(any(sample.descendant_pids for sample in samples))
            self.assertTrue(any(sample.rss_bytes and sample.rss_bytes > 0 for sample in samples))
            self.assertTrue(any(sample.cpu_percent and sample.cpu_percent > 0.0 for sample in samples))
            self.assertTrue(any(sample.worker_pid and sample.worker_pid != process.pid for sample in samples))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    unittest.main()
