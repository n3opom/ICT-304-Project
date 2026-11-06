
import csv
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import monotonic


class AlertNotifier:
    def __init__(
        self,
        mode="telegram",
        cooldown_seconds=60.0,
        output_path="results/caregiver_alerts.csv",
    ):
        self.mode = mode
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = float("-inf")

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if mode not in ("telegram", "simulated"):
            raise ValueError("Invalid notifier mode.")

        if mode == "telegram":
            if not self.bot_token or not self.chat_id:
                raise ValueError(
                    "Missing Telegram environment variables."
                )

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        self.executor = ThreadPoolExecutor(
            max_workers=1
        )

        self.log_file = self.output_path.open(
            "a", newline="", encoding="utf-8"
        )
        self.writer = csv.writer(self.log_file)

        if self.output_path.stat().st_size == 0:
            self.writer.writerow([
                "timestamp",
                "event",
                "fall_probability",
                "notification_mode",
                "status",
            ])
            self.log_file.flush()

    def notify(self, fall_probability, timestamp=None):
        now = monotonic() if timestamp is None else timestamp

        # Prevent repeated caregiver notifications.
        if now - self.last_alert_time < self.cooldown_seconds:
            print("[NOTIFIER] Cooldown active.")
            return False

        self.last_alert_time = now

        event_time = datetime.now().astimezone()
        time_text = event_time.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        )

        probability = float(fall_probability)

        message = (
            "⚠️ POTENTIAL FALL DETECTED!\n\n"
            "ICT304 Fall Detection System\n\n"
            f"Detection Time: {time_text}\n"
            "Detection Model: Random Forest V3\n"
            f"Fall Classification Score: {probability:.2%}\n\n"
            "Please check on the monitored person.\n\n"
            "This is an automated prototype alert."
        )

        print("[NOTIFIER] Potential fall detected.")

        if self.mode == "telegram":
            self.executor.submit(
                self._send_telegram,
                message,
                event_time.isoformat(),
                probability,
            )
        else:
            self._write_log(
                event_time.isoformat(),
                probability,
                "SIMULATED",
            )
            print("[NOTIFIER] Simulated notification.")

        return True

    def notify_shutdown(self):
        """Notify the caretaker when monitoring stops normally.

        This is a system-status message, independent of fall cooldown.
        close() waits for this message to finish sending.
        """
        event_time = datetime.now().astimezone()
        time_text = event_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        message = (
            "⚠️ FALL DETECTION MONITORING STOPPED\n\n"
            "ICT304 Fall Detection System\n\n"
            f"Shutdown Time: {time_text}\n\n"
            "The monitoring program has stopped. "
            "Fall detection and notifications are no longer active.\n\n"
            "Please check that monitoring is restarted if required.\n\n"
            "This is an automated prototype status notification."
        )
        print("[NOTIFIER] Monitoring stopped.")

        if self.mode == "telegram":
            self.executor.submit(
                self._send_shutdown_telegram,
                message,
            )
        else:
            print("[NOTIFIER] Simulated shutdown notification.")

        return True

    def _write_log(self, event_time, probability, status):
        self.writer.writerow([
            event_time,
            "POTENTIAL_FALL",
            f"{probability:.4f}",
            self.mode,
            status,
        ])
        self.log_file.flush()

    def _send_telegram(
        self, message, event_time, probability,
    ):
        url = (
            f"https://api.telegram.org/"
            f"bot{self.bot_token}/sendMessage"
        )

        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": message,
        }).encode("utf-8")

        status = "FAILED"

        try:
            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
            )

            with urllib.request.urlopen(
                request, timeout=10
            ) as response:
                result = json.load(response)

            if result.get("ok"):
                status = "SENT"
                print("[NOTIFIER] Telegram message sent.")
            else:
                print("[NOTIFIER] Telegram rejected message.")

        except Exception as error:
            print(
                "[NOTIFIER] Sending failed:",
                type(error).__name__,
            )

        finally:
            self._write_log(
                event_time, probability, status
            )

    def _send_shutdown_telegram(self, message):
        """Send shutdown notification without CSV logging."""

        url = (
            f"https://api.telegram.org/"
            f"bot{self.bot_token}/sendMessage"
        )

        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": message,
        }).encode("utf-8")

        try:
            request = urllib.request.Request(
                url,
                data=data,
                method="POST",
            )

            with urllib.request.urlopen(
                request, timeout=10
            ) as response:
                result = json.load(response)

            if result.get("ok"):
                print("[NOTIFIER] Shutdown message sent.")
            else:
                print("[NOTIFIER] Telegram rejected shutdown message.")

        except Exception as error:
            print(
                "[NOTIFIER] Shutdown sending failed:",
                type(error).__name__,
            )

    def close(self):
        self.executor.shutdown(wait=True)
        self.log_file.close()
