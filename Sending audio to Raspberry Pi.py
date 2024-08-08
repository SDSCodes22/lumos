import asyncio
import subprocess

class AudioSender:
    def __init__(self, raspberry_pi_ip, port):
        self.raspberry_pi_ip = raspberry_pi_ip
        self.port = port

    async def run_command(self, command):
        process = await asyncio.create_subprocess_shell(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Command failed with error: {stderr.decode('utf-8')}")
        return stdout.decode('utf-8')

    async def bluetooth_pairing(self, device_name):
        # macOS does not use bluetoothctl; use AppleScript for Bluetooth pairing
        apple_script = f"""
        tell application "System Events"
            tell process "SystemUIServer"
                click menu bar item "Bluetooth" of menu bar 1
                delay 1
                click menu item "{device_name}" of menu 1 of menu bar item "Bluetooth" of menu bar 1
                delay 1
                click button "Connect" of window "{device_name}" of process "SystemUIServer"
            end tell
        end tell
        """
        await self.run_command(f"osascript -e '{apple_script}'")

    async def check_wifi_connection(self):
        try:
            await self.run_command("ping -c 1 8.8.8.8")  # Ping a public IP to check connectivity
            print("Wi-Fi connection is stable.")
        except Exception as e:
            raise Exception("No stable Wi-Fi connection detected. Please check your network.")

    async def stream_audio(self, server_input_file):
        server_command = f"ffmpeg -re -i {server_input_file} -f rtp rtp://{self.raspberry_pi_ip}:{self.port}"
        client_command = f"ffmpeg -i rtp://@:{self.port} -acodec copy output.wav"

        # Start server (on the transmitting device)
        server_process = await asyncio.create_subprocess_shell(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Start client (on the Raspberry Pi)
        client_process = await asyncio.create_subprocess_shell(client_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        await server_process.communicate()
        await client_process.communicate()

if __name__ == "__main__":
    device_name = "Your Bluetooth Device Name"  # Replace with your Bluetooth device's name
    server_input_file = "path/to/audio/file.mp3"  # Replace with the path to your input audio file
    raspberry_pi_ip = "192.168.1.100"  # Replace with your Raspberry Pi's IP address
    audio_stream_port = 5004  # Unique port for the audio stream

    async def main():
        sender = AudioSender(raspberry_pi_ip, audio_stream_port)
        
        try:
            # Step 1: Pair Bluetooth Device
            await sender.bluetooth_pairing(device_name)
            
            # Step 2: Check Wi-Fi Connection
            await sender.check_wifi_connection()
            
            # Step 3: Start Wi-Fi Audio Streaming
            await sender.stream_audio(server_input_file)
        
        except Exception as e:
            print(f"An error occurred: {e}")

    # Run the main coroutine
    asyncio.run(main())
