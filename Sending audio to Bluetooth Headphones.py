import asyncio
import subprocess

class AudioSender:
    def __init__(self, bluetooth_device_name):
        self.bluetooth_device_name = bluetooth_device_name

    async def run_command(self, command):
        process = await asyncio.create_subprocess_shell(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"Command failed with error: {stderr.decode('utf-8')}")
        return stdout.decode('utf-8')

    async def bluetooth_pairing(self):
        # macOS Bluetooth pairing with AppleScript
        apple_script = f"""
        tell application "System Events"
            tell process "SystemUIServer"
                click menu bar item "Bluetooth" of menu bar 1
                delay 1
                click menu item "{self.bluetooth_device_name}" of menu 1 of menu bar item "Bluetooth" of menu bar 1
                delay 1
                click button "Connect" of window "{self.bluetooth_device_name}" of process "SystemUIServer"
            end tell
        end tell
        """
        await self.run_command(f"osascript -e '{apple_script}'")

    async def stream_live_audio_to_bluetooth(self):
        # Replace "0" with the correct input device ID obtained from the list_devices command
        ffmpeg_command = f"ffmpeg -f avfoundation -i :0 -f alsa default"
        await self.run_command(ffmpeg_command)

if __name__ == "__main__":
    device_name = "Your Bluetooth Headphones"  # Replace with your Bluetooth device's name

    async def main():
        sender = AudioSender(device_name)
        
        try:
            # Step 1: Pair and Connect Bluetooth Headphones
            await sender.bluetooth_pairing()
            
            # Step 2: Stream Live Audio to Bluetooth Headphones
            await sender.stream_live_audio_to_bluetooth()
        
        except Exception as e:
            print(f"An error occurred: {e}")

    # Run the main coroutine
    asyncio.run(main())
