import asyncio
import logging

logger = logging.getLogger(__name__)

class TerminalTool:
    def __init__(self) -> None:
        pass

    async def run_command(self, command: str, timeout: int = 10) -> str:
        """
        Run a shell command with a timeout.
        Restrictions: No interactive commands, no sudo (unless user allows, but we block mostly).
        """
        # specialized security checks can be added here
        forbidden = ["sudo", "rm -rf /", ":(){ :|:& };:"]
        if any(f in command for f in forbidden):
            return "Error: Command blocked for security."

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode().strip()
                error = stderr.decode().strip()
                result = output + ("\nstderr: " + error if error else "")
                return result or "Success (No Output)"
            except TimeoutError:
                proc.kill()
                return "Error: Command timed out."

        except Exception as e:
            return f"Error: {e}"
