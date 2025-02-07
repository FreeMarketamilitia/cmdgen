#!/usr/bin/env python3
"""
Terminal Assistant (cmdgen) with Google Gemini API

This module accepts natural language queries from the user and uses the Google Gemini API
to generate a terminal command. It previews the generated command and, upon confirmation,
executes it.

On the very first run (or when --setup is passed), it prompts you for your Gemini API key
and lets you choose a Gemini model from a predefined list. The API key and model selection
are saved in a configuration file for later runs.
"""

import argparse
import platform
import subprocess
import logging
import os
import sys
import json
import time
from datetime import datetime

# Try to import rich for a colorful UI.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, IntPrompt
except ImportError:
    sys.exit("Please install the 'rich' package (pip install rich) to run this script with a nice UI.")

# Optionally try to import pyperclip for clipboard support.
try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

# Try to import the Google Gemini API SDK.
try:
    import google.generativeai as genai
    from google.genai import types
except ImportError:
    sys.exit("Please install the 'google-generativeai' package (pip install google-generativeai) to run this script.")

# Define version and file paths.
__version__ = "2.0.0"
DEFAULT_HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".cmdgen_history.txt")
DEFAULT_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".cmdgen_config.json")

# Configure logging.
logging.basicConfig(
    filename='cmdgen.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Create a global Console instance for rich output.
console = Console()

def persist_env_vars(config: dict) -> None:
    """
    Persist environment variables permanently.
    On Unix-like systems, append export statements to the shell startup file.
    On Windows, use the 'setx' command to set user-level environment variables.
    """
    import platform
    if platform.system() == "Windows":
        mapping = {
            "gemini_api_key": "GEMINI_API_KEY",
            "model_id": "MODEL_ID",
            "USER": "USER",
            "HOME": "HOME",
            "SHELL": "SHELL",
            "PATH": "PATH",
            "OS": "OS",
            "CWD": "CWD",
        }
        for key, env_name in mapping.items():
            value = config.get(key, "")
            os.system(f'setx {env_name} "{value}"')
        console.print("[green]Environment variables set permanently using setx.[/green]")
    else:
        shell = config.get("SHELL", "")
        if "bash" in shell:
            rc_file = os.path.join(os.path.expanduser("~"), ".bashrc")
        elif "zsh" in shell:
            rc_file = os.path.join(os.path.expanduser("~"), ".zshrc")
        elif "fish" in shell:
            rc_file = os.path.join(os.path.expanduser("~"), ".config", "fish", "config.fish")
        else:
            rc_file = os.path.join(os.path.expanduser("~"), ".profile")
    
        env_block = "\n# Added by CommandGen Setup\n"
        env_block += f'export GEMINI_API_KEY="{config.get("gemini_api_key")}"\n'
        env_block += f'export MODEL_ID="{config.get("model_id")}"\n'
        env_block += f'export USER="{config.get("USER")}"\n'
        env_block += f'export HOME="{config.get("HOME")}"\n'
        env_block += f'export SHELL="{config.get("SHELL")}"\n'
        env_block += f'export PATH="{config.get("PATH")}"\n'
        env_block += f'export OS="{config.get("OS")}"\n'
        env_block += f'export CWD="{config.get("CWD")}"\n'
    
        try:
            with open(rc_file, "a") as f:
                f.write(env_block)
            console.print(f"[green]Environment variables appended to {rc_file}.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to write to {rc_file}: {e}[/red]")

def choose_model() -> str:
    """
    Allow the user to select a Gemini model from a predefined list.
    """
    models = [
        "gemini-2.0-flash",
        "gemini-2.0-pro"
    ]
    table = Table(title="Available Gemini Models")
    table.add_column("No.", justify="right", style="cyan", no_wrap=True)
    table.add_column("Model ID", style="magenta")
    for idx, model in enumerate(models, start=1):
        table.add_row(str(idx), model)
    console.print(table)
    while True:
        index = IntPrompt.ask("Select a model by number", default=1)
        if 1 <= index <= len(models):
            selected = models[index - 1]
            console.print(f"[green]✔ Selected Model: [bold]{selected}[/bold][/green]")
            return selected
        else:
            console.print(f"[yellow]⚠ Please enter a number between 1 and {len(models)}.[/yellow]")

def first_run_setup(config_file: str, gemini_api_key: str) -> str:
    """
    Run interactive setup: prompt for Gemini API key, select a model,
    capture system environment variables, save settings in a config file,
    and persist environment variables.
    """
    console.print("[bold green]Welcome to CommandGen Setup![/bold green] This is your first run.\n")
    if not gemini_api_key:
        gemini_api_key = Prompt.ask("Enter your Gemini API key", default="")
    os.environ["GEMINI_API_KEY"] = gemini_api_key

    selected_model = choose_model()

    config = {
        "model_id": selected_model,
        "gemini_api_key": gemini_api_key,
        "USER": os.getenv("USER") or os.getenv("USERNAME"),
        "HOME": os.getenv("HOME") or os.getenv("HOMEPATH"),
        "SHELL": os.getenv("SHELL") if platform.system() != "Windows" else "cmd.exe",
        "PATH": os.getenv("PATH"),
        "OS": platform.system(),
        "CWD": os.getcwd(),
    }

    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        console.print("\n[bold green]Configuration saved successfully![/bold green]")
    except Exception as e:
        logging.error("Error saving config: %s", e)
        console.print("[red]Warning: Could not save configuration.[/red]")

    persist_env_vars(config)
    console.print(f"\n[bold green]Setup complete.[/bold green] Selected model: {selected_model}.\n")
    return selected_model

def load_config(config_file: str, gemini_api_key: str) -> str:
    """
    Load configuration from file. If not found or if there is an error,
    run first_run_setup to prompt for API key and model selection.
    """
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                model_id = config.get("model_id", None)
                if model_id:
                    os.environ["GEMINI_API_KEY"] = config.get("gemini_api_key", "")
                    os.environ["MODEL_ID"] = model_id
                    return model_id
        except Exception as e:
            logging.error("Error reading config file: %s", e)
    return first_run_setup(config_file, gemini_api_key)

def generate_command(query: str, os_name: str, shell: str = None) -> str:
    """
    Generate a terminal command based on the natural language query using the Gemini API.
    This function uses the GenerativeModel class and sets its generation_config.
    """
    # Gather environment variables.
    env_vars = {
        "CWD": os.getcwd(),
        "OS": os_name,
        "SHELL": os.getenv("SHELL") if os.name != "nt" else "cmd.exe",
        "USER": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "HOME": os.getenv("HOME") or os.getenv("HOMEPATH") or "unknown",
        "PATH": os.getenv("PATH") or "unknown",
    }
    
    env_vars_str = (
        f"USER: {env_vars['USER']}\n"
        f"HOME: {env_vars['HOME']}\n"
        f"PATH: {env_vars['PATH']}\n"
    )
    
    # Compose the prompt. This instructs the model to output only the terminal command.
    prompt = (
        "You are a command-line assistant that generates accurate terminal commands based on natural language instructions. "
        "Your response must contain ONLY the terminal command (no extra commentary, formatting, or explanations).\n\n"
        f"Target OS: {env_vars['OS']}\n"
        f"Target Shell: {env_vars['SHELL']}\n"
        f"Current Working Directory: {env_vars['CWD']}\n"
        "\nDetected Environment Variables:\n"
        f"{env_vars_str}\n"
        "Guidelines for output:\n"
        "- Output ONLY the terminal command with no additional text.\n"
        "- Use standard shell utilities (e.g., ls, rm, grep, find, awk).\n"
        "- Default to Bash syntax unless otherwise specified.\n"
        "- Avoid destructive commands unless explicitly requested.\n"
        "- Use absolute paths when needed.\n"
        "\nExamples:\n"
        "Example 1:\n"
        "  Input: 'List all files in the current directory.'\n"
        "  Output: `ls`\n\n"
        "Example 2:\n"
        "  Input: 'Delete the file named \"old.log\" in the current directory.'\n"
        "  Output: `rm old.log`\n\n"
        "Example 3:\n"
        "  Input: 'Find all .txt files in /home/user/documents.'\n"
        "  Output: `find /home/user/documents -type f -name \"*.txt\"`\n\n"
        f"Instruction: {query}\n"
        "Command:"
    )
    
    logging.debug(f"Prompt sent to Gemini API:\n{prompt}")
    
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    model_id = os.environ.get("MODEL_ID", "gemini-2.0-flash")
    
    try:
        # Configure the API key using the environment variable.
        genai.configure(api_key=gemini_api_key)
        # Create a model instance using the GenerativeModel class.
        model = genai.GenerativeModel(model_name=model_id)
        # Set the generation configuration on the model instance.
        model.generation_config = types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.0
        )
        # Generate content using the prompt.
        response = model.generate_content(prompt)
        command = response.text.strip()
        logging.debug(f"Extracted command: {command}")
        return command
    except Exception as e:
        logging.error("Error during command generation: %s", e)
        return ""

def confirm_execution(command: str) -> bool:
    """
    Display the generated command and ask the user to confirm before executing it.
    """
    console.print("\n[bold blue]Generated command:[/bold blue]\n------------------")
    console.print(command, style="bold")
    console.print("------------------")
    if any(word in command.lower() for word in ["rm", "del", "delete"]):
        console.print("[red]WARNING:[/red] This command appears to be destructive.")
    choice = Prompt.ask("Do you want to execute this command? [y/N]", default="N")
    return choice.strip().lower() == 'y'

def execute_command(command: str, target_shell: str = None) -> None:
    """
    Execute the command using the appropriate shell.
    """
    current_os = platform.system()
    try:
        if current_os == 'Windows':
            subprocess.run(command, shell=True)
        else:
            executable = target_shell if target_shell else '/bin/bash'
            subprocess.run(command, shell=True, executable=executable)
        logging.info("Executed command: %s", command)
    except Exception as e:
        logging.error("Error executing command: %s", e)
        console.print(f"[red]Execution failed:[/red] {e}")

def append_history(command: str, history_file: str):
    """
    Append the generated command with a timestamp to the history file.
    """
    try:
        with open(history_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: {command}\n")
    except Exception as e:
        logging.error("Error appending history: %s", e)

def show_history(history_file: str):
    """
    Print previously generated commands from the history file.
    """
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            console.print(f.read(), style="bold green")
    else:
        console.print("No history available.", style="red")

def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute terminal commands from natural language instructions using the Google Gemini API."
    )
    parser.add_argument('-query', '-q', nargs='*', help='Your natural language instruction for command generation')
    parser.add_argument('--run', '-r', action='store_true',
                        help='Executes the generated command immediately (with confirmation by default).')
    parser.add_argument('--shell', '-s', type=str,
                        help='Specify which shell to generate commands for (bash, zsh, fish).')
    parser.add_argument('--history', '-H', action='store_true',
                        help='Show previously generated commands.')
    parser.add_argument('--copy', '-c', action='store_true',
                        help='Copy the generated command to clipboard.')
    parser.add_argument('--save', '-S', type=str,
                        help='Save the generated command output to a file.')
    parser.add_argument('--config', type=str,
                        help='Load a custom config file for cmdgen.')
    parser.add_argument('--no-confirm', '-n', action='store_true',
                        help='Skip confirmation when using --run.')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output the generated command in JSON format.')
    parser.add_argument('--setup', action='store_true',
                        help='Run interactive setup to input all required variables and select your Gemini model.')
    parser.add_argument('--version', '-v', action='version', version=f'cmdgen {__version__}')
    args = parser.parse_args()

    history_file = DEFAULT_HISTORY_FILE
    if args.history:
        show_history(history_file)
        sys.exit(0)

    config_file = args.config if args.config else DEFAULT_CONFIG_FILE
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
   
    if args.setup:
        selected_model = first_run_setup(config_file, gemini_api_key)
        os.environ["MODEL_ID"] = selected_model
        console.print("[bold green]Setup complete.[/bold green]")
        sys.exit(0)
    else:
        selected_model = load_config(config_file, gemini_api_key)
        os.environ["MODEL_ID"] = selected_model

    if not args.query:
        parser.print_help()
        sys.exit(0)
    user_query = " ".join(args.query)
    logging.info("Received query: %s", user_query)

    os_name = platform.system()
    command = generate_command(user_query, os_name, shell=args.shell)
    if not command:
        console.print("[red]Failed to generate a command. Please try again.[/red]")
        sys.exit(1)

    append_history(command, history_file)

    if args.copy:
        if HAS_CLIPBOARD:
            pyperclip.copy(command)
            console.print("[bold green]Command copied to clipboard.[/bold green]")
        else:
            console.print("[red]pyperclip is not installed. Unable to copy to clipboard.[/red]")

    if args.save:
        try:
            with open(args.save, "w") as f:
                f.write(command)
            console.print(f"[bold green]Command saved to {args.save}.[/bold green]")
        except Exception as e:
            logging.error("Error saving command to file: %s", e)
            console.print(f"[red]Error saving command to file:[/red] {e}")

    if args.json:
        console.print_json(json.dumps({"command": command}))
        sys.exit(0)

    if args.run:
        if args.no_confirm or confirm_execution(command):
            logging.info("User confirmed execution of command: %s", command)
            execute_command(command, target_shell=args.shell)
        else:
            console.print("[yellow]Command execution cancelled.[/yellow]")
            logging.info("User cancelled the command execution.")
    else:
        console.print("[bold blue]Generated command:[/bold blue]\n", command)

if __name__ == "__main__":
    main()
