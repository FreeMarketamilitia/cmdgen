
# 🚀 CommandGen: Your AI-Powered Terminal Buddy

Ever wished you had a **tech wizard** in your terminal? 🧙‍♂️✨  
Tired of Googling how to use `grep` or `find`? 🥴  
Fear not, **CommandGen** is here! 🎉  

CommandGen is your **AI-powered terminal assistant** that translates **plain English** into shell commands using the **Google Gemini API**. Just **describe** what you need, and CommandGen will generate and (optionally) run the command for you! 🖥️⚡

## 🔑 **How to Obtain Your API Key from Google AI Studio**

To use **CommandGen** with the **Google Gemini API**, follow these easy steps:

| Step | Action |
|------|--------|
| 1️⃣ | Visit [Google AI Studio](https://aistudio.google.com) and get ready for some magic! |
| 2️⃣ | Sign in (or create a Google Cloud account if you don’t have one). |
| 3️⃣ | Create a new project: **Create New Project** ➡️ Name it something cool (e.g., "CommandGenGenius"). |
| 4️⃣ | Enable the API: **APIs & Services** ➡️ Search for **Google Gemini API** ➡️ **Enable** it! |
| 5️⃣ | Generate API Key: **Credentials** ➡️ Click **Create Credentials** ➡️ Select **API Key** 🎟️ |
| 6️⃣ | Copy and save your key securely! |

Now you're ready to set up **CommandGen** with your shiny new API key! 🎉


## 📦 Installation

To install `CommandGen`, run the following command:

```
pip install cmdgen
sudo apt-get install xclip
#pip install grpcio==1.60.1 optional 
#but if until google updates their package you'll recieve a text warning

```

This will install `cmdgen` and automatically add it to your system's `PATH`.

---

### 🔧 Setup Option
- `--setup`: Initiates the first-time setup for your API key and model selection. After running this command, the model ID will be saved, and setup will be complete. Example:

```
cmdgen --setup
```
---


## 🔥 Features

| Feature                           | Purpose                                               |
|-----------------------------------|------------------------------------------------------------|
| ✅ **Natural Language Magic**      | No more `awk`-ward moments, just type what you want!        |
| ✅ **Preview Before You Wreck**    | Avoid accidental `rm -rf /` disasters. 😬                   |
| ✅ **Custom Shell Support**        | Bash? Zsh? Fish? We got you. 🐠                             |
| ✅ **Clipboard & Save Options**    | Because typing is hard. 🤷                                 |
| ✅ **Command History**             | Never forget that genius one-liner again. 📝                |
| ✅ **Configurable & Secure**       | Your API key stays safe. 🔐                                |
## ⚙️ CLI Options

| **Option**           | **Description**                                           |
|----------------------|-----------------------------------------------------------|
| `-q, --query`        | 💬 Your natural language command request.                |
| `--run`              | 🚀 Executes the command immediately (with confirmation). |
| `--shell`            | 🐚 Specify a shell (bash, zsh, fish).                    |
| `--history`          | 📜 View past generated commands.                         |
| `--copy`             | 📋 Copy command to clipboard.                            |
| `--save FILE`        | 💾 Save the generated command to a file.                  |
| `--json`             | 📦 Output the generated command in JSON format.           |
| `--setup`            | 🔧 Run interactive setup for API key and model selection. |
| `--version`          | 🏷️ Show the current version of cmdgen.                   |


Here's the updated table format for the example runs, similar to the previous table for CLI options:

### Example Runs

| **Command** | **Description** | **Output** |
|-------------|-----------------|-----------|
| `cmdgen --query "list all files"` | Generate a command from a natural language query. | `Generated command: ls -al` |
| `cmdgen --query "list all files" --run` | Generate and run a command immediately with user confirmation. | `Generated command: ls -al` <br> `Do you want to execute this command? (y/n): y` <br> `Executing command: ls -al` |
| `cmdgen --query "list all files" --run --no-confirm` | Run the command immediately without confirmation. | `Generated command: ls -al` <br> `Executing command: ls -al` |
| `cmdgen --query "list all files" --save output.sh` | Save the generated command to a file. | `Command saved to output.sh.` |
| `cmdgen --query "list all files" --copy` | Copy the generated command to clipboard (requires pyperclip). | `Command copied to clipboard.` |
| `cmdgen --history` | View the command history. | `Command history:` <br> `1. ls -al` <br> `2. pwd` <br> `3. mkdir new_folder` |
| `cmdgen --query "list all files" --json` | Output the generated command in JSON format. | `{"command": "ls -al"}` |
| `cmdgen --query "list all files" --shell zsh` | Generate a command for a specific shell (e.g., zsh). | `Generated command for zsh: ls -al` |
| `cmdgen --setup` | Run the interactive setup for first-time users (API key & model selection). | `Running interactive setup...` <br> `Enter your API key: ***************` <br> `Select a model: Gemini-1` <br> `Setup complete.` |
| `cmdgen --version` | Show the current version of cmdgen. | `cmdgen version: 1.0.0` |
| `cmdgen --query "list all files" --json` | Output the generated command in JSON format. | `{"command": "ls -al"}` |

---

# 🚀 cmdgen: Your AI-Powered CLI Assistant

### **Run Terminal Commands Like a Wizard! 🧙‍♂️✨**

`cmdgen` is a command-line tool powered by the Google Gemini API that converts natural language into terminal commands. No more Googling weird syntax—just ask, confirm, and run.

---

## **📢 Test It Out!**

Once installed, you can test `cmdgen` by running:

```bash
cmdgen --help
```

If it works, you’re all set! 🚀

---

## 📝 License

MIT License. Use it, improve it, but don’t blame me if you `rm -rf /` your system. 😆
