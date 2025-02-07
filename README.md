# 🚀 CommandGen: Your AI-Powered Terminal Buddy

Ever wished you had a **tech wizard** in your terminal? 🧙‍♂️✨  
Tired of Googling how to use `grep` or `find`? 🥴  
Fear not, **CommandGen** is here! 🎉  

CommandGen is your **AI-powered terminal assistant** that translates **plain English** into shell commands using the **Google Gemini API**. Just **describe** what you need, and CommandGen will generate and (optionally) run the command for you! 🖥️⚡

## 🔥 Features

✅ **Natural Language Magic** – No more `awk`-ward moments, just type what you want!  
✅ **Preview Before You Wreck** – Avoid accidental `rm -rf /` disasters. 😬  
✅ **Custom Shell Support** – Bash? Zsh? Fish? We got you. 🐠  
✅ **Clipboard & Save Options** – Because typing is hard. 🤷  
✅ **Command History** – Never forget that genius one-liner again. 📝  
✅ **Configurable & Secure** – Your API key stays safe. 🔐  

---

## ⚙️ CLI Options

```bash
  -q, --query       💬 Your natural language command request.
  --run             🚀 Executes the command immediately (with confirmation).
  --shell           🐚 Specify a shell (bash, zsh, fish).
  --history         📜 View past generated commands.
  --copy            📋 Copy command to clipboard.
  --save FILE       💾 Save the generated command to a file.
  --json            📦 Output the generated command in JSON format.
  --setup           🔧 Run interactive setup for API key and model selection.
  --version         🏷️ Show the current version of CommandGen.
```
---

## 📦 Installation

```bash
git clone https://github.com/freemarketamilitia/cmdgen.git
cd commandgen
pip install -e .
```

# 🚀 cmdgen: Your AI-Powered CLI Assistant

### **Run Terminal Commands Like a Wizard! 🧙‍♂️✨**

`cmdgen` is a command-line tool powered by the Google Gemini API that converts natural language into terminal commands. No more Googling weird syntax—just ask, confirm, and run.

---

## **🔧 Setting Up `cmdgen` as a Global Command**
Want to run `cmdgen` from anywhere instead of typing `python cli.py`? Here’s how! 🛠️

---

### **🛠️ 1️⃣ Rename `cli.py` to `cmdgen` (Optional)**
Make your script executable:
```bash
mv cli.py cmdgen
chmod +x cmdgen  # Make it executable
```

---

### **🔗 2️⃣ Move It to a Directory in Your System PATH**
Move the script to a directory already in `$PATH`, such as `/usr/local/bin` on macOS/Linux or the user scripts directory on Windows.

#### **📌 On macOS/Linux**
```bash
mv cmdgen /usr/local/bin/
```
Now you can run:
```bash
cmdgen
```
from anywhere. 🎉

---

#### **📌 On Windows**
1. **Find your Python scripts directory:**
   ```powershell
   python -c "import sys; print(sys.executable)"
   ```
2. **Copy `cmdgen` to this folder**
3. **Add it to the PATH variable**:
   - Open **Control Panel → System → Advanced System Settings**
   - Click **Environment Variables**
   - Find **Path** under System Variables → Click **Edit**
   - Click **New** and paste the path of the directory containing `cmdgen`
   - Click **OK**, close everything, and restart your terminal.

Now you can run:
```powershell
cmdgen
```
from anywhere on Windows! 🎉

---

## **📢 Test It Out!**
After adding `cmdgen` to your path, test it by running:
```bash
cmdgen --help
```

If it works, you’re all set! 🚀

---

### **💡 Bonus: Add Autocomplete**
For an even smoother experience, add shell autocomplete:

#### **Bash/Zsh (Mac/Linux)**
```bash
echo 'complete -W "$(cmdgen --help)" cmdgen' >> ~/.bashrc
source ~/.bashrc
```

#### **PowerShell (Windows)**
```powershell
Set-PSReadLineOption -PredictionSource History
```

Now enjoy AI-powered command generation with `cmdgen`! 🚀🔥

---

## 🛠️ How to Use

Just ask CommandGen **like a normal human:**
```bash
commandgen -q "list all files in the current directory"
```
➡️ This outputs:  
```bash
ls
```
Simple, right? **No more Stack Overflow rabbit holes!** 🕳️🐇

---



---

## 🏆 Why Use CommandGen?

🚀 **Boost Productivity** – No more wasting time figuring out syntax.  
😎 **Look Like a Pro** – Impress your coworkers with obscure one-liners.  
🛡️ **Safer Execution** – Always review before running.  

---

## 🤖 Under the Hood

CommandGen uses **Google Gemini API** to convert plain English into **accurate shell commands**.  
The generated command is **previewed before execution** for safety.  

---

## 🤝 Contributing

Want to improve CommandGen? Found a bug? PRs welcome! 🚀  
Clone the repo, make changes, and submit a pull request.  

```bash
git clone https://github.com/yourusername/commandgen.git
cd commandgen
```

---

## 📝 License

MIT License. Use it, improve it, but don’t blame us if you `rm -rf /` your system. 😆  
