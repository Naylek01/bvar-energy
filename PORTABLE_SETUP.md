# Portable Windows setup

Put these files at the repository root while preserving the `.vscode/` and `scripts/` folders.

Expected layout:

```text
bvar-energy/
├─ setup.ps1
├─ requirements.txt
├─ .vscode/
│  ├─ settings.json
│  └─ extensions.json
├─ scripts/
│  └─ configure_notebook_kernels.py
├─ notebooks/
└─ src/
```

On a new Windows PC:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
```

The script:

1. creates `.venv` with Python 3.11 if needed;
2. installs `requirements.txt`;
3. registers the Jupyter kernel `Python (bvar-energy)`;
4. stamps the notebooks with that kernelspec;
5. verifies the core scientific stack.

After the first setup, if VS Code was already open, run:

```text
Ctrl+Shift+P → Developer: Reload Window
```

Then a notebook should use `Python (bvar-energy)`. You can verify with:

```python
import sys
print(sys.executable)
```

It should end in:

```text
bvar-energy\.venv\Scripts\python.exe
```

Do not commit `.venv/` to Git.
