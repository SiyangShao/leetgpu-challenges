(function () {
  "use strict";

  const body = document.body;
  const slug = body.dataset.slug;
  const codeStorageKey = `leetgpu_code_${slug}`;
  const historyStorageKey = `leetgpu_history_${slug}`;
  const vimPreferenceKey = "leetgpu_vim_mode";

  const starterCode = JSON.parse(document.getElementById("starter-code-json").textContent);
  const editorHost = document.getElementById("editor");
  const vimToggle = document.getElementById("vim-toggle");
  const vimStatus = document.getElementById("vim-status");
  const saveStatus = document.getElementById("save-status");
  const consoleOutput = document.getElementById("console");
  const runButton = document.getElementById("btn-run");
  const submitButton = document.getElementById("btn-submit");
  const historyButton = document.getElementById("btn-history");
  const resetButton = document.getElementById("btn-reset");

  function storageGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (_error) {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
      return true;
    } catch (_error) {
      return false;
    }
  }

  function storageRemove(key) {
    try {
      window.localStorage.removeItem(key);
    } catch (_error) {
      // The editor remains usable when browser storage is unavailable.
    }
  }

  const savedCode = storageGet(codeStorageKey);
  const initialCode = savedCode === null ? starterCode : savedCode;
  let vimEnabled = storageGet(vimPreferenceKey) === "true";
  let cmEditor = null;

  function insertIndent(cm) {
    if (cm.getOption("keyMap") === "vim" && (!cm.state.vim || !cm.state.vim.insertMode)) {
      return window.CodeMirror.Pass;
    }
    if (cm.somethingSelected()) {
      cm.indentSelection("add");
    } else {
      const spaces = " ".repeat(cm.getOption("indentUnit"));
      cm.replaceSelection(spaces, "end", "+input");
    }
    return undefined;
  }

  function removeIndent(cm) {
    if (cm.getOption("keyMap") === "vim" && (!cm.state.vim || !cm.state.vim.insertMode)) {
      return window.CodeMirror.Pass;
    }
    if (!cm.somethingSelected()) {
      return window.CodeMirror.Pass;
    }
    cm.indentSelection("subtract");
    return undefined;
  }

  function createEditor() {
    if (typeof window.CodeMirror !== "function") {
      const fallback = document.createElement("textarea");
      fallback.className = "editor-fallback";
      fallback.value = initialCode;
      fallback.setAttribute("aria-label", "CUDA solution editor");
      fallback.spellcheck = false;
      editorHost.appendChild(fallback);
      fallback.addEventListener("input", saveCurrentCode);
      fallback.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
          event.preventDefault();
          runCode(event.shiftKey ? "submit" : "run");
        }
      });
      vimToggle.disabled = true;
      vimToggle.dataset.unavailable = "true";
      vimToggle.title = "Vim keybindings could not load";
      saveStatus.textContent = "Editor fallback active";
      return {
        getValue: () => fallback.value,
        setValue: (value) => {
          fallback.value = value;
          saveCurrentCode();
        },
        focus: () => fallback.focus(),
        refresh: () => {},
      };
    }

    const hasVimKeymap = Boolean(window.CodeMirror.keyMap && window.CodeMirror.keyMap.vim);
    if (!hasVimKeymap) {
      vimEnabled = false;
      vimToggle.disabled = true;
      vimToggle.dataset.unavailable = "true";
      vimToggle.title = "Vim keybindings could not load";
    }

    const editor = window.CodeMirror(editorHost, {
      value: initialCode,
      mode: "text/x-c++src",
      theme: "default",
      keyMap: vimEnabled && hasVimKeymap ? "vim" : "default",
      lineNumbers: true,
      matchBrackets: true,
      autoCloseBrackets: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      viewportMargin: 30,
      extraKeys: {
        Tab: insertIndent,
        "Shift-Tab": removeIndent,
        "Ctrl-Enter": () => runCode("run"),
        "Cmd-Enter": () => runCode("run"),
        "Ctrl-Shift-Enter": () => runCode("submit"),
        "Cmd-Shift-Enter": () => runCode("submit"),
      },
    });

    const input = editor.getInputField();
    input.setAttribute("aria-label", "CUDA solution editor");
    input.setAttribute("aria-multiline", "true");
    editor.on("change", saveCurrentCode);
    editor.on("vim-mode-change", function (...eventArgs) {
      const modeInfo = eventArgs.find(
        (value) => value && typeof value === "object" && typeof value.mode === "string",
      );
      updateVimStatus(modeInfo || { mode: "normal" });
    });

    return editor;
  }

  function saveCurrentCode() {
    if (!cmEditor) return;
    const saved = storageSet(codeStorageKey, cmEditor.getValue());
    saveStatus.textContent = saved ? "Saved locally" : "Local save unavailable";
  }

  function updateVimStatus(modeInfo) {
    if (!vimEnabled) {
      vimStatus.hidden = true;
      return;
    }

    const mode = String(modeInfo.mode || "normal").toUpperCase();
    const subMode = modeInfo.subMode ? ` · ${String(modeInfo.subMode).toUpperCase()}` : "";
    vimStatus.textContent = `-- ${mode}${subMode} --`;
    vimStatus.hidden = false;
  }

  function setVimEnabled(enabled, persist) {
    const hasVimKeymap = Boolean(
      cmEditor &&
      typeof cmEditor.setOption === "function" &&
      window.CodeMirror &&
      window.CodeMirror.keyMap &&
      window.CodeMirror.keyMap.vim,
    );

    const nextEnabled = Boolean(enabled && hasVimKeymap);
    if (
      vimEnabled &&
      !nextEnabled &&
      window.CodeMirror &&
      window.CodeMirror.Vim &&
      typeof window.CodeMirror.Vim.handleKey === "function" &&
      cmEditor.state &&
      cmEditor.state.vim
    ) {
      window.CodeMirror.Vim.handleKey(cmEditor, "<Esc>");
    }

    vimEnabled = nextEnabled;
    if (hasVimKeymap) {
      cmEditor.setOption("keyMap", vimEnabled ? "vim" : "default");
    }

    vimToggle.setAttribute("aria-pressed", String(vimEnabled));
    if (vimToggle.dataset.unavailable !== "true") {
      vimToggle.title = vimEnabled ? "Disable Vim keybindings" : "Enable Vim keybindings";
    }
    updateVimStatus({ mode: "normal" });

    if (persist) {
      storageSet(vimPreferenceKey, String(vimEnabled));
    }
    if (cmEditor) cmEditor.focus();
  }

  cmEditor = createEditor();
  window.getCode = () => cmEditor.getValue();
  window.leetgpuEditor = cmEditor;
  setVimEnabled(vimEnabled, false);

  vimToggle.addEventListener("click", () => {
    setVimEnabled(!vimEnabled, true);
  });

  runButton.addEventListener("click", () => runCode("run"));
  submitButton.addEventListener("click", () => runCode("submit"));
  resetButton.addEventListener("click", resetCode);

  // Render challenge math after both the document and KaTeX helpers are available.
  if (typeof window.renderMathInElement === "function") {
    window.renderMathInElement(document.getElementById("problem-content"), {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }

  // Resizable problem/editor and editor/result panes, with pointer and keyboard support.
  const workspace = document.getElementById("workspace");
  const descriptionPanel = document.getElementById("desc-panel");
  const rightPanel = document.querySelector(".right-panel");
  const consoleContainer = document.getElementById("console-container");
  const horizontalResizer = document.getElementById("h-resizer");
  const verticalResizer = document.getElementById("v-resizer");
  const mobileLayout = window.matchMedia("(max-width: 900px)");
  let activeResize = null;

  function setDescriptionWidth(width) {
    const rect = workspace.getBoundingClientRect();
    const min = Math.min(300, rect.width * 0.42);
    const max = Math.max(min, rect.width - 340);
    const next = Math.min(Math.max(width, min), max);
    descriptionPanel.style.flexBasis = `${next}px`;
    const percent = Math.round((next / rect.width) * 100);
    horizontalResizer.setAttribute("aria-valuemin", String(Math.round((min / rect.width) * 100)));
    horizontalResizer.setAttribute("aria-valuemax", String(Math.round((max / rect.width) * 100)));
    horizontalResizer.setAttribute("aria-valuenow", String(percent));
    cmEditor.refresh();
  }

  function setConsoleHeight(height) {
    const rect = rightPanel.getBoundingClientRect();
    const min = 100;
    const max = Math.max(min, rect.height - 220);
    const next = Math.min(Math.max(height, min), max);
    consoleContainer.style.flexBasis = `${next}px`;
    const percent = Math.round((next / rect.height) * 100);
    verticalResizer.setAttribute("aria-valuemin", String(Math.round((min / rect.height) * 100)));
    verticalResizer.setAttribute("aria-valuemax", String(Math.round((max / rect.height) * 100)));
    verticalResizer.setAttribute("aria-valuenow", String(percent));
    cmEditor.refresh();
  }

  horizontalResizer.addEventListener("pointerdown", (event) => {
    if (mobileLayout.matches) return;
    activeResize = "horizontal";
    horizontalResizer.setPointerCapture(event.pointerId);
    body.style.cursor = "col-resize";
    body.style.userSelect = "none";
  });

  verticalResizer.addEventListener("pointerdown", (event) => {
    if (mobileLayout.matches) return;
    activeResize = "vertical";
    verticalResizer.setPointerCapture(event.pointerId);
    body.style.cursor = "row-resize";
    body.style.userSelect = "none";
  });

  document.addEventListener("pointermove", (event) => {
    if (activeResize === "horizontal") {
      const rect = workspace.getBoundingClientRect();
      setDescriptionWidth(event.clientX - rect.left);
    } else if (activeResize === "vertical") {
      const rect = rightPanel.getBoundingClientRect();
      setConsoleHeight(rect.bottom - event.clientY);
    }
  });

  document.addEventListener("pointerup", () => {
    activeResize = null;
    body.style.cursor = "";
    body.style.userSelect = "";
  });

  horizontalResizer.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const step = event.shiftKey ? 48 : 16;
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    setDescriptionWidth(descriptionPanel.getBoundingClientRect().width + direction * step);
  });

  verticalResizer.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    const step = event.shiftKey ? 48 : 16;
    const direction = event.key === "ArrowUp" ? 1 : -1;
    setConsoleHeight(consoleContainer.getBoundingClientRect().height + direction * step);
  });

  function updateSeparatorRanges() {
    const workspaceRect = workspace.getBoundingClientRect();
    const rightRect = rightPanel.getBoundingClientRect();
    if (workspaceRect.width > 0) {
      const minWidth = Math.min(300, workspaceRect.width * 0.42);
      const maxWidth = Math.max(minWidth, workspaceRect.width - 340);
      horizontalResizer.setAttribute("aria-valuemin", String(Math.round((minWidth / workspaceRect.width) * 100)));
      horizontalResizer.setAttribute("aria-valuemax", String(Math.round((maxWidth / workspaceRect.width) * 100)));
      horizontalResizer.setAttribute(
        "aria-valuenow",
        String(Math.round((descriptionPanel.getBoundingClientRect().width / workspaceRect.width) * 100)),
      );
    }
    if (rightRect.height > 0) {
      const minHeight = 100;
      const maxHeight = Math.max(minHeight, rightRect.height - 220);
      verticalResizer.setAttribute("aria-valuemin", String(Math.round((minHeight / rightRect.height) * 100)));
      verticalResizer.setAttribute("aria-valuemax", String(Math.round((maxHeight / rightRect.height) * 100)));
      verticalResizer.setAttribute(
        "aria-valuenow",
        String(Math.round((consoleContainer.getBoundingClientRect().height / rightRect.height) * 100)),
      );
    }
  }

  function syncResponsiveLayout() {
    if (mobileLayout.matches) {
      descriptionPanel.style.flexBasis = "auto";
      consoleContainer.style.flexBasis = "250px";
    } else {
      descriptionPanel.style.flexBasis = "";
      consoleContainer.style.flexBasis = "";
    }
    window.requestAnimationFrame(() => {
      updateSeparatorRanges();
      cmEditor.refresh();
    });
  }

  if (typeof mobileLayout.addEventListener === "function") {
    mobileLayout.addEventListener("change", syncResponsiveLayout);
  } else {
    mobileLayout.addListener(syncResponsiveLayout);
  }

  if (typeof window.ResizeObserver === "function") {
    let refreshFrame = 0;
    const editorObserver = new window.ResizeObserver(() => {
      window.cancelAnimationFrame(refreshFrame);
      refreshFrame = window.requestAnimationFrame(() => cmEditor.refresh());
    });
    editorObserver.observe(document.getElementById("editor-container"));
  }
  window.requestAnimationFrame(updateSeparatorRanges);

  // Reusable confirmation dialog with focus restoration and no accumulating listeners.
  const modalOverlay = document.getElementById("modal-overlay");
  const modalTitle = document.getElementById("modal-title");
  const modalMessage = document.getElementById("modal-msg");
  const modalOk = document.getElementById("modal-ok");
  const modalCancel = document.getElementById("modal-cancel");
  let modalResolve = null;
  let modalPreviousFocus = null;

  function closeModal(result) {
    if (!modalResolve) return;
    const resolve = modalResolve;
    modalResolve = null;
    modalOverlay.classList.remove("active");
    modalOverlay.setAttribute("aria-hidden", "true");
    modalOverlay.inert = true;
    resolve(result);
    const focusTarget = modalPreviousFocus;
    window.requestAnimationFrame(() => {
      if (focusTarget && typeof focusTarget.focus === "function" && !focusTarget.disabled) {
        focusTarget.focus();
      } else if (cmEditor) {
        cmEditor.focus();
      }
    });
  }

  function showModal(message, options) {
    const config = options || {};
    if (modalResolve) closeModal(false);
    modalPreviousFocus = document.activeElement;
    modalTitle.textContent = config.title || "Confirm action";
    modalMessage.textContent = message;
    modalOk.textContent = config.confirmLabel || "Continue";
    modalOverlay.classList.add("active");
    modalOverlay.setAttribute("aria-hidden", "false");
    modalOverlay.inert = false;
    window.requestAnimationFrame(() => modalOk.focus());
    return new Promise((resolve) => {
      modalResolve = resolve;
    });
  }

  modalOk.addEventListener("click", () => closeModal(true));
  modalCancel.addEventListener("click", () => closeModal(false));
  modalOverlay.addEventListener("click", (event) => {
    if (event.target === modalOverlay) closeModal(false);
  });

  modalOverlay.addEventListener("keydown", (event) => {
    if (event.key !== "Tab" || !modalResolve) return;
    const focusable = [modalCancel, modalOk];
    const current = focusable.indexOf(document.activeElement);
    if (event.shiftKey && current <= 0) {
      event.preventDefault();
      modalOk.focus();
    } else if (!event.shiftKey && current === focusable.length - 1) {
      event.preventDefault();
      modalCancel.focus();
    }
  });

  // Per-challenge submission history.
  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const historyClose = document.getElementById("history-close");
  const drawerScrim = document.getElementById("drawer-scrim");
  let historyPreviousFocus = null;

  function getHistory() {
    try {
      const entries = JSON.parse(storageGet(historyStorageKey) || "[]");
      return Array.isArray(entries) ? entries : [];
    } catch (_error) {
      return [];
    }
  }

  function saveHistory(entries) {
    storageSet(historyStorageKey, JSON.stringify(entries));
  }

  function addHistoryEntry(code, action, tests, allPassed) {
    const entries = getHistory();
    const totalTime = tests.reduce((sum, test) => sum + (Number(test.time_ms) || 0), 0);
    const performanceTest = tests.find((test) => test.is_performance);
    entries.unshift({
      ts: Date.now(),
      action,
      code,
      passed: tests.filter((test) => test.passed).length,
      total: tests.length,
      allPassed: Boolean(allPassed),
      totalTimeMs: Math.round(totalTime * 1000) / 1000,
      perfTimeMs: performanceTest ? performanceTest.time_ms : null,
    });
    if (entries.length > 50) entries.length = 50;
    saveHistory(entries);
  }

  function renderHistory() {
    const entries = getHistory();
    if (entries.length === 0) {
      historyList.innerHTML =
        '<div class="history-empty"><span class="history-empty-mark" aria-hidden="true">↺</span><strong>No submissions yet</strong><span>Submit a solution to keep a local snapshot here.</span></div>';
      return;
    }

    historyList.innerHTML = entries
      .map((entry, index) => {
        const time = new Date(Number(entry.ts)).toLocaleString();
        const label = entry.action === "submit" ? "Submit" : "Run";
        const resultClass = entry.allPassed ? "passed" : "failed";
        const resultText = entry.allPassed
          ? "All Passed"
          : `${Number(entry.passed) || 0}/${Number(entry.total) || 0} passed`;
        const performance =
          entry.perfTimeMs == null
            ? ""
            : `<span class="history-runtime">perf: ${esc(entry.perfTimeMs)}ms</span>`;
        return `<article class="history-item">
          <div class="history-meta">
            <span class="history-time">${esc(time)} · ${label}</span>
            <span class="history-result ${resultClass}">${esc(resultText)}</span>
          </div>
          <div class="history-stats">Total: ${esc(entry.totalTimeMs)}ms${performance}</div>
          <div class="history-actions">
            <button class="history-action load" type="button" data-action="load" data-index="${index}">Load code</button>
            <button class="history-action delete" type="button" data-action="delete" data-index="${index}">Delete</button>
          </div>
        </article>`;
      })
      .join("");
  }

  function toggleHistory(force) {
    const shouldOpen = typeof force === "boolean" ? force : !historyPanel.classList.contains("open");
    historyPanel.classList.toggle("open", shouldOpen);
    drawerScrim.classList.toggle("active", shouldOpen);
    historyPanel.setAttribute("aria-hidden", String(!shouldOpen));
    drawerScrim.setAttribute("aria-hidden", String(!shouldOpen));
    historyButton.setAttribute("aria-expanded", String(shouldOpen));
    historyPanel.inert = !shouldOpen;

    if (shouldOpen) {
      historyPreviousFocus = document.activeElement;
      renderHistory();
      window.requestAnimationFrame(() => historyClose.focus());
    } else if (historyPreviousFocus && typeof historyPreviousFocus.focus === "function") {
      historyPreviousFocus.focus();
    }
  }

  historyButton.addEventListener("click", () => toggleHistory());
  historyClose.addEventListener("click", () => toggleHistory(false));
  drawerScrim.addEventListener("click", () => toggleHistory(false));

  historyList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const index = Number(button.dataset.index);
    const entries = getHistory();
    if (!entries[index]) return;

    if (button.dataset.action === "load") {
      cmEditor.setValue(entries[index].code || "");
      toggleHistory(false);
      cmEditor.focus();
    } else if (button.dataset.action === "delete") {
      entries.splice(index, 1);
      saveHistory(entries);
      renderHistory();
    }
  });

  historyPanel.addEventListener("keydown", (event) => {
    if (event.key !== "Tab" || !historyPanel.classList.contains("open")) return;
    const focusable = Array.from(
      historyPanel.querySelectorAll('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'),
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (modalResolve) {
      event.preventDefault();
      closeModal(false);
    } else if (historyPanel.classList.contains("open")) {
      event.preventDefault();
      toggleHistory(false);
    }
  });

  function setRunningState(running, action) {
    runButton.disabled = running;
    submitButton.disabled = running;
    historyButton.disabled = running;
    resetButton.disabled = running;
    vimToggle.disabled = running || vimToggle.dataset.unavailable === "true";
    consoleOutput.setAttribute("aria-busy", String(running));
    runButton.textContent = running && action === "run" ? "Running…" : "Run";
    submitButton.textContent = running && action === "submit" ? "Submitting…" : "Submit";
  }

  async function readJsonResponse(response) {
    try {
      return await response.json();
    } catch (_error) {
      throw new Error(`Server returned ${response.status} ${response.statusText}`.trim());
    }
  }

  async function runCode(action) {
    if (runButton.disabled || submitButton.disabled) return;
    const code = cmEditor.getValue();
    setRunningState(true, action);
    consoleOutput.innerHTML = `<span class="info">${action === "submit" ? "Submitting" : "Running"}… compiling CUDA source</span>\n`;

    try {
      const response = await window.fetch(`/api/challenges/${slug}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await readJsonResponse(response);

      if (data.error && !data.compilation) {
        consoleOutput.innerHTML += `<span class="fail">Error: ${esc(data.error)}</span>\n`;
        return;
      }

      if (!data.compilation || !data.compilation.success) {
        consoleOutput.innerHTML = '<span class="fail">Compilation failed</span>\n\n';
        consoleOutput.innerHTML += `<span class="dim">${esc(data.compilation && data.compilation.stderr)}</span>`;
        return;
      }

      consoleOutput.innerHTML = '<span class="pass">Compilation successful</span>\n';
      if (data.compilation.stderr) {
        consoleOutput.innerHTML += `<span class="warn">${esc(data.compilation.stderr)}</span>\n`;
      }

      if (data.error) {
        consoleOutput.innerHTML += `\n<span class="fail">Runtime error:\n${esc(data.error)}</span>\n`;
        return;
      }

      const tests = Array.isArray(data.tests) ? data.tests : [];
      const functionalTests = tests.filter((test) => !test.is_performance);
      const performanceTests = tests.filter((test) => test.is_performance);

      if (functionalTests.length > 0) {
        consoleOutput.innerHTML += `\n<span class="info">Functional tests · ${functionalTests.length}</span>\n`;
        for (const test of functionalTests) {
          if (test.passed) {
            consoleOutput.innerHTML += `  <span class="pass">✓ Test ${esc(test.index)}/${tests.length} passed</span> <span class="dim">(${esc(test.time_ms)}ms)</span>\n`;
          } else {
            consoleOutput.innerHTML += `  <span class="fail">✕ Test ${esc(test.index)}/${tests.length} failed</span>`;
            if (test.error) consoleOutput.innerHTML += ` <span class="dim">— ${esc(test.error)}</span>`;
            consoleOutput.innerHTML += "\n";
          }
        }
      }

      if (performanceTests.length > 0) {
        consoleOutput.innerHTML += '\n<span class="info">Performance test</span>\n';
        for (const test of performanceTests) {
          if (test.passed) {
            consoleOutput.innerHTML += `  <span class="pass">✓ Performance passed</span> <span class="dim">(${esc(test.time_ms)}ms)</span>\n`;
          } else {
            consoleOutput.innerHTML += '  <span class="fail">✕ Performance failed</span>';
            if (test.error) consoleOutput.innerHTML += ` <span class="dim">— ${esc(test.error)}</span>`;
            consoleOutput.innerHTML += "\n";
          }
        }
      }

      if (tests.length === 0) {
        consoleOutput.innerHTML += '\n<span class="warn">No test results were returned.</span>\n';
      }

      const passed = tests.filter((test) => test.passed).length;
      const allPassed = Boolean(data.all_passed);
      const resultClass = tests.length > 0 && passed === tests.length ? "pass" : "fail";
      consoleOutput.innerHTML += `\n<span class="${resultClass}">Result: ${passed}/${tests.length} tests passed</span>\n`;

      if (action === "submit") {
        addHistoryEntry(code, action, tests, allPassed);
      }

      if (action === "submit" && allPassed) {
        const shouldSave = await showModal("All functional and performance tests passed. Save this solution to disk?", {
          title: "All tests passed",
          confirmLabel: "Save solution",
        });
        if (shouldSave) {
          const saveResponse = await window.fetch(`/api/challenges/${slug}/save`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code }),
          });
          const saveData = await readJsonResponse(saveResponse);
          if (saveData.saved) {
            consoleOutput.innerHTML += `<span class="pass">Solution saved to ${esc(saveData.saved)}</span>\n`;
          } else if (saveData.error) {
            consoleOutput.innerHTML += `<span class="fail">Could not save solution: ${esc(saveData.error)}</span>\n`;
          }
        }
      }
    } catch (error) {
      consoleOutput.innerHTML += `<span class="fail">Request failed: ${esc(error.message)}</span>\n`;
    } finally {
      setRunningState(false, action);
      consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }
  }

  async function resetCode() {
    const shouldReset = await showModal("Replace your current code with the starter template? This cannot be undone.", {
      title: "Reset solution?",
      confirmLabel: "Reset code",
    });
    if (!shouldReset) return;
    storageRemove(codeStorageKey);
    cmEditor.setValue(starterCode);
    cmEditor.focus();
  }

  function esc(value) {
    const element = document.createElement("div");
    element.textContent = value == null ? "" : String(value);
    return element.innerHTML;
  }
})();
