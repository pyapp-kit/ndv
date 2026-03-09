document.addEventListener("DOMContentLoaded", () => {
  const tableData = {
    Source: ["PyPI", "Conda", "Github (dev version)"],
    Graphics: ["VisPy", "Pygfx"],
    Frontend: ["PyQt6", "PySide6", "wxPython", "Jupyter"],
  };

  const pipCommand = 'pip install'
  const condaCommand = 'conda install -c conda-forge'
  const gitCommand = 'pip install "git+https://github.com/pyapp-kit/ndv.git@main#egg'
  const condaVispy = 'vispy pyopengl';
  const condaJupyter = 'jupyter jupyter-rfb pyglfw ipywidgets';
  const commandMap = {
    "PyPI,VisPy,PyQt6": `${pipCommand} "ndv[vispy,pyqt]"`,
    "PyPI,VisPy,PySide6": `${pipCommand} "ndv[vispy,pyside]"`,
    "PyPI,VisPy,wxPython": `${pipCommand} "ndv[vispy,wxpython]"`,
    "PyPI,VisPy,Jupyter": `${pipCommand} "ndv[vispy,jupyter]"`,
    "PyPI,Pygfx,PyQt6": `${pipCommand} "ndv[pygfx,pyqt]"`,
    "PyPI,Pygfx,PySide6": `${pipCommand} "ndv[pygfx,pyside]"`,
    "PyPI,Pygfx,wxPython": `${pipCommand} "ndv[pygfx,wxpython]"`,
    "PyPI,Pygfx,Jupyter": `${pipCommand} "ndv[pygfx,jupyter]"`,
    "Conda,VisPy,PyQt6": `pyqt6 is not available in conda-forge, use PySide6`,
    "Conda,VisPy,PySide6": `${condaCommand} ndv ${condaVispy} 'pyside6<6.8'`,
    "Conda,VisPy,wxPython": `${condaCommand} ndv ${condaVispy} wxpython`,
    "Conda,VisPy,Jupyter": `${condaCommand} ndv ${condaVispy} ${condaJupyter}`,
    "Conda,Pygfx,PyQt6": `${condaCommand} ndv pygfx qt6-main`,
    "Conda,Pygfx,PySide6": `${condaCommand} ndv pygfx 'pyside6<6.8'`,
    "Conda,Pygfx,wxPython": `${condaCommand} ndv pygfx wxpython`,
    "Conda,Pygfx,Jupyter": `${condaCommand} ndv pygfx ${condaJupyter}`,
    "Github (dev version),VisPy,PyQt6": `${gitCommand}=ndv[vispy,pyqt]"`,
    "Github (dev version),VisPy,PySide6": `${gitCommand}=ndv[vispy,pyside]"`,
    "Github (dev version),VisPy,wxPython": `${gitCommand}=ndv[vispy,wx]"`,
    "Github (dev version),VisPy,Jupyter": `${gitCommand}=ndv[vispy,jupyter]"`,
    "Github (dev version),Pygfx,PyQt6": `${gitCommand}=ndv[pygfx,pyqt]"`,
    "Github (dev version),Pygfx,PySide6": `${gitCommand}=ndv[pygfx,pyside]"`,
    "Github (dev version),Pygfx,wxPython": `${gitCommand}=ndv[pygfx,wx]"`,
    "Github (dev version),Pygfx,Jupyter": `${gitCommand}=ndv[pygfx,jupyter]"`,
  };

  const container = document.getElementById("install-table");

  const createTable = () => {
    Object.keys(tableData).forEach((category) => {
      const label = document.createElement("div");
      label.classList.add("category-label");
      label.textContent = category;

      const buttonsContainer = document.createElement("div");
      buttonsContainer.classList.add("grid-right", "buttons");

      tableData[category].forEach((item, index) => {
        const button = document.createElement("button");
        button.textContent = item;
        button.dataset.value = item;

        // Activate the first button in the category
        if (index === 0) {
          button.classList.add("active");
        }

        // Event listener for button click
        button.addEventListener("click", (event) => {
          // Deactivate all buttons in this category
          Array.from(buttonsContainer.children).forEach((btn) => btn.classList.remove("active"));

          // Activate the clicked button
          button.classList.add("active");

          // Update command
          updateCommand();
        });

        buttonsContainer.appendChild(button);
      });

      container.appendChild(label);
      container.appendChild(buttonsContainer);
    });

    const label = document.createElement("div");
    label.classList.add("category-label", "command-label");
    label.textContent = "Command:";

    const commandDiv = document.createElement("div");
    commandDiv.classList.add("grid-right", "command-section");
    commandDiv.innerHTML = `
    <p id="command-output">Select options to generate command</p>
    <button class="md-clipboard md-icon" title="Copy to clipboard"></button>
    `;
    container.appendChild(label);
    container.appendChild(commandDiv);

    // Add copy functionality
    const copyButton = commandDiv.querySelector(".md-clipboard");
    copyButton.addEventListener("click", copyToClipboard);

    // Update the command output initially
    updateCommand();
  };

  const updateCommand = () => {
    const activeButtons = document.querySelectorAll("button.active");
    const selectedOptions = Array.from(activeButtons).map((btn) => btn.dataset.value);
    const commandOutput = document.getElementById("command-output");
    console.log();

    if (selectedOptions.length === 0) {
      commandOutput.textContent = "Select options to generate command";
    } else {
      commandOutput.textContent = commandMap[selectedOptions.join(",")];
    }
  };
  const copyToClipboard = () => {
    const commandOutput = document.getElementById("command-output").textContent;
    navigator.clipboard
      .writeText(commandOutput)
      .then(() => {
        // give a little animated feedback
        const commandDiv = document.querySelector(".command-section .md-clipboard");
        commandDiv.classList.add("copied");
        setTimeout(() => {
          commandDiv.classList.remove("copied");
        }, 500);
      })
      .catch((error) => {
        console.error("Failed to copy to clipboard", error);
      });
  };

  if (container) {
    createTable();
  }
});
