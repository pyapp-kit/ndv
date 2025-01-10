document.addEventListener("DOMContentLoaded", () => {
  const tableData = {
    Source: ["PyPI", "Conda", "Github (dev version)"],
    Graphics: ["VisPy", "Pygfx"],
    Frontend: ["PyQt6", "PySide6", "wxPython", "Jupyter"],
  };

  const commandMap = {
    "PyPI,VisPy,PyQt6": "pip install ndv[vispy,pyqt]",
    "PyPI,VisPy,PySide6": "pip install ndv[vispy,pyside]",
    "PyPI,VisPy,wxPython": "pip install ndv[vispy,wx]",
    "PyPI,VisPy,Jupyter": "pip install ndv[vispy,jupyter]",
    "PyPI,Pygfx,PyQt6": "pip install ndv[pygfx,pyqt]",
    "PyPI,Pygfx,PySide6": "pip install ndv[pygfx,pyside]",
    "PyPI,Pygfx,wxPython": "pip install ndv[pygfx,wx]",
    "PyPI,Pygfx,Jupyter": "pip install ndv[pygfx,jupyter]",
    "Conda,VisPy,PyQt6": "conda install -c conda-forge ndv vispy pyopengl qt6-main",
    "Conda,VisPy,PySide6": "conda install -c conda-forge ndv vispy pyopengl 'pyside6<6.8'",
    "Conda,VisPy,wxPython": "conda install -c conda-forge ndv vispy pyopengl wxpython",
    "Conda,VisPy,Jupyter": "conda install -c conda-forge ndv vispy pyopengl jupyter jupyter_rfb glfw ipywidgets",
    "Conda,Pygfx,PyQt6": "conda install -c conda-forge ndv pygfx qt6-main",
    "Conda,Pygfx,PySide6": "conda install -c conda-forge ndv pygfx 'pyside6<6.8'",
    "Conda,Pygfx,wxPython": "conda install -c conda-forge ndv pygfx wxpython",
    "Conda,Pygfx,Jupyter": "conda install -c conda-forge ndv pygfx jupyter jupyter_rfb glfw ipywidgets",
    "Github (dev version),VisPy,PyQt6": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[vispy,pyqt]",
    "Github (dev version),VisPy,PySide6": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[vispy,pyside]",
    "Github (dev version),VisPy,wxPython": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[vispy,wx]",
    "Github (dev version),VisPy,Jupyter": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[vispy,jupyter]",
    "Github (dev version),Pygfx,PyQt6": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[pygfx,pyqt]",
    "Github (dev version),Pygfx,PySide6": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[pygfx,pyside]",
    "Github (dev version),Pygfx,wxPython": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[pygfx,wx]",
    "Github (dev version),Pygfx,Jupyter": "pip install git+https://github.com/pyapp-kit/ndv.git@main#egg=ndv[pygfx,jupyter]",
  };

  const container = document.getElementById("install-table");

  const createTable = () => {
    Object.keys(tableData).forEach((category) => {
      const label = document.createElement("div");
      label.classList.add("category-label");
      label.textContent = category;

      const buttonsContainer = document.createElement("div");
      buttonsContainer.classList.add("buttons");

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
    commandDiv.classList.add("buttons", "command-section");
    commandDiv.innerHTML = `
    <p id="command-output">Select options to generate command</p>
    `;
    container.appendChild(label);
    container.appendChild(commandDiv);

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

  createTable();
});
