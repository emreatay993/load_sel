# **Technical Documentation: we\_mechload\_viewer Architecture**

## **Introduction**

This document provides a deep dive into the internal architecture of the we\_mechload\_viewer application (modular\_version\_v2). It details the responsibility of each module and, crucially, maps out the **signal and slot communication network** that allows the application's components to interact.

The architecture follows a **Model-View-Controller (MVC)** pattern. Think of it as a well-organized system with specialized roles: the **Model** manages the data, the **View** displays it, and the **Controller** acts as the go-between. Understanding this separation is key to understanding the codebase.

[a Model-View-Controller diagram resmi](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTuBn_3gui4RmmDeClGEaqQwqKPY7MNavFe6MGRFnn8XjGIrrgMyC68ggF9DbEESa3mqMu1iW1rBKSkWoJnAcMM1sazSf3VmZsbKt4LuyPQU1NJvzc)

## **Part 1: Application Bootstrap**

### **main.py**

* **Role:** The single entry point for the application. It's the "general contractor" that assembles everything.  
* **Responsibilities:**  
  1. Initializes the QApplication.  
  2. Instantiates all the core components:  
     * MainWindow (the main View).  
     * DataManager, ConfigManager (the Model).  
     * ActionHandler, PlotController (the Controllers).  
  3. **Dependency Injection:** It passes instances of the Model and View to the Controllers. This is a critical step that **decouples** the components; the controllers don't create their dependencies, they receive them.  
  4. Starts the Qt event loop.

\# main.py \- Simplified  
app \= QApplication(sys.argv)

\# 1\. Instantiate all components  
main\_win \= MainWindow()  
data\_manager \= DataManager()  
config\_manager \= ConfigManager(config\_path)  
plotter \= main\_win.plotter \# Get plotter from MainWindow

\# 2\. Instantiate controllers and inject dependencies  
action\_handler \= ActionHandler(main\_win, data\_manager, config\_manager)  
plot\_controller \= PlotController(main\_win, data\_manager, plotter)

\# 3\. Show the main window and run  
main\_win.show()  
sys.exit(app.exec\_())

## **Part 2: The Model Layer (Data and Logic)**

The Model layer is responsible for managing data and business logic. It **knows nothing about the UI**.

### **app/data\_manager.py**

* **Role:** The central repository for application data. It's the **single source of truth**.  
* **Key Attributes:** self.df (the main pandas DataFrame), self.df\_compare (the second DataFrame for comparisons).  
* **Key Methods:** load\_data(), get\_dataframe(), set\_dataframe().  
* **Signals:** It defines a custom signal, data\_loaded, which it emits after a new file has been successfully loaded into the DataFrame. This allows other components (like controllers) to react to new data becoming available.

### **app/config\_manager.py**

* **Role:** Handles saving and loading user settings and application configuration from a JSON file.

### **app/analysis/data\_processing.py**

* **Role:** A library of **pure, stateless functions** for all numerical analysis. This is where the "heavy lifting" happens.  
* **Responsibilities:** Contains all calculation logic, such as:  
  * calculate\_fft()  
  * calculate\_rolling\_envelope()  
  * apply\_butterworth\_filter()  
* **Interaction:** These functions are called exclusively by the **Controllers**. They take data as input and return results, without modifying any application state directly.

### **app/analysis/ansys\_exporter.py**

* **Role:** A specialized module for exporting data to the Ansys APDL format.  
* **Interaction:** Called by the ActionHandler controller when the user triggers an export action.

## **Part 3: The View Layer (UI)**

The View layer is responsible for everything the user sees. Its components are designed to be "dumb"—they display information and **emit signals** when the user interacts with them, but they don't contain processing logic.

### **app/main\_window.py**

* **Role:** The top-level QMainWindow that acts as a container for all other UI elements.  
* **Responsibilities:**  
  1. Initializes and lays out the main UI structure (docks, tabs, menu bar).  
  2. Instantiates all the individual UI tabs from the app/ui/ directory.  
  3. Instantiates the Plotter widget.  
  4. Provides accessors (.get\_tab\_single\_data(), etc.) so the controllers can access the UI components to connect signals.

### **app/plotting/plotter.py**

* **Role:** A specialized QWidget whose only job is to render a Plotly figure.  
* **Responsibilities:** It has one primary slot/method: plot(fig), which takes a Plotly figure object, saves it to a temporary HTML file, and loads that file into its web browser view.

### **app/ui/\*.py (The Tabs)**

These modules are the primary source of user-interaction **signals**.

* **tab\_single\_data.py**, **tab\_compare\_data.py**:  
  * **Purpose:** Display data statistics and allow column selection.  
  * **Signals Emitted:**  
    * self.column\_selector.currentTextChanged  
    * self.plot\_checkbox.stateChanged  
    * self.filter\_checkbox.stateChanged  
    * self.butterworth\_order.valueChanged  
    * self.butterworth\_cutoff.valueChanged  
* **tab\_time\_domain\_represent.py**:  
  * **Purpose:** Configure time-domain plot representations like rolling FFTs and envelopes.  
  * **Signals Emitted:**  
    * self.checkbox\_roll\_fft.stateChanged  
    * self.spin\_box\_roll\_fft\_window.valueChanged  
    * self.spin\_box\_roll\_fft\_freq.valueChanged  
    * self.checkbox\_roll\_env.stateChanged  
    * self.spin\_box\_roll\_env.valueChanged  
* **directory\_tree\_dock.py**:  
  * **Purpose:** A file browser dock.  
  * **Signals Emitted:** self.tree.doubleClicked (emits a QModelIndex).

## **Part 4: The Controller Layer (The "Glue")**

The Controllers are the heart of the application's interactive logic, connecting the View's signals to the Model's functions. They **listen for user actions** and orchestrate the application's response.

### **app/controllers/action\_handler.py**

* **Role:** Manages general, non-plotting actions like file I/O and settings.  
* **Signal-Slot Connections (in \_\_init\_\_)**:  
  * main\_win.open\_action.triggered **connects to** self.handle\_open\_file  
  * main\_win.open\_folder\_action.triggered **connects to** self.handle\_open\_folder  
  * main\_win.directory\_dock.tree.doubleClicked **connects to** self.handle\_tree\_selection  
  * data\_manager.data\_loaded **connects to** main\_win.update\_ui\_after\_data\_load *(Controller connects a Model signal to a View slot)*  
  * main\_win.settings\_tab.save\_settings\_button.clicked **connects to** self.handle\_save\_settings  
  * main\_win.interface\_data\_tab.export\_button.clicked **connects to** self.handle\_export\_to\_ansys

### **app/controllers/plot\_controller.py**

* **Role:** Manages all logic related to updating the plot. This is the most complex controller because many different UI inputs can trigger a plot refresh.  
* **Signal-Slot Connections (in \_\_init\_\_)**:  
  * **From tab\_single\_data & tab\_compare\_data**:  
    * tab.column\_selector.currentTextChanged **connects to** self.update\_plot  
    * tab.plot\_checkbox.stateChanged **connects to** self.update\_plot  
    * tab.filter\_checkbox.stateChanged **connects to** self.update\_plot  
    * tab.butterworth\_order.valueChanged **connects to** self.update\_plot  
    * tab.butterworth\_cutoff.valueChanged **connects to** self.update\_plot  
  * **From tab\_time\_domain\_represent**:  
    * tab\_time.checkbox\_roll\_fft.stateChanged **connects to** self.update\_plot  
    * tab\_time.spin\_box\_roll\_fft\_window.valueChanged **connects to** self.update\_plot  
    * tab\_time.spin\_box\_roll\_fft\_freq.valueChanged **connects to** self.update\_plot  
    * tab\_time.checkbox\_roll\_env.stateChanged **connects to** self.update\_plot  
    * tab\_time.spin\_box\_roll\_env.valueChanged **connects to** self.update\_plot  
  * **From data\_manager (Model)**:  
    * self.data\_manager.data\_loaded **connects to** self.update\_plot  
* **Core Logic (update\_plot slot)**: This central slot is the destination for almost every plot-related signal. When triggered, it executes the following sequence:  
  1. Gathers the current configuration from **all relevant UI tabs** (which columns are selected, are filters active, what are the FFT settings, etc.).  
  2. Gets the raw DataFrame from the DataManager.  
  3. If required, it calls functions in data\_processing.py to perform filtering, FFT calculations, etc., on the raw data.  
  4. Constructs a Plotly Figure object with the final data traces.  
  5. Calls self.plotter.plot(fig) to render the result.