# **WE MechLoad Viewer**

## **Technical Architecture and Data Flow Documentation**

### **Table of Contents**

1. **Introduction & High-Level Architecture**  
   * 1.1. Purpose of the Application  
   * 1.2. Architectural Pattern (MVC-inspired)  
   * 1.3. Key Components Overview  
2. **Core Components: The Application Backbone**  
   * 2.1. main.py: The Application Entry Point  
   * 2.2. DataManager: The Data Engine (Model)  
   * 2.3. MainWindow: The Central Hub (View-Controller)  
3. **The Controller Layer: Handling Logic**  
   * 3.1. Role of Controllers  
   * 3.2. PlotController: The Visualization Coordinator  
   * 3.3. ActionHandler: The Action & Export Manager  
4. **The View Layer: User Interaction Tabs**  
   * 4.1. SingleDataTab  
   * 4.2. InterfaceDataTab  
   * 4.3. PartLoadsTab  
   * 4.4. CompareDataTab  
   * 4.5. TimeDomainRepresentTab  
   * 4.6. SettingsTab  
5. **Signal & Slot Communication Maps**  
   * 5.1. Application Startup and Initial Data Load  
   * 5.2. Plotting Workflow: SingleDataTab  
   * 5.3. Plotting Workflow: InterfaceDataTab  
   * 5.4. Plotting Workflow: PartLoadsTab  
   * 5.5. Data Comparison Workflow  
   * 5.6. Action Workflow: Ansys Export  
   * 5.7. Action Workflow: Time Domain Data Extraction  
6. **Backend Modules: Analysis & Utilities**  
   * 6.1. data\_processing.py  
   * 6.2. ansys\_exporter.py  
   * 6.3. plotter.py  
7. **State Management**  
   * 7.1. Primary Data State (self.df)  
   * 7.2. Comparison Data State (self.df\_compare)  
   * 7.3. UI State  
8. **Conclusion and Future Development**

## **1\. Introduction & High-Level Architecture**

### **1.1. Purpose of the Application**

The **WE MechLoad Viewer** is a specialized desktop application designed for mechanical engineers to load, visualize, analyze, and export mechanical load data. It handles both time-domain and frequency-domain data, providing various plotting capabilities and export functions for further analysis in software like Ansys.

### **1.2. Architectural Pattern (MVC-inspired)**

The application's architecture is heavily inspired by the **Model-View-Controller (MVC)** design pattern. This pattern separates the application's concerns into three interconnected components, which makes the code more organized, scalable, and easier to maintain.

[a Model-View-Controller diagram resmi](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcR8pCTVQOofaTrB0a3OdDayOHSbr0CjMyhpAz7AePmyl2xew5qtvUheSB1LVZy6jfjQUDbgqAVAE4j3UdREWlGiXhq47X2cA1cYQ3_CN2RqnVu8G4Q)

* **Model (DataManager)**: Manages the application's data and business logic. It's responsible for reading data from files, processing it, and storing it. It knows nothing about the user interface.  
* **View (UI Tabs like SingleDataTab)**: The user interface. It displays the data from the Model and sends user actions (button clicks, selections) to the Controller.  
* **Controller (PlotController, ActionHandler)**: Acts as the intermediary between the Model and the View. It receives user input from the View, processes it (sometimes by querying the Model for data), and updates the View with the results.

This separation is key. For example, the DataManager can be tested independently of the UI, and the UI can be changed without affecting the data loading logic.

### **1.3. Key Components Overview**

Here’s a tree view of the main components and their roles:

WE MechLoad Viewer  
├── main.py (Entry Point)  
│  
├── app/  
│   ├── data\_manager.py (Model)  
│   │   └── Handles all file I/O and data parsing. Holds the raw data logic.  
│   │  
│   ├── main\_window.py (Main Container)  
│   │   └── Holds all UI elements and acts as the central communication hub.  
│   │  
│   ├── controllers/  
│   │   ├── plot\_controller.py (Controller)  
│   │   │   └── Manages all logic related to updating plots.  
│   │   └── action\_handler.py (Controller)  
│   │       └── Manages logic for exports and other complex actions.  
│   │  
│   ├── ui/ (Views)  
│   │   ├── tab\_single\_data.py  
│   │   ├── tab\_interface\_data.py  
│   │   └── ... (Other UI tabs and components)  
│   │  
│   ├── plotting/  
│   │   └── plotter.py (Utility)  
│   │       └── Contains the actual plotting code (using Plotly).  
│   │  
│   └── analysis/  
│       ├── data\_processing.py (Utility)  
│       │   └── Contains functions for filtering, windowing, etc.  
│       └── ansys\_exporter.py (Utility)  
│           └── Contains logic to generate Ansys script files.

## **2\. Core Components: The Application Backbone**

### **2.1. main.py: The Application Entry Point**

This is the script that starts everything. Its role is simple but critical:

1. **Create the QApplication instance**: The fundamental object for any PyQt app.  
2. **Instantiate Core Objects**: It creates the two most important objects:  
   * data\_manager \= DataManager()  
   * main\_window \= MainWindow(data\_manager)  
3. **Establish Initial Connection**: It makes the very first signal-slot connection, which is the foundation for the entire application's data flow:  
   data\_manager.dataLoaded.connect(main\_window.on\_data\_loaded)

4. **Show the Window**: It displays the MainWindow.  
5. **Trigger Initial Data Load**: It uses QTimer.singleShot to call data\_manager.load\_data\_from\_directory(). Using a timer ensures the UI is fully rendered and responsive before the potentially blocking file dialog appears.

### **2.2. DataManager: The Data Engine (Model)**

The DataManager is the sole authority on data loading and parsing. It is completely decoupled from the UI.

#### **Data Loading Process (load\_data\_from\_paths)**

This is the core method. When called, it performs the following steps for each folder path it receives:

1. **Validation**: It checks for the existence of required full.pld and max.pld files. If they're missing, the folder is skipped.  
2. **Read Raw Data**: It reads all full.pld files into a temporary Pandas DataFrame.  
3. **Determine Domain**: It inspects the DataFrame's columns to see if it contains 'TIME' or 'FREQ', thus determining the data's domain.  
4. **Domain Consistency Check**: It ensures that all folders being loaded share the same domain. If a mismatch is found, the inconsistent folder is skipped.  
5. **Read Headers**: It calls \_read\_pld\_log\_file to parse the max.pld file, which contains the column headers.  
6. **Assign Columns**: It uses \_get\_column\_headers to format the headers correctly (adding 'Phase\_' columns for frequency data) and assigns them to the temporary DataFrame.  
7. **Tag Data**: It adds a 'DataFolder' column to the DataFrame to identify the source of the data. This is crucial for multi-folder analysis.  
8. **Combine and Sort**: After processing all valid folders, it concatenates all temporary DataFrames into one final DataFrame and sorts it by the domain column ('TIME' or 'FREQ').

#### **Emitted Signals**

The DataManager communicates with the rest of the application using these signals:

* dataLoaded(pd.DataFrame, str, str): Emitted upon a successful data load. It sends the final DataFrame, the data domain ('TIME' or 'FREQ'), and the path of the first valid folder.  
* dataLoadFailed(str): Emitted if no valid data could be loaded.  
* comparisonDataLoaded(pd.DataFrame): Emitted when a secondary dataset for comparison has been successfully loaded.

### **2.3. MainWindow: The Central Hub (View-Controller)**

MainWindow is the most central class. It owns all the UI elements, the controllers, and holds the application's state.

#### **Responsibilities:**

1. **UI Setup (\_setup\_ui)**: Initializes and arranges all the main UI components: the menu bar, the DirectoryTreeDock, and the main QTabWidget which contains all the individual view tabs.  
2. **Signal Aggregation (\_connect\_signals)**: This is where the application is wired together. It connects signals from the UI tabs and DataManager to the slots in the controllers (PlotController, ActionHandler) and MainWindow itself. This method is the blueprint for the application's interactivity.  
3. **State Management**: It holds the primary data in self.df and comparison data in self.df\_compare. By holding the state, it ensures that all controllers and views have a single source of truth for the data.

#### **Key Slots:**

* **on\_data\_loaded(self, data, data\_domain, folder\_path)**: This is arguably the most important slot in the application. When the DataManager emits its dataLoaded signal, this method is triggered. It does the following:  
  1. Receives the data and stores it in self.df.  
  2. Updates the window title.  
  3. Populates the DirectoryTreeDock.  
  4. Calls \_populate\_all\_selectors() to fill all the dropdowns in the UI tabs with the columns and interfaces from the new data.  
  5. Adjusts UI visibility based on the data (e.g., shows/hides the "Time Domain Rep." tab).  
  6. Triggers an initial plot update by calling self.plot\_controller.update\_all\_plots\_from\_settings().  
* **on\_comparison\_data\_loaded(self, df\_compare)**: Stores the comparison DataFrame in self.df\_compare and populates the relevant UI controls in the CompareDataTab.

## **3\. The Controller Layer: Handling Logic**

### **3.1. Role of Controllers**

The controllers listen for signals from the View (UI tabs) and decide what to do. They contain the "business logic" of the application. This design prevents the UI tabs from becoming bloated with complex logic.

### **3.2. PlotController: The Visualization Coordinator**

The PlotController's only job is to create plots. It listens for ...\_parameters\_changed signals from the various UI tabs.

#### **Workflow for a Typical Slot (e.g., update\_single\_data\_plots):**

1. **Trigger**: The user changes a setting in SingleDataTab (e.g., selects a new column). SingleDataTab emits plot\_parameters\_changed.  
2. **Activation**: update\_single\_data\_plots slot is called.  
3. **Gather Information**:  
   * It accesses the main DataFrame: self.main\_window.df.  
   * It reads the current UI settings from SingleDataTab (e.g., self.main\_window.tab\_single\_data.column\_selector.currentText()).  
   * It reads global plot settings from SettingsTab.  
4. **Delegate to Plotter**: It passes the data and settings to a method in the Plotter class (e.g., self.main\_window.plotter.create\_single\_plot(...)).  
5. **Receive Figure**: The Plotter returns a Plotly figure object.  
6. **Update View**: It calls the appropriate display method on the SingleDataTab to render the figure: self.main\_window.tab\_single\_data.display\_regular\_plot(fig).

This cycle—**View Event \-\> Controller Logic \-\> Model Data \-\> View Update**—is repeated for every plot in the application.

### **3.3. ActionHandler: The Action & Export Manager**

The ActionHandler is similar to the PlotController but handles non-plotting actions.

#### **Key Slots:**

* **handle\_compare\_data\_selection()**: This is a simple passthrough. It receives the request from CompareDataTab and calls self.data\_manager.load\_comparison\_data(), effectively initiating the data loading flow for the comparison dataset.  
* **handle\_time\_domain\_represent\_export()**:  
  1. Triggered by the "Extract Data" button in TimeDomainRepresentTab.  
  2. Reads the plot data currently stored on the tab (tab.current\_plot\_data).  
  3. Formats this data into a new Pandas DataFrame.  
  4. Opens a QFileDialog to ask the user where to save the CSV file.  
  5. Saves the DataFrame to the selected location.  
* **handle\_ansys\_export()**: This is the most complex action.  
  1. Triggered by the "Export to Ansys" button in PartLoadsTab.  
  2. Opens a dialog to allow the user to select which "sides" to export.  
  3. **Filters Data**: Creates a new DataFrame containing only the columns relevant to the selected sides.  
  4. **Processes Data**: It accesses the UI settings on PartLoadsTab to see if time-slicing (apply\_data\_section) or a Tukey window (apply\_tukey\_window) should be applied. It calls these functions from data\_processing.py.  
  5. **Unit Conversion**: It multiplies the relevant load columns by 1000\.  
  6. **Delegate to Exporter**: It creates an AnsysExporter instance and calls the appropriate method (create\_harmonic\_template or create\_transient\_template) to generate the final Ansys script files.

## **4\. The View Layer: User Interaction Tabs**

Each tab is a self-contained QWidget responsible for its own layout and user inputs. Their primary role is to emit signals when the user interacts with them.

### **4.1. SingleDataTab**

* **Purpose**: To visualize a single data column against TIME or FREQ.  
* **Key Widgets**: column\_selector, spectrum\_checkbox, filter\_checkbox.  
* **Emitted Signals**:  
  * plot\_parameters\_changed: Emitted when any control that affects the main plot or the existence of the spectrum plot is changed.  
  * spectrum\_parameters\_changed: Emitted only when controls that *only* affect the spectrum plot's appearance (e.g., plot\_type\_selector) are changed. This is an optimization to avoid re-calculating the main plot unnecessarily.

### **4.2. InterfaceDataTab**

* **Purpose**: To visualize all components (e.g., Fx, Fy, Fz) of a specific interface.  
* **Key Widgets**: interface\_selector.  
* **Emitted Signals**: plot\_parameters\_changed.

### **4.3. PartLoadsTab**

* **Purpose**: To visualize loads grouped by "side" (e.g., all forces on the "Left-Hand-Side").  
* **Key Widgets**: side\_filter\_selector, component\_selector, tukey\_checkbox.  
* **Emitted Signals**: plot\_parameters\_changed, export\_to\_ansys\_requested.

### **4.4. CompareDataTab**

* **Purpose**: To plot a column from the primary dataset against the same column from a secondary (comparison) dataset.  
* **Key Widgets**: column\_selector, compare\_column\_selector.  
* **Emitted Signals**: plot\_parameters\_changed, select\_compare\_data\_requested.

### **4.5. TimeDomainRepresentTab**

* **Purpose**: For frequency-domain data, this tab reconstructs and visualizes the time-domain representation at a single selected frequency.  
* **Key Widgets**: data\_point\_selector (to select the frequency).  
* **Emitted Signals**: plot\_parameters\_changed, extract\_data\_requested.

### **4.6. SettingsTab**

* **Purpose**: To control global plotting settings that affect all tabs.  
* **Key Widgets**: Checkboxes for legend visibility, grid lines, etc.  
* **Emitted Signals**: settings\_changed. This signal is connected to the PlotController's update\_all\_plots\_from\_settings slot, causing a full refresh of every visible plot.

## **5\. Signal & Slot Communication Maps**

This section provides a visual trace of the most common user workflows.

### **5.1. Application Startup and Initial Data Load**

graph TD  
    A\[main.py\] \--\>|1. Creates| B(DataManager);  
    A \--\>|2. Creates| C(MainWindow);  
    A \--\>|3. Connects| D{dataLoaded Signal};  
    C \--\>|4. Listens on| E\[on\_data\_loaded Slot\];  
    D \-.-\> E;  
    A \--\>|5. QTimer calls| F\[load\_data\_from\_directory\];  
    B \-- owns \--\> F;  
    F \--\>|6. User selects folder| G\[File Dialog\];  
    F \--\>|7. Parses .pld/.log files| H\[pandas DataFrame\];  
    F \--\>|8. Emits| D;

### **5.2. Plotting Workflow: SingleDataTab**

USER ACTION: Selects 'I1 Fx' from column\_selector in SingleDataTab  
     │  
     │  
     ▼  
SingleDataTab.column\_selector.currentIndexChanged  
     │  
     ├─ emits signal: plot\_parameters\_changed()  
     │  
     │  
     ▼  
MainWindow.\_connect\_signals (Connection was made here)  
     │  
     │  
     ▼  
PlotController.update\_single\_data\_plots() \[SLOT\]  
     │  
     ├─ 1\. Accesses MainWindow.df (The main data)  
     ├─ 2\. Reads current column 'I1 Fx' from SingleDataTab  
     ├─ 3\. Reads plot settings from SettingsTab  
     │  
     ├─ 4\. Calls Plotter.create\_single\_plot(data, column='I1 Fx', ...)  
     │    │  
     │    └─ Returns a Plotly figure object \`fig\`  
     │  
     ├─ 5\. Calls SingleDataTab.display\_regular\_plot(fig)  
     │    │  
     │    └─ Renders \`fig\` into the QWebEngineView  
     │  
     └─ \[PROCESS COMPLETE\]

### **5.3. Plotting Workflow: InterfaceDataTab**

This follows the same pattern as SingleDataTab, but is triggered by the interface\_selector.

USER ACTION: Selects 'I3' from interface\_selector in InterfaceDataTab  
     │  
     ▼  
InterfaceDataTab.interface\_selector.currentIndexChanged  
     │  
     ├─ emits signal: plot\_parameters\_changed()  
     │  
     ▼  
PlotController.update\_interface\_data\_plots() \[SLOT\]  
     │  
     ├─ 1\. Accesses MainWindow.df  
     ├─ 2\. Reads current interface 'I3'  
     ├─ 3\. Finds all columns related to 'I3' (e.g., 'I3 Fx', 'I3 Fy', ...)  
     ├─ 4\. Calls Plotter.create\_interface\_plot(data, interface='I3', ...)  
     │  
     └─ Calls InterfaceDataTab.display\_plot(fig)

### **5.4. Plotting Workflow: PartLoadsTab**

This workflow is slightly more complex as it involves filtering by both "side" and "component".

USER ACTION: Selects 'LHS' from side\_filter\_selector  
     │  
     ▼  
PartLoadsTab.side\_filter\_selector.currentIndexChanged  
     │  
     ├─ emits signal: plot\_parameters\_changed()  
     │  
     ▼  
PlotController.update\_part\_loads\_plots() \[SLOT\]  
     │  
     ├─ 1\. Accesses MainWindow.df  
     ├─ 2\. Reads side 'LHS' and component 'Mz' from PartLoadsTab  
     ├─ 3\. Filters df to get all 'Mz' columns that also contain 'LHS'  
     ├─ 4\. Calls Plotter.create\_part\_loads\_plot(...)  
     │  
     └─ Calls PartLoadsTab.display\_plot(fig)

### **5.5. Data Comparison Workflow**

This is a two-step process: loading the data, then plotting.

**Step 1: Loading Comparison Data**

USER ACTION: Clicks "Select Data to Compare" button in CompareDataTab  
     │  
     ▼  
CompareDataTab.select\_compare\_data\_requested() \[SIGNAL\]  
     │  
     ▼  
ActionHandler.handle\_compare\_data\_selection() \[SLOT\]  
     │  
     └─ Calls DataManager.load\_comparison\_data()  
          │  
          ├─ Opens File Dialog, parses files...  
          │  
          └─ emits comparisonDataLoaded(df\_compare) \[SIGNAL\]  
               │  
               ▼  
          MainWindow.on\_comparison\_data\_loaded(df\_compare) \[SLOT\]  
               │  
               ├─ Stores data in self.df\_compare  
               └─ Populates selectors in CompareDataTab

Step 2: Plotting Comparison Data  
This follows the standard plotting flow, but the PlotController's update\_compare\_data\_plots slot will access both self.main\_window.df and self.main\_window.df\_compare.

### **5.6. Action Workflow: Ansys Export**

USER ACTION: Clicks "Export to Ansys" button in PartLoadsTab  
     │  
     ▼  
PartLoadsTab.export\_to\_ansys\_requested() \[SIGNAL\]  
     │  
     ▼  
ActionHandler.handle\_ansys\_export() \[SLOT\]  
     │  
     ├─ 1\. Opens dialog to select "sides" for export.  
     ├─ 2\. Filters MainWindow.df based on selection.  
     ├─ 3\. Checks UI controls on PartLoadsTab (e.g., tukey\_checkbox).  
     │  
     ├─ 4\. Calls functions from data\_processing.py (e.g., apply\_tukey\_window).  
     │    │  
     │    └─ Returns a processed DataFrame \`df\_processed\`.  
     │  
     ├─ 5\. Performs unit conversions on \`df\_processed\`.  
     │  
     ├─ 6\. Creates AnsysExporter instance.  
     │  
     ├─ 7\. Calls AnsysExporter.create\_transient\_template(df\_processed, ...).  
     │    │  
     │    └─ Writes .dat and .inp files to disk.  
     │  
     └─ \[PROCESS COMPLETE\]

### **5.7. Action Workflow: Time Domain Data Extraction**

USER ACTION: Clicks "Extract Data" button in TimeDomainRepresentTab  
     │  
     ▼  
TimeDomainRepresentTab.extract\_data\_requested() \[SIGNAL\]  
     │  
     ▼  
ActionHandler.handle\_time\_domain\_represent\_export() \[SLOT\]  
     │  
     ├─ 1\. Reads plot data from TimeDomainRepresentTab.current\_plot\_data.  
     ├─ 2\. Constructs a new DataFrame from this plot data.  
     ├─ 3\. Opens a QFileDialog to get a save path from the user.  
     │  
     └─ 4\. Saves the new DataFrame as a .csv file.

## **6\. Backend Modules: Analysis & Utilities**

These modules contain pure, non-GUI logic.

### **6.1. data\_processing.py**

This is a library of functions that operate on DataFrames.

* apply\_data\_section: Slices a DataFrame based on a min/max time.  
* apply\_tukey\_window: Applies a Tukey (tapered cosine) window to the data to reduce spectral leakage, which is important for FFT analysis.  
* apply\_low\_pass\_filter: Applies a Butterworth filter to remove high-frequency noise from time-domain data.

### **6.2. ansys\_exporter.py**

This class is responsible for converting a Pandas DataFrame into a format that Ansys Mechanical can understand.

* It creates .dat files which are essentially tables of load-vs-time or load-vs-frequency.  
* It also creates a corresponding .inp (APDL script) file that tells Ansys how to read the .dat file and apply it as a load in a transient or harmonic analysis.

### **6.3. plotter.py**

This class isolates the plotting library (Plotly) from the rest of the application. The PlotController tells the Plotter *what* to plot, and the Plotter handles the *how*. This design means you could swap out Plotly for Matplotlib or another library by only changing plotter.py, without touching any of the controller logic.

## **7\. State Management**

### **7.1. Primary Data State (self.df)**

The primary loaded data is stored in MainWindow.df. This is the **single source of truth**. All plotting and analysis operations read from this DataFrame. It is considered immutable after loading; any processing (like filtering) creates a temporary copy, rather than modifying the original df.

### **7.2. Comparison Data State (self.df\_compare)**

Similarly, MainWindow.df\_compare holds the secondary dataset. It's kept separate to avoid confusion with the primary data.

### **7.3. UI State**

The state of all checkboxes, selectors, and input fields across all UI tabs is the "UI State". The controllers read this state at the moment a plot or action is requested to get the user's desired parameters.

## **8\. Conclusion and Future Development**

The WE MechLoad Viewer employs a robust, decoupled architecture based on MVC principles and the Qt signal/slot mechanism. This design promotes:

* **Maintainability**: Code is organized by function, making it easy to locate and modify logic.  
* **Scalability**: New tabs, plots, or export formats can be added by creating new view/controller pairs without disrupting existing functionality.  
* **Testability**: The data and controller layers can be tested independently of the GUI.

**Potential Future Enhancements:**

* **Unit Testing**: Implement unit tests for DataManager, PlotController, ActionHandler, and the analysis modules.  
* **Asynchronous Data Loading**: For very large datasets, the data loading process could be moved to a separate QThread to prevent the GUI from freezing.  
* **User-Defined Calculations**: Add a feature allowing users to create new data columns based on mathematical operations on existing columns (e.g., New\_Force \= I1\_Fx \* 2.5).  
* **Plugin System**: Develop a plugin system that would allow new plot types or data exporters to be added as separate modules.