# File: main.py

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer

from app.data_manager import DataManager
from app.main_window import MainWindow

if __name__ == "__main__":
    # 1. Create the application instance
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)

    # 2. Create the main objects
    data_manager = DataManager()
    main_window = MainWindow(data_manager)

    # 3. Connect the signal from the data_manager to the slot in the main_window
    data_manager.dataLoaded.connect(main_window.on_data_loaded)

    # 4. Show the main window
    main_window.showMaximized()

    # 5. Trigger the initial data loading process.
    # QTimer is used to ensure the main window is fully shown before the blocking
    # file dialog appears (for better user experience).
    QTimer.singleShot(100, data_manager.load_data_from_directory)

    # 6. Start the application's event loop
    sys.exit(app.exec_())