# File: app/ui/directory_tree_dock.py

import os
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDockWidget, QTreeView, QFileSystemModel


class DirectoryTreeDock(QDockWidget):
    # Signal to emit a list of selected directory paths
    directories_selected = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__("Data Folders", parent)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._setup_ui()
        self.tree_view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def _setup_ui(self):
        self.tree_view = QTreeView(self)
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)

        # Set a default root path
        self.set_root_path(os.getcwd())

        self.tree_view.setModel(self.file_model)
        self.tree_view.setColumnHidden(1, True)
        self.tree_view.setColumnHidden(2, True)
        self.tree_view.setColumnHidden(3, True)
        self.tree_view.header().setVisible(False)
        self.tree_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setWidget(self.tree_view)

    def set_root_path(self, path):
        parent_folder = os.path.dirname(path) if path else os.getcwd()
        self.file_model.setRootPath(parent_folder)
        self.tree_view.setRootIndex(self.file_model.index(parent_folder))

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def _on_selection_changed(self, selected, deselected):
        """
        Gathers selected directory paths and emits a signal.
        """
        indexes = self.tree_view.selectionModel().selectedRows()
        if not indexes:
            return

        # Get the file paths from the model indexes
        folder_paths = [self.file_model.filePath(index) for index in indexes]

        # Filter out any selections that are not directories (e.g., if you change the filter later)
        folder_paths = [path for path in folder_paths if os.path.isdir(path)]

        if folder_paths:
            self.directories_selected.emit(folder_paths)
