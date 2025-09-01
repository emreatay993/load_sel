# File: app/config_manager.py
# This file contains all the Qt Style Sheet (QSS) strings for the application.

TREEVIEW_STYLE = """
    QTreeView {
        background-color: #f7f7f7;
        border: none;
    }
    QTreeView::item {
        padding: 5px;
    }
    QTreeView::item:selected {
        background-color: #00838f;
        color: white;
    }
"""

TABWIDGET_STYLE = """
    QTabBar::tab {
        background: #00838f;
        color: white;
        min-width: 120px;
        padding: 5px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        margin-right: 5px;
    }
    QTabBar::tab:selected {
        background: #00acc1;
        font-weight: normal;
    }
    QTabBar::tab:disabled {
        background: #cccccc;
        color: #777777;
    }
    QTabWidget::pane {
        border-top: 2px solid #ccc;
        border-left: 1px solid #ccc;
        border-right: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
        border-radius: 10px;
        padding: 5px;
    }
"""

GROUPBOX_STYLE = """
    QGroupBox {
        color: #00838f;
        background-color: #f0f0f0;
        border: 1px solid lightgray;
        border-radius: 5px;
        margin-top: 1ex;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
    }
"""

COMPARE_BUTTON_STYLE = """
    QPushButton {
        background-color: #00838f;
        color: white;
        border: 2px solid #006064;
        border-radius: 5px;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #00acc1;
        border-color: #006064;
    }
    QPushButton:pressed {
        background-color: #006064;
        border-color: #004d40;
    }
"""
