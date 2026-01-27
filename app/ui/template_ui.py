# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'template.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGraphicsView, QGroupBox,
    QHBoxLayout, QLineEdit, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QSpinBox,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1048, 842)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 50, 951, 681))
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.widget = QWidget(self.groupBox)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(80, 40, 793, 546))
        self.verticalLayout_4 = QVBoxLayout(self.widget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.lineEdit = QLineEdit(self.widget)
        self.lineEdit.setObjectName(u"lineEdit")

        self.horizontalLayout.addWidget(self.lineEdit)

        self.spinBox = QSpinBox(self.widget)
        self.spinBox.setObjectName(u"spinBox")

        self.horizontalLayout.addWidget(self.spinBox)

        self.lineEdit_2 = QLineEdit(self.widget)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.horizontalLayout.addWidget(self.lineEdit_2)

        self.spinBox_2 = QSpinBox(self.widget)
        self.spinBox_2.setObjectName(u"spinBox_2")

        self.horizontalLayout.addWidget(self.spinBox_2)

        self.pushButton = QPushButton(self.widget)
        self.pushButton.setObjectName(u"pushButton")

        self.horizontalLayout.addWidget(self.pushButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.graphicsView = QGraphicsView(self.widget)
        self.graphicsView.setObjectName(u"graphicsView")

        self.verticalLayout_3.addWidget(self.graphicsView)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lineEdit_4 = QLineEdit(self.widget)
        self.lineEdit_4.setObjectName(u"lineEdit_4")

        self.horizontalLayout_2.addWidget(self.lineEdit_4)

        self.spinBox_4 = QSpinBox(self.widget)
        self.spinBox_4.setObjectName(u"spinBox_4")

        self.horizontalLayout_2.addWidget(self.spinBox_4)

        self.pushButton_2 = QPushButton(self.widget)
        self.pushButton_2.setObjectName(u"pushButton_2")

        self.horizontalLayout_2.addWidget(self.pushButton_2)

        self.checkBox_2 = QCheckBox(self.widget)
        self.checkBox_2.setObjectName(u"checkBox_2")

        self.horizontalLayout_2.addWidget(self.checkBox_2)

        self.pushButton_3 = QPushButton(self.widget)
        self.pushButton_3.setObjectName(u"pushButton_3")

        self.horizontalLayout_2.addWidget(self.pushButton_3)

        self.pushButton_4 = QPushButton(self.widget)
        self.pushButton_4.setObjectName(u"pushButton_4")

        self.horizontalLayout_2.addWidget(self.pushButton_4)

        self.pushButton_6 = QPushButton(self.widget)
        self.pushButton_6.setObjectName(u"pushButton_6")

        self.horizontalLayout_2.addWidget(self.pushButton_6)

        self.pushButton_5 = QPushButton(self.widget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        self.pushButton_5.setMaximumSize(QSize(1000202, 16777215))
        font = QFont()
        font.setFamilies([u"MS Gothic"])
        font.setPointSize(20)
        self.pushButton_5.setFont(font)

        self.horizontalLayout_2.addWidget(self.pushButton_5)


        self.verticalLayout_3.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.lineEdit_7 = QLineEdit(self.widget)
        self.lineEdit_7.setObjectName(u"lineEdit_7")

        self.horizontalLayout_6.addWidget(self.lineEdit_7)

        self.spinBox_7 = QSpinBox(self.widget)
        self.spinBox_7.setObjectName(u"spinBox_7")

        self.horizontalLayout_6.addWidget(self.spinBox_7)

        self.pushButton_7 = QPushButton(self.widget)
        self.pushButton_7.setObjectName(u"pushButton_7")

        self.horizontalLayout_6.addWidget(self.pushButton_7)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.lineEdit_5 = QLineEdit(self.widget)
        self.lineEdit_5.setObjectName(u"lineEdit_5")

        self.horizontalLayout_7.addWidget(self.lineEdit_5)

        self.spinBox_5 = QSpinBox(self.widget)
        self.spinBox_5.setObjectName(u"spinBox_5")

        self.horizontalLayout_7.addWidget(self.spinBox_5)

        self.spinBox_8 = QSpinBox(self.widget)
        self.spinBox_8.setObjectName(u"spinBox_8")

        self.horizontalLayout_7.addWidget(self.spinBox_8)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.lineEdit_8 = QLineEdit(self.widget)
        self.lineEdit_8.setObjectName(u"lineEdit_8")

        self.horizontalLayout_8.addWidget(self.lineEdit_8)

        self.spinBox_10 = QSpinBox(self.widget)
        self.spinBox_10.setObjectName(u"spinBox_10")

        self.horizontalLayout_8.addWidget(self.spinBox_10)

        self.spinBox_11 = QSpinBox(self.widget)
        self.spinBox_11.setObjectName(u"spinBox_11")

        self.horizontalLayout_8.addWidget(self.spinBox_11)


        self.verticalLayout.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.lineEdit_6 = QLineEdit(self.widget)
        self.lineEdit_6.setObjectName(u"lineEdit_6")

        self.horizontalLayout_9.addWidget(self.lineEdit_6)

        self.spinBox_6 = QSpinBox(self.widget)
        self.spinBox_6.setObjectName(u"spinBox_6")

        self.horizontalLayout_9.addWidget(self.spinBox_6)

        self.spinBox_9 = QSpinBox(self.widget)
        self.spinBox_9.setObjectName(u"spinBox_9")

        self.horizontalLayout_9.addWidget(self.spinBox_9)


        self.verticalLayout.addLayout(self.horizontalLayout_9)


        self.horizontalLayout_11.addLayout(self.verticalLayout)

        self.checkBox_3 = QCheckBox(self.widget)
        self.checkBox_3.setObjectName(u"checkBox_3")

        self.horizontalLayout_11.addWidget(self.checkBox_3)

        self.pushButton_8 = QPushButton(self.widget)
        self.pushButton_8.setObjectName(u"pushButton_8")

        self.horizontalLayout_11.addWidget(self.pushButton_8)

        self.pushButton_9 = QPushButton(self.widget)
        self.pushButton_9.setObjectName(u"pushButton_9")

        self.horizontalLayout_11.addWidget(self.pushButton_9)

        self.pushButton_10 = QPushButton(self.widget)
        self.pushButton_10.setObjectName(u"pushButton_10")

        self.horizontalLayout_11.addWidget(self.pushButton_10)

        self.pushButton_11 = QPushButton(self.widget)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setEnabled(True)
        sizePolicy.setHeightForWidth(self.pushButton_11.sizePolicy().hasHeightForWidth())
        self.pushButton_11.setSizePolicy(sizePolicy)
        self.pushButton_11.setMaximumSize(QSize(1000202, 16777215))
        self.pushButton_11.setFont(font)

        self.horizontalLayout_11.addWidget(self.pushButton_11)


        self.verticalLayout_3.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_37 = QHBoxLayout()
        self.horizontalLayout_37.setObjectName(u"horizontalLayout_37")
        self.lineEdit_9 = QLineEdit(self.widget)
        self.lineEdit_9.setObjectName(u"lineEdit_9")

        self.horizontalLayout_37.addWidget(self.lineEdit_9)

        self.spinBox_12 = QSpinBox(self.widget)
        self.spinBox_12.setObjectName(u"spinBox_12")

        self.horizontalLayout_37.addWidget(self.spinBox_12)

        self.pushButton_12 = QPushButton(self.widget)
        self.pushButton_12.setObjectName(u"pushButton_12")

        self.horizontalLayout_37.addWidget(self.pushButton_12)

        self.checkBox_4 = QCheckBox(self.widget)
        self.checkBox_4.setObjectName(u"checkBox_4")

        self.horizontalLayout_37.addWidget(self.checkBox_4)

        self.pushButton_13 = QPushButton(self.widget)
        self.pushButton_13.setObjectName(u"pushButton_13")

        self.horizontalLayout_37.addWidget(self.pushButton_13)

        self.pushButton_14 = QPushButton(self.widget)
        self.pushButton_14.setObjectName(u"pushButton_14")

        self.horizontalLayout_37.addWidget(self.pushButton_14)

        self.pushButton_15 = QPushButton(self.widget)
        self.pushButton_15.setObjectName(u"pushButton_15")

        self.horizontalLayout_37.addWidget(self.pushButton_15)

        self.pushButton_16 = QPushButton(self.widget)
        self.pushButton_16.setObjectName(u"pushButton_16")
        self.pushButton_16.setEnabled(True)
        sizePolicy.setHeightForWidth(self.pushButton_16.sizePolicy().hasHeightForWidth())
        self.pushButton_16.setSizePolicy(sizePolicy)
        self.pushButton_16.setMaximumSize(QSize(1000202, 16777215))
        self.pushButton_16.setFont(font)

        self.horizontalLayout_37.addWidget(self.pushButton_16)


        self.verticalLayout_3.addLayout(self.horizontalLayout_37)

        self.horizontalLayout_38 = QHBoxLayout()
        self.horizontalLayout_38.setObjectName(u"horizontalLayout_38")
        self.pushButton_19 = QPushButton(self.widget)
        self.pushButton_19.setObjectName(u"pushButton_19")

        self.horizontalLayout_38.addWidget(self.pushButton_19)

        self.pushButton_17 = QPushButton(self.widget)
        self.pushButton_17.setObjectName(u"pushButton_17")

        self.horizontalLayout_38.addWidget(self.pushButton_17)


        self.verticalLayout_3.addLayout(self.horizontalLayout_38)


        self.verticalLayout_4.addLayout(self.verticalLayout_3)

        self.horizontalLayout_39 = QHBoxLayout()
        self.horizontalLayout_39.setObjectName(u"horizontalLayout_39")
        self.lineEdit_41 = QLineEdit(self.widget)
        self.lineEdit_41.setObjectName(u"lineEdit_41")

        self.horizontalLayout_39.addWidget(self.lineEdit_41)

        self.spinBox_56 = QSpinBox(self.widget)
        self.spinBox_56.setObjectName(u"spinBox_56")

        self.horizontalLayout_39.addWidget(self.spinBox_56)

        self.horizontalSpacer_11 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_39.addItem(self.horizontalSpacer_11)


        self.verticalLayout_4.addLayout(self.horizontalLayout_39)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1048, 23))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0434\u0430\u0439\u0442\u0435 \u0440\u0430\u0437\u043c\u0435\u0440\u044b \u043f\u043e\u043b\u043e\u0442\u043d\u0430", None))
        self.lineEdit.setText(QCoreApplication.translate("MainWindow", u"\u0428\u0438\u0440\u0438\u043d\u0430:", None))
        self.lineEdit_2.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0441\u043e\u0442\u0430:", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c", None))
        self.lineEdit_4.setText(QCoreApplication.translate("MainWindow", u"\u041a\u043e\u043b-\u0432\u043e \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u043e\u0432", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0435\u0449\u0451", None))
        self.pushButton_4.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0432\u0441\u0451", None))
        self.pushButton_6.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.pushButton_5.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.lineEdit_7.setText(QCoreApplication.translate("MainWindow", u"\u041a\u043e\u043b-\u0432\u043e \u0440\u0430\u0437\u043b\u043e\u043c\u043e\u0432", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c", None))
        self.lineEdit_5.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0434\u043b\u0438\u043d\u044b", None))
        self.lineEdit_8.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0443\u0433\u043b\u043e\u0432", None))
        self.lineEdit_6.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0430\u043f\u043b\u0438\u0442\u0443\u0434\u044b", None))
        self.checkBox_3.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.pushButton_8.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0435\u0449\u0451", None))
        self.pushButton_9.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0432\u0441\u0451", None))
        self.pushButton_10.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.pushButton_11.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.lineEdit_9.setText(QCoreApplication.translate("MainWindow", u"\u0418\u0441\u043a\u0430\u0436\u0435\u043d\u0438\u0435", None))
        self.pushButton_12.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c", None))
        self.checkBox_4.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.pushButton_13.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0436\u0430\u0442\u044c", None))
        self.pushButton_14.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u0442\u044f\u043d\u0443\u0442\u044c", None))
        self.pushButton_15.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.pushButton_16.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.pushButton_19.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u043c\u0430\u0441\u043a\u0443", None))
        self.pushButton_17.setText(QCoreApplication.translate("MainWindow", u"GAN \u0421\u0435\u0439\u0441\u043c\u0438\u043a\u0430", None))
        self.lineEdit_41.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0435\u043f\u0440\u043e\u0437\u0440\u0430\u043d\u043e\u0441\u0442\u044c \u0441\u0435\u0439\u0441\u043c\u0438\u043a\u0438:", None))
    # retranslateUi

