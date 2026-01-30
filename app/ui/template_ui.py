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
    QHBoxLayout, QLabel, QMainWindow, QMenuBar,
    QPushButton, QScrollArea, QSizePolicy, QSpacerItem, QSpinBox,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1048, 842)
        MainWindow.setMinimumSize(QSize(860, 1000))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralLayout = QVBoxLayout(self.centralwidget)
        self.centralLayout.setObjectName(u"centralLayout")
        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollAreaLayout.setObjectName(u"scrollAreaLayout")
        self.groupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.groupBoxLayout = QVBoxLayout(self.groupBox)
        self.groupBoxLayout.setObjectName(u"groupBoxLayout")
        self.layoutWidget = QWidget(self.groupBox)
        self.layoutWidget.setObjectName(u"layoutWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.groupBox.setSizePolicy(sizePolicy)
        self.layoutWidget.setSizePolicy(sizePolicy)
        self.elementsvVerticalLayout = QVBoxLayout(self.layoutWidget)
        self.elementsvVerticalLayout.setObjectName(u"elementsvVerticalLayout")
        self.elementsvVerticalLayout.setContentsMargins(0, 0, 0, 0)
        self.canvasHorizontalLayout = QHBoxLayout()
        self.canvasHorizontalLayout.setObjectName(u"canvasHorizontalLayout")
        self.leftHorizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.canvasHorizontalLayout.addItem(self.leftHorizontalSpacer)

        self.WidthLabel = QLabel(self.layoutWidget)
        self.WidthLabel.setObjectName(u"WidthLabel")

        self.canvasHorizontalLayout.addWidget(self.WidthLabel)

        self.widthSpinBox = QSpinBox(self.layoutWidget)
        self.widthSpinBox.setObjectName(u"widthSpinBox")

        self.canvasHorizontalLayout.addWidget(self.widthSpinBox)

        self.HeightLabel = QLabel(self.layoutWidget)
        self.HeightLabel.setObjectName(u"HeightLabel")

        self.canvasHorizontalLayout.addWidget(self.HeightLabel)

        self.heightSpinBox = QSpinBox(self.layoutWidget)
        self.heightSpinBox.setObjectName(u"heightSpinBox")

        self.canvasHorizontalLayout.addWidget(self.heightSpinBox)

        self.canvasSizeButton = QPushButton(self.layoutWidget)
        self.canvasSizeButton.setObjectName(u"canvasSizeButton")

        self.canvasHorizontalLayout.addWidget(self.canvasSizeButton)

        self.rightHorizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.canvasHorizontalLayout.addItem(self.rightHorizontalSpacer)


        self.elementsvVerticalLayout.addLayout(self.canvasHorizontalLayout)

        self.canvasGraphicsView = QGraphicsView(self.layoutWidget)
        self.canvasGraphicsView.setObjectName(u"canvasGraphicsView")
        self.canvasGraphicsView.setSizePolicy(sizePolicy)

        self.elementsvVerticalLayout.addWidget(self.canvasGraphicsView)

        self.horizonsHorizontalLayout = QHBoxLayout()
        self.horizonsHorizontalLayout.setObjectName(u"horizonsHorizontalLayout")
        self.horizonsLabel = QLabel(self.layoutWidget)
        self.horizonsLabel.setObjectName(u"horizonsLabel")

        self.horizonsHorizontalLayout.addWidget(self.horizonsLabel)

        self.horizonsSpinBox = QSpinBox(self.layoutWidget)
        self.horizonsSpinBox.setObjectName(u"horizonsSpinBox")

        self.horizonsHorizontalLayout.addWidget(self.horizonsSpinBox)

        self.horizonsCreateButton = QPushButton(self.layoutWidget)
        self.horizonsCreateButton.setObjectName(u"horizonsCreateButton")

        self.horizonsHorizontalLayout.addWidget(self.horizonsCreateButton)

        self.horizonsCheckBox = QCheckBox(self.layoutWidget)
        self.horizonsCheckBox.setObjectName(u"horizonsCheckBox")

        self.horizonsHorizontalLayout.addWidget(self.horizonsCheckBox)

        self.horizonsAddButton = QPushButton(self.layoutWidget)
        self.horizonsAddButton.setObjectName(u"horizonsAddButton")

        self.horizonsHorizontalLayout.addWidget(self.horizonsAddButton)

        self.horizonsClearButton = QPushButton(self.layoutWidget)
        self.horizonsClearButton.setObjectName(u"horizonsClearButton")

        self.horizonsHorizontalLayout.addWidget(self.horizonsClearButton)

        self.horizonsSaveButton = QPushButton(self.layoutWidget)
        self.horizonsSaveButton.setObjectName(u"horizonsSaveButton")

        self.horizonsHorizontalLayout.addWidget(self.horizonsSaveButton)

        self.horizonsDrawButton = QPushButton(self.layoutWidget)
        self.horizonsDrawButton.setObjectName(u"horizonsDrawButton")
        self.horizonsDrawButton.setEnabled(True)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizonsDrawButton.sizePolicy().hasHeightForWidth())
        self.horizonsDrawButton.setSizePolicy(sizePolicy)
        self.horizonsDrawButton.setMaximumSize(QSize(1000202, 16777215))
        font = QFont()
        font.setFamilies([u"MS Gothic"])
        font.setPointSize(20)
        self.horizonsDrawButton.setFont(font)

        self.horizonsHorizontalLayout.addWidget(self.horizonsDrawButton)


        self.elementsvVerticalLayout.addLayout(self.horizonsHorizontalLayout)

        self.riftsHorizontalLayout = QHBoxLayout()
        self.riftsHorizontalLayout.setObjectName(u"riftsHorizontalLayout")
        self.riftsVerticalLayout = QVBoxLayout()
        self.riftsVerticalLayout.setObjectName(u"riftsVerticalLayout")
        self.lengthHorizontalLayout = QHBoxLayout()
        self.lengthHorizontalLayout.setObjectName(u"lengthHorizontalLayout")
        self.lengthLabel = QLabel(self.layoutWidget)
        self.lengthLabel.setObjectName(u"lengthLabel")

        self.lengthHorizontalLayout.addWidget(self.lengthLabel)

        self.lengthFromSpinBox = QSpinBox(self.layoutWidget)
        self.lengthFromSpinBox.setObjectName(u"lengthFromSpinBox")

        self.lengthHorizontalLayout.addWidget(self.lengthFromSpinBox)

        self.lengthToSpinBox = QSpinBox(self.layoutWidget)
        self.lengthToSpinBox.setObjectName(u"lengthToSpinBox")

        self.lengthHorizontalLayout.addWidget(self.lengthToSpinBox)


        self.riftsVerticalLayout.addLayout(self.lengthHorizontalLayout)

        self.anglesHorizontalLayout = QHBoxLayout()
        self.anglesHorizontalLayout.setObjectName(u"anglesHorizontalLayout")
        self.anglesLabel = QLabel(self.layoutWidget)
        self.anglesLabel.setObjectName(u"anglesLabel")

        self.anglesHorizontalLayout.addWidget(self.anglesLabel)

        self.anglesFromSpinBox = QSpinBox(self.layoutWidget)
        self.anglesFromSpinBox.setObjectName(u"anglesFromSpinBox")

        self.anglesHorizontalLayout.addWidget(self.anglesFromSpinBox)

        self.anglesToSpinBox = QSpinBox(self.layoutWidget)
        self.anglesToSpinBox.setObjectName(u"anglesToSpinBox")

        self.anglesHorizontalLayout.addWidget(self.anglesToSpinBox)


        self.riftsVerticalLayout.addLayout(self.anglesHorizontalLayout)

        self.amplitudeHorizontalLayout = QHBoxLayout()
        self.amplitudeHorizontalLayout.setObjectName(u"amplitudeHorizontalLayout")
        self.amplitudeLabel = QLabel(self.layoutWidget)
        self.amplitudeLabel.setObjectName(u"amplitudeLabel")

        self.amplitudeHorizontalLayout.addWidget(self.amplitudeLabel)

        self.amplitudeFromSpinBox = QSpinBox(self.layoutWidget)
        self.amplitudeFromSpinBox.setObjectName(u"amplitudeFromSpinBox")

        self.amplitudeHorizontalLayout.addWidget(self.amplitudeFromSpinBox)

        self.amplitudeToSpinBox = QSpinBox(self.layoutWidget)
        self.amplitudeToSpinBox.setObjectName(u"amplitudeToSpinBox")

        self.amplitudeHorizontalLayout.addWidget(self.amplitudeToSpinBox)


        self.riftsVerticalLayout.addLayout(self.amplitudeHorizontalLayout)

        self.riftsAmountHorizontalLayout = QHBoxLayout()
        self.riftsAmountHorizontalLayout.setObjectName(u"riftsAmountHorizontalLayout")
        self.riftsAmountLabel = QLabel(self.layoutWidget)
        self.riftsAmountLabel.setObjectName(u"riftsAmountLabel")

        self.riftsAmountHorizontalLayout.addWidget(self.riftsAmountLabel)

        self.riftsAmountSpinBox = QSpinBox(self.layoutWidget)
        self.riftsAmountSpinBox.setObjectName(u"riftsAmountSpinBox")

        self.riftsAmountHorizontalLayout.addWidget(self.riftsAmountSpinBox)

        self.riftsCreateButton = QPushButton(self.layoutWidget)
        self.riftsCreateButton.setObjectName(u"riftsCreateButton")

        self.riftsAmountHorizontalLayout.addWidget(self.riftsCreateButton)


        self.riftsVerticalLayout.addLayout(self.riftsAmountHorizontalLayout)


        self.riftsHorizontalLayout.addLayout(self.riftsVerticalLayout)

        self.riftsCheckBox = QCheckBox(self.layoutWidget)
        self.riftsCheckBox.setObjectName(u"riftsCheckBox")

        self.riftsHorizontalLayout.addWidget(self.riftsCheckBox)

        self.riftsAddButton = QPushButton(self.layoutWidget)
        self.riftsAddButton.setObjectName(u"riftsAddButton")

        self.riftsHorizontalLayout.addWidget(self.riftsAddButton)

        self.riftsClearButton = QPushButton(self.layoutWidget)
        self.riftsClearButton.setObjectName(u"riftsClearButton")

        self.riftsHorizontalLayout.addWidget(self.riftsClearButton)

        self.riftsSaveButton = QPushButton(self.layoutWidget)
        self.riftsSaveButton.setObjectName(u"riftsSaveButton")

        self.riftsHorizontalLayout.addWidget(self.riftsSaveButton)

        self.riftsDrawButton = QPushButton(self.layoutWidget)
        self.riftsDrawButton.setObjectName(u"riftsDrawButton")
        self.riftsDrawButton.setEnabled(True)
        sizePolicy.setHeightForWidth(self.riftsDrawButton.sizePolicy().hasHeightForWidth())
        self.riftsDrawButton.setSizePolicy(sizePolicy)
        self.riftsDrawButton.setMaximumSize(QSize(1000202, 16777215))
        self.riftsDrawButton.setFont(font)

        self.riftsHorizontalLayout.addWidget(self.riftsDrawButton)


        self.elementsvVerticalLayout.addLayout(self.riftsHorizontalLayout)

        self.distortionHorizontalLayout = QHBoxLayout()
        self.distortionHorizontalLayout.setObjectName(u"distortionHorizontalLayout")
        self.distortionLabel = QLabel(self.layoutWidget)
        self.distortionLabel.setObjectName(u"distortionLabel")

        self.distortionHorizontalLayout.addWidget(self.distortionLabel)

        self.distortionSpinBox = QSpinBox(self.layoutWidget)
        self.distortionSpinBox.setObjectName(u"distortionSpinBox")

        self.distortionHorizontalLayout.addWidget(self.distortionSpinBox)

        self.distortionApplyButton = QPushButton(self.layoutWidget)
        self.distortionApplyButton.setObjectName(u"distortionApplyButton")

        self.distortionHorizontalLayout.addWidget(self.distortionApplyButton)

        self.distortionCheckBox = QCheckBox(self.layoutWidget)
        self.distortionCheckBox.setObjectName(u"distortionCheckBox")

        self.distortionHorizontalLayout.addWidget(self.distortionCheckBox)

        self.distortionCompressButton = QPushButton(self.layoutWidget)
        self.distortionCompressButton.setObjectName(u"distortionCompressButton")

        self.distortionHorizontalLayout.addWidget(self.distortionCompressButton)

        self.distortionStretchButton = QPushButton(self.layoutWidget)
        self.distortionStretchButton.setObjectName(u"distortionStretchButton")

        self.distortionHorizontalLayout.addWidget(self.distortionStretchButton)

        self.distortionSaveButton = QPushButton(self.layoutWidget)
        self.distortionSaveButton.setObjectName(u"distortionSaveButton")

        self.distortionHorizontalLayout.addWidget(self.distortionSaveButton)

        self.distortionDrawButton = QPushButton(self.layoutWidget)
        self.distortionDrawButton.setObjectName(u"distortionDrawButton")
        self.distortionDrawButton.setEnabled(True)
        sizePolicy.setHeightForWidth(self.distortionDrawButton.sizePolicy().hasHeightForWidth())
        self.distortionDrawButton.setSizePolicy(sizePolicy)
        self.distortionDrawButton.setMaximumSize(QSize(1000202, 16777215))
        self.distortionDrawButton.setFont(font)

        self.distortionHorizontalLayout.addWidget(self.distortionDrawButton)


        self.elementsvVerticalLayout.addLayout(self.distortionHorizontalLayout)

        self.actionsHorizontalLayout = QHBoxLayout()
        self.actionsHorizontalLayout.setObjectName(u"actionsHorizontalLayout")
        self.saveMaskButton = QPushButton(self.layoutWidget)
        self.saveMaskButton.setObjectName(u"saveMaskButton")

        self.actionsHorizontalLayout.addWidget(self.saveMaskButton)

        self.GANSeismicButton = QPushButton(self.layoutWidget)
        self.GANSeismicButton.setObjectName(u"GANSeismicButton")

        self.actionsHorizontalLayout.addWidget(self.GANSeismicButton)


        self.elementsvVerticalLayout.addLayout(self.actionsHorizontalLayout)

        self.opacityHorizontalLayout = QHBoxLayout()
        self.opacityHorizontalLayout.setObjectName(u"opacityHorizontalLayout")
        self.opacityLabel = QLabel(self.layoutWidget)
        self.opacityLabel.setObjectName(u"opacityLabel")

        self.opacityHorizontalLayout.addWidget(self.opacityLabel)

        self.opacitySpinBox = QSpinBox(self.layoutWidget)
        self.opacitySpinBox.setObjectName(u"opacitySpinBox")

        self.opacityHorizontalLayout.addWidget(self.opacitySpinBox)

        self.rightOpacityHorizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.opacityHorizontalLayout.addItem(self.rightOpacityHorizontalSpacer)


        self.elementsvVerticalLayout.addLayout(self.opacityHorizontalLayout)

        self.groupBoxLayout.addWidget(self.layoutWidget)
        self.scrollAreaLayout.addWidget(self.groupBox)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.centralLayout.addWidget(self.scrollArea)

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
        self.WidthLabel.setText(QCoreApplication.translate("MainWindow", u"\u0428\u0438\u0440\u0438\u043d\u0430:", None))
        self.HeightLabel.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0441\u043e\u0442\u0430:", None))
        self.canvasSizeButton.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c", None))
        self.horizonsLabel.setText(QCoreApplication.translate("MainWindow", u"\u041a\u043e\u043b-\u0432\u043e \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u043e\u0432", None))
        self.horizonsCreateButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c", None))
        self.horizonsCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.horizonsAddButton.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0435\u0449\u0451", None))
        self.horizonsClearButton.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0432\u0441\u0451", None))
        self.horizonsSaveButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.horizonsDrawButton.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.lengthLabel.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0434\u043b\u0438\u043d\u044b", None))
        self.anglesLabel.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0443\u0433\u043b\u043e\u0432", None))
        self.amplitudeLabel.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0437\u0431\u0440\u043e\u0441 \u0430\u043f\u043b\u0438\u0442\u0443\u0434\u044b", None))
        self.riftsAmountLabel.setText(QCoreApplication.translate("MainWindow", u"\u041a\u043e\u043b-\u0432\u043e \u0440\u0430\u0437\u043b\u043e\u043c\u043e\u0432", None))
        self.riftsCreateButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c", None))
        self.riftsCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.riftsAddButton.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0431\u0430\u0432\u0438\u0442\u044c \u0435\u0449\u0451", None))
        self.riftsClearButton.setText(QCoreApplication.translate("MainWindow", u"\u041e\u0447\u0438\u0441\u0442\u0438\u0442\u044c \u0432\u0441\u0451", None))
        self.riftsSaveButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.riftsDrawButton.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.distortionLabel.setText(QCoreApplication.translate("MainWindow", u"\u0418\u0441\u043a\u0430\u0436\u0435\u043d\u0438\u0435", None))
        self.distortionApplyButton.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0440\u0438\u043c\u0435\u043d\u0438\u0442\u044c", None))
        self.distortionCheckBox.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0437\u0434\u0430\u0442\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e ", None))
        self.distortionCompressButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0436\u0430\u0442\u044c", None))
        self.distortionStretchButton.setText(QCoreApplication.translate("MainWindow", u"\u0420\u0430\u0441\u0442\u044f\u043d\u0443\u0442\u044c", None))
        self.distortionSaveButton.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c", None))
        self.distortionDrawButton.setText(QCoreApplication.translate("MainWindow", u"\u270d\ufe0f", None))
        self.saveMaskButton.setText(QCoreApplication.translate("MainWindow", "Наложить маску", None))
        self.GANSeismicButton.setText(QCoreApplication.translate("MainWindow", u"GAN \u0421\u0435\u0439\u0441\u043c\u0438\u043a\u0430", None))
        self.opacityLabel.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0435\u043f\u0440\u043e\u0437\u0440\u0430\u043d\u043e\u0441\u0442\u044c \u0441\u0435\u0439\u0441\u043c\u0438\u043a\u0438:", None))
    # retranslateUi

