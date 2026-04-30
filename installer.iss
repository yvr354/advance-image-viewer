[Setup]
AppName=VyuhaAI Image Viewer
AppVersion=1.0.0
AppPublisher=VyuhaAI
AppPublisherURL=https://github.com/yvr354/advance-image-viewer
DefaultDirName={autopf}\VyuhaAI Image Viewer
DefaultGroupName=VyuhaAI Image Viewer
OutputDir=dist
OutputBaseFilename=VyuhaAI_ImageViewer_Setup_v1.0
SetupIconFile=resources\icons\logo.ico
WizardImageFile=resources\icons\logo.png
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checked

[Files]
Source: "dist\VyuhaAI_ImageViewer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VyuhaAI Image Viewer";  Filename: "{app}\VyuhaAI_ImageViewer.exe"; IconFilename: "{app}\VyuhaAI_ImageViewer.exe"
Name: "{group}\Uninstall VyuhaAI Image Viewer"; Filename: "{uninstallexe}"
Name: "{autodesktop}\VyuhaAI Image Viewer"; Filename: "{app}\VyuhaAI_ImageViewer.exe"; Tasks: desktopicon; IconFilename: "{app}\VyuhaAI_ImageViewer.exe"

[Run]
Filename: "{app}\VyuhaAI_ImageViewer.exe"; Description: "Launch VyuhaAI Image Viewer"; Flags: nowait postinstall skipifsilent
