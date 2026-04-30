[Setup]
AppName=VyuhaAI Image Viewer
AppVersion=1.0.0
AppPublisher=VyuhaAI
AppPublisherURL=https://github.com/yvr354/advance-image-viewer
DefaultDirName={autopf}\VyuhaAI Image Viewer
DefaultGroupName=VyuhaAI Image Viewer
OutputDir=dist
OutputBaseFilename=VyuhaAI_ImageViewer_Setup_v1.0
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
SetupIconFile=resources\icons\logo.ico
ArchitecturesInstallIn64BitMode=x64
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";      Description: "Create a desktop shortcut";          GroupDescription: "Additional icons:"; Flags: checked
Name: "assoctiff";        Description: "Open .tiff / .tif files with VyuhaAI"; GroupDescription: "File associations:"; Flags: checked
Name: "assocpng";         Description: "Open .png files with VyuhaAI";        GroupDescription: "File associations:"; Flags: checked
Name: "assocbmp";         Description: "Open .bmp files with VyuhaAI";        GroupDescription: "File associations:"; Flags: checked
Name: "assocjpg";         Description: "Open .jpg / .jpeg files with VyuhaAI";GroupDescription: "File associations:"; Flags: checked
Name: "assocpgm";         Description: "Open .pgm / .ppm files with VyuhaAI"; GroupDescription: "File associations:"

[Files]
Source: "dist\VyuhaAI_ImageViewer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; Register app for "Open With" in right-click menu for all image types
Root: HKCU; Subkey: "Software\Classes\VyuhaAI.ImageFile"; ValueType: string; ValueName: ""; ValueData: "VyuhaAI Image Viewer"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\VyuhaAI.ImageFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\VyuhaAI_ImageViewer.exe,0"
Root: HKCU; Subkey: "Software\Classes\VyuhaAI.ImageFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\VyuhaAI_ImageViewer.exe"" ""%1"""

; .tiff / .tif
Root: HKCU; Subkey: "Software\Classes\.tiff"; ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assoctiff; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.tif";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assoctiff; Flags: uninsdeletevalue

; .png
Root: HKCU; Subkey: "Software\Classes\.png";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocpng;  Flags: uninsdeletevalue

; .bmp
Root: HKCU; Subkey: "Software\Classes\.bmp";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocbmp;  Flags: uninsdeletevalue

; .jpg / .jpeg
Root: HKCU; Subkey: "Software\Classes\.jpg";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocjpg;  Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.jpeg"; ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocjpg;  Flags: uninsdeletevalue

; .pgm / .ppm
Root: HKCU; Subkey: "Software\Classes\.pgm";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocpgm;  Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\.ppm";  ValueType: string; ValueName: ""; ValueData: "VyuhaAI.ImageFile"; Tasks: assocpgm;  Flags: uninsdeletevalue

[Icons]
Name: "{group}\VyuhaAI Image Viewer";            Filename: "{app}\VyuhaAI_ImageViewer.exe"
Name: "{group}\Uninstall VyuhaAI Image Viewer";  Filename: "{uninstallexe}"
Name: "{autodesktop}\VyuhaAI Image Viewer";      Filename: "{app}\VyuhaAI_ImageViewer.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VyuhaAI_ImageViewer.exe"; Description: "Launch VyuhaAI Image Viewer"; Flags: nowait postinstall skipifsilent
