Skapar normaldata för (programmet)

Öppna (detta program) och välj för vilken undersökning du vill skapa nytt normalmaterial för (O-15 vatten / O-15 vatten (stress/basline) / F-18 PE2I / C-11 PE2I)

Välj zip mappen (eller en vanlig mapp) med sparade (K_1 eller R_I) .nni/.nii.gz bilder (måste vara sparade från (programmet) och i MNI space)

Välj mapp där normalmaterialet ska vara

Klicka på 'Starta' (och vänta medans programmet beräknar ny normaldata)

När programmet är klart dyker det upp en ny (window) som säger att det är klart

Klicka på tillbaka om du vill skapa normalmaterial för någon anna undersökning eller stäng sidan

(Allt gammalt normalmaterial sparas i samma mapp med tillägget '-backup', om det nya normalmaterialet inte blir bra kan man gå in i mappen och ta bort de nya och ta bort '-backup' från de gamla)

(Skrivet i python 3.9)

För att skapa .exe fil:

1. Ladda ner auto-py-tp-exe

2. Skriv in "auto-py-to-exe" i din virituella miljö med python 3.9

3. Script location: Välj "Create_mean_and_std.py"

4. Välj "One File"

5. Välj "Window Based (hide the console)"

6. Lägg till icon (frivillig) (i .ico format)

7. Additional Files: Välj "Add Files" och välj alla filer och scripts som finns här förutom "Create_mean_and_std.py"

8. Strata genom att trycka på "CONVERT .PY TO .EXE"

9. Filen genereras och finns sedan under "C:\Users\'namn'\output"
