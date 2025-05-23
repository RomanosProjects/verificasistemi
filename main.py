import cv2
import numpy as np

def calcola_aree(path_immagine):
    # Carica e prepara l'immagine
    image = cv2.imread(path_immagine)
    if image is None:
        print("Immagine non trovata. Controlla il percorso.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Trova i contorni
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Trovate {len(contours)} forme.")

    output = image.copy()
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostra area sull'immagine
        text = f"{int(area)} px"
        cv2.putText(output, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"Forma #{i + 1}: Area = {area:.2f}")

    # Mostra e salva il risultato
    cv2.imshow("Forme rilevate con aree", output)
    cv2.imwrite("output_forme_con_aree.jpg", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Esegui
calcola_aree("Sistemi/romano_negro.jpg")
