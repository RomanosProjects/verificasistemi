import cv2
import numpy as np
from sklearn.cluster import KMeans

def trova_colore_medio(immagine, maschera):
    colori = immagine[maschera == 255]
    if len(colori) == 0:
        return (0, 0, 0)
    kmeans = KMeans(n_clusters=1, n_init='auto').fit(colori)
    colore_medio = kmeans.cluster_centers_[0]
    return tuple(map(int, colore_medio))

def calcola_geometria(path_immagine):
    img = cv2.imread(path_immagine)
    if img is None:
        print("Immagine non trovata.")
        return

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Trovate {len(contours)} forme.")

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 100:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        maschera = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(maschera, [c], -1, 255, -1)
        colore = trova_colore_medio(img, maschera)

        figura = "Sconosciuta"
        if len(approx) == 3:
            figura = "Triangolo"
            perimetro = peri
            info = f"{figura}: A={area:.1f}, P={perimetro:.1f}"
        elif len(approx) == 4:
            figura = "Rettangolo"
            perimetro = peri
            x, y, w, h = cv2.boundingRect(approx)
            info = f"{figura}: A={w*h}, P={2*(w+h)}"
        else:
            figura = "Cerchio"
            raggio = np.sqrt(area / np.pi)
            circonferenza = 2 * np.pi * raggio
            info = f"{figura}: A={area:.1f}, C={circonferenza:.1f}"

        cv2.drawContours(output, [c], -1, (0,255,0), 2)
        cv2.circle(output, (cx, cy), 3, (255,0,0), -1)
        cv2.putText(output, info, (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(output, f"Colore: {colore}", (cx - 60, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        print(f"{figura} #{i+1} - Area: {area:.2f}, Centro: ({cx},{cy}), Colore: {colore}")

    cv2.imwrite("forme_annotate.jpg", output)
    cv2.imshow("Risultato", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

calcola_geometria("figure.jpg")
