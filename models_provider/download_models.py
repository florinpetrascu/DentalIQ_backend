import requests



def download_file(url, output_file):
    """
    Descarcă un fișier binar de la un URL și îl salvează local.
    """
    try:
        # Descarcă fișierul
        response = requests.get(url, stream=True)  # Stream pentru fișiere mari
        if response.status_code == 200:
            with open(output_file, "wb") as file:  # Deschide fișierul în modul binar
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Fișierul a fost descărcat cu succes *************** în {output_file}.")
        else:
            print(f"Eroare la descărcare: {response.status_code}")
            return None
    except Exception as e:
        print(f"A apărut o eroare: {e}")
        return None
    return output_file  # Returnează calea către fișierul salvat