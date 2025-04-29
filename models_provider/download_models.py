import requests
import os

# def download_file(url, output_file):
#     """
#     Descarcă un fișier binar de la un URL și îl salvează local.
#     """
#     try:
#         # Descarcă fișierul
#         response = requests.get(url, stream=True)  # Stream pentru fișiere mari
#         if response.status_code == 200:
#             with open(output_file, "wb") as file:  # Deschide fișierul în modul binar
#                 for chunk in response.iter_content(chunk_size=8192):
#                     file.write(chunk)
#             print(f"Fișierul a fost descărcat cu succes *************** în {output_file}.")
#         else:
#             print(f"Eroare la descărcare: {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"A apărut o eroare: {e}")
#         return None
#     return output_file  # Returnează calea către fișierul salvat

# def download_file(url, output_file):
#     """
#     Descarcă un fișier binar de la un URL (inclusiv cu redirect) și îl salvează local.
#     """
#     try:
#         with requests.Session() as session:
#             # GitHub sometimes needs special headers
#             headers = {'User-Agent': 'Mozilla/5.0'}
#             response = session.get(url, stream=True, headers=headers, allow_redirects=True)
#             response.raise_for_status()  # va arunca excepție dacă status != 200
#
#             with open(output_file, 'wb') as file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         file.write(chunk)
#
#         print(f"✅ Fișierul descărcat în {output_file}")
#         return output_file
#
#     except Exception as e:
#         print(f"❌ Eroare la descărcare: {e}")
#         return None


def download_file(url, output_file):
    """
    Descarcă un fișier binar de la un URL (inclusiv cu redirect) și îl salvează local,
    cu gestionare a erorilor și posibilitatea de a relua descărcarea în caz de întrerupere.
    """
    try:
        # Verifică dacă fișierul există deja parțial și află dimensiunea curentă
        resume_header = {}
        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            resume_header = {'Range': f'bytes={current_size}-'}  # Continuă de la unde s-a oprit
        else:
            current_size = 0

        with requests.Session() as session:
            # GitHub și alte site-uri uneori necesită User-Agent special
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = session.get(url, stream=True, headers={**headers, **resume_header}, allow_redirects=True)
            response.raise_for_status()  # aruncă excepție dacă status != 200

            # Obține dimensiunea totală a fișierului, dacă serverul oferă această informație
            total_size = int(response.headers.get('Content-Length', 0)) + current_size

            with open(output_file, 'ab') as file:  # 'ab' pentru a adăuga la fișierul existent
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        current_size += len(chunk)
                        print(f"Descărcat {current_size}/{total_size} bytes...", end="\r")

        print(f"\n✅ Fișierul descărcat complet în {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Eroare la descărcare: {e}")
        return None