import requests

# URL of the CSV file
url = 'https://raw.githubusercontent.com/fourthrevlxd/cam_dsb/main/engine.csv'

# Send GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Save the content to a local file
    with open('../data/ship-data.csv', 'wb') as file:
        file.write(response.content)
    print("CSV file saved successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")