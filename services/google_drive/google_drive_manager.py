import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload

class GoogleDriveManager:
    def __init__(self, service_account_file):
        """Initialize the Google Drive Manager with service account credentials."""
        self.SCOPES = ['https://www.googleapis.com/auth/drive']
        self.SERVICE_ACCOUNT_FILE = service_account_file
        
        # create credentials and build service
        self.credentials = service_account.Credentials.from_service_account_file(
            self.SERVICE_ACCOUNT_FILE, 
            scopes=self.SCOPES
        )
        self.drive_service = build('drive', 'v3', credentials=self.credentials)

    def make_file_public(self, file_id):
        """Make a file publicly accessible and return its sharing link."""
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            self.drive_service.permissions().create(
                fileId=file_id,
                body=permission
            ).execute()

            file = self.drive_service.files().get(
                fileId=file_id,
                fields='webViewLink, webContentLink'
            ).execute()

            print(f"File is now public.")
            print(f"View Link: {file.get('webViewLink')}")
            print(f"Download Link: {file.get('webContentLink')}")
            
            return file.get('webContentLink')
        except Exception as e:
            print(f"Error making file public: {str(e)}")
            return None

    def create_folder(self, folder_name, parent_folder_id=None):
        """Create a folder in Google Drive or return existing folder ID."""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        
        results = self.drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        files = results.get('files', [])
        
        if files:
            print(f'Folder already exists - ID: {files[0]["id"]}')
            return files[0]['id']
        
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        
        folder = self.drive_service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        print(f'Created new folder - ID: {folder["id"]}')
        return folder['id']

    def list_files(self, parent_folder_id=None, delete=False):
        """List folders and files in Google Drive."""
        query = f"'{parent_folder_id}' in parents and trashed=false" if parent_folder_id else "trashed=false"
        results = self.drive_service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)"
        ).execute()
        items = results.get('files', [])

        if not items:
            print("No folders or files found in Google Drive.")
        else:
            print("Folders and files in Google Drive:")
            for item in items:
                print(f"Name: {item['name']}, ID: {item['id']}, Type: {item['mimeType']}")
                if delete:
                    self.delete_files(item['id'])

    def delete_files(self, file_or_folder_id):
        """Delete a file or folder in Google Drive by ID."""
        try:
            self.drive_service.files().delete(fileId=file_or_folder_id).execute()
            print(f"Successfully deleted file/folder with ID: {file_or_folder_id}")
        except Exception as e:
            print(f"Error deleting file/folder with ID: {file_or_folder_id}")
            print(f"Error details: {str(e)}")

    def upload_file(self, local_file_path, folder_id=None):
        """Upload a file to Google Drive and make it publicly accessible."""
        if not os.path.exists(local_file_path):
            print(f"File not found: {local_file_path}")
            return None, None

        file_metadata = {'name': os.path.basename(local_file_path)}
        if folder_id:
            file_metadata['parents'] = [folder_id]

        media = MediaFileUpload(local_file_path, resumable=True)

        try:
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            file_id = file['id']
            print(f"Uploaded File ID: {file_id}")

            sharing_link = self.make_file_public(file_id)
            
            return file_id, sharing_link
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None, None

    def download_file(self, file_id, destination_path):
        """Download a file from Google Drive by its ID."""
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(destination_path, mode='wb')
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download {int(status.progress() * 100)}%.")
            print(f"File downloaded to: {destination_path}")
        except Exception as e:
            print(f"Error downloading file: {str(e)}")


# if __name__ == '__main__':
#     gdrive = GoogleDriveManager("video-448518-c557bc123b1b.json")
    
#     folder_id = gdrive.create_folder("uploaded_video")
    
#     gdrive.list_files(parent_folder_id=folder_id)
    
#     uploaded_file_id, sharing_link = gdrive.upload_file(
#         "/Users/mac/Documents/work/video-streaming/vidaio-subnet/services/upscaling/videos/7dacc160-ef37-40a6-a7c4-2da0cc8f2e8b.mp4", 
#         folder_id
#     )

#     if sharing_link:
#         print(f"Public download link: {sharing_link}")

#     if uploaded_file_id:
#         gdrive.download_file(
#             uploaded_file_id,
#             "/Users/mac/Documents/work/video-streaming/vidaio-subnet/services/upscaling/videos/downloaded_file.mp4"
#         )

    # uncomment to delete the file
    # if uploaded_file_id:
    #     gdrive.delete_files(uploaded_file_id)
