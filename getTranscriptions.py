import json
import requests

'''
Get all course ids
'''
def getAllCourseIds():
    headers = {
        'Content-Type': 'application/json',
    }
    courses = []
    api_url = 'https://classtranscribe.ncsa.illinois.edu/api/Offerings/ByStudent'
    response = requests.get(api_url, headers = headers)
    if response.status_code == 200:
        content = json.loads(response.content.decode('utf-8'))
        for course in content:
            courses.append({
                'sectionName': course['sectionName'],
                'id': course['id'],
            })
    return courses

'''
Get all transcriptions of videos of a course by offspring id
return a double array of transcriptions
each index is a video with an array of transcriptions in different language
'''
def getAllTranscriptionsByoffspringID(offspringId):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ5aWZhbmM3QGlsbGlub2lzLmVkdSIsImp0aSI6IjI0Y2Y2YzhhLTZhNzEtNDY1Ny05NDg5LTM2Mzc2ZWZmMzdlNSIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL25hbWVpZGVudGlmaWVyIjoiNDkwYWNiZDAtNTQ4My00YzAzLWI5MDEtMWM4OWJlNWJlMTc1IiwiZXhwIjoxNTcxMjUxNDE0LCJpc3MiOiJodHRwczovL2NsYXNzdHJhbnNjcmliZS5uY3NhLmlsbGlub2lzLmVkdSIsImF1ZCI6Imh0dHBzOi8vY2xhc3N0cmFuc2NyaWJlLm5jc2EuaWxsaW5vaXMuZWR1In0.QFlDqGqUJoFP1sS6ibfNyUEIxZ1blpsU4HWmgKfTCgk',
    }
    videos = []
    api_url = 'https://classtranscribe.ncsa.illinois.edu/api/Playlists/ByOffering/'+ offspringId
    response = requests.get(api_url, headers = headers)
    if response.status_code == 200:
        content = json.loads(response.content.decode('utf-8'))
        for video in content:
            medias = video['medias']
            for media in medias:
                videos.append(media['transcriptions'])
    return videos

'''
Get all transcriptions of a language of a video by transcription ids
'''
def getTranscriptionById(transcriptionId):
    headers = {
        'Content-Type': 'application/json',
    }
    text = ''
    api_url = 'https://classtranscribe.ncsa.illinois.edu/api/Captions/ByTranscription/' + transcriptionId
    response = requests.get(api_url, headers = headers)
    if response.status_code == 200:
        content = json.loads(response.content.decode('utf-8'))
        for transcription in content:
            text += transcription['text']
    return text

def main():
    getAllCourseIds()
    # testing
    getAllTranscriptionsByoffspringID('bb6ba819-3d08-4c68-accd-272778493362')
    getTranscriptionById('5b3ee68c-5407-4a1f-a330-e55d5d239ffb')

if __name__ == '__main__':
    main()