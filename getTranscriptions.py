import json
import requests


# Get all course ids
def get_all_course_ids():
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
                'courseName': course['courses'][0]['courseName'],
                'offeringId': course['offering']['id'],
            })
    return courses


'''
Get all transcriptions of videos of a course by offspring id
return a double array of transcriptions
each index is a video with an array of transcriptions in different language
'''
def get_all_transcriptions_by_offspring_id(offspringId):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': '',
    }
    videos = []
    api_url = 'https://classtranscribe.ncsa.illinois.edu/api/Playlists/ByOffering/' + offspringId
    response = requests.get(api_url, headers=headers)
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
def get_transcription_by_id(transcription_id):
    headers = {
        'Content-Type': 'application/json',
    }
    text = ''
    api_url = 'https://classtranscribe.ncsa.illinois.edu/api/Captions/ByTranscription/' + transcription_id
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        content = json.loads(response.content.decode('utf-8'))
        for transcription in content:
            text += transcription['text']
            text += " "
    return text


def main():
    get_all_course_ids()
    # testing
    get_all_transcriptions_by_offspring_id('bb6ba819-3d08-4c68-accd-272778493362')
    text = get_transcription_by_id('5b3ee68c-5407-4a1f-a330-e55d5d239ffb')
    print(text)


if __name__ == '__main__':
    main()
