import requests
from lxml import html



session_requests = requests.session()

login_url = "https://bitbucket.org/account/signin/?next=/"
result = session_requests.get(login_url)

tree = html.fromstring(result.text)
authenticity_token = list(set(tree.xpath("//input[@name='csrfmiddlewaretoken']/@value")))[0]

payload = {
    "username": "rjhaveri41@gmail.com",
    "password": "Rutanshu14@",
    "csrfmiddlewaretoken": authenticity_token
}

result = session_requests.post(
    login_url,
    data = payload,
    headers = dict(referer=login_url)
)

url = 'https://bitbucket.org/dashboard/overview'
result = session_requests.get(
    url,
    headers = dict(referer = url)
)

tree = html.fromstring(result.content)
bucket_names = tree.xpath("//div[@class='repo-list--repo']/a/text()")

print(bucket_names)