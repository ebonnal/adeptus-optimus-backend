import requests
import json

api_key = "AIzaSyDIPVEEqYlGMePVvjh4h7Wbh6kosaxQuMk"
domain = "https://adeptusoptimus.page.link"
url = "https://firebasedynamiclinks.googleapis.com/v1/shortLinks"
query_string = f"key={api_key}"


def get_short_dynamic_link(link):
    return json.loads(requests.post(
        f"{url}?{query_string}",
        json={"longDynamicLink": f"{domain}/?link={link}"},
        headers={"Content-Type": "application/json"}).content)["shortLink"]
