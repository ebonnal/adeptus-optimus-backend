import requests
from urllib.parse import quote
import json

api_key = "AIzaSyDIPVEEqYlGMePVvjh4h7Wbh6kosaxQuMk"
domain = "https://adeptusoptimus.page.link"
url = "https://firebasedynamiclinks.googleapis.com/v1/shortLinks"
query_string = f"key={api_key}"
"""
st 	The title to use when the Dynamic Link is shared in a social post.
sd 	The description to use when the Dynamic Link is shared in a social post.
si 	The URL to an image related to this link. The image should be at least 300x200 px, and less than 300 KB.
"""
st = "Adeptus Optimus"
sd = "Damage calculator tool for 40k"
si = "https://adeptus-optimus.web.app/images/logo.png"
social_media_params = f"st={quote(st)}&sd={quote(sd)}&si={quote(si)}"


def get_long_dynamic_link(share_settings):
    link = f"https://adeptus-optimus.web.app?share_settings={quote(share_settings)}"
    return f"{domain}/?link={quote(link)}&{social_media_params}"


def get_short_dynamic_link(share_settings):
    return json.loads(requests.post(
        f"{url}?{query_string}",
        json={"longDynamicLink": get_long_dynamic_link(share_settings)},
        headers={"Content-Type": "application/json"}).content)["shortLink"]
