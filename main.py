# Google Cloud Function entry point
from adeptus_optimus_backend import ddos_tanking_treat_request


def run(request):
    return ddos_tanking_treat_request(request, "https://adeptus-optimus.web.app")
