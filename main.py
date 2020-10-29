# Google Cloud Function entry point
from adeptus_optimus_backend import compare


def run(request):
    return compare(request)
