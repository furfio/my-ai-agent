from typing import TypedDict, List

class Provider(TypedDict):
    internalId: str
    providerName: str
    asn: int
    emails: List[str]
    newProviderName: str
    isNameChanged: bool
    finalName: str