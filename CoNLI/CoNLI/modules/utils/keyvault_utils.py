
from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
import os

def load_secret_from_keyvault(keyvault_url : str, secret_name : str, managed_identity_client_env_var : str = None):
    try:
        credential = DefaultAzureCredential()
        secret_client = SecretClient(
            vault_url=keyvault_url,
            credential=credential
        )
        secret = secret_client.get_secret(secret_name)
    except Exception as e:
        if managed_identity_client_env_var is not None:
            client_id = os.environ.get(managed_identity_client_env_var)
            credential = ManagedIdentityCredential(client_id=client_id)
            client = SecretClient(vault_url=keyvault_url, credential=credential)
            secret = client.get_secret(secret_name)
    return secret.value
