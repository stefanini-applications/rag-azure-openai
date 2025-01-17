How to:
1 - Install azure CLI
https://learn.microsoft.com/pt-br/cli/azure/install-azure-cli-windows?tabs=azure-cli#install-or-update
Login with your MS account via: az login

2 - Create the services and connect them via azure portal
https://github.com/MicrosoftDocs/azure-ai-docs/blob/main/articles/ai-studio/tutorials/copilot-sdk-create-resources.md

3 - Place your credentials in .env (follow tutorials/tutorial-credentials.pdf)

4 - Run the app with: python -m streamlit run app.py



Obs:
In case of tenant error:
az login --tenant <TENANT_ID>
