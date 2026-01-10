# Introduction
Document ingestion focuses on minimal text extraction sufficient to study downstream retrieval and agent behaviour. Rich document structure (tables, images, styling) is intentionally ignored to keep the prototype evaluation-focused.

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

## Documents

**The documents' heading structure has been amended for ease of parsing.**

State Treaty on the modernisation of media legislation in Germany (N-2020-0026-000-EN.DOCX)
https://technical-regulation-information-system.ec.europa.eu/en/notification/15957/text/D/EN

REPORT FROM THE COMMISSION TO THE EUROPEAN PARLIAMENT, THE COUNCIL AND THE EUROPEAN ECONOMIC AND SOCIAL COMMITTEE ON THE OPERATION OF THE SINGLE MARKET TRANSPARENCY DIRECTIVE FROM 2016 ΤΟ 2020 (1_EN_ACT_part1_v4.docx)
https://secure.ipex.eu/IPEXL-WEB/download/file/082d29088354edb301836a5c43790652

REPORT FROM THE COMMISSION TO THE EUROPEAN PARLIAMENT, THE COUNCIL AND THE EUROPEAN ECONOMIC AND SOCIAL COMMITTEEON THE OPERATION OF DIRECTIVE (EU) 2015/1535 FROM 2014 ΤΟ 2015 (1_EN_ACT_part1_v5.docx)
https://secure.ipex.eu/IPEXL-WEB/download/file/082dbcc5618c772b01618ff34350045d

Although several chunks exceed the effective context window of the embedding model, this setup is retained to illustrate realistic failure modes of retrieval in long, densely structured documents rather than to maximise benchmark performance.

Despite exceeding the embedding model’s optimal context length, the oversized chunk is often retrieved correctly due to reduced intra-document competition, illustrating how corpus-level dynamics can offset suboptimal chunk granularity in small collections.

# Build and Test
TODO: Describe and show how to build your code and run the tests.

# Contribute
TODO: Explain how other users and developers can contribute to make your code better.

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
