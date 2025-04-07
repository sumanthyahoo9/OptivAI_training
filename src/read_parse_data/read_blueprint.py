"""
This script helps us read a blueprint in the pdf format and get it
ready for OCR models.
The facility layout is critical since it adds a lot of context beyond
raw numbers for the LLM to make sense of the agent's actions.
This script assumes that the blueprints are in the PDF format.
Once we extract the blueprint using OCR, we can integrate it
with the epJSON file with the data about the zones
"""
