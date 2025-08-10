# prompts.py

def generic_prompt(category, name):
    print("Prompt genérico: ")
    return f"""
Summarize the essence of the brand {category} and the project {name} in a single, compact paragraph, conveying its core idea clearly and directly.
"""

def structured_prompt(category_name, name, blurb):
    print("Prompt estructurado: ")
    return f"""
Generate a professional and structured pitch deck summary for a project titled **'{name}'**, which falls under the **'{category_name}'** category. This pitch should be a fluent and engaging extension of the original project description (blurb) provided below.

Original Project Blurb:
"{blurb}"

**Instructions for Generation (Max: 150 words):**

1. **Structure**:
- **Problem (1–2 sentences):** Clearly describe the central problem addressed by the project. Include at least 2 keywords or phrases from the blurb.
- **Solution (2–3 sentences):** Explain how the project '{name}' solves the problem, using at least 3 phrases from the blurb with synonyms.
- **Value Proposition (1 sentence):** Rephrase key nouns and verbs from the blurb.

2. **Lexical Guidelines**:
- At least 70% lexical overlap.
- Preserve verbs/nouns/adjectives; add synonyms.
- Avoid redundancy.

3. **Style**:
- Natural, persuasive, concise.
- Ensure coherence and logical flow.
"""
