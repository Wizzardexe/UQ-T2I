ambiguous_prompts = {
    # lexical ambiguity
    "a photo of the bat on a table": [
        "a photo of the bat, the flying animal, on a table",
        "a photo of the baseball bat on a table",
    ],
    "a photo of a bat near the field": [
        "a photo of a bat, the flying animal, near the field",
        "a photo of a baseball bat near the field",
    ],
    "a photo of a bow displayed in the market": [
        "a photo of a bow for archery displayed in the market",
        "a photo of a ribbon bow displayed in the market",
    ],
    "The bank next to the park": [
        "The bank building, where people keep money, next to the park",
        "The river bank next to the park",
    ],
    # figurative vs. literal reading
    "The night sky was a blanket of stars.": [
        "The night sky was full of stars.",
        "A blanket with a pattern of stars.",
    ],
    # syntactic ambiguity
    "a girl looking at a boy with a telescope": [
        "a girl using a telescope to look at a boy",
        "a girl looking at a boy who is holding a telescope",
    ],
    "old men and women sitting on a bench": [
        "old men and old women sitting on a bench",
        "old men and young women sitting on a bench",
    ],
}

underspecified_prompts = {
    "a dog in the park": [
        "a golden retriever puppy sitting on the grass in the park, front view, afternoon sun",
        "a black labrador running along a path in the park, side view, overcast day",
        "a small white poodle lying under a tree in the park, close-up, evening light",
    ],
    "a house": [
        "a small red wooden house with a white door by a lake, wide shot, sunset",
        "a modern white concrete house with large windows on a hill, front view, noon",
        "an old stone cottage with a thatched roof in a meadow, side view, foggy morning",
    ],
    "a car on a street": [
        "a red vintage convertible parked on a cobblestone street, side view, at noon",
        "a black SUV driving on a wet city street at night, front view, neon lights",
        "a small yellow taxi waiting at a crossing, rear view, rainy afternoon",
    ],
}

specificity_ladders = [
    ["", "a dog", "a dog in the park", "a golden retriever in the park",
     "a golden retriever puppy sitting on the grass in the park, front view, afternoon sun"],
    ["", "a house", "a wooden house", "a red wooden house by a lake",
     "a small red wooden house with a white door by a lake, wide shot, sunset"],
]

paraphrases = [
    ["a dog in a park", "a dog at a park", "in a park, a dog"],
    ["a red apple on a wooden table", "a red apple lying on a wooden table",
     "on a wooden table, a red apple"],
    ["a cat sleeping on a sofa", "a cat asleep on a sofa", "a sleeping cat on a couch"],
    ["a man riding a bicycle down a street", "a man cycling down a street",
     "a man on a bicycle going down a street"],
]

rare_concepts = [
    "a photo of a coelacanth",
    "a photo of an astrolabe on a desk",
    "a photo of a shoebill standing in a swamp",
    "a photo of a theremin on a stage",
    "a photo of a dhole in a forest",
]

novel_compositions = [
    "a horse riding an astronaut",
    "a blue banana next to a yellow strawberry",
    "a giraffe smaller than a mouse",
    "a cat to the left of a dog",
    "three red cubes and two green spheres",
    "thirty sheep in a field",
]

ood_prompts = [
    "a satellite image of a harbor",
    "an electron microscope image of pollen grains",
    "a histology slide of liver tissue",
    "an X-ray image of a hand holding a pen",
    "a thermal camera image of a house at night",
]

counting_prompts = [
    "one apple on a table",
    "two apples on a table",
    "three apples on a table",
    "four apples on a table",
    "10 sheep in a field",
    "20 sheep in a field",
    "30 sheep in a field",
    "3 red cubes and 2 green spheres",
    "5 red cubes and 7 green spheres",
    "10 red cubes and 20 green spheres",
]

common_prompts = [
    "a photo of a cat",
    "a photo of a car",
    "a cup of coffee on a table",
    "a photo of a beach",
    "a bowl of fruit",
]


def generate_prompts_list() -> list[tuple[str, str]]:
    seen = set()
    output = []

    def create_prompt(group, prompt):
        if prompt not in seen:
            seen.add(prompt)
            output.append((group, prompt))

    for prompt, clarifications in ambiguous_prompts.items():
        create_prompt("ambiguous", prompt)
        for c in clarifications:
            create_prompt("clarification", c)
    for prompt, refinements in underspecified_prompts.items():
        create_prompt("underspecified", prompt)
        for r in refinements:
            create_prompt("specified", r)
    for ladder in specificity_ladders:
        for p in ladder:
            create_prompt("ladder", p)
    for group in paraphrases:
        for p in group:
            create_prompt("paraphrase", p)
    for name, prompts in [("rare", rare_concepts), ("composition", novel_compositions),
                          ("ood", ood_prompts), ("common", common_prompts), ("counting", counting_prompts)]:
        for p in prompts:
            create_prompt(name, p)
    
    return output


if __name__ == "__main__":
    from collections import Counter
    prompts = generate_prompts_list()
    print(prompts[:10])
    print(Counter(g for g, _ in prompts))
    print(len(prompts), "prompts; with 20 seeds:", 20 * len(prompts), "images")