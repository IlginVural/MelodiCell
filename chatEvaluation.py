from openai import OpenAI
client = OpenAI(api_key="")

#Evaluates a given chat and outputs the necessery parameters for music generation
def evaluateChat(string):
    response = client.responses.create(
    model="gpt-4o",
    input=string,
    text={
        "format": {
        "type": "json_schema",
        "name": "chat_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "emotional_state": {
                "type": "object",
                "properties": {
                "calm": {
                    "type": "integer",
                    "description": "Rating of calmness on a scale of 1 to 10."
                },
                "anxious": {
                    "type": "integer",
                    "description": "Rating of anxiousness on a scale of 1 to 10."
                },
                "tired": {
                    "type": "integer",
                    "description": "Rating of tiredness on a scale of 1 to 10."
                },
                "focused": {
                    "type": "integer",
                    "description": "Rating of focus on a scale of 1 to 10."
                },
                "sad": {
                    "type": "integer",
                    "description": "Rating of sadness on a scale of 1 to 10."
                },
                "neutral": {
                    "type": "integer",
                    "description": "Rating of neutrality on a scale of 1 to 10."
                }
                },
                "required": [
                "calm",
                "anxious",
                "tired",
                "focused",
                "sad",
                "neutral"
                ],
                "additionalProperties": False
            },
            "preferred_instrumentation": {
                "type": "array",
                "description": "List of preferred types of instruments or styles.",
                "items": {
                "type": "string",
                "enum": [
                    "Piano",
                    "Strings",
                    "Synthesizers",
                    "Nature sounds (water, rain, birds)",
                    "Soft guitar",
                    "Flutes",
                    "Ambient pads"
                ]
                }
            }
            },
            "required": [
            "emotional_state",
            "preferred_instrumentation"
            ],
            "additionalProperties": False
        }
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=2048,
    top_p=1,
    store=True
    )
    return response.output_text


#print(evaluateChat("Sad and Heart Broke, I prefer guitar"))


