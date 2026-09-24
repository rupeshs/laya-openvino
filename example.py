import laya

agent = laya.OVAgent(".ov/laya-ov-int8")

state = "I was charged twice for my Pro subscription and support hasn't replied."
questions = {
    "intent":  {"type": "choice", "instructions": "What does the customer want?",
                "criteria": {"refund": "money back", "support": "help", "cancel": "end subscription"}},
    "urgency": {"type": "score",  "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat", "urgent", "critical"]},
    "angry":   {"type": "noul",   "instructions": "The customer is frustrated."},
}

result = agent.predict(state, questions)
print(result["answers"]["intent"]["choice"])         # refund
print(result["answers"]["urgency"]["score"])         # 1.87
print(result["answers"]["angry"]["noul"])            # 0.84
