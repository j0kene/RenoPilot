"""Pricing: landed cost in SGD, and comparable offers from other retailers.

Deterministic money maths (fx, landed_cost) is kept apart from the one judgement a model
makes (comparables: "is this really the same kind of product?"). The graph node lives in
node.py and plugs in between shortlist and planner.
"""
