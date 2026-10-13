"""Agent: 整合计划器和解决器 实现一个完整的Agent"""
from typing import List
from .models import Planner, Solver

class PlanAndSolveAgent:
    def __init__(self, planner: Planner, solver: Solver):
        self.planner = planner
        self.solver = solver

    def run(self, question: str):
        print("PlanAndSolveAgent is running...")
        plans: List[str] = self.planner.plan(question)
        if not plans:
            print("PlanAndSolveAgent failed to plan.")
            return
        final_answer: str = self.solver.solve(question, plans)
        print(f"PlanAndSolveAgent is finished. Final answer: {final_answer}")