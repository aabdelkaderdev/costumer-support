from langchain_core.tools import BaseTool

class CompatibilityChecker(BaseTool):
    name: str = "compatibility_checker"
    description: str = "Check if two products are compatible. Input format must be: 'product1, product2'"

    def _run(self, query: str) -> str:
        """Check compatibility between two products"""
        if "," not in query:
            return "Please provide two products separated by a comma. Example: Product A, Product B"

        products = [p.strip() for p in query.split(",")]
        p1, p2 = products[0], products[1]
        
        lower_query = query.lower()
        if "cloud" in lower_query and "hardware" in lower_query:
            return f"{p1} and {p2} require an intermediate bridge gateway for compatibility."
        elif "legacy" in lower_query:
            return f"Unfortunately, {p1} and {p2} are not compatible due to outdated architecture."
            
        return f"Yes, {p1} and {p2} are fully compatible."

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")
