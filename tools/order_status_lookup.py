from langchain_core.tools import BaseTool

class OrderStatusLookup(BaseTool):
    name: str = "order_status_lookup"
    description: str = "Look up the status of a specific order. Input should be the exact order ID."
    
    def _run(self, order_id: str) -> str:
        order_id = order_id.strip()
        
        if not order_id:
            return "Please provide a valid order ID."
            
        if order_id.startswith("ORD-"):
            return f"Order {order_id} has been Shipped. Expected delivery in 3 days."
        elif order_id.isdigit():
            return f"Order {order_id} is currently Processing. It will ship within 24 hours."
        else:
            return f"Error: Order {order_id} not found in our database. Please verify your order ID."
            
    async def _arun(self, order_id: str) -> str:
        raise NotImplementedError("Async not implemented")
