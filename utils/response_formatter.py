class ResponseFormatter:
    def __init__(self, company_name, support_email, support_phone):
        self.company_name = company_name
        self.support_email = support_email
        self.support_phone = support_phone

    def format_response(self, response_content, knowledge_base=None):
        """Format response with consistent structure"""
        kb_text = f" (Sourced from {knowledge_base} Knowledge Base)" if knowledge_base else ""
        return (
            f"**{self.company_name} Support**\n\n"
            f"{response_content}\n\n"
            f"---\n"
            f"*Disclaimer: This information is provided by our automated support system{kb_text}. "
            f"For further assistance, please contact us at {self.support_email} or {self.support_phone}.*"
        )

    def format_error_response(self, error_type):
        """Format error response"""
        return (
            f"**System Notification**\n\n"
            f"We encountered an issue while processing your request (Error: {error_type}).\n"
            f"Please try again or contact our human support team at {self.support_email}."
        )

    def format_escalation_response(self):
        """Format response for issues that need escalation"""
        return (
            f"**Escalation Notice**\n\n"
            f"This issue appears to be complex and requires specialized attention.\n"
            f"I have created a priority ticket for our advanced support team. They will reach out to you via {self.support_email} shortly."
        )
