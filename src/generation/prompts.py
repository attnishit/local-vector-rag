"""
Prompt Template System

Author: Nishit Attrey

This module provides a Jinja2-based prompt template system for LLM generation.
Templates allow flexible prompt engineering for different use cases (Q&A,
summarization, chat, etc.) while maintaining consistent formatting.

Key Features:
- Jinja2-based templating with variable substitution
- Default templates for common use cases
- Custom template loading from files
- Template validation

Example:
    >>> from src.generation.prompts import load_prompt_template
    >>> template = load_prompt_template("qa")
    >>> prompt = template.render(query="What is HNSW?", context="[1] HNSW...")
    >>> print(prompt)
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Template, Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)

# Default prompt templates
DEFAULT_TEMPLATES = {
    "qa": """You are a helpful assistant answering questions based on provided context.

Context information:
{{ context }}

Question: {{ query }}

Instructions:
- Answer the question using ONLY the information from the context above
- Cite sources using [1], [2], etc. when referencing specific information
- If the context doesn't contain enough information, say so clearly
- Be concise and accurate
- Do not make up information not present in the context

Answer:""",

    "summarize": """You are a helpful assistant that summarizes information from documents.

Context to summarize:
{{ context }}

Instructions:
- Provide a concise summary of the key points from the context
- Cite sources using [1], [2], etc. for each major point
- Organize information logically with clear structure
- Keep the summary focused and relevant
- Use bullet points or numbered lists if appropriate

Summary:""",

    "chat": """You are a helpful assistant having a conversation with a user.

Relevant context from documents:
{{ context }}

{% if history %}
Previous conversation:
{% for msg in history %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}
{% endif %}

User: {{ query }}

Instructions:
- Use the context to inform your response when relevant
- Maintain conversation continuity with previous messages
- Cite sources with [1], [2], etc. when using document information
- Be conversational but accurate
- If the context isn't relevant, acknowledge that and provide general help
Assistant:""",
}


class PromptTemplate:
    """
    Wrapper around Jinja2 Template for prompt engineering.
    
    Provides a consistent interface for rendering prompts with variable substitution,
    validation, and error handling.
    
    Example:
        >>> template = PromptTemplate(DEFAULT_TEMPLATES["qa"], name="qa")
        >>> prompt = template.render(query="What is HNSW?", context="[1] HNSW is...")
        >>> print(prompt)
    """
    
    def __init__(self, template_string: str, name: str = "custom"):
        """
        Initialize prompt template.
        
        Args:
            template_string: Jinja2 template string
            name: Template name for logging/debugging
        """
        self.name = name
        self.template_string = template_string
        
        try:
            self.template = Template(template_string)
        except Exception as e:
            logger.error(f"Failed to parse template '{name}': {e}")
            raise ValueError(f"Invalid Jinja2 template '{name}': {e}")
        
        logger.debug(f"Initialized template: {name}")
    
    def render(self, **variables) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **variables: Variable values to substitute in template
        
        Returns:
            Rendered prompt string
        
        Raises:
            ValueError: If required variables are missing
        
        Example:
            >>> template.render(query="What is HNSW?", context="[1] Context...")
        """
        try:
            rendered = self.template.render(**variables)
            logger.debug(
                f"Rendered template '{self.name}' with variables: {list(variables.keys())}"
            )
            return rendered
        except Exception as e:
            logger.error(
                f"Failed to render template '{self.name}': {e}. "
                f"Variables provided: {list(variables.keys())}"
            )
            raise ValueError(
                f"Template rendering failed for '{self.name}': {e}. "
                f"Check that all required variables are provided."
            )
    
    def get_required_variables(self) -> List[str]:
        """
        Extract variable names from template.
        
        Returns:
            List of variable names used in template
        
        Example:
            >>> template.get_required_variables()
            ['query', 'context']
        """
        # Get undeclared variables from Jinja2 AST
        from jinja2 import meta
        env = Environment()
        ast = env.parse(self.template_string)
        variables = meta.find_undeclared_variables(ast)
        return sorted(list(variables))
    
    def __repr__(self) -> str:
        return f"PromptTemplate(name='{self.name}', variables={self.get_required_variables()})"


def load_prompt_template(name: str) -> PromptTemplate:
    """
    Load a default prompt template by name.
    
    Args:
        name: Template name ("qa", "summarize", "chat")
    
    Returns:
        PromptTemplate instance
    
    Raises:
        ValueError: If template name doesn't exist
    
    Example:
        >>> template = load_prompt_template("qa")
        >>> prompt = template.render(query="...", context="...")
    """
    if name not in DEFAULT_TEMPLATES:
        available = ", ".join(DEFAULT_TEMPLATES.keys())
        raise ValueError(
            f"Template '{name}' not found. Available templates: {available}"
        )
    
    logger.debug(f"Loading default template: {name}")
    return PromptTemplate(DEFAULT_TEMPLATES[name], name=name)


def load_custom_template(path: Path) -> PromptTemplate:
    """
    Load a custom prompt template from a file.
    
    Args:
        path: Path to template file (.txt or .j2 extension)
    
    Returns:
        PromptTemplate instance
    
    Raises:
        FileNotFoundError: If template file doesn't exist
        ValueError: If template is invalid
    
    Example:
        >>> template = load_custom_template(Path("templates/my_custom.j2"))
        >>> prompt = template.render(query="...", context="...")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        template_string = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read template file {path}: {e}")
    
    template_name = path.stem  # Filename without extension
    logger.info(f"Loaded custom template from {path}")
    
    return PromptTemplate(template_string, name=template_name)


def validate_template(template: PromptTemplate, required_vars: List[str]) -> bool:
    """
    Validate that a template contains all required variables.
    
    Args:
        template: PromptTemplate to validate
        required_vars: List of variable names that must be present
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If required variables are missing
    
    Example:
        >>> template = load_prompt_template("qa")
        >>> validate_template(template, ["query", "context"])
        True
    """
    template_vars = set(template.get_required_variables())
    required_set = set(required_vars)
    
    missing_vars = required_set - template_vars
    
    if missing_vars:
        raise ValueError(
            f"Template '{template.name}' is missing required variables: {sorted(missing_vars)}. "
            f"Template has: {sorted(template_vars)}"
        )
    
    logger.debug(f"Template '{template.name}' validated successfully")
    return True
