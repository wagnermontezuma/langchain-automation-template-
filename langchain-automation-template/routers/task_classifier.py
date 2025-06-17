import re
from typing import Dict, List, Tuple, Optional, Any
from core.types import TaskClassification # Assuming this is defined in core.types
from core.logger import get_logger

logger = get_logger(__name__)

# Define keywords for different task types
TASK_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "code", "python", "javascript", "java", "c++", "ruby", "golang", "swift", "kotlin",
        "function", "class", "method", "algorithm", "script", "debug", "compile", "fix",
        "implement", "develop", "program", "software", "api", "django", "flask", "react", "vue", "angular",
        "docker", "kubernetes", "aws", "azure", "gcp", "sql", "database", "query", "html", "css", "typescript"
    ],
    "analysis": [
        "analyze", "analysis", "data", "statistics", "report", "insights", "trends", "compare",
        "evaluate", "examine", "interpret", "visualize", "chart", "graph", "metrics", "kpi",
        "spreadsheet", "excel", "csv", "json", "log", "forecast", "predict"
    ],
    "creative": [
        "write", "story", "poem", "essay", "blog", "post", "script", "lyrics", "marketing", "copy",
        "slogan", "tagline", "brainstorm", "ideas", "create", "design", "narrative", "dialogue",
        "novel", "article", "content"
    ],
    "research": [
        "research", "find", "information", "what is", "who is", "when was", "explain", "summary",
        "summarize", "topic", "study", "paper", "discovery", "evidence", "source", "cite", "bibliography"
    ],
    "translation": [
        "translate", "translation", "language", "english", "spanish", "french", "german", "chinese",
        "japanese", "korean", "italian", "portuguese", "dutch", "russian", "arabic", "hindi", "interpret"
    ],
    "math": [
        "math", "mathematics", "calculate", "equation", "algebra", "geometry", "calculus", "statistics",
        "problem", "solve", "formula", "proof", "theorem", "number", "integral", "derivative", "vector", "matrix"
    ],
    "general": [ # General keywords, or if no other type is strongly matched
        "how to", "help", "question", "advice", "general", "conversation", "chat", "discuss", "tell me about"
    ]
}

# Define keywords for complexity
COMPLEXITY_KEYWORDS: Dict[str, List[str]] = {
    "high": ["complex", "advanced", "difficult", "hard", "in-depth", "thorough", "extensive", "multi-step", "large scale"],
    "medium": ["moderate", "intermediate", "detailed", "several parts", "some complexity"],
    "low": ["simple", "basic", "easy", "quick", "small", "minor", "trivial", "brief", "summarize briefly"]
}

# Define default task type if no other is found
DEFAULT_TASK_TYPE = "general"
DEFAULT_COMPLEXITY = "medium"
DEFAULT_CONFIDENCE = 0.5 # Default confidence if only default type is matched

class TaskClassifier:
    """
    Classifies tasks based on input query text.
    Determines task type, complexity, estimated context length, and confidence.
    """

    def __init__(self, task_keywords: Optional[Dict[str, List[str]]] = None,
                 complexity_keywords: Optional[Dict[str, List[str]]] = None):
        self.task_keywords = task_keywords or TASK_TYPE_KEYWORDS
        self.complexity_keywords = complexity_keywords or COMPLEXITY_KEYWORDS
        logger.info("TaskClassifier initialized.")

    def _normalize_text(self, text: str) -> str:
        """Normalizes text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text

    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Counts occurrences of keywords in the text."""
        count = 0
        # Ensure text is not empty after normalization
        if not text.strip():
            return 0
        for keyword in keywords:
            # Use  for word boundaries to match whole words
            # Ensure keyword is not empty
            if not keyword.strip():
                continue
            try:
                count += len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
            except re.error as e:
                logger.warning(f"Regex error counting keyword '{keyword}': {e}")
                continue
        return count

    def classify_task_type(self, query: str) -> Tuple[str, float]:
        """
        Classifies the task type based on keywords.
        Returns the classified task type and a confidence score.
        """
        normalized_query = self._normalize_text(query)
        scores: Dict[str, int] = {task_type: 0 for task_type in self.task_keywords}
        total_keywords_matched = 0

        for task_type, keywords in self.task_keywords.items():
            if task_type == "general": # Skip general for initial scoring, handle later
                continue
            score = self._count_keywords(normalized_query, keywords)
            scores[task_type] = score
            total_keywords_matched += score

        if not total_keywords_matched: # If no specific keywords matched
            logger.debug(f"No specific task type keywords found in query: '{query}'. Defaulting to '{DEFAULT_TASK_TYPE}'.")
            return DEFAULT_TASK_TYPE, DEFAULT_CONFIDENCE

        # Determine the best match
        # Filter out types with 0 score before finding max to avoid issues if all specific scores are 0
        # (though covered by total_keywords_matched == 0 check)
        scored_types = {t: s for t, s in scores.items() if s > 0 and t != "general"}
        if not scored_types: # Should mean only general keywords would match, or none
             return DEFAULT_TASK_TYPE, DEFAULT_CONFIDENCE

        best_type = max(scored_types, key=scored_types.get)

        confidence = scored_types[best_type] / total_keywords_matched if total_keywords_matched > 0 else DEFAULT_CONFIDENCE
        confidence = min(0.95, confidence * 1.2) if confidence > 0.6 else confidence # Boost and cap

        logger.debug(f"Classified task type for query '{query}': {best_type} (Confidence: {confidence:.2f})")
        return best_type, round(confidence, 2)

    def classify_complexity(self, query: str, task_type: Optional[str] = None) -> Tuple[str, float]:
        """
        Classifies the complexity of the task.
        Returns complexity level (low, medium, high) and a confidence score.
        """
        normalized_query = self._normalize_text(query)
        scores: Dict[str, int] = {level: 0 for level in self.complexity_keywords}
        total_keywords_matched = 0

        for level, keywords in self.complexity_keywords.items():
            score = self._count_keywords(normalized_query, keywords)
            scores[level] = score
            total_keywords_matched += score

        if not total_keywords_matched:
            logger.debug(f"No complexity keywords found in query: '{query}'. Defaulting to '{DEFAULT_COMPLEXITY}'.")
            return DEFAULT_COMPLEXITY, DEFAULT_CONFIDENCE

        best_complexity = max(scores, key=scores.get)
        confidence = scores[best_complexity] / total_keywords_matched if total_keywords_matched > 0 else DEFAULT_CONFIDENCE
        confidence = min(0.95, confidence * 1.1) # Slight boost, cap

        logger.debug(f"Classified complexity for query '{query}': {best_complexity} (Confidence: {confidence:.2f})")
        return best_complexity, round(confidence, 2)

    def estimate_context_length(self, query: str, task_type: Optional[str] = None, complexity: Optional[str] = None) -> int:
        """
        Estimates the necessary context length for the task.
        (Expressed in approximate number of tokens)
        """
        base_length = len(query.split()) * 2

        if complexity == "high":
            base_length *= 3
        elif complexity == "medium":
            base_length *= 2
        else: # low
            base_length *= 1.5

        if task_type in ["coding", "research", "analysis"] and complexity == "high":
            base_length += 2000
        elif task_type in ["coding", "research", "analysis"]:
             base_length += 500

        estimated_tokens = max(100, min(int(base_length), 8000))
        logger.debug(f"Estimated context length for query '{query}': {estimated_tokens} tokens")
        return estimated_tokens

    def classify(self, query: str) -> TaskClassification:
        """
        Performs full task classification.
        """
        task_type, type_confidence = self.classify_task_type(query)
        # If task_type defaults to general, and no specific keywords were found, complexity might also be uncertain.
        if task_type == DEFAULT_TASK_TYPE and type_confidence == DEFAULT_CONFIDENCE:
            complexity = DEFAULT_COMPLEXITY
            complexity_confidence = DEFAULT_CONFIDENCE
        else:
            complexity, complexity_confidence = self.classify_complexity(query, task_type)

        overall_confidence = round((type_confidence + complexity_confidence) / 2, 2)
        context_length_estimate = self.estimate_context_length(query, task_type, complexity)

        classification_details: Dict[str, Any] = {
            "task_type_confidence": type_confidence,
            "complexity_level": complexity,
            "complexity_confidence": complexity_confidence,
            "estimated_context_tokens": context_length_estimate
        }

        result: TaskClassification = {
            "task_description": query,
            "category": task_type,
            "confidence": overall_confidence,
            "details": classification_details
        }
        logger.info(f"Full classification for query '{query}': {result}")
        return result

if __name__ == '__main__':
    classifier = TaskClassifier()
    queries = [
        "Can you write a python script to fetch data from an API and store it in a CSV file? It's quite complex.",
        "analyze the sales data for the last quarter and provide insights. Make it simple.",
        "Tell me a short creative story about a dragon.",
        "what is the capital of France?",
        "translate 'hello world' to Spanish",
        "calculate the integral of x^2 from 0 to 1. This is a hard problem.",
        "how to make a good cup of coffee?",
        "Debug this javascript function for me, it's not working as expected and involves multiple asynchronous calls.",
        "Research the impact of AI on climate change and provide a detailed report with citations.",
        "Provide a simple explanation of quantum entanglement."
    ]

    for q in queries:
        classification = classifier.classify(q)
        print(f"Query: {q}")
        print(f"  Type: {classification['category']} (Confidence: {classification['details']['task_type_confidence']})")
        print(f"  Complexity: {classification['details']['complexity_level']} (Confidence: {classification['details']['complexity_confidence']})")
        print(f"  Overall Confidence: {classification['confidence']}")
        print(f"  Estimated Context Tokens: {classification['details']['estimated_context_tokens']}")
        print("-" * 20)
