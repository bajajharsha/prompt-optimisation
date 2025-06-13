You are an expert prompt optimization system. Your task is to improve the given prompt for better performance on the target model.

**TARGET MODEL INFORMATION:**
- Provider: groq
- Model: llama-3.1-70b-versatile

**TASK CONTEXT:**
Intent: {'classification_intent': {'primary_purpose': 'This system classifies user requests for code generation to route them to appropriate code generation engines or templates. It determines what type of code to generate, for which platform, using which framework, and what specific coding approach to take.'}, 'schema_understanding': {'field_purposes': {'action': 'Determines if the request is actually asking for code generation or something else entirely - acts as the primary filter', 'subAction': "Specifies the type of coding work needed - whether it's functional coding, UI/visual work, error handling, or general non-coding tasks", 'platform': 'Identifies the target deployment platform and whether the application needs dynamic functionality or is purely static', 'framework': 'Determines the specific technology stack to use for code generation - currently supports React and Flutter', 'languageType': 'Specifies the exact programming language and framework combination for more granular code generation targeting'}, 'classification_logic': 'The system should first determine if code generation is needed, then progressively narrow down the technical requirements from platform type to specific framework and language. Each field acts as a filter to route the request to the most appropriate code generation pipeline.', 'decision_criteria': 'Classification should be based on explicit mentions of technologies, platform indicators (web/mobile), functionality requirements (dynamic vs static), and UI-focused vs logic-focused language in the user request.', 'field_relationships': 'The fields form a hierarchy: action gates everything, platform determines the broad category, framework specifies the technology, and languageType provides the final technical specification. There are logical dependencies - REACT framework should typically pair with REACT_JAVASCRIPT languageType.'}, 'domain_insights': {'input_patterns': "Inputs follow patterns like 'create [app/page/screen] for [purpose] in/using [platform/technology]'. Common requests include login screens, mobile apps with specific functionality, and web applications. Users often specify the target platform (web/mobile) and sometimes the desired framework."}, 'classification_challenges': {'ambiguous_cases': "When users don't specify a framework explicitly, the system must decide whether to default to a specific framework or use NOT_FOUND. Requests mentioning 'web' without specifying static vs dynamic functionality create ambiguity.", 'edge_cases': "Generic requests like 'create a web app' without framework specification, requests for platforms not in the schema, and requests that could work on multiple platforms (like a login screen that could be web or mobile).", 'human_decision_factors': 'A human expert would consider: explicit technology mentions, implied complexity (dynamic features suggest DYNAMIC platform), industry standards (React is common for web), and whether the request focuses on UI design (VISUAL_EDITS) vs functionality (CODING).', 'current_confusion_points': "The system is over-predicting specific frameworks (REACT, FLUTTER) when the ground truth expects NOT_FOUND, suggesting it should be more conservative about framework assignment. It's also incorrectly inferring REACT_JAVASCRIPT when no specific framework is mentioned. The system seems to assume frameworks rather than requiring explicit specification."}}
Current Metrics: {}
Iteration: 1

**CURRENT PROMPT:**

    You are a classification model. Classify the input into the correct category. Return the result in JSON format. 
    The schema is as follows:
    {{
    "action": ["CODE_GENERATION", "NOT_FOUND"],
    "subAction": ["CODING", "VISUAL_EDITS", "ERROR", "GENERAL"],
    "platform": ["DYNAMIC_WEB_APPLICATION", "STATIC_WEB_APPLICATION", "DYNAMIC_MOBILE_APP", "STATIC_MOBILE_APP", "NOT_FOUND"],
    "framework": ["REACT", "FLUTTER", "NOT_FOUND"],
    "languageType": ["REACT_JAVASCRIPT", "NOT_FOUND"]
    }}
    
    IMPORTANT: Respond with a valid JSON object only. Do not include any explanations or text outside the JSON. Do not add any comments inside the JSON.
    

**JSON SCHEMA REQUIREMENT:**
{
  "action": [
    "CODE_GENERATION",
    "NOT_FOUND"
  ],
  "subAction": [
    "CODING",
    "VISUAL_EDITS",
    "ERROR",
    "GENERAL"
  ],
  "platform": [
    "DYNAMIC_WEB_APPLICATION",
    "STATIC_WEB_APPLICATION",
    "DYNAMIC_MOBILE_APP",
    "STATIC_MOBILE_APP",
    "NOT_FOUND"
  ],
  "framework": [
    "REACT",
    "FLUTTER",
    "NOT_FOUND"
  ],
  "languageType": [
    "REACT_JAVASCRIPT",
    "NOT_FOUND"
  ]
}

**FAILED CASES ANALYSIS:**
No specific failed cases provided

**HISTORY AND FEEDBACK:**



**OPTIMIZATION INSTRUCTIONS:**
1. Analyze the failed cases to identify patterns in misclassification
2. Consider the target model characteristics when optimizing the prompt
3. Make the prompt more generalizable and robust to edge cases
4. Ensure the prompt works well with the specified JSON schema
5. Maintain consistency with the task intent while improving accuracy

Provide your optimization as a JSON response with this structure:
{
    "optimized_prompt": "Your improved prompt here",
    "reasoning": "Detailed explanation of what you changed and why",
    "confidence": 0.8,
    "changes_made": ["List of specific changes made"]
}

Focus on making prompts that are:
- Clear and unambiguous for the target model
- Robust to edge cases and variations
- Well-suited for the target model's capabilities
- Consistent with the JSON schema requirements

NOTE: Do not add strict statements according to the failed cases.
