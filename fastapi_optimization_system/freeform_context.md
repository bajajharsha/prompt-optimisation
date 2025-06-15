You are an expert prompt optimization system. Your task is to improve the given prompt for better performance on the target model.

**TARGET MODEL INFORMATION:**
- Provider: groq
- Model: llama-3.3-70b-versatile

**TASK CONTEXT:**
Intent: {'classification_intent': {'primary_purpose': 'This system classifies software development requests/queries to understand what type of code generation or development assistance is needed, specifically focusing on web and mobile application development with React and Flutter frameworks.'}, 'schema_understanding': {'field_purposes': {'action': 'Determines if the request is asking for code generation or if the intent cannot be determined - this is the primary action classifier', 'subAction': "Specifies the type of development work needed - whether it's general coding, visual/UI work, error resolution, or general assistance", 'platform': 'Identifies the target deployment platform - distinguishing between web applications, mobile apps, and whether they are static or dynamic', 'framework': 'Determines the specific development framework being used or requested (React for web, Flutter for mobile)', 'languageType': 'Specifies the programming language variant, currently only supporting React JavaScript with room for expansion'}, 'classification_logic': "The system should work hierarchically: first determine if it's a code generation request, then identify the specific type of work (UI vs logic vs error-fixing), then determine the target platform and framework, and finally the language specifics.", 'decision_criteria': 'Classification should be based on keywords and context clues in the user query: framework mentions (Flutter, React), platform indicators (mobile, web, app), task type indicators (UI, screen, error messages, integration), and technical terminology.', 'field_relationships': 'The fields are interdependent: platform and framework should align (Flutter→mobile, React→web), languageType should match the framework, and subAction should reflect the complexity and nature of the request.'}, 'domain_insights': {'input_patterns': 'Training data shows queries are typically short, informal requests for creating UI components (login screens, splash screens), adding functionality (sign up screens, database integration), or solving technical problems. Queries often mention specific frameworks explicitly and focus heavily on mobile app development with Flutter.'}, 'classification_challenges': {'ambiguous_cases': "Queries that don't specify a framework or platform (like 'add a sign up screen') require inference from context or previous conversation. Generic requests for applications without technical specifics are difficult to classify precisely.", 'edge_cases': 'Error messages with technical details need to be classified as ERROR subAction while still identifying the correct framework and platform. Requests for multi-screen applications may not clearly indicate if they want static or dynamic functionality.', 'human_decision_factors': 'A human expert would consider: explicit framework mentions, implied platform from context, complexity of the request (simple UI vs full functionality), presence of error messages or technical problems, and whether the request is about visual design or backend integration.', 'current_confusion_points': "The system is failing to produce valid JSON responses and is using incorrect case formatting (lowercase 'flutter' instead of 'FLUTTER'). It's also generating explanatory text instead of just classification JSON, and appears to be missing required fields in responses."}}
Current Metrics: {'validation_metrics': {'valid_json_accuracy': 0.6938775510204082, 'exact_match_accuracy': 0.0, 'schema_compliance_accuracy': 0.9117647058823529, 'all_fields_present_accuracy': 0.0}, 'enum_field_metrics': {'action': {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0, 'accuracy': 0.0, 'per_class': {}}, 'subAction': {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0, 'accuracy': 0.0, 'per_class': {}}, 'platform': {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0, 'accuracy': 0.0, 'per_class': {}}, 'framework': {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0, 'accuracy': 0.0, 'per_class': {}}, 'languageType': {'macro_precision': 0.0, 'macro_recall': 0.0, 'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0, 'micro_f1': 0.0, 'accuracy': 0.0, 'per_class': {}}}, 'field_insights': {'action': {'missing_rate': 0.673469387755102, 'most_common_invalid_values': {"Replace 'Airbnb' with 'Homely'": 1}, 'most_common_confusions': {}, 'error_severity': 'HIGH'}, 'subAction': {'missing_rate': 0.6938775510204082, 'most_common_invalid_values': {}, 'most_common_confusions': {}, 'error_severity': 'HIGH'}, 'platform': {'missing_rate': 0.6938775510204082, 'most_common_invalid_values': {}, 'most_common_confusions': {}, 'error_severity': 'HIGH'}, 'framework': {'missing_rate': 0.6530612244897959, 'most_common_invalid_values': {'flutter': 1, 'Flutter': 1}, 'most_common_confusions': {}, 'error_severity': 'HIGH'}, 'languageType': {'missing_rate': 0.6938775510204082, 'most_common_invalid_values': {}, 'most_common_confusions': {}, 'error_severity': 'HIGH'}}, 'optimization_opportunities': {'prompt_engineering': ['Add explicit JSON formatting instructions and examples'], 'schema_clarification': ['Provide clearer enum value definitions and constraints'], 'example_enhancement': ["Provide more examples for 'action' field classifications", "Provide more examples for 'framework' field classifications"], 'instruction_refinement': ['Emphasize requirement for all fields to be present', "Add specific guidance for 'action' field", "Add specific guidance for 'subAction' field", "Add specific guidance for 'platform' field", "Add specific guidance for 'framework' field", "Add specific guidance for 'languageType' field"]}, 'summary': {'average_enum_macro_f1': 0.0, 'average_enum_macro_precision': 0.0, 'average_enum_macro_recall': 0.0, 'total_examples': 49}, 'overall_accuracy': 0.0, 'evaluation_metadata': {'evaluation_type': 'train_baseline', 'model_used': 'llama-3.3-70b-versatile', 'num_samples': 49}}
Iteration: 1

**CURRENT PROMPT:**
You are a classification model. Classify the input into the correct category. Return the result in JSON format.

-

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
- Input: 'create flutter fb splash screen ui(query)' | Expected: {"action": "CODE_GENERATION", "platform": "DYNAMIC_MOBILE_APP", "framework": "FLUTTER", "subAction": "VISUAL_EDITS", "languageType": "NOT_FOUND"} | Got: Here's an example of how you can create a Facebook-style splash screen UI in Flutter:

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Facebook Splash Screen',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => HomeScreen()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFF4267B2),
              Color(0xFF3A5998),
            ],
          ),
        ),
        child: Center(
          child: Image.asset(
            'assets/facebook_logo.png',
            height: 100,
            width: 100,
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Home Screen'),
      ),
      body: Center(
        child: Text('Welcome to Home Screen'),
      ),
    );
  }
}
```

In this example, we create a `SplashScreen` widget that displays a Facebook-style logo on a gradient background. After 3 seconds, the `SplashScreen` widget navigates to the `HomeScreen` widget.

You'll need to replace `'assets/facebook_logo.png'` with the actual path to the Facebook logo image in your assets.

Make sure to add the Facebook logo image to your `pubspec.yaml` file:

```yml
flutter:
  assets:
    - assets/facebook_logo.png
```

You can customize the splash screen as per your requirements.

**JSON Output:**

```json
{
  "query": "create flutter fb splash screen ui",
  "category": "Flutter",
  "subCategory": "UI/UX",
  "exampleCode": "provided above"
}
```
- Input: 'ADD SUPABASE IN THIS CODE(query)' | Expected: {"action": "CODE_GENERATION", "platform": "NOT_FOUND", "framework": "NOT_FOUND", "subAction": "CODING", "languageType": "NOT_FOUND"} | Got: To classify your request, I need the actual code or query you're referring to. However, since you've mentioned "Supabase," I'll assume you're looking to integrate Supabase into an existing project or query, possibly for database management.

Supabase is an open-source alternative to Firebase, providing a PostgreSQL database, authentication, and real-time capabilities. Without the specific code, I can only provide a general example of how you might integrate Supabase into a project.

Let's say you have a simple Node.js application and you want to use Supabase as your database. Here's a basic example:

```javascript
const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase
const supabaseUrl = 'https://your-supabase-url.supabase.io';
const supabaseKey = 'your-supabase-key';
const supabaseSecret = 'your-supabase-secret';
const supabase = createClient(supabaseUrl, supabaseKey, supabaseSecret);

// Example query to fetch data from a table
async function fetchData() {
  try {
    const { data, error } = await supabase
      .from('your_table_name') // Replace 'your_table_name' with your actual table name
      .select('*');
    
    if (error) {
      console.log(error);
    } else {
      console.log(data);
    }
  } catch (error) {
    console.error(error);
  }
}

fetchData();
```

To classify this input based on the information provided:

```json
{
  "classification": "Database Integration",
  "technology": "Supabase",
  "programming_language": "JavaScript",
  "application": "Node.js"
}
```

Please provide more details or the specific code you're working with for a more tailored response.
- Input: 'make an airbnb clone (home page only) for desktop view(query)' | Expected: {"action": "CODE_GENERATION", "platform": "STATIC_WEB_APPLICATION", "framework": "NOT_FOUND", "subAction": "CODING", "languageType": "NOT_FOUND"} | Got: Below is a basic HTML and CSS structure for an Airbnb clone's home page for desktop view. Please note that for a full-fledged application, you would need a backend and possibly a frontend framework like React. However, for the sake of simplicity, this example will focus on the frontend aspect.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Airbnb Clone</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#">Become a Host</a></li>
                <li><a href="#">Help</a></li>
                <li><a href="#">Sign Up</a></li>
                <li><a href="#">Log In</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <section class="hero">
            <h1>Find your dream vacation rental</h1>
            <form>
                <input type="search" placeholder="Where are you going?">
                <input type="date" placeholder="Check-in">
                <input type="date" placeholder="Check-out">
                <input type="number" placeholder="Guests">
                <button>Search</button>
            </form>
        </section>
        <section class="explore">
            <h2>Explore</h2>
            <div class="explore-grid">
                <div class="explore-item">
                    <img src="https://picsum.photos/200/300" alt="New York">
                    <h3>New York</h3>
                </div>
                <div class="explore-item">
                    <img src="https://picsum.photos/200/301" alt="Paris">
                    <h3>Paris</h3>
                </div>
                <div class="explore-item">
                    <img src="https://picsum.photos/200/302" alt="Tokyo">
                    <h3>Tokyo</h3>
                </div>
                <div class="explore-item">
                    <img src="https://picsum.photos/200/303" alt="Sydney">
                    <h3>Sydney</h3>
                </div>
            </div>
        </section>
        <section class="discover">
            <h2>Discover</h2>
            <div class="discover-grid">
                <div class="discover-item">
                    <img src="https://picsum.photos/200/300" alt="Beachside Getaway">
                    <h3>Beachside Getaway</h3>
                </div>
                <div class="discover-item">
                    <img src="https://picsum.photos/200/301" alt="Mountain Retreat">
                    <h3>Mountain Retreat</h3>
                </div>
                <div class="discover-item">
                    <img src="https://picsum.photos/200/302" alt="City Break">
                    <h3>City Break</h3>
                </div>
                <div class="discover-item">
                    <img src="https://picsum.photos/200/303" alt="Countryside Escape">
                    <h3>Countryside Escape</h3>
                </div>
            </div>
        </section>
    </main>
    <footer>
        <p>&copy; 2024 Airbnb Clone</p>
    </footer>
</body>
</html>
```

And the CSS (in style.css file):

```css
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: Arial, sans-serif;
}

header {
    background-color: #f7f7f7;
    padding: 20px;
    text-align: right;
}

header nav ul {
    list-style: none;
}

header nav ul li {
    display: inline-block;
    margin-left: 20px;
}

header nav a {
    text-decoration: none;
    color: #333;
}

main {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
}

.hero {
    background-image: linear-gradient(to bottom, #f7f7f7, #fff);
    padding: 50px;
    text-align: center;
}

.hero h1 {
    font-size: 36px;
    margin-bottom: 20px;
}

.hero form {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.hero input, .hero button {
    padding: 10px;
    margin-bottom: 10px;
    border: none;
    border-radius: 5px;
   
- Input: 'Create only UI login screen in Flutter(query)' | Expected: {"action": "CODE_GENERATION", "platform": "DYNAMIC_MOBILE_APP", "framework": "FLUTTER", "subAction": "VISUAL_EDITS", "languageType": "NOT_FOUND"} | Got: ```json
{
  "category": "frontend development",
  "subcategory": "mobile app development",
  "language": "dart",
  "framework": "flutter",
  "task": "ui design"
}
```
- Input: 'lib/widgets/animated_background.dart:115:47: Error: The getter 'currentPoint' isn't defined for the class 'Path'. - 'Path' is from 'dart:ui'. Try correcting the name to the name of an existing getter, or defining a getter or field named 'currentPoint'. path.quadraticBezierTo(controlX, path.currentPoint!.dy, x, y); ^^^^^^^^^^^^ lib/widgets/animated_background.dart:135:59: Error: The getter 'currentPoint' isn't defined for the class 'Path'. - 'Path' is from 'dart:ui'. Try correcting the name to the name of an existing getter, or defining a getter or field named 'currentPoint'. secondPath.quadraticBezierTo(controlX, secondPath.currentPoint!.dy, x, y); ^^^^^^^^^^^^
Flutter process exited unexpectedly with code 1

GETTING ERROR(query)' | Expected: {"action": "CODE_GENERATION", "platform": "DYNAMIC_MOBILE_APP", "framework": "FLUTTER", "subAction": "ERROR", "languageType": "NOT_FOUND"} | Got: ```json
{
  "category": "Error Message",
  "subcategory": "Compiler Error",
  "language": "Dart",
  "framework": "Flutter",
  "error_type": "Undefined Getter"
}
```

**HISTORY AND FEEDBACK:**



**OPTIMIZATION INSTRUCTIONS:**
You MUST make significant improvements to the prompt. Analyze the failed cases carefully and make aggressive changes:

1. **CRITICAL ANALYSIS**: Look at each failed case and identify exactly why the model made the wrong prediction
2. **PATTERN IDENTIFICATION**: Find common patterns in the failures (e.g., specific fields being misclassified)
3. **AGGRESSIVE OPTIMIZATION**: Make substantial changes to the prompt structure, not just minor tweaks
4. **SPECIFIC GUIDANCE**: Add explicit instructions for handling the types of cases that are failing
5. **SCHEMA ENFORCEMENT**: Ensure the model strictly follows the JSON schema format
6. **EDGE CASE HANDLING**: Add specific instructions for ambiguous or edge cases
7. **NOT STRICT STATEMENTS**: Do not add strict statements according to the failed cases
8. **INTENT MATCHING**: The optimized prompt MUST be more aligned with the intent of the task
9. **JSON SCHEMA**: On the basis of sample data, the schema is provided. The optimized prompt MUST be more aligned with the schema.

**REQUIRED CHANGES:**
- Add explicit examples of correct classifications for problematic cases but not strict statements according to the failed cases
- Include specific instructions for each field in the schema but not strict statements according to the failed cases
- Add clear decision-making criteria for ambiguous inputs
- Restructure the prompt for better clarity and specificity
- Add validation instructions to ensure JSON format compliance

**FAILED CASE FOCUS:**
Pay special attention to the failed cases provided. For each pattern you see:
- Add specific instructions to handle that type of input but not strict statements according to the failed cases
- Include examples that demonstrate the correct classification
- Add decision-making criteria to avoid similar mistakes

Provide your optimization as a JSON response with this structure:
{
    "optimized_prompt": "Your significantly improved prompt here - MUST be substantially different from the original",
    "reasoning": "Detailed explanation of the major changes you made and why they address the specific failed cases",
    "confidence": 0.8,
    "changes_made": ["List of specific major changes made - should be substantial improvements"]
}

**REQUIREMENTS:**
- The optimized prompt MUST be significantly better than the original
- MUST address the specific patterns seen in failed cases
- MUST include explicit examples and decision criteria but not strict statements according to the failed cases
- MUST be more structured and comprehensive than the original
- MUST include field-specific guidance for the JSON schema
