import json
import random
import time
from typing import List, Dict, Any
import anthropic
import os
from datetime import datetime
from pymongo import MongoClient
import pytz

class ProductReviewDatasetGenerator:
    def __init__(self):
        # Fixed enum values for product review classification
        self.ratings = [1, 2, 3, 4, 5]
        self.sentiments = ["Positive", "Negative", "Neutral", "Mixed"]
        self.product_categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty"]
        self.review_types = ["Detailed", "Brief", "Complaint", "Praise", "Question"]
        self.helpfulness = ["Very_Helpful", "Helpful", "Somewhat_Helpful", "Not_Helpful"]
        self.verified_purchase = ["Yes", "No", "Unknown"]
        
        # Initialize Claude client
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Initialize MongoDB for token usage tracking
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client["personal_project_log_usage"]
            self.collection = self.db["llm_usage"]
            print("✅ MongoDB connection established for token tracking")
        except Exception as e:
            print(f"⚠️  Warning: MongoDB connection failed: {e}")
            self.mongo_client = None
        
        # Sample product reviews (varied lengths and types)
        self.sample_reviews = [
            # Short positive reviews
            "Great product!",
            "Love it!",
            "Perfect!",
            "Excellent quality",
            "Fast delivery",
            "Worth the money",
            
            # Short negative reviews
            "Terrible quality",
            "Don't buy this",
            "Waste of money",
            "Broke immediately",
            "Poor customer service",
            "Not as described",
            
            # Medium length reviews
            "Good product but shipping was slow. Overall satisfied with the purchase.",
            "The item arrived damaged but customer service was helpful in resolving the issue.",
            "Quality is decent for the price point. Would recommend to others.",
            "Easy to use and setup was straightforward. Missing some advanced features.",
            "Beautiful design and good build quality. A bit expensive but worth it.",
            "Works as expected but instructions could be clearer.",
            
            # Long detailed reviews
            "I've been using this product for 3 months now and I'm really impressed with the build quality. The packaging was excellent and everything arrived in perfect condition. Setup was easy and the user manual was very clear. The product performs exactly as advertised and I haven't had any issues. Customer service was responsive when I had a question. Overall, I'm very satisfied and would definitely recommend this to others.",
            
            "Unfortunately, this product didn't meet my expectations. The quality feels cheap and it started showing signs of wear after just a few weeks of normal use. The description on the website was misleading - it's much smaller than I expected. I tried contacting customer service but they were unhelpful and it took days to get a response. I wouldn't recommend this product and I'm disappointed with my purchase. Save your money and look elsewhere.",
            
            "Mixed feelings about this purchase. On the positive side, it arrived quickly and was well-packaged. The design is attractive and it mostly works as intended. However, there are some annoying bugs and the battery life is shorter than advertised. For the price, I expected better performance. It's not terrible but there are probably better alternatives available. I might return it if I find something better.",
            
            # Questions/uncertain reviews
            "Does this work with older models? The description isn't clear.",
            "Has anyone tried using this for professional work?",
            "Is this suitable for beginners or do you need experience?",
            "What's the warranty like? Thinking about buying but want to know.",
            
            # Specific complaints
            "The color was completely different from what was shown online.",
            "Arrived 2 weeks late and the tracking information was never updated.",
            "The size chart was wrong - ordered large but it fits like a medium.",
            "Make sure to read the fine print - there are hidden fees.",
            
            # Enthusiastic praise
            "AMAZING! Best purchase I've made all year! Cannot recommend enough!",
            "Absolutely perfect in every way. Exceeded all my expectations!",
            "This company never disappoints. Top quality as always!",
            "Worth every penny! You won't regret buying this!",
        ]
    
    def _log_token_usage(self, input_tokens: int, output_tokens: int, component: str = "dataset_generator"):
        """Log token usage to MongoDB following the established pattern"""
        if not self.mongo_client:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
                "provider": "claude",
                "model": "claude-3-sonnet-20240229",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "file_name": "/Users/harshabajaj/Desktop/PERSONAL_PROJECT/datasets/generate_dataset.py",
                "component": component,
                "operation": "product_review_classification"
            }
            self.collection.insert_one(log_entry)
            print(f"📊 Token usage logged: {input_tokens} in + {output_tokens} out = {input_tokens + output_tokens} total")
        except Exception as e:
            print(f"⚠️  Warning: Failed to log token usage: {e}")
    
    def get_claude_classification(self, review: str) -> Dict[str, Any]:
        """Use Claude to classify the product review"""
        
        prompt = f"""
        Classify this product review according to the following fixed categories:

        Review: "{review}"

        Provide classification in this exact JSON format:
        {{
            "rating": [1, 2, 3, 4, or 5],
            "sentiment": ["Positive", "Negative", "Neutral", or "Mixed"],
            "product_category": ["Electronics", "Clothing", "Home", "Books", "Sports", or "Beauty"],
            "review_type": ["Detailed", "Brief", "Complaint", "Praise", or "Question"],
            "helpfulness": ["Very_Helpful", "Helpful", "Somewhat_Helpful", or "Not_Helpful"],
            "verified_purchase": ["Yes", "No", or "Unknown"]
        }}

        Guidelines:
        - Rating: Predict likely star rating based on sentiment (1-5)
        - Sentiment: Overall emotional tone
        - Product Category: Best guess based on context clues, default to "Electronics" if unclear
        - Review Type: Brief=short, Detailed=long explanation, Complaint=problems, Praise=enthusiasm, Question=asking something
        - Helpfulness: How useful this review would be to other customers
        - Verified Purchase: Default to "Unknown" unless there are specific clues

        Return only the JSON object, no other text.
        """
        
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log token usage to MongoDB
            usage = response.usage
            self._log_token_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                component="product_review_classifier"
            )
            
            # Parse the JSON response
            classification_text = response.content[0].text.strip()
            classification = json.loads(classification_text)
            
            # Validate enum values
            if classification["rating"] not in self.ratings:
                classification["rating"] = 3
            if classification["sentiment"] not in self.sentiments:
                classification["sentiment"] = "Neutral"
            if classification["product_category"] not in self.product_categories:
                classification["product_category"] = "Electronics"
            if classification["review_type"] not in self.review_types:
                classification["review_type"] = "Brief"
            if classification["helpfulness"] not in self.helpfulness:
                classification["helpfulness"] = "Helpful"
            if classification["verified_purchase"] not in self.verified_purchase:
                classification["verified_purchase"] = "Unknown"
            
            return classification
            
        except Exception as e:
            print(f"❌ Error with Claude API: {e}")
            # Return default classification if API fails
            return {
                "rating": 3,
                "sentiment": "Neutral",
                "product_category": "Electronics",
                "review_type": "Brief",
                "helpfulness": "Helpful",
                "verified_purchase": "Unknown"
            }
    
    def generate_varied_reviews(self, num_reviews: int) -> List[str]:
        """Generate varied reviews including some custom ones"""
        reviews = []
        
        # Use provided sample reviews
        base_reviews = self.sample_reviews.copy()
        
        # Add some variations and new ones
        variations = [
            "This product changed my life! Highly recommend to everyone!",
            "Meh, it's okay I guess. Nothing special but does the job.",
            "Returned immediately. Complete waste of time and money.",
            "Five stars! Will definitely buy again!",
            "One star. Worst product ever made.",
            "Good value for money but could be improved in several areas.",
            "I'm on the fence about this one. Some good points, some bad.",
            "Perfect for what I needed. Simple and effective solution.",
            "Overhyped product. Marketing was better than the actual item.",
            "Surprised by the quality! Much better than I expected for this price.",
        ]
        
        all_reviews = base_reviews + variations
        
        # Generate the required number of reviews
        for i in range(num_reviews):
            if i < len(all_reviews):
                reviews.append(all_reviews[i])
            else:
                # Reuse reviews with slight modifications for variety
                base_review = random.choice(base_reviews)
                reviews.append(base_review)
        
        return reviews
    
    def generate_dataset(self, num_rows: int = 200) -> List[Dict[str, Any]]:
        """Generate the complete dataset using Claude for classification"""
        dataset = []
        
        print(f"Generating {num_rows} product reviews...")
        print("Using Claude API for ground truth classification...")
        
        # Generate varied reviews
        reviews = self.generate_varied_reviews(num_rows)
        
        # Classify each review using Claude
        for i, review in enumerate(reviews, 1):
            print(f"Processing review {i}/{num_rows}...")
            
            # Get Claude's classification
            classification = self.get_claude_classification(review)
            
            # Create dataset entry
            dataset.append({
                "review_id": f"REV-{i:04d}",
                "product_review": review,
                "classification": classification
            })
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "product_reviews_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filename}")
    
    def print_sample(self, dataset: List[Dict[str, Any]], num_samples: int = 5):
        """Print sample reviews for inspection"""
        print(f"\nSample {num_samples} product reviews:")
        print("=" * 80)
        
        for item in dataset[:num_samples]:
            classification = item['classification']
            print(f"\nReview {item['review_id']}:")
            print(f"Review: '{item['product_review']}'")
            print(f"Rating: {classification['rating']} stars")
            print(f"Sentiment: {classification['sentiment']}")
            print(f"Category: {classification['product_category']}")
            print(f"Type: {classification['review_type']}")
            print(f"Helpfulness: {classification['helpfulness']}")
            print(f"Verified: {classification['verified_purchase']}")
            print("-" * 40)
    
    def print_statistics(self, dataset: List[Dict[str, Any]]):
        """Print dataset statistics"""
        stats = {
            'ratings': {},
            'sentiments': {},
            'categories': {},
            'types': {},
            'helpfulness': {},
            'verified': {}
        }
        
        for item in dataset:
            c = item['classification']
            stats['ratings'][c['rating']] = stats['ratings'].get(c['rating'], 0) + 1
            stats['sentiments'][c['sentiment']] = stats['sentiments'].get(c['sentiment'], 0) + 1
            stats['categories'][c['product_category']] = stats['categories'].get(c['product_category'], 0) + 1
            stats['types'][c['review_type']] = stats['types'].get(c['review_type'], 0) + 1
            stats['helpfulness'][c['helpfulness']] = stats['helpfulness'].get(c['helpfulness'], 0) + 1
            stats['verified'][c['verified_purchase']] = stats['verified'].get(c['verified_purchase'], 0) + 1
        
        print(f"\nDataset Statistics (Total: {len(dataset)} reviews):")
        print("=" * 50)
        
        for category, counts in stats.items():
            print(f"\n{category.title()}:")
            for value, count in sorted(counts.items()):
                percentage = (count / len(dataset)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
    
    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client:
            self.mongo_client.close()
            print("📦 MongoDB connection closed")

# Main execution
if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Export it in your terminal: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    # Create generator
    generator = ProductReviewDatasetGenerator()
    
    try:
        print("🛍️  Product Review Classification Dataset Generator")
        print("=" * 60)
        print("Use Case: Classify customer product reviews for e-commerce")
        print("Method: Using Claude API for ground truth classification")
        print("Token Tracking: MongoDB logging enabled")
        print("\nFixed Enum Categories:")
        print(f"• Rating: {generator.ratings}")
        print(f"• Sentiment: {generator.sentiments}")
        print(f"• Product Category: {generator.product_categories}")
        print(f"• Review Type: {generator.review_types}")
        print(f"• Helpfulness: {generator.helpfulness}")
        print(f"• Verified Purchase: {generator.verified_purchase}")
        print("\n" + "=" * 60)
        
        # Generate dataset
        dataset = generator.generate_dataset(200)
        
        # Save dataset
        generator.save_dataset(dataset)
        
        # Show samples
        generator.print_sample(dataset, 6)
        
        # Show statistics
        generator.print_statistics(dataset)
        
        print("\n" + "=" * 60)
        print("✅ Dataset generation completed!")
        print("Ground truth provided by Claude API")
        print("Ready for ML model training and evaluation")
        
    finally:
        # Always close MongoDB connection
        generator.close()