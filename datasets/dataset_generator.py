import json
import asyncio
import aiohttp
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

# Configuration
CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Simplified, practical classification schema
CLASSIFICATION_SCHEMA = {
    "product_category": {
        "department": ["electronics", "clothing", "home", "sports", "books", "health"],
        "category": {
            "electronics": ["smartphones", "laptops", "audio", "gaming", "accessories"],
            "clothing": ["mens", "womens", "shoes", "accessories"],
            "home": ["furniture", "kitchen", "garden", "tools"],
            "sports": ["fitness", "outdoor", "team_sports", "equipment"],
            "books": ["books", "movies", "music", "games"],
            "health": ["skincare", "supplements", "personal_care"]
        }
    },
    "price_range": ["budget", "mid_range", "premium", "luxury"]
}

@dataclass
class ProductListing:
    title: str
    description: str
    price: float
    brand: str
    condition: str
    images_count: int
    seller_rating: float

class DatasetGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_sample_products(self, count: int = 50) -> List[ProductListing]:
        """Generate sample product listings for classification"""
        products = []
        
        sample_data = [
            # Electronics
            ("iPhone 14 Pro Max 256GB Space Black", "Latest Apple iPhone with ProRAW camera and A16 Bionic chip. Includes original box and charger.", 899.99, "Apple", "like_new", 8, 4.8),
            ("Gaming Laptop ASUS ROG Strix", "High performance laptop with RTX 4070, 32GB RAM, 1TB SSD. Perfect for gaming and content creation.", 1599.99, "ASUS", "new", 12, 4.9),
            ("Sony WH-1000XM4 Headphones", "Noise cancelling wireless headphones with 30hr battery life. Minor scuffs on headband.", 249.99, "Sony", "good", 5, 4.6),
            ("Nintendo Switch Console", "Portable gaming console with Joy-Con controllers. Screen has light scratches.", 199.99, "Nintendo", "fair", 4, 4.2),
            
            # Clothing
            ("Vintage Levi's 501 Jeans Size 32", "Classic straight leg denim jeans from the 1990s. Some fading but no holes.", 45.00, "Levi's", "good", 5, 4.3),
            ("Nike Air Jordan 1 Retro High", "Basketball shoes in Chicago colorway, size 10. Brand new in box with tags.", 180.00, "Nike", "new", 10, 4.7),
            ("Patagonia Fleece Jacket Large", "Outdoor fleece jacket in excellent condition. Minimal pilling.", 89.99, "Patagonia", "like_new", 6, 4.5),
            ("Designer Handbag Coach", "Authentic leather handbag with dust bag. Minor wear on corners.", 299.99, "Coach", "good", 8, 4.4),
            
            # Home
            ("Wooden Coffee Table Oak", "Handcrafted solid oak coffee table with storage drawer. Some water rings on surface.", 199.99, "Custom Furniture", "fair", 6, 4.0),
            ("Instant Pot Duo 6Qt", "Multi-functional electric pressure cooker. Used a few times, like new condition.", 79.99, "Instant Pot", "like_new", 4, 4.8),
            ("Garden Tool Set", "Complete set of gardening tools including spades, pruners, and gloves.", 45.00, "Fiskars", "good", 3, 4.1),
            
            # Sports
            ("Mountain Bike Trek 29er", "Full suspension mountain bike with 21 speeds. Well maintained, some scratches.", 450.00, "Trek", "good", 9, 4.4),
            ("Yoga Mat Premium", "High-quality non-slip yoga mat, 6mm thick. Barely used.", 39.99, "Manduka", "like_new", 3, 4.6),
            
            # Books/Media
            ("The Great Gatsby First Edition", "Rare first edition hardcover from 1925. Cover worn but pages intact.", 850.00, "Scribner", "fair", 5, 3.9),
            ("Marvel Movie Collection", "Complete Phase 1-3 Blu-ray collection in original packaging.", 149.99, "Disney", "new", 7, 4.7),
            
            # Health/Beauty
            ("Skincare Set Luxury", "Complete anti-aging skincare routine with vitamin C serum and retinol cream.", 89.99, "The Ordinary", "new", 7, 4.3),
            ("Protein Powder Whey", "Unopened 5lb container of vanilla whey protein. Expires in 2025.", 59.99, "Optimum Nutrition", "new", 2, 4.5)
        ]
        
        for i in range(count):
            base_product = sample_data[i % len(sample_data)]
            # Add variation
            variation_suffix = f" - #{i+1:03d}"
            price_variation = random.uniform(0.85, 1.15)
            rating_variation = random.uniform(-0.2, 0.1)
            
            product = ProductListing(
                title=base_product[0] + variation_suffix,
                description=base_product[1],
                price=round(base_product[2] * price_variation, 2),
                brand=base_product[3],
                condition=base_product[4],
                images_count=max(1, base_product[5] + random.randint(-2, 2)),
                seller_rating=max(1.0, min(5.0, base_product[6] + rating_variation))
            )
            products.append(product)
        
        return products
    
    async def classify_product(self, product: ProductListing) -> Dict[str, Any]:
        """Send a product to Claude API for classification"""
        
        prompt = f"""
You are an expert e-commerce product classifier. Classify this product listing according to the schema below.
Respond with ONLY a valid JSON object, no additional text.

CLASSIFICATION SCHEMA:
{{
    "product_category": {{
        "department": "one of: {', '.join(CLASSIFICATION_SCHEMA['product_category']['department'])}",
        "category": "choose from the appropriate category list based on department"
    }},
    "condition": "one of: {', '.join(CLASSIFICATION_SCHEMA['condition'])}",
    "price_range": "one of: {', '.join(CLASSIFICATION_SCHEMA['price_range'])} (budget: <$50, mid_range: $50-200, premium: $200-500, luxury: >$500)",
    "listing_completeness": "one of: {', '.join(CLASSIFICATION_SCHEMA['listing_completeness'])} (complete: detailed description + many images, partial: basic info, minimal: very little info)",
    "authenticity_confidence": "one of: {', '.join(CLASSIFICATION_SCHEMA['authenticity_confidence'])} (based on brand, price, description quality, seller rating)"
}}

PRODUCT TO CLASSIFY:
Title: {product.title}
Description: {product.description}
Price: ${product.price}
Brand: {product.brand}
Listed Condition: {product.condition}
Images: {product.images_count} photos
Seller Rating: {product.seller_rating}/5.0

Respond with ONLY the JSON classification:
"""

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with self.session.post(CLAUDE_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    classification_text = result['content'][0]['text'].strip()
                    
                    try:
                        classification = json.loads(classification_text)
                        return classification
                    except json.JSONDecodeError:
                        print(f"Failed to parse: {classification_text}")
                        return self._get_fallback_classification()
                else:
                    print(f"API error: {response.status}")
                    return self._get_fallback_classification()
                    
        except Exception as e:
            print(f"Error: {e}")
            return self._get_fallback_classification()
    
    def _get_fallback_classification(self) -> Dict[str, Any]:
        """Generate a fallback classification"""
        dept = random.choice(CLASSIFICATION_SCHEMA['product_category']['department'])
        return {
            "product_category": {
                "department": dept,
                "category": random.choice(CLASSIFICATION_SCHEMA['product_category']['category'].get(dept, ["other"]))
            },
            "condition": random.choice(CLASSIFICATION_SCHEMA['condition']),
            "price_range": random.choice(CLASSIFICATION_SCHEMA['price_range']),
            "listing_completeness": random.choice(CLASSIFICATION_SCHEMA['listing_completeness']),
            "authenticity_confidence": random.choice(CLASSIFICATION_SCHEMA['authenticity_confidence'])
        }
    
    async def generate_dataset(self, num_samples: int = 50, batch_size: int = 3) -> List[Dict[str, Any]]:
        """Generate complete dataset"""
        print(f"Generating {num_samples} classified samples...")
        
        products = self.generate_sample_products(num_samples)
        dataset = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            tasks = [self.classify_product(product) for product in batch]
            classifications = await asyncio.gather(*tasks)
            
            for product, classification in zip(batch, classifications):
                dataset_entry = {
                    "input": {
                        "title": product.title,
                        "description": product.description,
                        "price": product.price,
                        "brand": product.brand,
                        "listed_condition": product.condition,
                        "images_count": product.images_count,
                        "seller_rating": product.seller_rating
                    },
                    "output": classification,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "sample_id": len(dataset) + 1
                    }
                }
                dataset.append(dataset_entry)
            
            print(f"Processed {min(i + batch_size, len(products))}/{len(products)}")
            await asyncio.sleep(0.5)  # Rate limiting
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "simple_ecommerce_dataset.json"):
        """Save dataset to file"""
        output_data = {
            "metadata": {
                "dataset_name": "Simple E-commerce Classification Dataset",
                "description": "Simplified product classification with practical, objective categories",
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(dataset),
                "schema": CLASSIFICATION_SCHEMA
            },
            "data": dataset
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        return filename

async def main():
    if not CLAUDE_API_KEY:
        print("Error: Set ANTHROPIC_API_KEY environment variable")
        return
    
    num_samples = 25  # Start smaller
    
    async with DatasetGenerator(CLAUDE_API_KEY) as generator:
        try:
            dataset = await generator.generate_dataset(num_samples)
            filename = generator.save_dataset(dataset)
            
            print(f"\n✅ Complete! Generated {len(dataset)} samples")
            print(f"💾 Saved to: {filename}")
            
            if dataset:
                print(f"\n📝 Sample:")
                print(json.dumps(dataset[0], indent=2))
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 