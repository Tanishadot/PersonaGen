"""
Dataset creation script for PersonaGen
Creates a sample empathetic dialogues dataset
"""

import json
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset():
    """Create a sample empathetic dialogues dataset"""
    try:
        logger.info("Creating sample empathetic dialogues dataset...")
        
        # Sample empathetic dialogues data
        sample_data = [
            {
                "context": "friend is feeling overwhelmed with work",
                "prompt": "I have so much work to do and I don't know how I'm going to get it all done. I feel really stressed and overwhelmed.",
                "utterance": "I can hear how overwhelmed you're feeling right now. It sounds like you're carrying a heavy burden with all this work. Remember to take things one step at a time and be kind to yourself through this process. You've handled challenging situations before, and you have the strength to get through this too.",
                "speaker_idx": "speaker_2",
                "emotion": "stressed"
            },
            {
                "context": "friend is dealing with a breakup",
                "prompt": "My boyfriend and I broke up yesterday and I'm heartbroken. I don't know how to move forward.",
                "utterance": "I'm so sorry you're going through this heartbreak. Breakups are incredibly painful, and it's completely understandable that you're feeling this way. Allow yourself to grieve and feel these emotions - they're valid. Remember that healing takes time, and it's okay to not be okay right now. I'm here to support you through this.",
                "speaker_idx": "speaker_2", 
                "emotion": "sad"
            },
            {
                "context": "friend is anxious about a presentation",
                "prompt": "I have a big presentation tomorrow and I'm really nervous. What if I mess up?",
                "utterance": "It's completely normal to feel nervous before something important like this. Your feelings show how much you care about doing well. Remember that you've prepared for this, and you know your material. Even if you stumble a bit, that doesn't define your worth or capability. Take some deep breaths, and trust in the work you've put in.",
                "speaker_idx": "speaker_2",
                "emotion": "anxious"
            },
            {
                "context": "friend is celebrating an achievement",
                "prompt": "I got the job offer I was hoping for! I'm so excited!",
                "utterance": "That's absolutely wonderful news! All your hard work and dedication has paid off, and you truly deserve this achievement. I can feel your excitement, and I'm so happy celebrating this success with you. Take a moment to really savor this victory - you've earned it!",
                "speaker_idx": "speaker_2",
                "emotion": "excited"
            },
            {
                "context": "friend is feeling lonely",
                "prompt": "I've been feeling really lonely lately. It seems like everyone else has their life together except me.",
                "utterance": "I hear how lonely you're feeling, and I want you to know that you're not alone in this experience. Many people feel this way, even when it doesn't look like it from the outside. Your feelings are valid, and it's brave of you to share them. I'm here for you, and I value our connection. You're doing better than you think.",
                "speaker_idx": "speaker_2",
                "emotion": "lonely"
            },
            {
                "context": "friend is dealing with illness",
                "prompt": "I've been sick for over a week now and I'm getting frustrated. I just want to feel normal again.",
                "utterance": "I can understand how frustrating and draining it must be to deal with illness for this long. It's completely valid to feel this way - being sick takes such a toll on both body and mind. Be gentle with yourself as you heal. Your body is working hard to recover, and it's okay to need extra time and rest. I'm thinking of you and sending healing energy your way.",
                "speaker_idx": "speaker_2",
                "emotion": "frustrated"
            },
            {
                "context": "friend is worried about finances",
                "prompt": "I'm really worried about my financial situation. I don't know if I'll be able to pay all my bills this month.",
                "utterance": "Financial stress is incredibly heavy to carry, and I can hear the worry in your voice. These are challenging times, and your concerns are completely valid. You're being responsible by facing this head-on. Remember that you've navigated difficult situations before, and you have resourcefulness you might not even recognize right now. We can brainstorm some options together if you'd like.",
                "speaker_idx": "speaker_2",
                "emotion": "worried"
            },
            {
                "context": "friend is feeling proud of personal growth",
                "prompt": "I finally stood up for myself in a difficult situation today. I'm proud of how I handled it.",
                "utterance": "That's such a significant step in your personal growth, and you should absolutely feel proud of yourself! Standing up for yourself, especially in difficult situations, takes tremendous courage and self-respect. You're showing how much you've grown and how much you value yourself. This is a real achievement worth celebrating!",
                "speaker_idx": "speaker_2",
                "emotion": "proud"
            },
            {
                "context": "friend is feeling confused about life decisions",
                "prompt": "I don't know what I want to do with my life anymore. Everything feels so uncertain.",
                "utterance": "That feeling of uncertainty about your life path can be really unsettling and confusing. It's completely normal to question things and feel lost sometimes - this is actually often a sign that you're growing and evolving. Be patient with yourself as you navigate these questions. You don't need all the answers right now. Sometimes the path becomes clearer just by taking the next small step.",
                "speaker_idx": "speaker_2",
                "emotion": "confused"
            },
            {
                "context": "friend is dealing with disappointment",
                "prompt": "I didn't get into the graduate program I worked so hard for. I feel like such a failure.",
                "utterance": "I can only imagine how disappointing and heartbreaking this must feel for you, especially after putting in so much hard work. Please know that this outcome doesn't define your worth or your capabilities. You are not a failure - you're someone who was brave enough to pursue something meaningful. This is a setback, not a final verdict on your potential. Your journey isn't over.",
                "speaker_idx": "speaker_2",
                "emotion": "disappointed"
            }
        ]
        
        # Create variations to expand the dataset
        expanded_data = []
        
        # Add original samples
        expanded_data.extend(sample_data)
        
        # Add some variations with different emotions
        variations = [
            {
                "context": "friend is feeling overwhelmed with school",
                "prompt": "I have three exams this week and I haven't started studying. I feel so overwhelmed and I think I'm going to fail.",
                "utterance": "I can hear how overwhelmed you're feeling with all these exams coming up. That's a lot of pressure to handle at once. Take a deep breath - you're more capable than you think right now. Let's break this down into smaller, manageable steps. Even starting with one subject for 30 minutes is progress. You don't have to be perfect, you just have to start.",
                "speaker_idx": "speaker_2",
                "emotion": "overwhelmed"
            },
            {
                "context": "friend is dealing with family conflict",
                "prompt": "I had a big fight with my mom and now we're not talking. I feel terrible about it.",
                "utterance": "Family conflicts can be so painful and emotionally draining. I'm sorry you're going through this difficult situation with your mom. It's clear you care about your relationship, and that tension you're feeling shows how much it matters to you. Remember that conflicts happen even in loving relationships. Give yourself some grace, and know that things can get better with time and communication.",
                "speaker_idx": "speaker_2",
                "emotion": "hurt"
            },
            {
                "context": "friend is celebrating personal achievement",
                "prompt": "I finally finished that project I've been working on for months! I can't believe it's done.",
                "utterance": "That's amazing! Completing a long-term project like that takes such dedication and persistence. You should feel incredibly proud of yourself for seeing it through to the end. This is a testament to your commitment and hard work. Take time to celebrate this accomplishment - you've earned every bit of this success!",
                "speaker_idx": "speaker_2",
                "emotion": "accomplished"
            }
        ]
        
        expanded_data.extend(variations)
        
        # Add more empathetic responses with different contexts
        more_samples = [
            {
                "context": "friend is feeling insecure",
                "prompt": "Sometimes I feel like I'm not good enough at anything. Everyone seems so much more talented than me.",
                "utterance": "I hear how you're doubting yourself right now, and I want you to know that these feelings, while painful, don't reflect the truth of who you are. We all have moments of insecurity, even when we're capable and talented. You have unique strengths and qualities that make you special. Comparison is the thief of joy, and you're on your own beautiful journey.",
                "speaker_idx": "speaker_2",
                "emotion": "insecure"
            },
            {
                "context": "friend is dealing with burnout",
                "prompt": "I feel completely burned out. I have no energy or motivation for anything anymore.",
                "utterance": "Burnout is such an exhausting and overwhelming experience, and I'm sorry you're going through this. Your body and mind are telling you they need rest and care. It's okay to step back and slow down. You've been carrying so much for so long. Be gentle with yourself as you recover. This isn't permanent, and you will find your energy again.",
                "speaker_idx": "speaker_2",
                "emotion": "burned_out"
            }
        ]
        
        expanded_data.extend(more_samples)
        
        logger.info(f"Created dataset with {len(expanded_data)} samples")
        
        # Save to JSON file
        output_path = "data/empathetic_dialogues.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(expanded_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        
        # Print sample data for verification
        logger.info("Sample data:")
        for i in range(min(3, len(expanded_data))):
            sample = expanded_data[i]
            logger.info(f"  Sample {i+1}:")
            logger.info(f"    Context: {sample['context']}")
            logger.info(f"    Prompt: {sample['prompt'][:80]}...")
            logger.info(f"    Response: {sample['utterance'][:80]}...")
            logger.info(f"    Emotion: {sample['emotion']}")
            logger.info("---")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return False

if __name__ == "__main__":
    success = create_sample_dataset()
    if success:
        print("Dataset creation completed successfully!")
    else:
        print("Failed to create dataset")
