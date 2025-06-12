#!/usr/bin/env python3
"""
Script to add image_hashes table to existing database.
"""

import psycopg2
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Connect to database
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cur = conn.cursor()
        
        # Create image_hashes table
        print("Creating image_hashes table...")
        cur.execute('''
        CREATE TABLE IF NOT EXISTS image_hashes (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            media_id VARCHAR(36) NOT NULL UNIQUE,
            dhash VARCHAR(32),
            phash VARCHAR(32),
            ahash VARCHAR(32),
            dhash_16 VARCHAR(128),
            phash_16 VARCHAR(128),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        ''')
        
        # Create indexes
        print("Creating indexes...")
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_dhash ON image_hashes (dhash);',
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_phash ON image_hashes (phash);',
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_ahash ON image_hashes (ahash);',
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_dhash_16 ON image_hashes (dhash_16);',
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_phash_16 ON image_hashes (phash_16);',
            'CREATE INDEX IF NOT EXISTS idx_image_hashes_media_id ON image_hashes (media_id);'
        ]
        
        for idx in indexes:
            cur.execute(idx)
        
        # Create update trigger
        print("Creating update trigger...")
        cur.execute('''
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        ''')
        
        cur.execute('''
        CREATE OR REPLACE TRIGGER update_image_hashes_updated_at 
            BEFORE UPDATE ON image_hashes 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        print('✅ Image hashes table created successfully!')
        
    except Exception as e:
        print(f"❌ Error creating image_hashes table: {e}")
        raise

if __name__ == "__main__":
    main() 