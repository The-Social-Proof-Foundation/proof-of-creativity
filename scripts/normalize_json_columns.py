#!/usr/bin/env python3
"""
Script to normalize JSON columns by extracting important data into dedicated columns.
Uses hybrid approach: extract critical data while keeping JSON for less important fields.
"""

import psycopg2
import psycopg2.extras
import json
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Connect to database
        conn = psycopg2.connect(os.getenv('DATABASE_URL', 'postgres://tsdbadmin:jl5i45hpj49cox2d@hpdkpsps2i.rrpamcwlae.tsdb.cloud.timescale.com:37797/tsdb?sslmode=require'))
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        print("🔄 Adding new columns to normalize JSON data...")
        
        # ===== ATTRIBUTION RECORDS =====
        print("\n1. Attribution Records - Adding columns...")
        attribution_columns = [
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS file_hash VARCHAR(128);",
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS media_type VARCHAR(20);", 
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS matches_found INTEGER DEFAULT 0;",
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS high_confidence_matches INTEGER DEFAULT 0;",
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS max_similarity_score FLOAT DEFAULT 0.0;",
            "ALTER TABLE attribution_records ADD COLUMN IF NOT EXISTS processing_time_ms FLOAT DEFAULT 0.0;",
            
            # Add indexes for the new columns
            "CREATE INDEX IF NOT EXISTS idx_attribution_records_file_hash ON attribution_records (file_hash);",
            "CREATE INDEX IF NOT EXISTS idx_attribution_records_media_type ON attribution_records (media_type);",
            "CREATE INDEX IF NOT EXISTS idx_attribution_records_matches_found ON attribution_records (matches_found);",
            "CREATE INDEX IF NOT EXISTS idx_attribution_records_max_score ON attribution_records (max_similarity_score);"
        ]
        
        for sql in attribution_columns:
            cur.execute(sql)
        
        # Extract data from proof_data JSON
        print("   Extracting data from proof_data JSON...")
        cur.execute("SELECT id, proof_data FROM attribution_records WHERE proof_data IS NOT NULL;")
        attribution_records = cur.fetchall()
        
        for record in attribution_records:
            try:
                proof_data = record['proof_data']
                if proof_data:
                    update_sql = """
                    UPDATE attribution_records 
                    SET file_hash = %s, media_type = %s, matches_found = %s, 
                        high_confidence_matches = %s, max_similarity_score = %s, processing_time_ms = %s
                    WHERE id = %s
                    """
                    cur.execute(update_sql, (
                        proof_data.get('file_hash'),
                        proof_data.get('media_type'),
                        proof_data.get('matches_found', 0),
                        proof_data.get('high_confidence_matches', 0),
                        proof_data.get('max_similarity_score', 0.0),
                        proof_data.get('processing_time_ms', 0.0),
                        record['id']
                    ))
            except Exception as e:
                print(f"   Warning: Could not extract data from record {record['id']}: {e}")
        
        print(f"   ✅ Updated {len(attribution_records)} attribution records")
        
        # ===== MEDIA FILES =====
        print("\n2. Media Files - Adding columns...")
        media_files_columns = [
            "ALTER TABLE media_files ADD COLUMN IF NOT EXISTS matches_found INTEGER DEFAULT 0;",
            "ALTER TABLE media_files ADD COLUMN IF NOT EXISTS processing_time_ms FLOAT DEFAULT 0.0;",
            "ALTER TABLE media_files ADD COLUMN IF NOT EXISTS media_type VARCHAR(20);",
            
            # Add indexes
            "CREATE INDEX IF NOT EXISTS idx_media_files_matches_found ON media_files (matches_found);",
            "CREATE INDEX IF NOT EXISTS idx_media_files_processing_time ON media_files (processing_time_ms);",
            "CREATE INDEX IF NOT EXISTS idx_media_files_media_type ON media_files (media_type);"
        ]
        
        for sql in media_files_columns:
            cur.execute(sql)
        
        # Extract data from processing_results JSON
        print("   Extracting data from processing_results JSON...")
        cur.execute("SELECT media_id, processing_results FROM media_files WHERE processing_results IS NOT NULL;")
        media_records = cur.fetchall()
        
        for record in media_records:
            try:
                processing_results = record['processing_results']
                if processing_results:
                    update_sql = """
                    UPDATE media_files 
                    SET matches_found = %s, processing_time_ms = %s, media_type = %s
                    WHERE media_id = %s
                    """
                    cur.execute(update_sql, (
                        processing_results.get('matches_found', 0),
                        processing_results.get('processing_time_ms', 0.0),
                        processing_results.get('media_type'),
                        record['media_id']
                    ))
            except Exception as e:
                print(f"   Warning: Could not extract data from media {record['media_id']}: {e}")
        
        print(f"   ✅ Updated {len(media_records)} media files")
        
        # ===== SIMILARITY MATCHES =====
        print("\n3. Similarity Matches - Adding columns...")
        similarity_columns = [
            "ALTER TABLE similarity_matches ADD COLUMN IF NOT EXISTS match_category VARCHAR(50);",
            "ALTER TABLE similarity_matches ADD COLUMN IF NOT EXISTS embedding_type VARCHAR(20);",
            "ALTER TABLE similarity_matches ADD COLUMN IF NOT EXISTS fingerprint_hash VARCHAR(128);",
            "ALTER TABLE similarity_matches ADD COLUMN IF NOT EXISTS offset_seconds FLOAT;",
            
            # Add indexes
            "CREATE INDEX IF NOT EXISTS idx_similarity_matches_category ON similarity_matches (match_category);",
            "CREATE INDEX IF NOT EXISTS idx_similarity_matches_embedding_type ON similarity_matches (embedding_type);",
            "CREATE INDEX IF NOT EXISTS idx_similarity_matches_fingerprint_hash ON similarity_matches (fingerprint_hash);"
        ]
        
        for sql in similarity_columns:
            cur.execute(sql)
        
        # Extract data from match_details JSON
        print("   Extracting data from match_details JSON...")
        cur.execute("SELECT id, match_details FROM similarity_matches WHERE match_details IS NOT NULL;")
        similarity_records = cur.fetchall()
        
        for record in similarity_records:
            try:
                match_details = record['match_details']
                if match_details:
                    update_sql = """
                    UPDATE similarity_matches 
                    SET match_category = %s, embedding_type = %s, fingerprint_hash = %s, offset_seconds = %s
                    WHERE id = %s
                    """
                    cur.execute(update_sql, (
                        match_details.get('match_category'),
                        match_details.get('embedding_type'),
                        match_details.get('fingerprint_hash'),
                        match_details.get('offset'),
                        record['id']
                    ))
            except Exception as e:
                print(f"   Warning: Could not extract data from match {record['id']}: {e}")
        
        print(f"   ✅ Updated {len(similarity_records)} similarity matches")
        
        # Commit all changes
        conn.commit()
        cur.close()
        conn.close()
        
        print("\n🎉 JSON normalization completed successfully!")
        print("\n📊 New columns added:")
        print("   Attribution Records: file_hash, media_type, matches_found, high_confidence_matches, max_similarity_score, processing_time_ms")
        print("   Media Files: matches_found, processing_time_ms, media_type") 
        print("   Similarity Matches: match_category, embedding_type, fingerprint_hash, offset_seconds")
        print("\n📝 JSON columns retained for less critical data")
        
    except Exception as e:
        print(f"❌ Error normalizing JSON columns: {e}")
        raise

if __name__ == "__main__":
    main() 