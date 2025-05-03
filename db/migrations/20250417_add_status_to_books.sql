# Migration script: add status column to books table
# Usage: Run this SQL in your database to add the async job status tracking

ALTER TABLE books
ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'generating';
