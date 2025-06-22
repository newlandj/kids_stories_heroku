"""add_model_tracking_to_books

Revision ID: b3af9d782123
Revises: 07393ec6c42d
Create Date: 2025-06-19 10:18:44.499939

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3af9d782123'
down_revision: Union[str, None] = '07393ec6c42d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Add model tracking columns to books table."""
    # Add tracking columns to books table
    op.add_column('books', sa.Column('text_model', sa.String(length=100), nullable=True))
    op.add_column('books', sa.Column('image_model', sa.String(length=100), nullable=True))
    op.add_column('books', sa.Column('audio_model', sa.String(length=100), nullable=True))
    op.add_column('books', sa.Column('correct_first_try', sa.Boolean(), nullable=True))
    op.add_column('books', sa.Column('creation_duration', sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema - Remove model tracking columns from books table."""
    # Remove tracking columns from books table
    op.drop_column('books', 'creation_duration')
    op.drop_column('books', 'correct_first_try')
    op.drop_column('books', 'audio_model')
    op.drop_column('books', 'image_model')
    op.drop_column('books', 'text_model')
