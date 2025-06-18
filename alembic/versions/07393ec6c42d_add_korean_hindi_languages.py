"""add_korean_hindi_languages

Revision ID: 07393ec6c42d
Revises: 6bb1d7b8bd2c
Create Date: 2025-06-17 16:26:42.084942

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '07393ec6c42d'
down_revision: Union[str, None] = '6bb1d7b8bd2c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add new enum values to the existing supportedlanguage enum
    op.execute("ALTER TYPE supportedlanguage ADD VALUE 'KOREAN'")
    op.execute("ALTER TYPE supportedlanguage ADD VALUE 'HINDI'")


def downgrade() -> None:
    """Downgrade schema."""
    # Note: PostgreSQL doesn't support removing enum values directly
    # A full enum recreation would be needed, which is complex and potentially destructive
    # For now, we'll leave the enum values in place as they won't cause issues
    pass
