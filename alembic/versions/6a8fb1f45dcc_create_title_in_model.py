"""create title in model

Revision ID: 6a8fb1f45dcc
Revises: e12e84c4a725
Create Date: 2025-05-29 19:12:46.069393

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a8fb1f45dcc"
down_revision: Union[str, None] = "e12e84c4a725"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("books", sa.Column("title", sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("books", "title")
    # ### end Alembic commands ###
