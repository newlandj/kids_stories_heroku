"""fixed updated_at

Revision ID: e12e84c4a725
Revises: cbfb4bd82675
Create Date: 2025-05-04 14:11:21.529657

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e12e84c4a725"
down_revision: Union[str, None] = "cbfb4bd82675"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "books",
        "updated_at",
        server_default=sa.text("now()"),
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "books",
        "updated_at",
        server_default=None,
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
    )
    # ### end Alembic commands ###
