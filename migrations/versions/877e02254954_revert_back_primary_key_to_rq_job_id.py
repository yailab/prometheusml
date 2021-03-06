"""revert back primary key to rq job id

Revision ID: 877e02254954
Revises: 1e8647112b40
Create Date: 2022-02-22 22:29:33.229558

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "877e02254954"
down_revision = "1e8647112b40"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("task", "job_id")
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "task",
        sa.Column("job_id", sa.VARCHAR(length=36), autoincrement=False, nullable=True),
    )
    # ### end Alembic commands ###
