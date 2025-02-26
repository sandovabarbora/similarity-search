import json
from datetime import datetime
from typing import Any, Dict

import numpy as np
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import Session

from utils.logger import Logger, logger

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    feature_vector = Column(LargeBinary)  # Store feature vector as binary
    metadata = Column(String)  # JSON string for additional metadata

    # Relationships
    similar_to = relationship(
        "SimilarityMatch", foreign_keys="SimilarityMatch.image1_id", backref="source_image"
    )
    similar_from = relationship(
        "SimilarityMatch", foreign_keys="SimilarityMatch.image2_id", backref="target_image"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Creating new Image record: {self.filename}")

    def set_feature_vector(self, vector: np.ndarray) -> None:
        """Store feature vector as binary data."""
        try:
            self.feature_vector = vector.tobytes()
            logger.debug(
                f"Feature vector set for image {self.filename}",
                extra={"vector_shape": vector.shape},
            )
        except Exception as e:
            logger.error(f"Error setting feature vector for {self.filename}: {str(e)}")
            raise

    def get_feature_vector(self) -> np.ndarray:
        """Retrieve feature vector as numpy array."""
        try:
            vector = np.frombuffer(self.feature_vector, dtype=np.float32)
            logger.debug(
                f"Feature vector retrieved for image {self.filename}",
                extra={"vector_shape": vector.shape},
            )
            return vector
        except Exception as e:
            logger.error(f"Error retrieving feature vector for {self.filename}: {str(e)}")
            raise

    def set_metadata(self, metadata_dict: Dict[str, Any]) -> None:
        """Store metadata as JSON string."""
        try:
            self.metadata = json.dumps(metadata_dict)
            logger.debug(f"Metadata set for image {self.filename}")
        except Exception as e:
            logger.error(f"Error setting metadata for {self.filename}: {str(e)}")
            raise

    def get_metadata(self) -> Dict[str, Any]:
        """Retrieve metadata as dictionary."""
        try:
            metadata = json.loads(self.metadata) if self.metadata else {}
            logger.debug(f"Metadata retrieved for image {self.filename}")
            return metadata
        except Exception as e:
            logger.error(f"Error retrieving metadata for {self.filename}: {str(e)}")
            raise


class SimilarityMatch(Base):
    __tablename__ = "similarity_matches"

    id = Column(Integer, primary_key=True)
    image1_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    image2_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)
    match_date = Column(DateTime, default=datetime.utcnow)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(
            f"Creating new SimilarityMatch record",
            extra={
                "image1_id": self.image1_id,
                "image2_id": self.image2_id,
                "score": self.similarity_score,
            },
        )


# SQLAlchemy Event Listeners
@event.listens_for(Image, "after_insert")
def log_image_insert(mapper, connection, target):
    """Log when a new image is inserted."""
    logger.info(
        f"New image record created",
        extra={
            "image_id": target.id,
            "filename": target.filename,
            "upload_date": str(target.upload_date),
        },
    )


@event.listens_for(Image, "after_delete")
def log_image_delete(mapper, connection, target):
    """Log when an image is deleted."""
    logger.info(f"Image record deleted", extra={"image_id": target.id, "filename": target.filename})


@event.listens_for(SimilarityMatch, "after_insert")
def log_similarity_insert(mapper, connection, target):
    """Log when a new similarity match is created."""
    logger.info(
        f"New similarity match created",
        extra={
            "match_id": target.id,
            "image1_id": target.image1_id,
            "image2_id": target.image2_id,
            "score": target.similarity_score,
        },
    )


# Database Session Context Manager
class DatabaseSessionContext:
    def __init__(self, session: Session):
        self.session = session
        self.logger = Logger().add_extra_fields(session_id=id(session))

    def __enter__(self):
        self.logger.debug("Database session started")
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                "Error in database session",
                extra={"error_type": exc_type.__name__, "error_message": str(exc_val)},
            )
            self.session.rollback()
        else:
            try:
                self.session.commit()
                self.logger.debug("Database session committed successfully")
            except Exception as e:
                self.logger.error(f"Error committing session: {str(e)}")
                self.session.rollback()
                raise
        self.session.close()
        self.logger.debug("Database session closed")
