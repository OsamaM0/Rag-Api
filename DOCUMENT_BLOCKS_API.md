# Document Blocks API

The Document Blocks API provides functionality to manage and process document blocks that have a many-to-one relationship with documents. Each document can have multiple blocks that represent different sections, paragraphs, tables, or other structural elements.

## Features

- **CRUD Operations**: Create, read, update, and delete document blocks
- **Bulk Operations**: Create multiple blocks in a single request
- **JSON Upload**: Upload JSON files to automatically parse and create blocks
- **Search and Filtering**: Search blocks by content, document, content type, etc.
- **Hierarchical Structure**: Support for parent-child relationships between blocks
- **Rich Metadata**: Store coordinates, content types, section types, and priorities

## Database Schema

The `rag_document_blocks` table has the following structure:

```sql
CREATE TABLE rag_document_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    block_idx INTEGER NOT NULL,
    document_id UUID REFERENCES documents(uuid) ON DELETE CASCADE,
    name TEXT NOT NULL,
    content TEXT,
    level INTEGER NOT NULL,
    page_idx INTEGER NOT NULL,
    tag TEXT NOT NULL,
    block_class TEXT,
    x0 FLOAT,
    y0 FLOAT,
    x1 FLOAT,
    y1 FLOAT,
    parent_idx INTEGER,
    content_type TEXT DEFAULT 'regular',
    section_type TEXT,
    demand_priority INTEGER,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, block_idx)
);
```

## API Endpoints

### Base URL: `/document-blocks`

#### 1. List Document Blocks
- **GET** `/document-blocks/`
- **Query Parameters**:
  - `document_id` (optional): Filter by document UUID
  - `query` (optional): Full-text search query
  - `content_type` (optional): Filter by content type
  - `section_type` (optional): Filter by section type
  - `page` (default: 1): Page number
  - `page_size` (default: 10): Items per page

#### 2. Create Single Document Block
- **POST** `/document-blocks/?document_id={document_uuid}`
- **Body**: `DocumentBlockCreate` model

```json
{
  "block_idx": 1,
  "name": "intro_paragraph",
  "content": "This is the introduction paragraph.",
  "level": 0,
  "page_idx": 0,
  "tag": "para",
  "block_class": "text",
  "x0": 10.0,
  "y0": 20.0,
  "x1": 200.0,
  "y1": 40.0,
  "content_type": "regular"
}
```

#### 3. Bulk Create Document Blocks
- **POST** `/document-blocks/bulk`
- **Body**: `DocumentBlocksBulkCreate` model

```json
{
  "document_id": "document-uuid-here",
  "blocks": [
    {
      "block_idx": 1,
      "name": "block_1",
      "content": "First block content",
      "level": 0,
      "page_idx": 0,
      "tag": "para",
      "content_type": "regular"
    },
    {
      "block_idx": 2,
      "name": "block_2",
      "content": "Second block content",
      "level": 1,
      "page_idx": 0,
      "tag": "header",
      "content_type": "header"
    }
  ]
}
```

#### 4. Upload JSON File
- **POST** `/document-blocks/upload-json`
- **Form Data**:
  - `document_id`: Document UUID
  - `custom_id` (optional): Custom ID for block naming
  - `file`: JSON file containing block data

#### 5. Get Document Blocks by Document
- **GET** `/document-blocks/document/{document_id}`
- **Query Parameters**:
  - `page` (default: 1): Page number
  - `page_size` (default: 50): Items per page

#### 6. Get Single Document Block
- **GET** `/document-blocks/{block_id}`

#### 7. Update Document Block
- **PUT/PATCH** `/document-blocks/{block_id}`
- **Body**: `DocumentBlockUpdate` model (partial updates supported)

#### 8. Delete Document Block
- **DELETE** `/document-blocks/{block_id}`

#### 9. Delete All Document Blocks for a Document
- **DELETE** `/document-blocks/document/{document_id}`

## JSON File Format

The API supports multiple JSON formats for bulk block creation:

### Format 1: Direct Array
```json
[
  {
    "block_idx": 0,
    "sentences": [{"text": "First sentence"}],
    "level": 0,
    "page_idx": 0,
    "tag": "para",
    "bbox": [x0, y0, x1, y1]
  }
]
```

### Format 2: Object with Blocks Array
```json
{
  "blocks": [
    {
      "block_idx": 0,
      "content": "Direct content",
      "level": 0,
      "page_idx": 0,
      "tag": "para"
    }
  ]
}
```

### Format 3: LLMSherpa Format
```json
[
  {
    "block_idx": 0,
    "sentences": [{"text": "Paragraph content"}],
    "level": 0,
    "page_idx": 0,
    "tag": "para",
    "bbox": [10.0, 20.0, 100.0, 50.0]
  },
  {
    "block_idx": 1,
    "table_rows": [
      {
        "cells": [
          {"cell_value": "Column 1"},
          {"cell_value": "Column 2"}
        ]
      }
    ],
    "level": 0,
    "page_idx": 0,
    "tag": "table",
    "bbox": [10.0, 60.0, 200.0, 100.0]
  }
]
```

## Content Processing

The API automatically processes different content types:

1. **Sentences**: Extracts text from `sentences` array
2. **Tables**: Processes `table_rows` with `cells` containing `cell_value`
3. **Direct Content**: Uses `content` field directly
4. **Name Fallback**: Uses `name` field if no other content found

## Coordinate System

- `x0, y0`: Bottom-left coordinates
- `x1, y1`: Top-right coordinates
- All coordinates are stored as floating-point numbers

## Content Types

- `regular`: Standard text content
- `header`: Section headers
- `table`: Tabular data
- `list`: List items
- Custom types as needed

## Search Capabilities

The API provides full-text search using PostgreSQL's built-in text search:
- Automatic tokenization and indexing
- Support for complex queries
- Ranking by relevance
- Filtering by metadata

## Example Usage

### Upload JSON and Create Blocks

```bash
curl -X POST "http://localhost:8000/document-blocks/upload-json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "document_id=your-document-uuid" \
     -F "custom_id=my_custom_prefix" \
     -F "file=@sample_document_blocks.json"
```

### Search Blocks

```bash
curl -X GET "http://localhost:8000/document-blocks/?document_id=your-document-uuid&query=introduction&content_type=header" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

### Get All Blocks for a Document

```bash
curl -X GET "http://localhost:8000/document-blocks/document/your-document-uuid?page=1&page_size=50" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

## Integration with RAG System

The document blocks are designed to work seamlessly with the existing RAG system:

1. **Document Relationship**: Each block is linked to a parent document
2. **Search Integration**: Full-text search capabilities for content retrieval
3. **Hierarchical Structure**: Support for parent-child block relationships
4. **Metadata Enrichment**: Rich metadata for advanced filtering and classification

## Error Handling

The API provides detailed error messages for common scenarios:
- Document not found (404)
- Invalid JSON format (400)
- Validation errors (422)
- Server errors (500)

All errors include descriptive messages to help with debugging and integration.
