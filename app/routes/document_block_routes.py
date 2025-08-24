# app/routes/document_block_routes.py
import traceback
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app.config import logger
from app.models import (
    DocumentBlockCreate,
    DocumentBlockResponse,
    DocumentBlockUpdate,
    DocumentBlocksBulkCreate,
    PaginatedResponse,
    SuccessResponse
)
from app.services.database import (
    create_document_block,
    create_document_blocks_bulk,
    get_document_blocks_by_document,
    get_document_block_by_id,
    update_document_block,
    delete_document_block,
    delete_document_blocks_by_document,
    search_document_blocks,
    ensure_document_blocks_schema,
    ensure_document_images_schema,
    create_document_images_bulk,
    get_document_by_uuid
)

router = APIRouter(prefix="/document-blocks", tags=["Document Blocks"])


def parse_metadata(metadata):
    """Parse metadata field safely."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# Document blocks routes initialization is handled in lifespan
# The schema will be ensured during application startup


@router.get("/", response_model=PaginatedResponse)
async def list_document_blocks(
    request: Request,
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    query: Optional[str] = Query(None, description="Search query"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    section_type: Optional[str] = Query(None, description="Filter by section type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """List document blocks with pagination and filtering."""
    try:
        offset = (page - 1) * page_size
        
        blocks, total = await search_document_blocks(
            document_id=document_id,
            query=query,
            content_type=content_type,
            section_type=section_type,
            limit=page_size,
            offset=offset
        )
        
        # Convert to response format
        response_items = []
        for block in blocks:
            response_items.append(DocumentBlockResponse(
                id=str(block['id']),
                block_idx=block['block_idx'],
                document_id=str(block['document_id']),
                name=block['name'],
                content=block.get('content'),
                level=block['level'],
                page_idx=block['page_idx'],
                tag=block['tag'],
                block_class=block.get('block_class'),
                x0=block.get('x0'),
                y0=block.get('y0'),
                x1=block.get('x1'),
                y1=block.get('y1'),
                parent_idx=block.get('parent_idx'),
                content_type=block.get('content_type', 'regular'),
                section_type=block.get('section_type'),
                demand_priority=block.get('demand_priority'),
                created_at=block.get('created_at')
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=response_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list document blocks | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list document blocks: {str(e)}")


@router.post("/", response_model=DocumentBlockResponse)
async def create_document_block_endpoint(
    request: Request,
    block: DocumentBlockCreate,
    document_id: str = Query(..., description="Document UUID")
):
    """Create a new document block."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        created_block = await create_document_block(
            document_id=document_id,
            block_idx=block.block_idx,
            name=block.name,
            content=block.content,
            level=block.level,
            page_idx=block.page_idx,
            tag=block.tag,
            block_class=block.block_class,
            x0=block.x0,
            y0=block.y0,
            x1=block.x1,
            y1=block.y1,
            parent_idx=block.parent_idx,
            content_type=block.content_type,
            section_type=block.section_type,
            demand_priority=block.demand_priority
        )
        
        if not created_block:
            raise HTTPException(status_code=500, detail="Failed to create document block")
        
        return DocumentBlockResponse(
            id=str(created_block['id']),
            block_idx=created_block['block_idx'],
            document_id=str(created_block['document_id']),
            name=created_block['name'],
            content=created_block.get('content'),
            level=created_block['level'],
            page_idx=created_block['page_idx'],
            tag=created_block['tag'],
            block_class=created_block.get('block_class'),
            x0=created_block.get('x0'),
            y0=created_block.get('y0'),
            x1=created_block.get('x1'),
            y1=created_block.get('y1'),
            parent_idx=created_block.get('parent_idx'),
            content_type=created_block.get('content_type', 'regular'),
            section_type=created_block.get('section_type'),
            demand_priority=created_block.get('demand_priority'),
            created_at=created_block.get('created_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create document block | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create document block: {str(e)}")


@router.post("/bulk", response_model=SuccessResponse)
async def create_document_blocks_bulk_endpoint(
    request: Request,
    bulk_create: DocumentBlocksBulkCreate
):
    """Create multiple document blocks in bulk."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(bulk_create.document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {bulk_create.document_id} not found")
        
        # Convert blocks to dict format
        blocks_data = []
        for block in bulk_create.blocks:
            blocks_data.append({
                "block_idx": block.block_idx,
                "name": block.name,
                "content": block.content,
                "level": block.level,
                "page_idx": block.page_idx,
                "tag": block.tag,
                "block_class": block.block_class,
                "x0": block.x0,
                "y0": block.y0,
                "x1": block.x1,
                "y1": block.y1,
                "parent_idx": block.parent_idx,
                "content_type": block.content_type,
                "section_type": block.section_type,
                "demand_priority": block.demand_priority
            })
        
        created_blocks = await create_document_blocks_bulk(bulk_create.document_id, blocks_data)
        
        return SuccessResponse(
            message=f"Successfully created {len(created_blocks)} document blocks",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create document blocks bulk | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create document blocks bulk: {str(e)}")


@router.post("/upload-json")
async def upload_json_and_create_blocks(
    request: Request,
    document_id: str = Form(..., description="Document UUID"),
    custom_id: Optional[str] = Form(None, description="Custom ID for blocks"),
    file: UploadFile = File(..., description="JSON file containing block data")
):
    """Upload a JSON file and create document blocks from it."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Read and parse JSON file
        content = await file.read()
        try:
            json_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        
        # Process JSON data
        blocks_data = []
        images_data = []
        
        # Check if this is a Docling format JSON
        if isinstance(json_data, dict) and "schema_name" in json_data and json_data["schema_name"] == "DoclingDocument":
            # Process Docling format
            blocks_data = _process_docling_json(json_data, custom_id)
            
            # Process pages data for images
            images_data = _process_pages_data(json_data)
        else:
            # Handle other JSON structures (legacy support)
            if isinstance(json_data, list):
                # Direct list of blocks
                for i, block_data in enumerate(json_data):
                    processed_block = _process_block_data(block_data, i, custom_id)
                    if processed_block:
                        blocks_data.append(processed_block)
            elif isinstance(json_data, dict):
                # Object with blocks array or single block
                if 'blocks' in json_data and isinstance(json_data['blocks'], list):
                    for i, block_data in enumerate(json_data['blocks']):
                        processed_block = _process_block_data(block_data, i, custom_id)
                        if processed_block:
                            blocks_data.append(processed_block)
                else:
                    # Single block object
                    processed_block = _process_block_data(json_data, 0, custom_id)
                    if processed_block:
                        blocks_data.append(processed_block)
                
                # Check for pages data even in legacy format
                if 'pages' in json_data:
                    images_data = _process_pages_data(json_data)
        
        if not blocks_data:
            raise HTTPException(status_code=400, detail="No valid block data found in JSON file")
        
        # Filter out blocks with empty content
        blocks_data = [block for block in blocks_data if block.get("content", "").strip()]
        
        if not blocks_data:
            raise HTTPException(status_code=400, detail="No blocks with valid content found in JSON file")
        
        # Create blocks in bulk
        created_blocks = await create_document_blocks_bulk(document_id, blocks_data)
        
        # Create images in bulk if any images were found
        created_images = []
        if images_data:
            try:
                created_images = await create_document_images_bulk(document_id, images_data)
                logger.info(f"Created {len(created_images)} document images for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to create document images: {str(e)}")
                # Continue even if image creation fails
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully processed JSON file and created {len(created_blocks)} blocks" + 
                      (f" and {len(created_images)} images" if created_images else ""),
            "blocks_created": len(created_blocks),
            "images_created": len(created_images),
            "document_id": document_id,
            "format_detected": "DoclingDocument" if "schema_name" in json_data else "Legacy"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload JSON and create blocks | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process JSON file: {str(e)}")


def _extract_bbox_from_prov(prov_list: list) -> tuple:
    """Extract bounding box coordinates from provenance data."""
    if not prov_list or not isinstance(prov_list, list) or not prov_list:
        return (None, None, None, None)
    
    prov = prov_list[0]  # Use first provenance entry
    bbox = prov.get("bbox", {})
    
    if isinstance(bbox, dict):
        # Handle different coordinate origins
        coord_origin = bbox.get("coord_origin", "TOPLEFT")
        l, t, r, b = bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b")
        
        if coord_origin == "BOTTOMLEFT":
            # Convert from bottom-left to top-left coordinate system
            # Assuming page height for conversion (you might need to adjust this)
            return (l, t, r, b)  # Keep as is for now, adjust if needed
        else:
            return (l, t, r, b)
    
    return (None, None, None, None)

def _process_text_element(text_data: dict, idx: int) -> dict:
    """Process a text element from the texts array."""
    content = text_data.get("text", "").strip()
    if not content:
        content = text_data.get("orig", "").strip()
    
    # Extract bounding box from provenance
    prov = text_data.get("prov", [])
    x0, y0, x1, y1 = _extract_bbox_from_prov(prov)
    
    # Get page number from provenance (extract from first prov entry)
    page_idx = 0  # default to 0
    if prov and len(prov) > 0 and "page_no" in prov[0]:
        page_idx = prov[0]["page_no"]  # Keep 1-based indexing as in original data
    
    # Get the correct name from self_ref
    self_ref = text_data.get("self_ref", "")
    name = self_ref.replace("#/", "") if self_ref else f"text_{idx}"
    
    # Determine content type based on label
    label = text_data.get("label", "text")
    if label == "section_header":
        tag = "header"
        content_type = "header"
    elif label == "text":
        tag = "para"
        content_type = "regular"
    else:
        tag = "para"
        content_type = "regular"
    
    return {
        "block_idx": idx,
        "name": name,
        "content": content,
        "level": text_data.get("level", 0),
        "page_idx": page_idx,
        "tag": tag,
        "block_class": label,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "parent_idx": None,
        "content_type": content_type,
        "section_type": label,
        "demand_priority": None
    }

def _process_table_element(table_data: dict, idx: int) -> dict:
    """Process a table element from the tables array."""
    content = ""
    
    # Extract table data
    data = table_data.get("data", {})
    grid = data.get("grid", [])
    
    if grid:
        # Convert grid to markdown-like table format
        for row in grid:
            row_content = []
            for cell in row:
                cell_text = cell.get("text", "").strip()
                row_content.append(cell_text)
            if row_content:
                content += " | ".join(row_content) + "\n"
    
    # If no grid data, try table_cells
    if not content:
        table_cells = data.get("table_cells", [])
        if table_cells:
            content = "Table data: " + "; ".join([cell.get("text", "") for cell in table_cells if cell.get("text")])
    
    # Extract bounding box from provenance
    prov = table_data.get("prov", [])
    x0, y0, x1, y1 = _extract_bbox_from_prov(prov)
    
    # Get page number from provenance (extract from first prov entry)
    page_idx = 0  # default to 0
    if prov and len(prov) > 0 and "page_no" in prov[0]:
        page_idx = prov[0]["page_no"]  # Keep 1-based indexing as in original data
    
    # Get the correct name from self_ref
    self_ref = table_data.get("self_ref", "")
    name = self_ref.replace("#/", "") if self_ref else f"table_{idx}"
    
    return {
        "block_idx": idx,
        "name": name,
        "content": content.strip(),
        "level": 0,
        "page_idx": page_idx,
        "tag": "table",
        "block_class": "table",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "parent_idx": None,
        "content_type": "table",
        "section_type": "table",
        "demand_priority": None
    }

def _process_picture_element(picture_data: dict, idx: int) -> dict:
    """Process a picture element from the pictures array."""
    # Try to get description from annotations
    content = ""
    annotations = picture_data.get("annotations", [])
    
    for annotation in annotations:
        if annotation.get("kind") == "description":
            content = annotation.get("text", "")
            break
    
    if not content:
        content = f"Image: {picture_data.get('label', 'picture')}"
    
    # Extract image info
    image_info = picture_data.get("image", {})
    if image_info:
        size = image_info.get("size", {})
        if size:
            content += f" (Size: {size.get('width', 0)}x{size.get('height', 0)})"
    
    # Extract bounding box from provenance
    prov = picture_data.get("prov", [])
    x0, y0, x1, y1 = _extract_bbox_from_prov(prov)
    
    # Get page number from provenance (extract from first prov entry)
    page_idx = 0  # default to 0
    if prov and len(prov) > 0 and "page_no" in prov[0]:
        page_idx = prov[0]["page_no"]  # Keep 1-based indexing as in original data
    
    # Get the correct name from self_ref
    self_ref = picture_data.get("self_ref", "")
    name = self_ref.replace("#/", "") if self_ref else f"picture_{idx}"
    
    return {
        "block_idx": idx,
        "name": name,
        "content": content.strip(),
        "level": 0,
        "page_idx": page_idx,
        "tag": "figure",
        "block_class": "picture",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "parent_idx": None,
        "content_type": "image",
        "section_type": "figure",
        "demand_priority": None
    }

def _process_docling_json(json_data: dict, custom_id: str = None) -> list:
    """Process Docling JSON format and extract blocks."""
    blocks_data = []
    
    # Get the correct order from the body children
    body_children = json_data.get("body", {}).get("children", [])
    
    # Create a map of self_ref to index for proper ordering
    ref_to_index = {}
    for idx, child in enumerate(body_children):
        if "$ref" in child:
            ref_to_index[child["$ref"]] = idx
    
    # Collect all elements with their correct indices
    all_elements = []
    
    # Process texts with correct ordering
    texts = json_data.get("texts", [])
    for text_data in texts:
        self_ref = text_data.get("self_ref", "")
        if self_ref in ref_to_index:
            block_idx = ref_to_index[self_ref]
            processed_block = _process_text_element(text_data, block_idx)
            if processed_block and processed_block["content"].strip():
                all_elements.append(processed_block)
    
    # Process tables with correct ordering
    tables = json_data.get("tables", [])
    for table_data in tables:
        self_ref = table_data.get("self_ref", "")
        if self_ref in ref_to_index:
            block_idx = ref_to_index[self_ref]
            processed_block = _process_table_element(table_data, block_idx)
            if processed_block and processed_block["content"].strip():
                all_elements.append(processed_block)
    
    # Process pictures with correct ordering
    pictures = json_data.get("pictures", [])
    for picture_data in pictures:
        self_ref = picture_data.get("self_ref", "")
        if self_ref in ref_to_index:
            block_idx = ref_to_index[self_ref]
            processed_block = _process_picture_element(picture_data, block_idx)
            if processed_block and processed_block["content"].strip():
                all_elements.append(processed_block)
    
    # Sort blocks by their correct index
    all_elements.sort(key=lambda x: x.get('block_idx', 0))
    
    return all_elements


def _process_pages_data(json_data: dict) -> list:
    """Process pages data from JSON and extract image information."""
    images_data = []
    pages = json_data.get("pages", {})
    
    for page_key, page_data in pages.items():
        try:
            # Extract page number (could be string or int)
            page_no = int(page_key) if isinstance(page_key, str) and page_key.isdigit() else page_data.get("page_no", 1)
            
            # Extract image data
            image_data = page_data.get("image", {})
            if image_data and image_data.get("uri"):
                # Extract page size
                page_size = page_data.get("size", {})
                
                # Extract image size
                image_size = image_data.get("size", {})
                
                image_info = {
                    "page_no": page_no,
                    "mimetype": image_data.get("mimetype", "image/png"),
                    "uri": image_data.get("uri"),
                    "dpi": image_data.get("dpi"),
                    "width": image_size.get("width"),
                    "height": image_size.get("height"),
                    "page_width": page_size.get("width"),
                    "page_height": page_size.get("height")
                }
                
                images_data.append(image_info)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to process page data for page {page_key}: {str(e)}")
            continue
    
    return images_data

def _process_block_data(block_data: dict, default_idx: int, custom_id: str = None) -> dict:
    """Process individual block data from JSON (legacy format support)."""
    try:
        # Extract content from different possible structures
        content = ""
        if "sentences" in block_data:
            sentences = block_data["sentences"]
            if isinstance(sentences, list):
                if sentences and isinstance(sentences[0], dict) and "text" in sentences[0]:
                    content = " ".join(s["text"] for s in sentences)
                elif sentences and isinstance(sentences[0], str):
                    content = " ".join(sentences)
        elif "table_rows" in block_data:
            for row in block_data["table_rows"]:
                cells = row.get("cells", [])
                if cells:
                    content += " | ".join(str(cell.get('cell_value', '')) for cell in cells) + "\n"
                elif row.get("cell_value"):
                    content += str(row["cell_value"]) + "\n"
        elif "content" in block_data:
            content = str(block_data["content"])
        elif "name" in block_data:
            content = str(block_data["name"])
        
        # Extract bounding box
        bbox = block_data.get("bbox", [0, 0, 0, 0])
        x0 = bbox[0] if len(bbox) > 0 else None
        y0 = bbox[1] if len(bbox) > 1 else None
        x1 = bbox[2] if len(bbox) > 2 else None
        y1 = bbox[3] if len(bbox) > 3 else None
        
        # Build block name
        block_name = custom_id or block_data.get("name", f"block_{default_idx}")
        
        return {
            "block_idx": block_data.get("block_idx", default_idx),
            "name": block_name,
            "content": content,
            "level": block_data.get("level", 0),
            "page_idx": block_data.get("page_idx", 0),
            "tag": block_data.get("tag", "para"),
            "block_class": block_data.get("block_class"),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "parent_idx": block_data.get("parent_idx"),
            "content_type": block_data.get("content_type", "regular"),
            "section_type": block_data.get("section_type"),
            "demand_priority": block_data.get("demand_priority")
        }
        
    except Exception as e:
        logger.warning(f"Failed to process block data: {str(e)} | Data: {block_data}")
        return None


@router.get("/document/{document_id}", response_model=PaginatedResponse)
async def get_document_blocks(
    document_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get all blocks for a specific document."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        offset = (page - 1) * page_size
        blocks, total = await get_document_blocks_by_document(document_id, limit=page_size, offset=offset)
        
        # Convert to response format
        response_items = []
        for block in blocks:
            response_items.append(DocumentBlockResponse(
                id=str(block['id']),
                block_idx=block['block_idx'],
                document_id=str(block['document_id']),
                name=block['name'],
                content=block.get('content'),
                level=block['level'],
                page_idx=block['page_idx'],
                tag=block['tag'],
                block_class=block.get('block_class'),
                x0=block.get('x0'),
                y0=block.get('y0'),
                x1=block.get('x1'),
                y1=block.get('y1'),
                parent_idx=block.get('parent_idx'),
                content_type=block.get('content_type', 'regular'),
                section_type=block.get('section_type'),
                demand_priority=block.get('demand_priority'),
                created_at=block.get('created_at')
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=response_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document blocks | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document blocks: {str(e)}")


@router.get("/{block_id}", response_model=DocumentBlockResponse)
async def get_document_block_endpoint(block_id: str, request: Request):
    """Get a specific document block by ID."""
    try:
        block = await get_document_block_by_id(block_id)
        if not block:
            raise HTTPException(status_code=404, detail=f"Document block {block_id} not found")
        
        return DocumentBlockResponse(
            id=str(block['id']),
            block_idx=block['block_idx'],
            document_id=str(block['document_id']),
            name=block['name'],
            content=block.get('content'),
            level=block['level'],
            page_idx=block['page_idx'],
            tag=block['tag'],
            block_class=block.get('block_class'),
            x0=block.get('x0'),
            y0=block.get('y0'),
            x1=block.get('x1'),
            y1=block.get('y1'),
            parent_idx=block.get('parent_idx'),
            content_type=block.get('content_type', 'regular'),
            section_type=block.get('section_type'),
            demand_priority=block.get('demand_priority'),
            created_at=block.get('created_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document block | ID: {block_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document block: {str(e)}")


@router.put("/{block_id}", response_model=DocumentBlockResponse)
@router.patch("/{block_id}", response_model=DocumentBlockResponse)
async def update_document_block_endpoint(
    block_id: str,
    block_update: DocumentBlockUpdate,
    request: Request
):
    """Update a document block (supports both PUT and PATCH for partial updates)."""
    try:
        # Check if block exists
        existing_block = await get_document_block_by_id(block_id)
        if not existing_block:
            raise HTTPException(status_code=404, detail=f"Document block {block_id} not found")
        
        # Prepare updates
        updates = {}
        update_data = block_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if value is not None:
                updates[field] = value
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        updated_block = await update_document_block(block_id, **updates)
        
        if not updated_block:
            raise HTTPException(status_code=500, detail="Failed to update document block")
        
        return DocumentBlockResponse(
            id=str(updated_block['id']),
            block_idx=updated_block['block_idx'],
            document_id=str(updated_block['document_id']),
            name=updated_block['name'],
            content=updated_block.get('content'),
            level=updated_block['level'],
            page_idx=updated_block['page_idx'],
            tag=updated_block['tag'],
            block_class=updated_block.get('block_class'),
            x0=updated_block.get('x0'),
            y0=updated_block.get('y0'),
            x1=updated_block.get('x1'),
            y1=updated_block.get('y1'),
            parent_idx=updated_block.get('parent_idx'),
            content_type=updated_block.get('content_type', 'regular'),
            section_type=updated_block.get('section_type'),
            demand_priority=updated_block.get('demand_priority'),
            created_at=updated_block.get('created_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document block | ID: {block_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update document block: {str(e)}")


@router.delete("/{block_id}", response_model=SuccessResponse)
async def delete_document_block_endpoint(block_id: str, request: Request):
    """Delete a specific document block."""
    try:
        # Check if block exists
        existing_block = await get_document_block_by_id(block_id)
        if not existing_block:
            raise HTTPException(status_code=404, detail=f"Document block {block_id} not found")
        
        success = await delete_document_block(block_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document block")
        
        return SuccessResponse(
            message=f"Successfully deleted document block {block_id}",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document block | ID: {block_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document block: {str(e)}")


@router.delete("/document/{document_id}", response_model=SuccessResponse)
async def delete_document_blocks_endpoint(document_id: str, request: Request):
    """Delete all blocks for a specific document."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        deleted_count = await delete_document_blocks_by_document(document_id)
        
        return SuccessResponse(
            message=f"Successfully deleted {deleted_count} document blocks for document {document_id}",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document blocks | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document blocks: {str(e)}")
