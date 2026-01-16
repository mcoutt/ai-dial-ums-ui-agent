import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from agent.clients.dial_client import DialClient
from agent.clients.http_mcp_client import HttpMCPClient
from agent.clients.stdio_mcp_client import StdioMCPClient
from agent.conversation_manager import ConversationManager
from agent.models.message import Message

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

conversation_manager: Optional[ConversationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP clients, Redis, and ConversationManager on startup"""
    global conversation_manager

    logger.info("Application startup initiated")

    #TODO:
    # 1. Create empty list with dicts with name `tools`
    # 2. Create empty dict with name `tool_name_client_map` that applies as key `str` and sa value `HttpMCPClient | StdioMCPClient`
    # 3. Create HttpMCPClient for UMS MCP, url is "http://localhost:8005/mcp" (HttpMCPClient has static method create,
    #    don't forget that it is async and you need to await)
    # 4. Get tools for UMS MCP, iterate through them and add it to `tools` and and to the `tool_name_client_map`, key
    #    is tool name, value the UMS MCP Client
    # 5. Do the same as in 3 and 4 steps for Fetch MCP, url is "https://remote.mcpservers.org/fetch/mcp"
    # 6. Create StdioMCPClient for DuckDuckGo, docker image name is "mcp/duckduckgo:latest", and do the same as in 4th step
    # 7. Initialize DialClient with. Models: gpt-4o or claude-3-7-sonnet@20250219, endpoint is https://ai-proxy.lab.epam.com
    # 8. Create Redis client (redis.Redis). Host is localhost, port is 6379, and decode response
    # 9. ping to redis to check if `its alive (ping method in redis client)
    # 10. Create ConversationManager with DIAL clien and Redis client and assign to `conversation_manager` (global variable)
    tools: list[dict] = []
    tool_name_client_map: dict[str, HttpMCPClient | StdioMCPClient] = {}

    # Initialize UMS MCP client
    logger.info("Initializing UMS MCP client")
    ums_mcp_url = os.getenv("UMS_MCP_URL", "http://localhost:8005/mcp")
    logger.info("UMS MCP URL: %s", ums_mcp_url)
    ums_mcp_client = await HttpMCPClient.create(ums_mcp_url)

    for tool in await ums_mcp_client.get_tools():
        tool_name = tool.get('function', {}).get('name')
        tools.append(tool)
        tool_name_client_map[tool_name] = ums_mcp_client
        logger.info("Registered UMS tool", extra={"tool_name": tool_name})

    # Initialize DuckDuckGo MCP client
    logger.info("Initializing DuckDuckGo MCP client")
    duckduckgo_mcp_client = await StdioMCPClient.create(docker_image="khshanovskyi/ddg-mcp-server:latest")
    for tool in await duckduckgo_mcp_client.get_tools():
        tool_name = tool.get('function', {}).get('name')
        tools.append(tool)
        tool_name_client_map[tool_name] = duckduckgo_mcp_client
        logger.info("Registered DuckDuckGo tool", extra={"tool_name": tool_name})

    # Initialize DIAL client
    dial_api_key = os.getenv("DIAL_API_KEY")
    if not dial_api_key:
        logger.error("DIAL_API_KEY environment variable not set")
        raise ValueError("DIAL_API_KEY environment variable is required")

    # model = "gpt-4o"
    model = os.getenv("ORCHESTRATION_MODEL", "gpt-4o")
    endpoint=os.getenv("DIAL_URL", "https://ai-proxy.lab.epam.com")
    logger.info("Initializing DIAL client", extra={"url": endpoint, "model": model, "api_key": dial_api_key})

    dial_client = DialClient(
        api_key=dial_api_key,
        endpoint=endpoint,
        model=model,
        tools=tools,
        tool_name_client_map=tool_name_client_map
    )

    # Initialize Redis client
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    logger.info(
        "Connecting to Redis",
        extra={"host": redis_host, "port": redis_port}
    )

    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )

    await redis_client.ping()
    logger.info("Redis connection established successfully")

    # Initialize ConversationManager with both dependencies
    conversation_manager = ConversationManager(dial_client, redis_client)
    logger.info("ConversationManager initialized successfully")
    logger.info("Application startup completed")

    yield

    logger.info("Application shutdown initiated")
    await redis_client.close()
    logger.info("Application shutdown completed")


app = FastAPI(
    #TODO: add `lifespan` param from above, like:
    # - lifespan=lifespan
    lifespan=lifespan
)

# TODO:
# Since we will run it locally there will be some issues from FrontEnd side with CORS, and its okay for local setup to disable them:
#   - CORSMiddleware,
#   - allow_origins=["*"]
#   - allow_credentials=True
#   - allow_methods=["*"]
#   - allow_headers=["*"]
app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"]
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: Message
    stream: bool = True


class ChatResponse(BaseModel):
    content: str
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class CreateConversationRequest(BaseModel):
    title: str = None


# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "conversation_manager_initialized": conversation_manager is not None
    }


#TODO:
# Create such endpoints:
# 1. POST: "/conversations". Applies CreateConversationRequest and creates new conversation.
# 2. GET: "/conversations" Extracts all conversation from storage. Returns list of ConversationSummary objects
# 3. GET: "/conversations/{conversation_id}". Applies conversation_id string and extracts from storage full conversation
# 4. DELETE: "/conversations/{conversation_id}". Applies conversation_id string and deletes conversation. Returns dict
#    with message with info if conversation has been deleted
# 5. POST: "/conversations/{conversation_id}/chat". Chat endpoint that processes messages and returns assistant response.
#    Supports both streaming and non-streaming modes.
#    Applies conversation_id and ChatRequest.
#    If `request.stream` then return `StreamingResponse(result, media_type="text/event-stream")`, otherwise return `ChatResponse(**result)`
@app.post("/conversations")
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation"""
    if not conversation_manager:
        logger.error("Conversation manager not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("Creating new conversation", extra={"title": request.title})
    return await conversation_manager.create_conversation(request.title)


@app.get("/conversations")
async def list_conversations():
    """List all conversations sorted by last update time"""
    if not conversation_manager:
        logger.error("Conversation manager not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.debug("Listing conversations")
    conversations = await conversation_manager.list_conversations()
    return [ConversationSummary(**conv) for conv in conversations]


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    if not conversation_manager:
        logger.error("Conversation manager not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("Fetching conversation", extra={"conversation_id": conversation_id})
    conversation = await conversation_manager.get_conversation(conversation_id)

    if not conversation:
        logger.warning("Conversation not found", extra={"conversation_id": conversation_id})
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if not conversation_manager:
        logger.error("Conversation manager not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("Deleting conversation", extra={"conversation_id": conversation_id})
    deleted = await conversation_manager.delete_conversation(conversation_id)

    if not deleted:
        logger.warning("Conversation not found for deletion", extra={"conversation_id": conversation_id})
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"message": "Conversation deleted successfully"}


@app.post("/conversations/{conversation_id}/chat")
async def chat(conversation_id: str, request: ChatRequest):
    """
    Chat endpoint that processes messages and returns assistant response.
    Supports both streaming and non-streaming modes.
    Automatically saves conversation state.
    """
    if not conversation_manager:
        logger.error("Conversation manager not initialized")
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info(
        "Chat request received",
        extra={
            "conversation_id": conversation_id,
            "stream": request.stream,
            "message_role": request.message.role
        }
    )

    result = await conversation_manager.chat(
        user_message=request.message,
        conversation_id=conversation_id,
        stream=request.stream
    )

    if request.stream:
        logger.debug("Returning streaming response...")
        return StreamingResponse(
            result,
            media_type="text/event-stream"
        )
    else:
        logger.debug("Returning non-streaming response...")
        return ChatResponse(**result)


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting UMS Agent server")
    uvicorn.run(
         app,
         host="0.0.0.0",
         port=8011,
         log_level="debug"
    )
