"""
Context Window Stress Test Generator for CogCanvas Evaluation.

This script generates MASSIVE conversations (200 turns, >50k tokens) designed to
stress-test GPT-4o's context window and force lossy compression in summarization.

Design:
- Total turns: 200
- Token target: >50,000 tokens (to exceed typical summarization capacity)
- Long document injection: Every 20 turns, inject ~2500 tokens of technical noise
- Fact placement: Sparse - turns 10, 50, 100, 150 (needles in massive haystack)
- Compression point: Turn 180
- Test turns: 181-200

The key insight: Summarization MUST lose information when compressing 50k tokens
to ~2k tokens (25x compression). CogCanvas should preserve planted facts verbatim.

Usage:
    python -m experiments.data_gen_long
    python -m experiments.data_gen_long --preview  # See token counts
"""

from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from experiments.data_gen import (
    ConversationGenerator,
    ConversationTurn,
    SyntheticConversation,
    PlantedFact,
    EvaluationDataset,
    DifficultyLevel,
    DECISION_TEMPLATES,
    KEY_FACT_TEMPLATES,
    REMINDER_TEMPLATES,
    FILLER_EXCHANGES,
)


# =============================================================================
# Long Document Templates (~2500 tokens each)
# These simulate real-world scenarios: log dumps, API responses, code reviews
# =============================================================================

LONG_DOCUMENT_TEMPLATE_LOGS = '''```
[2024-01-15 10:23:45.123] INFO  [main] Application starting up...
[2024-01-15 10:23:45.234] DEBUG [config] Loading configuration from /etc/app/config.yaml
[2024-01-15 10:23:45.345] INFO  [config] Environment: production
[2024-01-15 10:23:45.456] DEBUG [db] Initializing database connection pool
[2024-01-15 10:23:45.567] INFO  [db] Connected to PostgreSQL 15.2 at db-primary.internal:5432
[2024-01-15 10:23:45.678] DEBUG [db] Connection pool size: min=5, max=20
[2024-01-15 10:23:45.789] INFO  [cache] Connecting to Redis cluster at redis.internal:6379
[2024-01-15 10:23:45.890] DEBUG [cache] Redis version: 7.0.11, cluster mode: enabled
[2024-01-15 10:23:46.001] INFO  [auth] Loading JWT signing keys from vault
[2024-01-15 10:23:46.112] DEBUG [auth] Key rotation: last rotated 2024-01-10, next 2024-01-20
[2024-01-15 10:23:46.223] INFO  [metrics] Prometheus metrics endpoint: /metrics:9090
[2024-01-15 10:23:46.334] DEBUG [metrics] Registered 47 custom metrics
[2024-01-15 10:23:46.445] INFO  [http] Starting HTTP server on :8080
[2024-01-15 10:23:46.556] INFO  [grpc] Starting gRPC server on :9000
[2024-01-15 10:23:46.667] DEBUG [health] Health check endpoints registered
[2024-01-15 10:23:46.778] INFO  [main] Application ready, accepting requests

[2024-01-15 10:24:00.123] INFO  [http] GET /api/v1/users/123 - 200 OK - 45ms
[2024-01-15 10:24:00.234] DEBUG [db] Query executed: SELECT * FROM users WHERE id = $1 [123]
[2024-01-15 10:24:00.345] DEBUG [cache] Cache HIT for key user:123
[2024-01-15 10:24:01.456] INFO  [http] POST /api/v1/orders - 201 Created - 123ms
[2024-01-15 10:24:01.567] DEBUG [db] Transaction started: txn_abc123
[2024-01-15 10:24:01.678] DEBUG [db] INSERT INTO orders (user_id, total, status) VALUES ($1, $2, $3)
[2024-01-15 10:24:01.789] DEBUG [db] INSERT INTO order_items (order_id, product_id, quantity) VALUES ...
[2024-01-15 10:24:01.890] DEBUG [db] Transaction committed: txn_abc123
[2024-01-15 10:24:01.901] INFO  [events] Published OrderCreated event to Kafka topic: orders.events
[2024-01-15 10:24:02.012] DEBUG [kafka] Message delivered to partition 3, offset 12345

[2024-01-15 10:24:05.123] WARN  [http] Request timeout approaching for /api/v1/reports/generate
[2024-01-15 10:24:05.234] DEBUG [db] Slow query detected (2.3s): SELECT * FROM transactions WHERE...
[2024-01-15 10:24:05.345] INFO  [http] GET /api/v1/reports/generate - 200 OK - 2456ms
[2024-01-15 10:24:06.456] INFO  [http] GET /api/v1/health - 200 OK - 2ms
[2024-01-15 10:24:07.567] DEBUG [metrics] Metric recorded: http_requests_total{method="GET",status="200"}
[2024-01-15 10:24:08.678] INFO  [http] GET /api/v1/products?page=1&limit=50 - 200 OK - 89ms
[2024-01-15 10:24:09.789] DEBUG [cache] Cache MISS for key products:page:1
[2024-01-15 10:24:09.890] DEBUG [db] Query executed: SELECT * FROM products ORDER BY created_at DESC LIMIT 50
[2024-01-15 10:24:10.901] DEBUG [cache] Cache SET for key products:page:1 (TTL: 300s)

[2024-01-15 10:25:00.123] ERROR [http] Request failed: POST /api/v1/payments
[2024-01-15 10:25:00.234] ERROR [payments] Payment gateway timeout after 30000ms
[2024-01-15 10:25:00.345] DEBUG [payments] Retry attempt 1/3 for payment txn_def456
[2024-01-15 10:25:01.456] DEBUG [payments] Retry attempt 2/3 for payment txn_def456
[2024-01-15 10:25:02.567] INFO  [payments] Payment successful on retry: txn_def456
[2024-01-15 10:25:02.678] INFO  [http] POST /api/v1/payments - 200 OK - 3234ms
[2024-01-15 10:25:03.789] DEBUG [notifications] Sending payment confirmation email to user@example.com
[2024-01-15 10:25:03.890] INFO  [notifications] Email queued: msg_id=email_789

[2024-01-15 10:26:00.123] INFO  [scheduler] Running scheduled job: cleanup_expired_sessions
[2024-01-15 10:26:00.234] DEBUG [db] DELETE FROM sessions WHERE expires_at < NOW()
[2024-01-15 10:26:00.345] INFO  [scheduler] Deleted 1247 expired sessions
[2024-01-15 10:26:01.456] INFO  [scheduler] Running scheduled job: sync_inventory
[2024-01-15 10:26:01.567] DEBUG [inventory] Fetching inventory from warehouse API
[2024-01-15 10:26:02.678] INFO  [inventory] Synced 3456 products, 23 updates applied
[2024-01-15 10:26:03.789] INFO  [scheduler] Running scheduled job: generate_daily_report
[2024-01-15 10:26:04.890] DEBUG [reports] Aggregating metrics for 2024-01-14
[2024-01-15 10:26:05.901] INFO  [reports] Daily report generated: /reports/2024-01-14.pdf

[2024-01-15 10:27:00.123] INFO  [http] WebSocket connection established: ws_conn_abc123
[2024-01-15 10:27:00.234] DEBUG [ws] Client subscribed to channel: orders.updates
[2024-01-15 10:27:01.345] DEBUG [ws] Broadcasting message to 47 connected clients
[2024-01-15 10:27:02.456] INFO  [ws] Client disconnected: ws_conn_xyz789 (normal closure)
[2024-01-15 10:27:03.567] DEBUG [ws] Active WebSocket connections: 234

[2024-01-15 10:28:00.123] INFO  [security] Rate limit triggered for IP: 203.0.113.45
[2024-01-15 10:28:00.234] WARN  [security] Potential DDoS detected: 1500 requests/min from subnet
[2024-01-15 10:28:00.345] INFO  [security] Temporarily blocking IP range: 203.0.113.0/24
[2024-01-15 10:28:01.456] DEBUG [security] Firewall rule added: BLOCK 203.0.113.0/24 (expires: 1h)
[2024-01-15 10:28:02.567] INFO  [security] Alert sent to security team via PagerDuty

[2024-01-15 10:29:00.123] INFO  [deploy] Deployment started: v2.3.1 -> v2.3.2
[2024-01-15 10:29:00.234] DEBUG [deploy] Rolling update: 0/10 pods updated
[2024-01-15 10:29:05.345] DEBUG [deploy] Rolling update: 3/10 pods updated
[2024-01-15 10:29:10.456] DEBUG [deploy] Rolling update: 7/10 pods updated
[2024-01-15 10:29:15.567] INFO  [deploy] Rolling update: 10/10 pods updated
[2024-01-15 10:29:16.678] INFO  [deploy] Deployment complete: v2.3.2
[2024-01-15 10:29:16.789] DEBUG [deploy] Health checks passing for all pods
[2024-01-15 10:29:17.890] INFO  [deploy] Traffic cutover complete

[2024-01-15 10:30:00.123] INFO  [backup] Starting daily backup job
[2024-01-15 10:30:00.234] DEBUG [backup] Creating snapshot of database: app_production
[2024-01-15 10:30:30.345] INFO  [backup] Database snapshot complete: 15.7 GB
[2024-01-15 10:30:31.456] DEBUG [backup] Uploading to S3: s3://backups/db/2024-01-15.sql.gz
[2024-01-15 10:31:00.567] INFO  [backup] Backup upload complete
[2024-01-15 10:31:01.678] DEBUG [backup] Cleaning up old backups (retention: 30 days)
[2024-01-15 10:31:02.789] INFO  [backup] Deleted 2 old backup files
```'''

LONG_DOCUMENT_TEMPLATE_API = '''```json
{
  "openapi": "3.0.3",
  "info": {
    "title": "E-Commerce Platform API",
    "version": "2.3.2",
    "description": "RESTful API for the e-commerce platform supporting orders, products, and user management."
  },
  "servers": [
    {"url": "https://api.example.com/v1", "description": "Production"},
    {"url": "https://staging-api.example.com/v1", "description": "Staging"}
  ],
  "paths": {
    "/users": {
      "get": {
        "summary": "List all users",
        "operationId": "listUsers",
        "tags": ["Users"],
        "parameters": [
          {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
          {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20, "maximum": 100}},
          {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["active", "inactive", "suspended"]}}
        ],
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "data": {"type": "array", "items": {"$ref": "#/components/schemas/User"}},
                    "meta": {"$ref": "#/components/schemas/PaginationMeta"}
                  }
                }
              }
            }
          },
          "401": {"description": "Unauthorized"},
          "403": {"description": "Forbidden"}
        }
      },
      "post": {
        "summary": "Create a new user",
        "operationId": "createUser",
        "tags": ["Users"],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {"$ref": "#/components/schemas/CreateUserRequest"}
            }
          }
        },
        "responses": {
          "201": {"description": "User created successfully"},
          "400": {"description": "Invalid request body"},
          "409": {"description": "User already exists"}
        }
      }
    },
    "/users/{userId}": {
      "get": {
        "summary": "Get user by ID",
        "operationId": "getUserById",
        "tags": ["Users"],
        "parameters": [
          {"name": "userId", "in": "path", "required": true, "schema": {"type": "string", "format": "uuid"}}
        ],
        "responses": {
          "200": {"description": "Successful response"},
          "404": {"description": "User not found"}
        }
      },
      "put": {
        "summary": "Update user",
        "operationId": "updateUser",
        "tags": ["Users"],
        "parameters": [
          {"name": "userId", "in": "path", "required": true, "schema": {"type": "string", "format": "uuid"}}
        ],
        "requestBody": {
          "required": true,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UpdateUserRequest"}}}
        },
        "responses": {
          "200": {"description": "User updated"},
          "404": {"description": "User not found"}
        }
      },
      "delete": {
        "summary": "Delete user",
        "operationId": "deleteUser",
        "tags": ["Users"],
        "parameters": [
          {"name": "userId", "in": "path", "required": true, "schema": {"type": "string", "format": "uuid"}}
        ],
        "responses": {
          "204": {"description": "User deleted"},
          "404": {"description": "User not found"}
        }
      }
    },
    "/products": {
      "get": {
        "summary": "List all products",
        "operationId": "listProducts",
        "tags": ["Products"],
        "parameters": [
          {"name": "category", "in": "query", "schema": {"type": "string"}},
          {"name": "minPrice", "in": "query", "schema": {"type": "number"}},
          {"name": "maxPrice", "in": "query", "schema": {"type": "number"}},
          {"name": "inStock", "in": "query", "schema": {"type": "boolean"}},
          {"name": "sort", "in": "query", "schema": {"type": "string", "enum": ["price_asc", "price_desc", "newest", "popular"]}}
        ],
        "responses": {
          "200": {"description": "Successful response"}
        }
      },
      "post": {
        "summary": "Create a new product",
        "operationId": "createProduct",
        "tags": ["Products"],
        "security": [{"bearerAuth": []}],
        "requestBody": {
          "required": true,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CreateProductRequest"}}}
        },
        "responses": {
          "201": {"description": "Product created"},
          "400": {"description": "Invalid request"}
        }
      }
    },
    "/orders": {
      "get": {
        "summary": "List orders for authenticated user",
        "operationId": "listOrders",
        "tags": ["Orders"],
        "security": [{"bearerAuth": []}],
        "parameters": [
          {"name": "status", "in": "query", "schema": {"type": "string", "enum": ["pending", "processing", "shipped", "delivered", "cancelled"]}},
          {"name": "startDate", "in": "query", "schema": {"type": "string", "format": "date"}},
          {"name": "endDate", "in": "query", "schema": {"type": "string", "format": "date"}}
        ],
        "responses": {
          "200": {"description": "Successful response"}
        }
      },
      "post": {
        "summary": "Create a new order",
        "operationId": "createOrder",
        "tags": ["Orders"],
        "security": [{"bearerAuth": []}],
        "requestBody": {
          "required": true,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CreateOrderRequest"}}}
        },
        "responses": {
          "201": {"description": "Order created"},
          "400": {"description": "Invalid request"},
          "402": {"description": "Payment required"}
        }
      }
    },
    "/orders/{orderId}": {
      "get": {
        "summary": "Get order by ID",
        "operationId": "getOrderById",
        "tags": ["Orders"],
        "parameters": [
          {"name": "orderId", "in": "path", "required": true, "schema": {"type": "string"}}
        ],
        "responses": {
          "200": {"description": "Successful response"},
          "404": {"description": "Order not found"}
        }
      }
    },
    "/orders/{orderId}/cancel": {
      "post": {
        "summary": "Cancel an order",
        "operationId": "cancelOrder",
        "tags": ["Orders"],
        "parameters": [
          {"name": "orderId", "in": "path", "required": true, "schema": {"type": "string"}}
        ],
        "requestBody": {
          "content": {"application/json": {"schema": {"type": "object", "properties": {"reason": {"type": "string"}}}}}
        },
        "responses": {
          "200": {"description": "Order cancelled"},
          "400": {"description": "Cannot cancel order in current state"}
        }
      }
    }
  },
  "components": {
    "schemas": {
      "User": {
        "type": "object",
        "properties": {
          "id": {"type": "string", "format": "uuid"},
          "email": {"type": "string", "format": "email"},
          "name": {"type": "string"},
          "status": {"type": "string", "enum": ["active", "inactive", "suspended"]},
          "createdAt": {"type": "string", "format": "date-time"},
          "updatedAt": {"type": "string", "format": "date-time"}
        }
      },
      "CreateUserRequest": {
        "type": "object",
        "required": ["email", "password", "name"],
        "properties": {
          "email": {"type": "string", "format": "email"},
          "password": {"type": "string", "minLength": 8},
          "name": {"type": "string", "minLength": 1, "maxLength": 100}
        }
      },
      "Product": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "description": {"type": "string"},
          "price": {"type": "number"},
          "currency": {"type": "string", "default": "USD"},
          "inventory": {"type": "integer"},
          "category": {"type": "string"},
          "images": {"type": "array", "items": {"type": "string", "format": "uri"}}
        }
      },
      "Order": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "userId": {"type": "string"},
          "items": {"type": "array", "items": {"$ref": "#/components/schemas/OrderItem"}},
          "total": {"type": "number"},
          "status": {"type": "string"},
          "shippingAddress": {"$ref": "#/components/schemas/Address"},
          "createdAt": {"type": "string", "format": "date-time"}
        }
      },
      "OrderItem": {
        "type": "object",
        "properties": {
          "productId": {"type": "string"},
          "quantity": {"type": "integer"},
          "unitPrice": {"type": "number"}
        }
      },
      "Address": {
        "type": "object",
        "properties": {
          "street": {"type": "string"},
          "city": {"type": "string"},
          "state": {"type": "string"},
          "postalCode": {"type": "string"},
          "country": {"type": "string"}
        }
      },
      "PaginationMeta": {
        "type": "object",
        "properties": {
          "currentPage": {"type": "integer"},
          "totalPages": {"type": "integer"},
          "totalItems": {"type": "integer"},
          "itemsPerPage": {"type": "integer"}
        }
      }
    },
    "securitySchemes": {
      "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
      }
    }
  }
}
```'''

LONG_DOCUMENT_TEMPLATE_CODE = '''```python
# ============================================================================
# Order Processing Service - Core Business Logic
# ============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
import asyncio
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PAYMENT_PROCESSING = "payment_processing"
    PAYMENT_FAILED = "payment_failed"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"


@dataclass
class Address:
    street: str
    city: str
    state: str
    postal_code: str
    country: str
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "street": self.street,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
            "is_default": self.is_default,
        }


@dataclass
class OrderItem:
    product_id: str
    product_name: str
    quantity: int
    unit_price: Decimal
    discount: Decimal = Decimal("0.00")
    tax_rate: Decimal = Decimal("0.08")

    @property
    def subtotal(self) -> Decimal:
        return (self.unit_price * self.quantity) - self.discount

    @property
    def tax_amount(self) -> Decimal:
        return self.subtotal * self.tax_rate

    @property
    def total(self) -> Decimal:
        return self.subtotal + self.tax_amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "quantity": self.quantity,
            "unit_price": str(self.unit_price),
            "discount": str(self.discount),
            "tax_rate": str(self.tax_rate),
            "subtotal": str(self.subtotal),
            "tax_amount": str(self.tax_amount),
            "total": str(self.total),
        }


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    items: List[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    shipping_address: Optional[Address] = None
    billing_address: Optional[Address] = None
    payment_method: Optional[PaymentMethod] = None
    payment_transaction_id: Optional[str] = None
    shipping_cost: Decimal = Decimal("0.00")
    discount_code: Optional[str] = None
    discount_amount: Decimal = Decimal("0.00")
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    shipped_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[datetime] = None

    @property
    def items_subtotal(self) -> Decimal:
        return sum(item.subtotal for item in self.items)

    @property
    def items_tax(self) -> Decimal:
        return sum(item.tax_amount for item in self.items)

    @property
    def grand_total(self) -> Decimal:
        return self.items_subtotal + self.items_tax + self.shipping_cost - self.discount_amount

    def add_item(self, item: OrderItem) -> None:
        existing = next((i for i in self.items if i.product_id == item.product_id), None)
        if existing:
            existing.quantity += item.quantity
        else:
            self.items.append(item)
        self.updated_at = datetime.utcnow()

    def remove_item(self, product_id: str) -> bool:
        for i, item in enumerate(self.items):
            if item.product_id == product_id:
                self.items.pop(i)
                self.updated_at = datetime.utcnow()
                return True
        return False

    def update_status(self, new_status: OrderStatus) -> None:
        valid_transitions = {
            OrderStatus.PENDING: [OrderStatus.PAYMENT_PROCESSING, OrderStatus.CANCELLED],
            OrderStatus.PAYMENT_PROCESSING: [OrderStatus.CONFIRMED, OrderStatus.PAYMENT_FAILED],
            OrderStatus.PAYMENT_FAILED: [OrderStatus.PAYMENT_PROCESSING, OrderStatus.CANCELLED],
            OrderStatus.CONFIRMED: [OrderStatus.PROCESSING, OrderStatus.CANCELLED, OrderStatus.REFUNDED],
            OrderStatus.PROCESSING: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
            OrderStatus.SHIPPED: [OrderStatus.DELIVERED, OrderStatus.CANCELLED],
            OrderStatus.DELIVERED: [OrderStatus.REFUNDED],
            OrderStatus.CANCELLED: [],
            OrderStatus.REFUNDED: [],
        }

        if new_status not in valid_transitions.get(self.status, []):
            raise ValueError(f"Invalid transition from {self.status} to {new_status}")

        self.status = new_status
        self.updated_at = datetime.utcnow()

        if new_status == OrderStatus.SHIPPED:
            self.shipped_at = datetime.utcnow()
            self.estimated_delivery = datetime.utcnow() + timedelta(days=5)
        elif new_status == OrderStatus.DELIVERED:
            self.delivered_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "items": [item.to_dict() for item in self.items],
            "status": self.status.value,
            "shipping_address": self.shipping_address.to_dict() if self.shipping_address else None,
            "billing_address": self.billing_address.to_dict() if self.billing_address else None,
            "payment_method": self.payment_method.value if self.payment_method else None,
            "payment_transaction_id": self.payment_transaction_id,
            "items_subtotal": str(self.items_subtotal),
            "items_tax": str(self.items_tax),
            "shipping_cost": str(self.shipping_cost),
            "discount_code": self.discount_code,
            "discount_amount": str(self.discount_amount),
            "grand_total": str(self.grand_total),
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "shipped_at": self.shipped_at.isoformat() if self.shipped_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "tracking_number": self.tracking_number,
            "estimated_delivery": self.estimated_delivery.isoformat() if self.estimated_delivery else None,
        }


class OrderService:
    def __init__(self, db_session, payment_gateway, inventory_service, notification_service):
        self.db = db_session
        self.payment = payment_gateway
        self.inventory = inventory_service
        self.notifications = notification_service

    async def create_order(self, user_id: str, items: List[Dict], shipping_address: Dict) -> Order:
        logger.info(f"Creating order for user {user_id}")

        # Validate inventory
        for item in items:
            available = await self.inventory.check_availability(item["product_id"], item["quantity"])
            if not available:
                raise ValueError(f"Insufficient inventory for product {item['product_id']}")

        # Create order
        order = Order(user_id=user_id, shipping_address=Address(**shipping_address))
        for item in items:
            product = await self.inventory.get_product(item["product_id"])
            order.add_item(OrderItem(
                product_id=product["id"],
                product_name=product["name"],
                quantity=item["quantity"],
                unit_price=Decimal(str(product["price"])),
            ))

        # Reserve inventory
        for item in order.items:
            await self.inventory.reserve(item.product_id, item.quantity)

        # Save order
        await self.db.save(order)
        logger.info(f"Order {order.id} created successfully")

        return order

    async def process_payment(self, order_id: str, payment_method: PaymentMethod, payment_details: Dict) -> bool:
        order = await self.db.get(Order, order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.update_status(OrderStatus.PAYMENT_PROCESSING)
        order.payment_method = payment_method

        try:
            result = await self.payment.charge(
                amount=float(order.grand_total),
                currency="USD",
                payment_method=payment_method.value,
                details=payment_details,
            )
            order.payment_transaction_id = result["transaction_id"]
            order.update_status(OrderStatus.CONFIRMED)
            await self.notifications.send_order_confirmation(order)
            logger.info(f"Payment successful for order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Payment failed for order {order_id}: {e}")
            order.update_status(OrderStatus.PAYMENT_FAILED)
            # Release reserved inventory
            for item in order.items:
                await self.inventory.release(item.product_id, item.quantity)
            return False

        finally:
            await self.db.save(order)

    async def ship_order(self, order_id: str, tracking_number: str) -> None:
        order = await self.db.get(Order, order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.update_status(OrderStatus.SHIPPED)
        order.tracking_number = tracking_number
        await self.db.save(order)

        await self.notifications.send_shipping_notification(order)
        logger.info(f"Order {order_id} shipped with tracking {tracking_number}")

    async def cancel_order(self, order_id: str, reason: str = "") -> None:
        order = await self.db.get(Order, order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.update_status(OrderStatus.CANCELLED)
        order.notes = f"Cancelled: {reason}"
        await self.db.save(order)

        # Release inventory
        for item in order.items:
            await self.inventory.release(item.product_id, item.quantity)

        # Process refund if payment was made
        if order.payment_transaction_id:
            await self.payment.refund(order.payment_transaction_id)

        await self.notifications.send_cancellation_notice(order)
        logger.info(f"Order {order_id} cancelled: {reason}")
```'''

LONG_DOCUMENT_TEMPLATES = [
    LONG_DOCUMENT_TEMPLATE_LOGS,
    LONG_DOCUMENT_TEMPLATE_API,
    LONG_DOCUMENT_TEMPLATE_CODE,
]

# Document injection prompts
DOCUMENT_INJECTION_PROMPTS = [
    ("Here are the server logs from the last hour:", "Let me analyze these logs... "),
    ("I found this API specification:", "I've reviewed the API spec. "),
    ("Here's the code we need to review:", "I've analyzed the code. "),
    ("Can you look at these logs?", "Examining the logs... "),
    ("This is the documentation:", "I've read through the documentation. "),
    ("Here's what the monitoring shows:", "Based on the monitoring data... "),
]


# =============================================================================
# Long Conversation Generator (Context Window Stress Test)
# =============================================================================

class LongConversationGenerator(ConversationGenerator):
    """
    Generates massive conversations to stress-test context windows.

    Key features:
    - Injects ~2500-token documents every 20 turns
    - Total target: >50,000 tokens
    - Forces summarization to perform aggressive lossy compression
    """

    def __init__(
        self,
        total_turns: int = 200,
        fact_turns: List[int] = None,
        compression_turn: int = 180,
        document_injection_interval: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__(
            total_turns=total_turns,
            fact_turns=fact_turns or [10, 50, 100, 150],
            compression_turn=compression_turn,
            seed=seed,
        )
        self.document_injection_interval = document_injection_interval
        # Generate document injection turns, excluding fact turns
        fact_turn_set = set(self.fact_turns)
        self.document_injection_turns = [
            t for t in range(document_injection_interval, compression_turn, document_injection_interval)
            if t not in fact_turn_set
        ]

    def generate(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        num_facts: int = 4,
        include_distractors: bool = False,
    ) -> SyntheticConversation:
        """Generate a massive conversation with document injections."""
        import uuid

        conversation_id = str(uuid.uuid4())[:8]
        turns: List[ConversationTurn] = []
        planted_facts: List[PlantedFact] = []

        # Select fact templates
        all_templates = (
            DECISION_TEMPLATES + KEY_FACT_TEMPLATES + REMINDER_TEMPLATES
        )
        selected_templates = self.rng.sample(
            all_templates, min(num_facts, len(all_templates))
        )

        # Map fact turns to templates
        fact_schedule = dict(zip(self.fact_turns[:num_facts], selected_templates))

        # Track document injections
        doc_index = 0

        # Generate turns
        for turn_id in range(1, self.total_turns + 1):
            if turn_id in fact_schedule:
                # Generate a fact turn
                template = fact_schedule[turn_id]
                turn, fact = self._generate_fact_turn(
                    turn_id, template, difficulty
                )
                turns.append(turn)
                planted_facts.append(fact)

            elif turn_id in self.document_injection_turns:
                # Generate a long document injection turn
                turn = self._generate_document_turn(turn_id, doc_index)
                turns.append(turn)
                doc_index = (doc_index + 1) % len(LONG_DOCUMENT_TEMPLATES)

            else:
                # Generate filler turn
                turn = self._generate_filler_turn(turn_id)
                turns.append(turn)

        # Calculate approximate token count
        total_chars = sum(
            len(t.user) + len(t.assistant)
            for t in turns
        )
        approx_tokens = total_chars // 4  # ~4 chars per token

        return SyntheticConversation(
            id=conversation_id,
            turns=turns,
            planted_facts=planted_facts,
            metadata={
                "total_turns": self.total_turns,
                "fact_turns": self.fact_turns[:num_facts],
                "compression_turn": self.compression_turn,
                "difficulty": difficulty.value,
                "has_distractors": include_distractors,
                "document_injection_turns": self.document_injection_turns,
                "num_document_injections": len(self.document_injection_turns),
                "approx_tokens": approx_tokens,
                "stress_test": True,
            },
        )

    def _generate_document_turn(
        self,
        turn_id: int,
        doc_index: int,
    ) -> ConversationTurn:
        """Generate a turn with a massive document injection."""
        prompt_template = self.rng.choice(DOCUMENT_INJECTION_PROMPTS)
        user_prompt, assistant_prefix = prompt_template

        document = LONG_DOCUMENT_TEMPLATES[doc_index % len(LONG_DOCUMENT_TEMPLATES)]

        # User provides the document
        user_msg = f"{user_prompt}\n\n{document}"

        # Assistant acknowledges and adds some analysis
        analysis = self.rng.choice([
            "I see several interesting patterns here. The main takeaway is that the system is operating normally with some minor issues to watch.",
            "After reviewing this, I notice a few areas that could be optimized. Overall, the implementation looks solid.",
            "This is comprehensive. I've identified the key points and we can proceed with our discussion.",
            "Good reference material. This will be helpful as we continue our conversation.",
            "I've noted the important details. Let me know if you want me to elaborate on any specific part.",
        ])

        assistant_msg = f"{assistant_prefix}{analysis}"

        return ConversationTurn(
            turn_id=turn_id,
            user=user_msg,
            assistant=assistant_msg,
        )


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_long_dataset(
    num_conversations: int = 10,
    seed: int = 42,
    max_workers: int = 10,
) -> EvaluationDataset:
    """
    Generate long conversation dataset (200 turns, >50k tokens each).

    Args:
        num_conversations: Number of conversations (default 10, since they're massive)
        seed: Random seed for reproducibility
        max_workers: Number of parallel workers (default 10; tune down if rate limits hit)

    Returns:
        EvaluationDataset with stress-test conversations
    """
    conversations = [None] * num_conversations

    def generate_one(idx: int) -> Tuple[int, 'SyntheticConversation']:
        # Each worker gets its own generator with a unique seed
        generator = LongConversationGenerator(
            total_turns=200,
            fact_turns=[10, 50, 100, 150],  # Sparse: 4 facts spread across 200 turns
            compression_turn=180,            # Late compression
            document_injection_interval=8,  # Document every 8 turns = ~22 documents for ~50k tokens
            seed=seed + idx,  # Unique seed per conversation
        )
        conv = generator.generate(
            difficulty=DifficultyLevel.MEDIUM,
            num_facts=4,
            include_distractors=False,
        )
        return idx, conv

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_one, i): i
            for i in range(num_conversations)
        }

        completed = 0
        for future in as_completed(futures):
            idx, conv = future.result()
            conversations[idx] = conv
            completed += 1
            print(f"Generated conversation {completed}/{num_conversations}: {conv.id}")

    # Calculate total tokens across all conversations
    total_tokens = sum(
        c.metadata.get("approx_tokens", 0)
        for c in conversations
    )
    avg_tokens = total_tokens // num_conversations if num_conversations > 0 else 0

    return EvaluationDataset(
        conversations=conversations,
        metadata={
            "num_conversations": num_conversations,
            "total_turns": 200,
            "fact_turns": [10, 50, 100, 150],
            "compression_turn": 180,
            "difficulty": "medium",
            "seed": seed,
            "generator_version": "2.0-stress-test",
            "purpose": "context_window_stress_test",
            "document_injection_interval": 20,
            "total_tokens_approx": total_tokens,
            "avg_tokens_per_conversation": avg_tokens,
            "target_compression_ratio": "25x",  # 50k -> 2k tokens
        },
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token for English)."""
    return len(text) // 4


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate context window stress test dataset"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview token counts without generating full dataset",
    )
    parser.add_argument(
        "--num-conversations", "-n",
        type=int,
        default=10,
        help="Number of conversations to generate",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(Path(__file__).parent / "data" / "eval_set_long.json"),
        help="Output path for generated dataset (default: experiments/data/eval_set_long.json)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CONTEXT WINDOW STRESS TEST GENERATOR")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Total turns: 200 (4x standard)")
    print("  - Fact turns: [10, 50, 100, 150] (sparse needles)")
    print("  - Document injection: Every 8 turns (~2000 tokens each)")
    print("  - Total document injections: ~22 per conversation")
    print("  - Target tokens per conversation: >40,000")
    print("  - Compression point: Turn 180")
    print()

    if args.preview:
        # Show token breakdown for one conversation
        print("Token Breakdown (Single Conversation):")
        print("-" * 50)

        generator = LongConversationGenerator(seed=42, document_injection_interval=8)
        conv = generator.generate(difficulty=DifficultyLevel.MEDIUM)

        doc_tokens = 0
        filler_tokens = 0
        fact_tokens = 0

        for turn in conv.turns:
            turn_tokens = estimate_tokens(turn.user + turn.assistant)
            if turn.turn_id in generator.document_injection_turns:
                doc_tokens += turn_tokens
            elif turn.contains_fact:
                fact_tokens += turn_tokens
            else:
                filler_tokens += turn_tokens

        total = doc_tokens + filler_tokens + fact_tokens

        print(f"  Document turns ({len(generator.document_injection_turns)}): ~{doc_tokens:,} tokens")
        print(f"  Fact turns (4): ~{fact_tokens:,} tokens")
        print(f"  Filler turns ({200 - len(generator.document_injection_turns) - 4}): ~{filler_tokens:,} tokens")
        print(f"  TOTAL: ~{total:,} tokens")
        print()
        print(f"  Compression ratio (to 2k summary): {total // 2000}x")
        print()

        # Show document injection turns
        print("Document Injection Schedule:")
        print(f"  Turns: {generator.document_injection_turns}")
        print()

        # Show fact turns vs compression point
        print("Fact vs Compression Timeline:")
        print("  [10]----[50]----[100]----[150]----[180:COMPRESS]----[200]")
        print("   F1      F2      F3       F4     |<-LOST IF NOT EXTRACTED")

    else:
        # Generate full dataset
        print(f"Generating {args.num_conversations} stress-test conversations...")
        print()

        dataset = generate_long_dataset(
            num_conversations=args.num_conversations,
            seed=42
        )

        output_path = Path(args.output)
        dataset.save(str(output_path))

        # Print summary
        print()
        print("=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        print(f"  Total conversations: {len(dataset.conversations)}")
        print(f"  Turns per conversation: 200")

        total_facts = sum(len(c.planted_facts) for c in dataset.conversations)
        print(f"  Total planted facts: {total_facts}")

        avg_tokens = dataset.metadata.get("avg_tokens_per_conversation", 0)
        total_tokens = dataset.metadata.get("total_tokens_approx", 0)
        print(f"  Avg tokens per conversation: ~{avg_tokens:,}")
        print(f"  Total tokens in dataset: ~{total_tokens:,}")

        # Show fact placement for first conversation
        if dataset.conversations:
            conv = dataset.conversations[0]
            print(f"\nSample conversation {conv.id}:")
            print(f"  Approx tokens: ~{conv.metadata.get('approx_tokens', 0):,}")
            print(f"  Document injections: {conv.metadata.get('num_document_injections', 0)}")
            print()
            for fact in conv.planted_facts:
                print(f"  - Turn {fact.turn_id}: {fact.content}")
                print(f"    Question: {fact.test_question}")
                print(f"    Answer: {fact.ground_truth}")

        print(f"\nSaved to: {output_path}")
        print()
        print("STRESS TEST GOAL:")
        print("  Summarization must compress ~50k tokens to ~2k (25x compression).")
        print("  This GUARANTEES lossy compression - facts WILL be lost.")
        print("  CogCanvas should preserve all 4 planted facts verbatim.")


if __name__ == "__main__":
    main()
