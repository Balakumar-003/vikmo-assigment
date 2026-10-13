from django.test import TestCase
from rest_framework.test import APIClient

from core.models import Product, Dealer, Order, ChatMessage


class AssistantAPITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.dealer = Dealer.objects.create(
            dealer_code="DLR-001",
            name="Apex Motors",
            city="Bengaluru",
        )
        self.product = Product.objects.create(
            sku="BRK-1044",
            name="Brake Pad Set - Bajaj Pulsar 150",
            category="Brakes",
            brand="Endurance",
            vehicle_fitment="Bajaj Pulsar 150",
            price=1459.50,
            stock=136,
            description="High friction brake pads for Pulsar 150",
        )

    def test_assistant_chat_greeting(self):
        res = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-1",
                "dealer_id": self.dealer.id,
                "message": "Hello, what can you help me with?",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data["reply"], str)
        self.assertIsInstance(data["cited_skus"], list)
        self.assertIsInstance(data["tool_calls"], list)
        self.assertIsInstance(data["actions"], list)

        # Check session message persistence
        msg_count = ChatMessage.objects.filter(session_id="test-session-1").count()
        self.assertEqual(msg_count, 2)  # 1 user + 1 assistant

    def test_assistant_chat_search_and_citation(self):
        res = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-2",
                "dealer_id": self.dealer.id,
                "message": "Do you have brake pads for a Pulsar 150?",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("BRK-1044", data["cited_skus"])
        self.assertTrue(any(tc["name"] == "search_parts" for tc in data["tool_calls"]))

    def test_assistant_chat_check_stock(self):
        res = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-3",
                "dealer_id": self.dealer.id,
                "message": "What is the stock for BRK-1044?",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("136", data["reply"])
        self.assertIn("BRK-1044", data["cited_skus"])
        self.assertTrue(any(tc["name"] == "check_stock" for tc in data["tool_calls"]))

    def test_assistant_chat_create_draft_order(self):
        res = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-4",
                "dealer_id": self.dealer.id,
                "message": "Please order 2 units of BRK-1044",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["actions"]), 1)
        action = data["actions"][0]
        self.assertEqual(action["type"], "draft_order_created")
        self.assertTrue(action["order_number"].startswith("ORD-"))

        # Check DB order
        order = Order.objects.get(order_number=action["order_number"])
        self.assertEqual(order.status, "DRAFT")
        self.assertEqual(order.dealer_id, self.dealer.id)
        self.assertEqual(order.items.count(), 1)
        self.assertEqual(order.items.first().quantity, 2)

    def test_assistant_chat_off_topic_decline(self):
        res = self.client.post(
            "/api/assistant/chat/",
            {
                "session_id": "test-session-5",
                "dealer_id": self.dealer.id,
                "message": "What is the weather in Delhi today?",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("only assist with", data["reply"].lower())
        self.assertEqual(len(data["actions"]), 0)

    def test_assistant_chat_validation_errors(self):
        # Missing session_id -> 400
        res1 = self.client.post(
            "/api/assistant/chat/",
            {"dealer_id": self.dealer.id, "message": "Hi"},
            format="json",
        )
        self.assertEqual(res1.status_code, 400)

        # Missing dealer_id -> 400
        res2 = self.client.post(
            "/api/assistant/chat/",
            {"session_id": "s1", "message": "Hi"},
            format="json",
        )
        self.assertEqual(res2.status_code, 400)

        # Nonexistent dealer -> 400
        res3 = self.client.post(
            "/api/assistant/chat/",
            {"session_id": "s1", "dealer_id": 99999, "message": "Hi"},
            format="json",
        )
        self.assertEqual(res3.status_code, 400)
