import unittest
from src.agents.base_agent import BaseAgent

class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self.agent = BaseAgent()

    def test_initialize(self):
        self.agent.initialize()
        self.assertTrue(self.agent.is_initialized)

    def test_act(self):
        self.agent.initialize()
        action = self.agent.act()
        self.assertIsNotNone(action)

    def test_learn(self):
        self.agent.initialize()
        initial_state = self.agent.get_state()
        self.agent.learn(initial_state, action='some_action', reward=1)
        self.assertNotEqual(initial_state, self.agent.get_state())

if __name__ == '__main__':
    unittest.main()