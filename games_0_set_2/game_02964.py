
# Generated: 2025-08-27T21:57:42.902248
# Source Brief: brief_02964.md
# Brief Index: 2964

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import defaultdict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player connects same-colored nodes on a grid.
    The goal is to form a single continuous line for each color before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select/deselect a node. Connect all nodes of the same color. Hold Shift to reset the cursor."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect-the-Nodes: Form a single line for each color before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    MARGIN_X = 40
    MARGIN_Y = 40
    CELL_W = (SCREEN_WIDTH - 2 * MARGIN_X) // GRID_COLS
    CELL_H = (SCREEN_HEIGHT - 2 * MARGIN_Y) // GRID_ROWS
    NODE_RADIUS = int(min(CELL_W, CELL_H) * 0.35)
    CONNECTION_WIDTH = 6
    MAX_MOVES = 15
    NUM_COLORS = 4
    NODES_PER_COLOR = 4

    # --- Colors ---
    COLOR_BG = (25, 28, 32)
    COLOR_GRID = (50, 55, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_SELECT = (255, 255, 255)
    NODE_COLORS = [
        (0, 200, 255),  # Cyan
        (255, 100, 200), # Pink
        (100, 255, 100), # Green
        (255, 180, 0),   # Orange
    ]

    class Node:
        """Helper class to store node properties."""
        def __init__(self, x, y, color_idx, node_id):
            self.x = x
            self.y = y
            self.color_idx = color_idx
            self.color = GameEnv.NODE_COLORS[color_idx]
            self.id = node_id
            self.screen_pos = (
                GameEnv.MARGIN_X + x * GameEnv.CELL_W + GameEnv.CELL_W // 2,
                GameEnv.MARGIN_Y + y * GameEnv.CELL_H + GameEnv.CELL_H // 2,
            )
        def __eq__(self, other):
            return isinstance(other, GameEnv.Node) and self.id == other.id
        def __hash__(self):
            return hash(self.id)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.nodes = []
        self.connections = []
        self.cursor_pos = [0, 0]
        self.selected_nodes = []
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.completed_colors = set()
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [0, 0]
        self.selected_nodes = []
        self.connections = []
        self.completed_colors = set()
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        self.nodes.clear()
        occupied_positions = set()
        node_id_counter = 0
        
        for color_idx in range(self.NUM_COLORS):
            for _ in range(self.NODES_PER_COLOR):
                while True:
                    pos = (
                        self.np_random.integers(0, self.GRID_COLS),
                        self.np_random.integers(0, self.GRID_ROWS),
                    )
                    if pos not in occupied_positions:
                        occupied_positions.add(pos)
                        node = self.Node(pos[0], pos[1], color_idx, node_id_counter)
                        self.nodes.append(node)
                        node_id_counter += 1
                        break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if shift_held:
            self.cursor_pos = [0, 0]
        
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        if space_held:
            # sound: 'select.wav'
            node_at_cursor = self._get_node_at(self.cursor_pos[0], self.cursor_pos[1])
            if node_at_cursor:
                if node_at_cursor in self.selected_nodes:
                    self.selected_nodes.remove(node_at_cursor)
                else:
                    self.selected_nodes.append(node_at_cursor)
                
                if len(self.selected_nodes) == 2:
                    reward += self._attempt_connection()
                    self.selected_nodes.clear()

        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            terminal_reward = 0
            if self.win:
                terminal_reward = 50
                # sound: 'win_game.wav'
            else:
                terminal_reward = -50
                # sound: 'lose_game.wav'
            reward += terminal_reward
            self.score += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _attempt_connection(self):
        node1, node2 = self.selected_nodes
        self.moves_left -= 1
        
        if node1.color_idx != node2.color_idx:
            # sound: 'error.wav'
            return -0.1 # Mismatched colors

        adj = defaultdict(list)
        for c_node1, c_node2 in self.connections:
            adj[c_node1.id].append(c_node2)
            adj[c_node2.id].append(c_node1)

        q, visited = [node1], {node1.id}
        while q:
            curr = q.pop(0)
            if curr.id == node2.id:
                # sound: 'error.wav'
                return -0.1 # Already in the same component
            for neighbor in adj[curr.id]:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    q.append(neighbor)
        
        for c_node1, c_node2 in self.connections:
            if self._line_segments_intersect(node1, node2, c_node1, c_node2):
                # sound: 'error.wav'
                return -0.1 # Path is blocked

        # --- Valid Connection ---
        # sound: 'connect.wav'
        self.connections.append((node1, node2))
        reward = 1.0

        if node1.color_idx not in self.completed_colors:
            adj[node1.id].append(node2)
            adj[node2.id].append(node1)
            
            q, visited, count = [node1], {node1.id}, 1
            while q:
                curr = q.pop(0)
                for neighbor in adj[curr.id]:
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        q.append(neighbor)
                        count += 1
            
            if count == self.NODES_PER_COLOR:
                # sound: 'complete_group.wav'
                self.completed_colors.add(node1.color_idx)
                reward += 5.0
        
        return reward

    def _check_termination(self):
        if len(self.completed_colors) == self.NUM_COLORS:
            self.win = True
            return True
        if self.moves_left <= 0:
            return True
        if self.steps >= 1000:
             return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "completed_groups": len(self.completed_colors),
        }

    def _render_game(self):
        for x in range(self.GRID_COLS + 1):
            px = self.MARGIN_X + x * self.CELL_W
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.MARGIN_Y), (px, self.SCREEN_HEIGHT - self.MARGIN_Y))
        for y in range(self.GRID_ROWS + 1):
            py = self.MARGIN_Y + y * self.CELL_H
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.MARGIN_X, py), (self.SCREEN_WIDTH - self.MARGIN_X, py))

        for node1, node2 in self.connections:
            pygame.draw.line(self.screen, node1.color, node1.screen_pos, node2.screen_pos, self.CONNECTION_WIDTH)

        pulse = (math.sin(self.steps * 0.15) + 1) / 2 * 3
        for node in self.nodes:
            radius = self.NODE_RADIUS
            if node.color_idx in self.completed_colors:
                radius = int(self.NODE_RADIUS + pulse)
            
            pygame.gfxdraw.filled_circle(self.screen, node.screen_pos[0], node.screen_pos[1], radius, node.color)
            pygame.gfxdraw.aacircle(self.screen, node.screen_pos[0], node.screen_pos[1], radius, node.color)
            
            if node in self.selected_nodes:
                pygame.draw.circle(self.screen, self.COLOR_CURSOR_SELECT, node.screen_pos, radius + 3, 3)

        cursor_rect = pygame.Rect(
            self.MARGIN_X + self.cursor_pos[0] * self.CELL_W,
            self.MARGIN_Y + self.cursor_pos[1] * self.CELL_H,
            self.CELL_W, self.CELL_H
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_node_at(self, x, y):
        for node in self.nodes:
            if node.x == x and node.y == y:
                return node
        return None

    def _on_segment(self, p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))

    def _orientation(self, p, q, r):
        val = (q.screen_pos[1] - p.screen_pos[1]) * (r.screen_pos[0] - q.screen_pos[0]) - \
              (q.screen_pos[0] - p.screen_pos[0]) * (r.screen_pos[1] - q.screen_pos[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    def _line_segments_intersect(self, p1, q1, p2, q2):
        if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
            return False

        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True
        return False

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    key_cooldown = 0
    COOLDOWN_FRAMES = 3 # Prevent multiple inputs from one key press

    while running:
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        if key_cooldown > 0:
            key_cooldown -= 1

        if key_cooldown == 0:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            if movement or space_held or shift_held:
                action = [movement, space_held, shift_held]
                obs, reward, terminated, truncated, info = env.step(action)
                key_cooldown = COOLDOWN_FRAMES
                action_taken = True
                if terminated or truncated:
                    print(f"Game Over! Final Score: {info['score']}")

        if not action_taken:
            obs = env._get_observation()

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()