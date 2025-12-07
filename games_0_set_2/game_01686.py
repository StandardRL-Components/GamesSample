
# Generated: 2025-08-27T17:56:30.029872
# Source Brief: brief_01686.md
# Brief Index: 1686

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: A minimalist puzzle game where the player connects same-colored nodes on a grid.

    The goal is to connect all pairs of nodes within a limited number of moves.
    Each successful connection between two nodes of the same color clears them from the board
    and awards points. Clearing all nodes of a single color gives a bonus, and clearing
    the entire board results in a large victory bonus. Running out of moves ends the game.

    The environment is designed with visual clarity and a satisfying user experience in mind,
    featuring clean geometric shapes, smooth antialiased rendering, and clear feedback for
    player actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a node, then "
        "move to its partner and press Space again to connect. Press Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect all pairs of same-colored nodes on the grid to solve the puzzle. "
        "You have a limited number of moves, so plan your path wisely!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Sizing
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    NODE_RADIUS = 12
    LINE_WIDTH = 5

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 0, 150)
    NODE_COLORS = [
        (255, 70, 70),   # Red
        (70, 150, 255),  # Blue
        (80, 220, 80),   # Green
        (255, 200, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    NUM_PAIRS = 5
    MAX_MOVES = 20
    MAX_STEPS = 1000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("sans", 40, bold=True)

        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.cursor_pos = [0, 0]
        self.nodes = {}
        self.node_pairs = {}
        self.connections = []
        self.connected_colors = set()
        self.selected_node_info = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.np_random = None
        self.feedback_msg = ""
        self.feedback_timer = 0
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.connections = []
        self.connected_colors = set()
        self.selected_node_info = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_msg = ""
        self.feedback_timer = 0

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def _generate_puzzle(self):
        self.nodes = {}
        self.node_pairs = {}
        
        grid_cells = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(grid_cells)

        for i in range(self.NUM_PAIRS):
            color = self.NODE_COLORS[i % len(self.NODE_COLORS)]
            pos1 = grid_cells.pop()
            pos2 = grid_cells.pop()
            self.nodes[pos1] = color
            self.nodes[pos2] = color
            self.node_pairs[color] = (pos1, pos2)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self._handle_movement(movement)
        
        if shift_pressed:
            self._handle_deselect()
        
        if space_pressed:
            reward += self._handle_selection()

        self.steps += 1
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_msg = ""

        # Check termination conditions
        if len(self.connections) == self.NUM_PAIRS:
            self.game_over = True
            terminated = True
            reward += 100  # Victory bonus
            self._set_feedback("PUZZLE COMPLETE!", 90)
        elif self.moves_left <= 0:
            self.game_over = True
            terminated = True
            reward += -100  # Loss penalty
            self._set_feedback("OUT OF MOVES!", 90)
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_H) % self.GRID_H
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_H
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_W) % self.GRID_W
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_W

    def _handle_deselect(self):
        if self.selected_node_info:
            self.selected_node_info = None
            self._set_feedback("Selection Canceled", 30)

    def _handle_selection(self):
        cursor_tuple = tuple(self.cursor_pos)
        reward = 0

        if cursor_tuple not in self.nodes or self._is_node_connected(cursor_tuple):
            return 0 # No action on empty or already connected cell

        if self.selected_node_info is None:
            # First selection
            self.selected_node_info = {
                "pos": cursor_tuple,
                "color": self.nodes[cursor_tuple]
            }
        else:
            # Second selection (connection attempt)
            self.moves_left = max(0, self.moves_left - 1)
            
            # Check for valid connection
            is_same_color = self.nodes[cursor_tuple] == self.selected_node_info["color"]
            is_different_node = cursor_tuple != self.selected_node_info["pos"]

            if is_same_color and is_different_node:
                # SUCCESSFUL CONNECTION
                # sound: connection_success.wav
                self.connections.append((self.selected_node_info["pos"], cursor_tuple))
                self.score += 1
                reward += 1

                # Check if this completes a color pair
                color = self.selected_node_info["color"]
                if color not in self.connected_colors:
                    self.connected_colors.add(color)
                    self.score += 10
                    reward += 10
                    self._set_feedback("COLOR CLEARED!", 45)
                else:
                    self._set_feedback("CONNECTED!", 30)

            else:
                # FAILED CONNECTION
                # sound: connection_fail.wav
                self._set_feedback("MISMATCH!", 30)

            self.selected_node_info = None
        return reward
    
    def _is_node_connected(self, pos):
        for p1, p2 in self.connections:
            if p1 == pos or p2 == pos:
                return True
        return False

    def _set_feedback(self, msg, duration):
        self.feedback_msg = msg
        self.feedback_timer = duration

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._draw_connections()
        self._draw_nodes()
        self._draw_cursor()

    def _draw_grid(self):
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.WIDTH, py))

    def _draw_connections(self):
        for p1, p2 in self.connections:
            color = self.nodes[p1]
            start_pos = (
                int(p1[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(p1[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            )
            end_pos = (
                int(p2[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(p2[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            )
            pygame.draw.line(self.screen, color, start_pos, end_pos, self.LINE_WIDTH)

    def _draw_nodes(self):
        for pos, color in self.nodes.items():
            if not self._is_node_connected(pos):
                center_px = (
                    int(pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                    int(pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
                )
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], self.NODE_RADIUS, color)
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], self.NODE_RADIUS, color)

    def _draw_cursor(self):
        # Draw selected node highlight
        if self.selected_node_info:
            pos = self.selected_node_info["pos"]
            color = self.selected_node_info["color"]
            center_px = (
                int(pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2),
                int(pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            )
            # Pulsing effect
            pulse_radius = self.NODE_RADIUS + 4 + int(2 * math.sin(self.steps * 0.2))
            pulse_alpha = 100 + int(50 * math.sin(self.steps * 0.2))
            pulse_color = (*color, pulse_alpha)
            
            # Create a temporary surface for the glowing circle
            temp_surface = pygame.Surface((pulse_radius*2, pulse_radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surface, pulse_radius, pulse_radius, pulse_radius, pulse_color)
            self.screen.blit(temp_surface, (center_px[0] - pulse_radius, center_px[1] - pulse_radius))

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE,
            self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Use a surface with alpha for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 2)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(moves_text, moves_rect)
        
        # Feedback message
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30.0)))
            feedback_surf = self.font_msg.render(self.feedback_msg, True, (*self.COLOR_TEXT[:3], alpha))
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(feedback_surf, feedback_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "completed_pairs": len(self.connections),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        # We need to reset to generate the first observation
        obs, _ = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Node Connector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0  # No-op
        space_held = False
        shift_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0

        # Since auto_advance is False, we need a small delay for human playability
        clock.tick(15) # Controls how fast human inputs are registered

    env.close()