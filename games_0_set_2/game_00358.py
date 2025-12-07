
# Generated: 2025-08-27T13:25:15.810330
# Source Brief: brief_00358.md
# Brief Index: 358

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A minimalist puzzle game where the player connects same-colored nodes on a grid.
    The goal is to connect all nodes of each color into a single contiguous group
    before running out of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a node, "
        "then move to another node of the same color and press Space again to connect. "
        "Press Shift to cancel a selection."
    )

    game_description = (
        "Connect same-colored nodes on a grid to clear the board within a limited number of moves. "
        "Plan your paths carefully to avoid getting stuck!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame and Display Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.grid_cols, self.grid_rows = 16, 10
        self.cell_size = 40
        self.grid_offset_x = (self.screen_width - self.grid_cols * self.cell_size) // 2
        self.grid_offset_y = (self.screen_height - self.grid_rows * self.cell_size) // 2
        
        self.initial_moves = 30
        self.max_steps = 500

        # --- Visuals ---
        self.color_bg = (20, 30, 40)
        self.color_grid = (40, 60, 80)
        self.color_text = (220, 220, 230)
        self.color_text_shadow = (10, 15, 20)
        self.node_colors = [
            (52, 152, 219),   # Blue
            (231, 76, 60),    # Red
            (46, 204, 113),   # Green
            (241, 196, 15),   # Yellow
        ]
        self.color_cursor = (255, 255, 255)
        self.color_invalid = (200, 0, 0)
        self.color_valid_preview = (255, 255, 255)

        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_remaining = 0
        self.cursor_pos = [0, 0]
        self.selected_node = None
        self.nodes = {}
        self.connections = set()
        self.color_node_map = {}
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_effect = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_remaining = self.initial_moves

        self.cursor_pos = [self.grid_cols // 2, self.grid_rows // 2]
        self.selected_node = None
        
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_effect = None

        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        self.nodes = {}  # (x, y) -> color_index
        self.connections = set()
        
        all_cells = [(x, y) for x in range(self.grid_cols) for y in range(self.grid_rows)]
        self.np_random.shuffle(all_cells)
        
        num_colors = self.np_random.integers(3, len(self.node_colors) + 1)
        num_nodes_per_color = self.np_random.integers(3, 5)
        
        self.color_node_map = {i: [] for i in range(num_colors)}
        
        cell_idx = 0
        for color_idx in range(num_colors):
            for _ in range(num_nodes_per_color):
                if cell_idx < len(all_cells):
                    pos = tuple(all_cells[cell_idx])
                    self.nodes[pos] = color_idx
                    self.color_node_map[color_idx].append(pos)
                    cell_idx += 1
                else:
                    break

    def step(self, action):
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_cols - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_rows - 1)

        # 2. Handle Actions
        if shift_pressed and self.selected_node:
            self.selected_node = None
            # sfx: cancel_select

        if space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            node_at_cursor = self.nodes.get(cursor_tuple)

            if node_at_cursor is not None:
                if not self.selected_node:
                    self.selected_node = cursor_tuple
                    # sfx: select_node
                elif self.selected_node == cursor_tuple:
                    self.selected_node = None # Deselect if clicking same node
                    # sfx: cancel_select
                else: # Attempt connection
                    is_valid, _ = self._is_valid_connection(self.selected_node, cursor_tuple)
                    if is_valid:
                        # sfx: connect_success
                        connection_reward = self._make_connection(self.selected_node, cursor_tuple)
                        reward += connection_reward
                        self.moves_remaining -= 1
                        self.feedback_effect = {"type": "success", "pos": cursor_tuple, "timer": 15}
                    else:
                        # sfx: connect_fail
                        reward -= 0.1
                        self.feedback_effect = {"type": "fail", "pos": cursor_tuple, "timer": 15}
                    self.selected_node = None # Clear selection after any attempt
            elif self.selected_node: # Clicked on empty space while a node is selected
                self.selected_node = None
                # sfx: cancel_select

        # 3. Update game state and check for termination
        self.steps += 1
        terminated = False
        
        if self._check_win_condition():
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50
        elif self.moves_remaining <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 50
        
        if self.steps >= self.max_steps:
            terminated = True
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_valid_connection(self, pos1, pos2):
        if self.nodes.get(pos1) != self.nodes.get(pos2):
            return False, "mismatched_color"
        
        dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        if dist != 1:
            return False, "not_adjacent"
            
        # Check if connection (in either direction) already exists
        if (pos1, pos2) in self.connections or (pos2, pos1) in self.connections:
            return False, "already_connected"
            
        return True, "ok"

    def _make_connection(self, pos1, pos2):
        # Ensure consistent key order for the set
        connection = tuple(sorted((pos1, pos2)))
        self.connections.add(connection)
        
        # Check if this connection completes the color group
        color_idx = self.nodes[pos1]
        if self._is_group_connected(color_idx):
            # sfx: group_complete
            return 5 + 1 # +5 for group completion, +1 for valid connection
        return 1 # +1 for valid connection

    def _is_group_connected(self, color_idx):
        nodes_in_group = self.color_node_map.get(color_idx, [])
        if not nodes_in_group or len(nodes_in_group) < 2:
            return True # A single node is a connected group

        q = deque([nodes_in_group[0]])
        visited = {nodes_in_group[0]}
        
        while q:
            current = q.popleft()
            for neighbor in nodes_in_group:
                if neighbor not in visited:
                    connection = tuple(sorted((current, neighbor)))
                    if connection in self.connections:
                        visited.add(neighbor)
                        q.append(neighbor)
        
        return len(visited) == len(nodes_in_group)

    def _check_win_condition(self):
        if not self.nodes: return True # Empty board is a win
        for color_idx in self.color_node_map:
            if not self._is_group_connected(color_idx):
                return False
        return True

    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return int(x), int(y)

    def _draw_text(self, text, pos, font, color, shadow_color=None, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        if shadow_color:
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.grid_rows + 1):
            y = self.grid_offset_y + r * self.cell_size
            pygame.draw.line(self.screen, self.color_grid, (self.grid_offset_x, y), (self.screen_width - self.grid_offset_x, y))
        for c in range(self.grid_cols + 1):
            x = self.grid_offset_x + c * self.cell_size
            pygame.draw.line(self.screen, self.color_grid, (x, self.grid_offset_y), (x, self.screen_height - self.grid_offset_y))

        # Draw connections
        for pos1, pos2 in self.connections:
            color = self.node_colors[self.nodes[pos1]]
            p1_px = self._grid_to_pixel(pos1)
            p2_px = self._grid_to_pixel(pos2)
            pygame.draw.line(self.screen, color, p1_px, p2_px, 6)

        # Draw connection preview
        if self.selected_node:
            start_px = self._grid_to_pixel(self.selected_node)
            end_px = self._grid_to_pixel(self.cursor_pos)
            is_valid_target = self._is_valid_connection(self.selected_node, tuple(self.cursor_pos))[0]
            color = self.color_valid_preview if is_valid_target else self.color_invalid
            pygame.draw.line(self.screen, color, start_px, end_px, 2)
            pygame.gfxdraw.filled_circle(self.screen, end_px[0], end_px[1], 4, color)


        # Draw nodes and selection highlight
        node_radius = self.cell_size // 2 - 6
        for pos, color_idx in self.nodes.items():
            px_pos = self._grid_to_pixel(pos)
            color = self.node_colors[color_idx]
            
            # Pulsing highlight for selected node
            if self.selected_node == pos:
                pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
                highlight_radius = int(node_radius + 4 + pulse * 3)
                highlight_color = tuple(min(255, c + 50) for c in color)
                pygame.gfxdraw.filled_circle(self.screen, px_pos[0], px_pos[1], highlight_radius, highlight_color)
                pygame.gfxdraw.aacircle(self.screen, px_pos[0], px_pos[1], highlight_radius, highlight_color)

            pygame.gfxdraw.filled_circle(self.screen, px_pos[0], px_pos[1], node_radius, color)
            pygame.gfxdraw.aacircle(self.screen, px_pos[0], px_pos[1], node_radius, color)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.cell_size,
            self.grid_offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.color_cursor, cursor_rect, 2, border_radius=4)
        
        # Draw feedback effects
        if self.feedback_effect and self.feedback_effect["timer"] > 0:
            effect = self.feedback_effect
            pos_px = self._grid_to_pixel(effect["pos"])
            alpha = int(255 * (effect["timer"] / 15))
            radius = int(self.cell_size * 0.8 * (1 - (effect["timer"] / 15)))
            
            if effect["type"] == "success":
                color = self.node_colors[self.nodes[effect["pos"]]]
            else: # fail
                color = self.color_invalid
            
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (radius, radius), radius, width=4)
            self.screen.blit(s, (pos_px[0] - radius, pos_px[1] - radius))
            
            effect["timer"] -= 1


    def _render_ui(self):
        # Display score and moves
        self._draw_text(f"MOVES: {self.moves_remaining}", (20, 15), self.font_ui, self.color_text, self.color_text_shadow)
        score_str = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_str, True, self.color_text)
        self._draw_text(score_str, (self.screen_width - score_surf.get_width() - 20, 15), self.font_ui, self.color_text, self.color_text_shadow)
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.node_colors[2] if self.win else self.node_colors[1]
            self._draw_text(msg, (self.screen_width // 2, self.screen_height // 2), self.font_msg, color, self.color_text_shadow, center=True)

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "selected_node": self.selected_node,
            "is_win": self.win if self.game_over else False
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Node Connector")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        space_held = False
        shift_held = False
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Win: {info['is_win']}")
            pygame.time.wait(3000) # Pause for 3 seconds on game over
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()