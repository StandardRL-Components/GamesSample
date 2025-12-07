import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class Node:
    """A simple class to hold node data."""
    def __init__(self, x, y, color_idx, group_id):
        self.x = x
        self.y = y
        self.color_idx = color_idx
        self.group_id = group_id

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a node, "
        "then move to an adjacent, same-colored node and press Space again to connect them."
    )

    game_description = (
        "A minimalist isometric puzzle. Connect all nodes of the same color into a single "
        "network to win. You have a limited number of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.MAX_MOVES = 40
        self.NUM_COLORS = 6
        self.NUM_NODE_PAIRS = 12
        self.NUM_SINGLE_NODES = 8

        # --- Observation and Action Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 60)

        # --- Colors ---
        self.COLOR_BG = pygame.Color("#1a1c2c")
        self.COLOR_GRID = pygame.Color("#282c4a")
        self.COLOR_CURSOR = pygame.Color("#ffffff")
        self.NODE_COLORS = [
            pygame.Color("#ff595e"), pygame.Color("#ffca3a"), pygame.Color("#8ac926"),
            pygame.Color("#1982c4"), pygame.Color("#6a4c93"), pygame.Color("#f08cae")
        ]
        self.COLOR_TEXT = pygame.Color("#e0e0e0")
        self.COLOR_TEXT_SHADOW = pygame.Color("#101010")

        # --- Isometric Projection ---
        self.TILE_WIDTH = 50
        self.TILE_HEIGHT = 25
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Game State Variables ---
        self.nodes = []
        self.grid = []
        self.connections = []
        self.cursor_pos = [0, 0]
        self.selected_node = None
        self.last_space_held = False
        self.steps = 0
        self.moves_made = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.moves_made = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.selected_node = None
        self.connections = []
        self.last_space_held = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        self._generate_board()

        return self._get_observation(), self._get_info()

    def _generate_board(self):
        """Generates a new board, ensuring it has valid starting moves."""
        while True:
            self.nodes = []
            self.grid = [[None for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            empty_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
            self.np_random.shuffle(empty_cells)
            
            group_id_counter = 0

            # 1. Place guaranteed pairs to ensure solvability
            for _ in range(self.NUM_NODE_PAIRS):
                if len(empty_cells) < 2: break
                
                color_idx = self.np_random.integers(0, self.NUM_COLORS)
                
                # Pick a starting cell
                x1, y1 = empty_cells.pop()
                
                # Find an adjacent empty cell
                adj_cells = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    x2, y2 = x1 + dx, y1 + dy
                    if (x2, y2) in empty_cells:
                        adj_cells.append((x2, y2))
                
                if adj_cells:
                    x2, y2 = adj_cells[self.np_random.integers(len(adj_cells))]
                    empty_cells.remove((x2, y2))

                    node1 = Node(x1, y1, color_idx, group_id_counter)
                    node2 = Node(x2, y2, color_idx, group_id_counter + 1)
                    self.nodes.extend([node1, node2])
                    self.grid[y1][x1] = node1
                    self.grid[y2][x2] = node2
                    group_id_counter += 2

            # 2. Place single nodes for complexity
            for _ in range(self.NUM_SINGLE_NODES):
                if not empty_cells: break
                x, y = empty_cells.pop()
                color_idx = self.np_random.integers(0, self.NUM_COLORS)
                node = Node(x, y, color_idx, group_id_counter)
                self.nodes.append(node)
                self.grid[y][x] = node
                group_id_counter += 1

            if self._count_valid_moves() >= 5:
                break

    def _count_valid_moves(self):
        """Counts the number of possible connections on the board."""
        count = 0
        for node1 in self.nodes:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = node1.x + dx, node1.y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    node2 = self.grid[ny][nx]
                    if node2 and node1.color_idx == node2.color_idx and node1.group_id != node2.group_id:
                        count += 1
        return count // 2 # Each pair is counted twice

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Movement ---
        self._handle_movement(movement)
        
        # --- Handle Action (Spacebar Press) ---
        is_press = space_held and not self.last_space_held
        if is_press:
            self.moves_made += 1
            reward += self._attempt_connection()
        self.last_space_held = space_held

        # --- Check Termination Conditions ---
        terminated = self._update_termination_status()
        if terminated:
            if self._check_win_condition():
                reward += 100.0
                self.win_message = "COMPLETE"
            else:
                reward += -10.0
                if self.moves_made >= self.MAX_MOVES:
                    self.win_message = "OUT OF MOVES"
                else:
                    self.win_message = "NO MOVES LEFT"
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        """Updates cursor position based on movement action."""
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

    def _attempt_connection(self):
        """Handles the logic of selecting and connecting nodes."""
        target_node = self.grid[self.cursor_pos[1]][self.cursor_pos[0]]
        
        if not self.selected_node:
            if target_node:
                self.selected_node = target_node
                return 0.0
            else:
                return -0.1
        
        source_node = self.selected_node
        self.selected_node = None

        if not target_node:
            return -0.2

        is_adjacent = abs(source_node.x - target_node.x) + abs(source_node.y - target_node.y) == 1
        is_same_color = source_node.color_idx == target_node.color_idx
        is_different_group = source_node.group_id != target_node.group_id

        if is_adjacent and is_same_color and is_different_group:
            reward = 1.0
            
            old_group_id = target_node.group_id
            group1_size = sum(1 for n in self.nodes if n.group_id == source_node.group_id)
            group2_size = sum(1 for n in self.nodes if n.group_id == old_group_id)
            if group1_size > 1 and group2_size > 1:
                reward += 5.0

            for node in self.nodes:
                if node.group_id == old_group_id:
                    node.group_id = source_node.group_id
            
            self.connections.append((source_node, target_node))
            return reward
        else:
            return -0.2

    def _update_termination_status(self):
        """Checks and sets the game_over flag."""
        if self._check_win_condition():
            self.game_over = True
        elif self.moves_made >= self.MAX_MOVES:
            self.game_over = True
        elif self._count_valid_moves() == 0 and not self._check_win_condition():
            self.game_over = True
        return self.game_over

    def _check_win_condition(self):
        """Returns True if all nodes of each color are fully connected."""
        if not self.nodes: return True
        
        color_groups = {}
        for node in self.nodes:
            if node.color_idx not in color_groups:
                color_groups[node.color_idx] = set()
            color_groups[node.color_idx].add(node.group_id)
            
        return all(len(groups) == 1 for groups in color_groups.values())

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_made": self.moves_made,
            "moves_left": self.MAX_MOVES - self.moves_made,
            "steps": self.steps,
        }

    def _iso_to_screen(self, x, y):
        """Converts grid coordinates to screen coordinates."""
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        """Renders all game elements (grid, connections, nodes, cursor)."""
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
            
        for n1, n2 in self.connections:
            p1 = self._iso_to_screen(n1.x, n1.y)
            p2 = self._iso_to_screen(n2.x, n2.y)
            color = self.NODE_COLORS[n1.color_idx]
            pygame.draw.line(self.screen, color, p1, p2, 4)

        pulse = (math.sin(self.steps * 0.2) + 1) / 2.0
        for node in self.nodes:
            pos = self._iso_to_screen(node.x, node.y)
            color = self.NODE_COLORS[node.color_idx]
            radius = 10
            
            is_selected = self.selected_node and self.selected_node.group_id == node.group_id
            is_valid_target = False
            if self.selected_node and not is_selected:
                is_adj = abs(node.x - self.selected_node.x) + abs(node.y - self.selected_node.y) == 1
                if is_adj and node.color_idx == self.selected_node.color_idx:
                    is_valid_target = True

            if is_selected:
                p_radius = int(radius + pulse * 4)
                p_color = color.lerp(pygame.Color("white"), 0.3 + pulse * 0.3)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p_radius, p_color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p_radius, p_color)
            elif is_valid_target:
                p_radius = int(radius + 2)
                p_color = color.lerp(pygame.Color("white"), 0.5)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p_radius, p_color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p_radius, p_color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        cursor_screen_pos = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        pulse_alpha = 100 + pulse * 100
        # FIX: Construct a 4-element RGBA tuple instead of a 5-element one.
        cursor_color = (self.COLOR_CURSOR.r, self.COLOR_CURSOR.g, self.COLOR_CURSOR.b, int(pulse_alpha))
        
        w, h = self.TILE_WIDTH, self.TILE_HEIGHT
        points = [
            (cursor_screen_pos[0], cursor_screen_pos[1] - h // 2),
            (cursor_screen_pos[0] + w // 2, cursor_screen_pos[1]),
            (cursor_screen_pos[0], cursor_screen_pos[1] + h // 2),
            (cursor_screen_pos[0] - w // 2, cursor_screen_pos[1]),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, cursor_color)
        
    def _render_ui(self):
        """Renders score, moves, and game over text."""
        score_text = f"SCORE: {int(self.score)}"
        moves_text = f"MOVES: {self.MAX_MOVES - self.moves_made}"

        self._draw_text(score_text, (10, 10), self.font_main)
        self._draw_text(moves_text, (self.SCREEN_WIDTH - 10, 10), self.font_main, align="right")
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.win_message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), self.font_large, align="center")

    def _draw_text(self, text, pos, font, color=None, shadow=True, align="left"):
        """Helper to draw text with a shadow."""
        if color is None: color = self.COLOR_TEXT
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        
        if align == "right": text_rect.topright = pos
        elif align == "center": text_rect.center = pos
        else: text_rect.topleft = pos
        
        if shadow:
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(topleft=(text_rect.left + 2, text_rect.top + 2))
            if align == "right": shadow_rect.topright = (pos[0] + 2, pos[1] + 2)
            elif align == "center": shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surf, shadow_rect)
            
        self.screen.blit(text_surf, text_rect)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you might need to unset the headless environment variable
    # and install pygame with display support.
    # For example:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Re-enable display for manual testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Node Connector")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # none
        space = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, 0] # shift is unused
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Keep showing the final screen for a moment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for manual play

    pygame.quit()