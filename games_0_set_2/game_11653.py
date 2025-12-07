import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:22:22.836662
# Source Brief: brief_01653.md
# Brief Index: 1653
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Weave a tapestry of dreams by connecting mystical nodes. Select word pairs to gather platform "
        "pieces and place them on the grid to complete the network."
    )
    user_guide = (
        "Controls: Use ↑↓ to select word pairs, space to claim a piece. "
        "Use ←→↑↓ to move the piece, shift to rotate, and space to place it on the grid."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20
    GRID_WIDTH, GRID_HEIGHT = GRID_COLS * CELL_SIZE, GRID_ROWS * CELL_SIZE
    MAX_STEPS = 1000

    # Colors
    COLOR_BG_START = (10, 5, 25)
    COLOR_BG_END = (30, 10, 60)
    COLOR_GRID = (40, 20, 80, 100)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TEXT_DIM = (150, 150, 180)
    COLOR_TEXT_HIGHLIGHT = (255, 255, 100)
    COLOR_CONN_POINT = (255, 80, 80)
    COLOR_CONN_POINT_GLOW = (255, 80, 80, 50)
    COLOR_GOLD = (255, 215, 0)
    COLOR_GOLD_GLOW = (255, 215, 0, 70)
    COLOR_CURSOR_VALID = (100, 255, 100)
    COLOR_CURSOR_INVALID = (255, 100, 100)

    PLATFORM_COLORS = [
        (0, 150, 255),  # Blue
        (150, 0, 255),  # Purple
        (0, 255, 150),  # Cyan
        (255, 150, 0),  # Orange
    ]

    # --- Platform Definitions ---
    PLATFORM_TYPES = {
        "I-Shape": {"shape": [(0, 0), (1, 0), (2, 0)], "unlock_score": 0},
        "L-Shape": {"shape": [(0, 0), (0, 1), (1, 1)], "unlock_score": 10},
        "T-Shape": {"shape": [(0, 0), (1, 0), (2, 0), (1, 1)], "unlock_score": 25},
        "Plus-Shape": {"shape": [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], "unlock_score": 50},
    }
    
    WORDS = ["AURA", "ECHO", "FADE", "VEIL", "MIST", "GLOW", "SONG", "DUSK", "DAWN", "VOID", "STAR", "FLUX"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self._create_background()

        # Persistent state across episodes
        self.total_connections_made = 0
        self.unlocked_platforms = self._get_unlocked_platforms()
        self.current_level = 0
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.word_pairs = []
        self.selected_word_idx = 0
        
        self.platform_inventory = {}
        self.held_platform_type = None
        self.held_platform_rotation = 0
        
        self.game_phase = "SELECT"  # "SELECT" or "PLACE"
        self.placement_cursor = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.connection_points = []
        self.placed_platforms = []
        self.grid = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=int)
        
        self.particles = []
        
        self.previous_space_held = 0
        self.previous_shift_held = 0

        self.union_find_parent = {}
        self.num_active_connections = 0

    def _create_background(self):
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_START[0] * (1 - interp) + self.COLOR_BG_END[0] * interp,
                self.COLOR_BG_START[1] * (1 - interp) + self.COLOR_BG_END[1] * interp,
                self.COLOR_BG_START[2] * (1 - interp) + self.COLOR_BG_END[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        
        self._generate_tapestry()
        
        self.placed_platforms = []
        self.grid.fill(0)
        self.particles = []

        self.platform_inventory = {p_type: 5 for p_type in self.unlocked_platforms}
        self.word_pairs = self._generate_word_pairs()
        
        self.game_phase = "SELECT"
        self.selected_word_idx = 0
        self.held_platform_type = self.unlocked_platforms[0] if self.unlocked_platforms else None
        self.held_platform_rotation = 0
        self.placement_cursor = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.previous_space_held = 0
        self.previous_shift_held = 0
        
        self._rebuild_connections()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = -0.01 # Small penalty for taking a step

        movement, space_held, shift_held = action[0], action[1], action[2]
        space_press = space_held and not self.previous_space_held
        shift_press = shift_held and not self.previous_shift_held

        self._handle_input(movement, space_press, shift_press)
        
        self._update_particles()

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        terminated = self._check_termination()
        
        # Final reward adjustment on termination
        if terminated:
            if self._is_victory():
                self.reward_this_step += 50
                self.current_level += 1 # Progress to next level on next reset
                # sound effect: victory fanfare
            else: # Max steps reached
                self.reward_this_step -= 10
                # sound effect: failure tone

        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_press, shift_press):
        if self.game_phase == "SELECT":
            if movement == 1: # Up
                self.selected_word_idx = max(0, self.selected_word_idx - 1)
            elif movement == 2: # Down
                self.selected_word_idx = min(len(self.word_pairs) - 1, self.selected_word_idx + 1)
            
            if shift_press:
                self._cycle_held_platform()
                self.game_phase = "PLACE"
                # sound effect: UI cycle
            
            if space_press and self.word_pairs:
                selected_pair = self.word_pairs.pop(self.selected_word_idx)
                p_type = selected_pair["platform"]
                self.platform_inventory[p_type] += 1
                self.reward_this_step += 0.1 # Small reward for gathering a resource
                self.selected_word_idx = min(self.selected_word_idx, len(self.word_pairs) - 1) if self.word_pairs else 0
                if len(self.word_pairs) < 4:
                    self.word_pairs.extend(self._generate_word_pairs(1))
                self.held_platform_type = p_type
                self.game_phase = "PLACE"
                # sound effect: word match success
        
        elif self.game_phase == "PLACE":
            if movement == 1: self.placement_cursor[1] = max(0, self.placement_cursor[1] - 1)
            elif movement == 2: self.placement_cursor[1] = min(self.GRID_ROWS - 1, self.placement_cursor[1] + 1)
            elif movement == 3: self.placement_cursor[0] = max(0, self.placement_cursor[0] - 1)
            elif movement == 4: self.placement_cursor[0] = min(self.GRID_COLS - 1, self.placement_cursor[0] + 1)
            
            if shift_press:
                self.held_platform_rotation = (self.held_platform_rotation + 1) % 4
                # sound effect: rotate piece
            
            if space_press:
                if self._try_place_platform():
                    self.game_phase = "SELECT"
                    # sound effect: place piece success
                else:
                    self.reward_this_step -= 0.5 # Penalty for invalid action
                    # sound effect: invalid action buzz

    def _cycle_held_platform(self):
        if not self.platform_inventory:
            return
        
        available_types = [p_type for p_type, count in self.platform_inventory.items() if count > 0]
        if not available_types:
            self.held_platform_type = None
            return

        if self.held_platform_type in available_types:
            current_idx = available_types.index(self.held_platform_type)
            next_idx = (current_idx + 1) % len(available_types)
            self.held_platform_type = available_types[next_idx]
        else:
            self.held_platform_type = available_types[0]
        self.held_platform_rotation = 0

    def _get_platform_cells(self, p_type, rotation, pos):
        base_shape = self.PLATFORM_TYPES[p_type]["shape"]
        cells = []
        for x, y in base_shape:
            # Rotate
            if rotation == 0:   rx, ry = x, y
            elif rotation == 1: rx, ry = -y, x
            elif rotation == 2: rx, ry = -x, -y
            else:               rx, ry = y, -x
            cells.append((pos[0] + rx, pos[1] + ry))
        return cells
    
    def _is_placement_valid(self, cells):
        for x, y in cells:
            if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
                return False # Out of bounds
            if self.grid[x, y] != 0:
                return False # Overlap
        return True

    def _try_place_platform(self):
        if self.held_platform_type is None or self.platform_inventory.get(self.held_platform_type, 0) <= 0:
            return False

        cells = self._get_platform_cells(self.held_platform_type, self.held_platform_rotation, self.placement_cursor)
        
        if not self._is_placement_valid(cells):
            return False

        platform_id = len(self.placed_platforms) + 1 # Use 1-based index for grid
        platform = {
            "id": platform_id,
            "type": self.held_platform_type,
            "pos": tuple(self.placement_cursor),
            "rotation": self.held_platform_rotation,
            "cells": cells
        }
        self.placed_platforms.append(platform)
        
        for x, y in cells:
            self.grid[x, y] = platform_id
            self._create_particles((x * self.CELL_SIZE, y * self.CELL_SIZE), self.PLATFORM_COLORS[list(self.PLATFORM_TYPES).index(self.held_platform_type)], 10)

        self.platform_inventory[self.held_platform_type] -= 1
        
        # Check for new connections and unlocks
        prev_connections = self.num_active_connections
        self._rebuild_connections()
        new_connections = self.num_active_connections - prev_connections
        if new_connections > 0:
            self.reward_this_step += new_connections * 1.0
            self.total_connections_made += new_connections
            
            # Check for unlocks
            prev_unlocked_count = len(self.unlocked_platforms)
            self.unlocked_platforms = self._get_unlocked_platforms()
            if len(self.unlocked_platforms) > prev_unlocked_count:
                self.reward_this_step += 5.0
                # sound effect: unlock new item
                # Add new platform types to inventory
                for p_type in self.unlocked_platforms:
                    if p_type not in self.platform_inventory:
                        self.platform_inventory[p_type] = 5

        if self.platform_inventory.get(self.held_platform_type, 0) == 0:
            self._cycle_held_platform()

        return True

    def _get_unlocked_platforms(self):
        return [p_type for p_type, data in self.PLATFORM_TYPES.items() if self.total_connections_made >= data["unlock_score"]]

    def _generate_tapestry(self):
        self.connection_points = []
        num_points = min(8, 4 + self.current_level)
        
        attempts = 0
        while len(self.connection_points) < num_points and attempts < 100:
            x = self.np_random.integers(1, self.GRID_COLS - 1)
            y = self.np_random.integers(1, self.GRID_ROWS - 1)
            
            # Ensure points are not too close to each other
            is_valid = True
            for px, py in self.connection_points:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                if dist < 4:
                    is_valid = False
                    break
            if is_valid:
                self.connection_points.append((x, y))
            attempts += 1
    
    def _generate_word_pairs(self, count=4):
        pairs = []
        available_platforms = self.unlocked_platforms if self.unlocked_platforms else list(self.PLATFORM_TYPES.keys())
        for _ in range(count):
            word1, word2 = self.np_random.choice(self.WORDS, 2, replace=False)
            p_type = self.np_random.choice(available_platforms)
            pairs.append({"words": (word1, word2), "platform": p_type})
        return pairs

    # --- Union-Find for Connection Checking ---
    def _uf_find(self, i):
        if self.union_find_parent[i] == i:
            return i
        self.union_find_parent[i] = self._uf_find(self.union_find_parent[i])
        return self.union_find_parent[i]

    def _uf_union(self, i, j):
        root_i = self._uf_find(i)
        root_j = self._uf_find(j)
        if root_i != root_j:
            self.union_find_parent[root_j] = root_i

    def _rebuild_connections(self):
        self.union_find_parent = {f"p{i}": f"p{i}" for i in range(len(self.connection_points))}
        for platform in self.placed_platforms:
            self.union_find_parent[platform["id"]] = platform["id"]

        # Connect platforms to adjacent platforms
        for i, p1 in enumerate(self.placed_platforms):
            for p2 in self.placed_platforms[i+1:]:
                for c1x, c1y in p1["cells"]:
                    for c2x, c2y in p2["cells"]:
                        if abs(c1x - c2x) + abs(c1y - c2y) == 1:
                            self._uf_union(p1["id"], p2["id"])

        # Connect connection points to adjacent platforms
        for i, (cpx, cpy) in enumerate(self.connection_points):
            for p in self.placed_platforms:
                for pcx, pcy in p["cells"]:
                    if abs(cpx - pcx) + abs(cpy - pcy) == 1:
                        self._uf_union(f"p{i}", p["id"])
        
        # Count active connections
        connected_pairs = set()
        for i in range(len(self.connection_points)):
            for j in range(i + 1, len(self.connection_points)):
                if self._uf_find(f"p{i}") == self._uf_find(f"p{j}"):
                    connected_pairs.add(tuple(sorted((i, j))))
        self.num_active_connections = len(connected_pairs)

    def _is_victory(self):
        if len(self.connection_points) < 2:
            return True
        root = self._uf_find("p0")
        for i in range(1, len(self.connection_points)):
            if self._uf_find(f"p{i}") != root:
                return False
        return True

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self._is_victory():
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.blit(self.bg_surface, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level, "connections": self.num_active_connections}

    # --- Rendering ---
    def _render_game(self):
        self._draw_placed_platforms()
        self._draw_connections()
        self._draw_connection_points()
        if self.game_phase == "PLACE":
            self._draw_placement_cursor()
        self._draw_particles()

    def _draw_connection_points(self):
        for x, y in self.connection_points:
            px, py = x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2
            # Glow
            for i in range(5, 0, -1):
                pygame.gfxdraw.aacircle(self.screen, px, py, 4 + i, self.COLOR_CONN_POINT_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, 4, self.COLOR_CONN_POINT)
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, self.COLOR_CONN_POINT)

    def _draw_placed_platforms(self):
        for platform in self.placed_platforms:
            p_type_idx = list(self.PLATFORM_TYPES).index(platform["type"])
            color = self.PLATFORM_COLORS[p_type_idx % len(self.PLATFORM_COLORS)]
            for x, y in platform["cells"]:
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), rect, 1, border_radius=3)

    def _draw_connections(self):
        if len(self.connection_points) < 2: return
        
        for i in range(len(self.connection_points)):
            for j in range(i + 1, len(self.connection_points)):
                if self._uf_find(f"p{i}") == self._uf_find(f"p{j}"):
                    p1 = self.connection_points[i]
                    p2 = self.connection_points[j]
                    px1 = p1[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                    py1 = p1[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    px2 = p2[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                    py2 = p2[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Glow
                    for k in range(1, 6):
                        pygame.draw.line(self.screen, self.COLOR_GOLD_GLOW, (px1, py1), (px2, py2), k*2)
                    pygame.draw.aaline(self.screen, self.COLOR_GOLD, (px1, py1), (px2, py2))

    def _draw_placement_cursor(self):
        if self.held_platform_type is None: return

        cells = self._get_platform_cells(self.held_platform_type, self.held_platform_rotation, self.placement_cursor)
        is_valid = self._is_placement_valid(cells)
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        for x, y in cells:
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(surf, color + (100,), surf.get_rect(), border_radius=3)
            pygame.draw.rect(surf, color + (200,), surf.get_rect(), 2, border_radius=3)
            self.screen.blit(surf, rect.topleft)

    def _render_ui(self):
        # Right-side panel for words
        panel_width = 160
        panel_x = self.SCREEN_WIDTH - panel_width
        pygame.draw.rect(self.screen, (0,0,0,100), (panel_x, 0, panel_width, self.SCREEN_HEIGHT))
        
        # Title
        title_surf = self.font_main.render("Word Pairs", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (panel_x + (panel_width - title_surf.get_width()) // 2, 10))

        # Word Pairs
        for i, pair_data in enumerate(self.word_pairs):
            y_pos = 50 + i * 40
            is_selected = (i == self.selected_word_idx and self.game_phase == "SELECT")
            color = self.COLOR_TEXT_HIGHLIGHT if is_selected else self.COLOR_TEXT
            
            p_type_idx = list(self.PLATFORM_TYPES).index(pair_data["platform"])
            p_color = self.PLATFORM_COLORS[p_type_idx % len(self.PLATFORM_COLORS)]
            pygame.draw.rect(self.screen, p_color, (panel_x + 5, y_pos, 5, 30), border_radius=2)

            word_text = f"{pair_data['words'][0]} / {pair_data['words'][1]}"
            word_surf = self.font_small.render(word_text, True, color)
            self.screen.blit(word_surf, (panel_x + 15, y_pos + 8))
        
        # Bottom panel for inventory
        inv_height = 60
        pygame.draw.rect(self.screen, (0,0,0,100), (0, self.SCREEN_HEIGHT - inv_height, panel_x, inv_height))
        
        # Platform Inventory
        for i, p_type in enumerate(self.unlocked_platforms):
            x_pos = 10 + i * 80
            p_type_idx = list(self.PLATFORM_TYPES).index(p_type)
            color = self.PLATFORM_COLORS[p_type_idx % len(self.PLATFORM_COLORS)]
            is_held = (p_type == self.held_platform_type and self.game_phase == "PLACE")
            
            # Draw icon
            icon_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            shape_cells = self.PLATFORM_TYPES[p_type]["shape"]
            min_x = min(c[0] for c in shape_cells)
            min_y = min(c[1] for c in shape_cells)
            for cx, cy in shape_cells:
                pygame.draw.rect(icon_surf, color, ((cx-min_x)*8+4, (cy-min_y)*8+4, 8, 8), border_radius=2)
            
            if is_held:
                pygame.draw.rect(self.screen, self.COLOR_TEXT_HIGHLIGHT, (x_pos-2, self.SCREEN_HEIGHT - inv_height + 8, 44, 44), 2, border_radius=4)
            
            self.screen.blit(icon_surf, (x_pos, self.SCREEN_HEIGHT - inv_height + 10))
            
            # Draw count
            count = self.platform_inventory.get(p_type, 0)
            count_surf = self.font_main.render(f"x{count}", True, self.COLOR_TEXT)
            self.screen.blit(count_surf, (x_pos + 45, self.SCREEN_HEIGHT - inv_height + 18))

        # Top-left Info
        score_surf = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        steps_surf = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (10, 35))

    # --- Particles ---
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append([list(pos), vel, lifespan, color])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
        self.particles = [p for p in self.particles if p[2] > 0]

    def _draw_particles(self):
        for pos, vel, life, color in self.particles:
            alpha = max(0, min(255, int(255 * (life / 30.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(life/10 + 1), color + (alpha,))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    manual_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dream Weavers")
    clock = pygame.time.Clock()
    
    total_reward = 0.0

    # Remove the validation call from the interactive test
    # env.validate_implementation() 

    while running:
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a Surface for display
        # The observation is (H, W, C), but pygame needs (W, H)
        # and surfarray.make_surface expects (W, H, C)
        display_obs = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(display_obs)
        manual_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
            total_reward = 0.0
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS for smooth manual play

    env.close()