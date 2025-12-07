import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:43:45.180343
# Source Brief: brief_00591.md
# Brief Index: 591
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A strategic puzzle game where you place dominoes on a grid. The goal is to connect all dominoes into a single chain before running out of turns."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press 'shift' to rotate the domino and 'space' to place it on the grid."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 7
    MAX_TURNS = 20
    MAX_DOMINOES = 7
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (25, 35, 55)
    COLOR_GRID = (50, 65, 90)
    COLOR_TEXT = (220, 230, 255)
    COLOR_TEXT_WARN = (255, 100, 100)
    COLOR_VALID_CURSOR = (100, 255, 100, 100)
    COLOR_INVALID_CURSOR = (255, 100, 100, 100)
    CHAIN_COLOR_START = pygame.Color(100, 180, 255)
    CHAIN_COLOR_END = pygame.Color(255, 255, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.dominoes_placed = None
        self.dominoes_to_place = None
        self.cursor_pos = None
        self.cursor_orientation = None # 0 for horizontal, 1 for vertical
        self.current_turn = None
        self.score = None
        self.game_over = None
        self.win_status = None
        self.steps = None
        self.last_shift_held = None
        self.last_space_held = None
        self.chain_map = None
        self.particles = None
        
        self.reset()
        # self.validate_implementation() # Removed for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Note: Pygame does not have a global RNG, so this seed won't affect `random`
            # For full determinism, you would seed the `random` module itself.
            pass

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.dominoes_placed = [] # List of tuples: (x, y, orientation)
        self.dominoes_to_place = self.MAX_DOMINOES
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.cursor_orientation = 0
        self.current_turn = 1
        self.score = 0
        self.game_over = False
        self.win_status = "IN_PROGRESS" # "WIN", "LOSE_TURNS", "LOSE_DISCONNECTED"
        self.steps = 0
        self.last_shift_held = False
        self.last_space_held = False
        self.chain_map = {} # Maps domino index to its chain ID
        self.particles = []

        # --- Grid rendering setup ---
        self.grid_area_width = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_area_width / self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_area_width) / 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_width) / 2

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Unpack Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        # Toggle orientation on shift press (rising edge)
        if shift_held and not self.last_shift_held:
            self.cursor_orientation = 1 - self.cursor_orientation
            # sfx: UI_TOGGLE

        # Move cursor
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        
        # Place domino on space press (rising edge)
        if space_held and not self.last_space_held:
            reward = self._attempt_place_domino()

        self.last_shift_held = shift_held
        self.last_space_held = space_held

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.dominoes_to_place == 0:
            terminated = True
            self._update_chains()
            if len(set(self.chain_map.values())) == 1:
                self.win_status = "WIN"
                reward += 50
                # sfx: WIN_JINGLE
            else:
                self.win_status = "LOSE_DISCONNECTED"
                reward -= 50
                # sfx: LOSE_SOUND

        elif self.current_turn > self.MAX_TURNS:
            terminated = True
            self.win_status = "LOSE_TURNS"
            reward -= 50
            # sfx: LOSE_SOUND
        
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Episode truncated by step limit
            self.win_status = "LOSE_TURNS" # Treat as a loss
            reward -= 50


        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _attempt_place_domino(self):
        x, y = self.cursor_pos
        is_valid, cells = self._is_valid_placement(x, y, self.cursor_orientation)

        if not is_valid:
            # sfx: INVALID_PLACEMENT
            return 0 # No reward for invalid move

        # --- Valid Placement Logic ---
        # sfx: PLACE_DOMINO_SUCCESS
        self.dominoes_placed.append((x, y, self.cursor_orientation))
        for cx, cy in cells:
            self.grid[cy, cx] = 1
        
        self.dominoes_to_place -= 1
        self.current_turn += 1

        # --- Calculate Reward ---
        reward = 0
        # Placement bonus
        reward += 10 if self.cursor_orientation == 1 else 5 # Vertical vs Horizontal
        
        # Chain bonus
        prev_chains = len(set(self.chain_map.values())) if self.chain_map else 0
        self._update_chains()
        new_chains = len(set(self.chain_map.values())) if self.chain_map else 0
        
        # Reward for connecting separate chains
        if prev_chains > 0:
            reward += (prev_chains - new_chains) * 5

        # Reward for chain length
        new_domino_idx = len(self.dominoes_placed) - 1
        if new_domino_idx in self.chain_map:
            chain_id = self.chain_map[new_domino_idx]
            chain_size = list(self.chain_map.values()).count(chain_id)
            reward += chain_size

        self.score += reward

        # --- Visual Effects ---
        self._create_particles(cells)

        return reward

    def _is_valid_placement(self, x, y, orientation):
        c1 = (x, y)
        c2 = (x + 1, y) if orientation == 0 else (x, y + 1)
        cells = [c1, c2]

        # Check bounds
        for cx, cy in cells:
            if not (0 <= cx < self.GRID_SIZE and 0 <= cy < self.GRID_SIZE):
                return False, []
        
        # Check overlap
        for cx, cy in cells:
            if self.grid[cy, cx] == 1:
                return False, []

        # Check connectivity (not required for the first domino)
        if len(self.dominoes_placed) > 0:
            is_connected = False
            for cx, cy in cells:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny, nx] == 1:
                        is_connected = True
                        break
                if is_connected: break
            if not is_connected:
                return False, []
        
        return True, cells

    def _update_chains(self):
        num_dominoes = len(self.dominoes_placed)
        if num_dominoes == 0:
            self.chain_map = {}
            return

        adj = [[] for _ in range(num_dominoes)]
        for i in range(num_dominoes):
            for j in range(i + 1, num_dominoes):
                if self._are_dominoes_adjacent(i, j):
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * num_dominoes
        chain_id_counter = 0
        self.chain_map = {}
        for i in range(num_dominoes):
            if not visited[i]:
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
                    self.chain_map[u] = chain_id_counter
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                chain_id_counter += 1

    def _get_domino_cells(self, domino_index):
        x, y, orientation = self.dominoes_placed[domino_index]
        return [(x, y), (x + 1, y) if orientation == 0 else (x, y + 1)]

    def _are_dominoes_adjacent(self, idx1, idx2):
        cells1 = self._get_domino_cells(idx1)
        cells2 = self._get_domino_cells(idx2)
        for c1x, c1y in cells1:
            for c2x, c2y in cells2:
                if abs(c1x - c2x) + abs(c1y - c2y) == 1:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            start_h = (self.grid_offset_x, self.grid_offset_y + i * self.cell_size)
            end_h = (self.grid_offset_x + self.grid_area_width, self.grid_offset_y + i * self.cell_size)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_h, end_h)
            start_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y)
            end_v = (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_area_width)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start_v, end_v)

        # Draw placed dominoes
        chain_sizes = {}
        if self.chain_map:
            for chain_id in set(self.chain_map.values()):
                chain_sizes[chain_id] = list(self.chain_map.values()).count(chain_id)

        for i, (x, y, orientation) in enumerate(self.dominoes_placed):
            color_ratio = 0
            if i in self.chain_map:
                chain_id = self.chain_map[i]
                size = chain_sizes.get(chain_id, 1)
                color_ratio = min(1.0, (size - 1) / (self.MAX_DOMINOES - 1))
            
            color = self.CHAIN_COLOR_START.lerp(self.CHAIN_COLOR_END, color_ratio)
            
            px = self.grid_offset_x + x * self.cell_size
            py = self.grid_offset_y + y * self.cell_size
            width = self.cell_size * 2 if orientation == 0 else self.cell_size
            height = self.cell_size if orientation == 0 else self.cell_size * 2
            
            rect = pygame.Rect(px, py, width, height)
            self._draw_rounded_rect(self.screen, rect.inflate(-8, -8), color, 0.3)

        # Draw cursor
        if not self.game_over:
            is_valid, _ = self._is_valid_placement(self.cursor_pos[0], self.cursor_pos[1], self.cursor_orientation)
            cursor_color = self.COLOR_VALID_CURSOR if is_valid else self.COLOR_INVALID_CURSOR
            px = self.grid_offset_x + self.cursor_pos[0] * self.cell_size
            py = self.grid_offset_y + self.cursor_pos[1] * self.cell_size
            width = self.cell_size * 2 if self.cursor_orientation == 0 else self.cell_size
            height = self.cell_size if self.cursor_orientation == 0 else self.cell_size * 2
            
            rect = pygame.Rect(px, py, width, height)
            self._draw_rounded_rect(self.screen, rect.inflate(-6, -6), cursor_color, 0.3, width=3)

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Turns
        turn_color = self.COLOR_TEXT if self.current_turn <= self.MAX_TURNS - 5 else self.COLOR_TEXT_WARN
        turn_surf = self.font_main.render(f"TURN: {self.current_turn}/{self.MAX_TURNS}", True, turn_color)
        turn_rect = turn_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(turn_surf, turn_rect)

        # Dominoes left
        domino_surf = self.font_small.render(f"DOMINOES LEFT: {self.dominoes_to_place}", True, self.COLOR_TEXT)
        domino_rect = domino_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 40))
        self.screen.blit(domino_surf, domino_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_status == "WIN":
                msg = "YOU WIN!"
                color = self.CHAIN_COLOR_END
            elif self.win_status == "LOSE_TURNS":
                msg = "OUT OF TURNS"
                color = self.COLOR_TEXT_WARN
            elif self.win_status == "LOSE_DISCONNECTED":
                msg = "NOT CONNECTED"
                color = self.COLOR_TEXT_WARN
            else:
                msg = "GAME OVER"
                color = self.COLOR_TEXT
                
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_rounded_rect(self, surface, rect, color, radius_ratio, width=0):
        r = int(min(rect.width, rect.height) * radius_ratio)
        if r * 2 > min(rect.width, rect.height):
             r = int(min(rect.width, rect.height) / 2)

        if width == 0: # Filled
            pygame.gfxdraw.aacircle(surface, rect.left + r, rect.top + r, r, color)
            pygame.gfxdraw.aacircle(surface, rect.right - r - 1, rect.top + r, r, color)
            pygame.gfxdraw.aacircle(surface, rect.left + r, rect.bottom - r - 1, r, color)
            pygame.gfxdraw.aacircle(surface, rect.right - r - 1, rect.bottom - r - 1, r, color)
            pygame.gfxdraw.filled_circle(surface, rect.left + r, rect.top + r, r, color)
            pygame.gfxdraw.filled_circle(surface, rect.right - r - 1, rect.top + r, r, color)
            pygame.gfxdraw.filled_circle(surface, rect.left + r, rect.bottom - r - 1, r, color)
            pygame.gfxdraw.filled_circle(surface, rect.right - r - 1, rect.bottom - r - 1, r, color)
            pygame.draw.rect(surface, color, (rect.left + r, rect.top, rect.width - 2 * r, rect.height))
            pygame.draw.rect(surface, color, (rect.left, rect.top + r, rect.width, rect.height - 2 * r))
        else: # Outline
            pygame.draw.rect(surface, color, rect, width, border_radius=r)

    def _create_particles(self, cells):
        center_x = self.grid_offset_x + (cells[0][0] + cells[1][0] + 1) * self.cell_size / 2
        center_y = self.grid_offset_y + (cells[0][1] + cells[1][1] + 1) * self.cell_size / 2
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(15, 30) # frames
            color = random.choice([self.CHAIN_COLOR_START, self.CHAIN_COLOR_END, (255,255,255)])
            self.particles.append([center_x, center_y, vx, vy, lifetime, color])

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # lifetime--
            if p[4] > 0:
                active_particles.append(p)
                size = int(max(0, p[4] / 10))
                if size > 0:
                    pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), size)
        self.particles = active_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "current_turn": self.current_turn,
            "dominoes_placed": len(self.dominoes_placed),
            "win_status": self.win_status,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display, so it will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
        import pygame
        
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Domino Strategy")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement = 0 # no-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Status: {info['win_status']}")
                pygame.time.wait(3000) # Pause for 3 seconds
                obs, info = env.reset()
                total_reward = 0

            clock.tick(30) # Run at 30 FPS

        env.close()
    except pygame.error as e:
        print(f"Could not initialize display for manual play: {e}")
        print("This is expected in a headless environment.")