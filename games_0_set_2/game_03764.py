
# Generated: 2025-08-28T00:21:20.186139
# Source Brief: brief_03764.md
# Brief Index: 3764

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move your dino. Press space to dig at your current location."
    )

    game_description = (
        "Guide a digging dinosaur to unearth 10 fossils within 20 moves. "
        "Complete sets of fossils for bonus points!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (10, 10)
        self.MAX_MOVES = 20
        self.WIN_FOSSIL_COUNT = 10

        # --- Color Palette ---
        self.COLOR_BG = (45, 35, 30)
        self.COLOR_GRID = (70, 60, 55)
        self.COLOR_GROUND = (125, 100, 80)
        self.COLOR_HOLE = (30, 20, 15)
        self.COLOR_DINO = (50, 205, 50) # Lime Green
        self.COLOR_DINO_SHADOW = (40, 160, 40)
        self.COLOR_UI_BG = (60, 50, 45, 200)
        self.COLOR_UI_TEXT = (240, 230, 220)
        self.COLOR_MOVES_BAR = (50, 205, 50)
        self.COLOR_MOVES_BAR_BG = (90, 80, 75)
        self.COLOR_GREEN = (100, 255, 100)
        self.COLOR_RED = (255, 100, 100)
        self.COLOR_GOLD = (255, 215, 0)
        
        self.FOSSIL_COLORS = {
            1: (220, 220, 200), # Skull (Bone)
            2: (210, 210, 190), # Ribs (Bone)
            3: (230, 230, 210), # Claw (Bone)
            4: (255, 180, 20)   # Amber
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 64)

        # --- Isometric Grid Setup ---
        self.TILE_WIDTH = 54
        self.TILE_HEIGHT = self.TILE_WIDTH / 2
        self.origin_x = self.WIDTH / 2
        self.origin_y = 80

        # Initialize state variables
        self.dino_pos = (0,0)
        self.grid = np.zeros((0,0))
        self.dug_grid = np.zeros((0,0))
        self.moves_left = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.collected_fossils = {}
        self.sets_completed = {}
        self.particles = []
        self.messages = []
        self.steps = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.dino_pos = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        self.grid = np.zeros(self.GRID_SIZE, dtype=int)
        self.dug_grid = np.zeros(self.GRID_SIZE, dtype=bool)
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.win = False
        self.collected_fossils = {1: 0, 2: 0, 3: 0, 4: 0}
        self.sets_completed = {1: False, 2: False, 3: False}
        self.particles = []
        self.messages = []
        self.steps = 0
        
        self._place_fossils()
        
        return self._get_observation(), self._get_info()

    def _place_fossils(self):
        fossils_to_place = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        locations = [(x, y) for x in range(self.GRID_SIZE[0]) for y in range(self.GRID_SIZE[1])]
        
        if self.dino_pos in locations:
            locations.remove(self.dino_pos)
            
        self.np_random.shuffle(locations)
        
        for i, fossil_type in enumerate(fossils_to_place):
            if i < len(locations):
                self.grid[locations[i]] = fossil_type

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        action_taken = False

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if space_held:
            action_taken = True
            reward += self._perform_dig()
        elif movement != 0:
            action_taken = True
            self._perform_move(movement)

        if action_taken:
            self.moves_left -= 1
        
        self.score += reward

        total_collected = sum(self.collected_fossils.values())
        if total_collected >= self.WIN_FOSSIL_COUNT:
            if not self.game_over: # Grant win bonus only once
                win_bonus = 50
                reward += win_bonus
                self.score += win_bonus
                self._create_message("ALL FOSSILS FOUND!", (self.WIDTH // 2, self.HEIGHT // 2 - 40), self.COLOR_GOLD, 48, 120)
            self.game_over = True
            self.win = True
        elif self.moves_left <= 0:
            if not self.game_over:
                self._create_message("OUT OF MOVES", (self.WIDTH // 2, self.HEIGHT // 2 - 40), self.COLOR_RED, 48, 120)
            self.game_over = True
            self.win = False

        terminated = self.game_over
        
        self._update_particles()
        self._update_messages()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_dig(self):
        ix, iy = self.dino_pos
        
        if self.dug_grid[ix, iy]:
            return -0.1

        # Sound: SFX_DIG
        self.dug_grid[ix, iy] = True
        self._create_dig_particles(ix, iy)
        
        fossil_type = self.grid[ix, iy]
        if fossil_type > 0:
            # Sound: SFX_FOSSIL_FOUND
            self.collected_fossils[fossil_type] += 1
            pos = self._iso_to_screen(ix, iy)
            self._create_message("+1 Fossil", pos, self.COLOR_GREEN, 28, 40)
            
            reward = 1.0
            if fossil_type in [1, 2, 3] and not self.sets_completed[fossil_type]:
                if self.collected_fossils[fossil_type] == 3:
                    # Sound: SFX_SET_COMPLETE
                    self.sets_completed[fossil_type] = True
                    set_bonus = 5.0
                    reward += set_bonus
                    self._create_message("Set Bonus!", (pos[0], pos[1] - 25), self.COLOR_GOLD, 32, 60)
            return reward
        else:
            # Sound: SFX_DIG_EMPTY
            self._create_message("Empty", self._iso_to_screen(ix, iy), self.COLOR_UI_TEXT, 24, 30)
            return -0.2

    def _perform_move(self, movement):
        ox, oy = self.dino_pos
        nx, ny = ox, oy

        if movement == 1: ny -= 1  # Up (iso north-west)
        elif movement == 2: ny += 1  # Down (iso south-east)
        elif movement == 3: nx -= 1  # Left (iso south-west)
        elif movement == 4: nx += 1  # Right (iso north-east)

        nx = max(0, min(self.GRID_SIZE[0] - 1, nx))
        ny = max(0, min(self.GRID_SIZE[1] - 1, ny))

        if (nx, ny) != (ox, oy):
            # Sound: SFX_DINO_MOVE
            self.dino_pos = (nx, ny)
            self._create_move_particles(ox, oy)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_fossils()
        self._render_particles()
        self._render_dino()
        self._render_messages()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "fossils_collected": sum(self.collected_fossils.values()),
        }

    def _iso_to_screen(self, ix, iy):
        screen_x = self.origin_x + (ix - iy) * (self.TILE_WIDTH / 2)
        screen_y = self.origin_y + (ix + iy) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_grid_and_fossils(self):
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                sx, sy = self._iso_to_screen(x, y)
                points = [
                    (sx, sy),
                    (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2),
                    (sx, sy + self.TILE_HEIGHT),
                    (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2)
                ]
                
                is_dug = self.dug_grid[x, y]
                tile_color = self.COLOR_HOLE if is_dug else self.COLOR_GROUND
                
                pygame.draw.polygon(self.screen, tile_color, points)
                pygame.draw.aalines(self.screen, self.COLOR_GRID, True, points)

                if is_dug:
                    fossil_type = self.grid[x,y]
                    if fossil_type > 0:
                        self._draw_fossil(fossil_type, (sx, sy))

    def _draw_fossil(self, f_type, pos):
        sx, sy = pos
        color = self.FOSSIL_COLORS[f_type]
        if f_type == 1: # Skull
            pygame.draw.circle(self.screen, color, (sx, sy + 10), 6)
            pygame.draw.circle(self.screen, self.COLOR_HOLE, (sx - 3, sy + 8), 1)
            pygame.draw.circle(self.screen, self.COLOR_HOLE, (sx + 3, sy + 8), 1)
        elif f_type == 2: # Ribs
            for i in range(3):
                pygame.draw.arc(self.screen, color, (sx-8, sy+5+i*3, 16, 8), math.pi, 2*math.pi, 2)
        elif f_type == 3: # Claw
            p1 = (sx, sy + 5)
            p2 = (sx - 6, sy + 18)
            p3 = (sx + 6, sy + 18)
            pygame.draw.polygon(self.screen, color, [p1, p2, p3])
        elif f_type == 4: # Amber
            rect = pygame.Rect(sx - 6, sy + 5, 12, 12)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.circle(self.screen, (0,0,0,50), (sx, sy + 11), 1)

    def _render_dino(self):
        ix, iy = self.dino_pos
        sx, sy = self._iso_to_screen(ix, iy)
        
        bob = math.sin(self.steps * 0.2) * 2
        
        # Shadow
        shadow_rect = (sx - 12, sy + self.TILE_HEIGHT - 3, 24, 6)
        pygame.gfxdraw.filled_ellipse(self.screen, int(shadow_rect[0]+shadow_rect[2]/2), int(shadow_rect[1]+shadow_rect[3]/2), int(shadow_rect[2]/2), int(shadow_rect[3]/2), self.COLOR_DINO_SHADOW)

        # Body
        body_pos = (sx, sy + 10 - bob)
        pygame.draw.circle(self.screen, self.COLOR_DINO, body_pos, 10)
        
        # Head
        head_pos = (sx + 8, sy + 2 - bob)
        pygame.draw.circle(self.screen, self.COLOR_DINO, head_pos, 7)
        pygame.draw.circle(self.screen, (0,0,0), (int(head_pos[0]+3), int(head_pos[1])), 2)

    def _render_ui(self):
        # Moves Left Bar
        bar_width = 200
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 15
        
        moves_ratio = max(0, self.moves_left / self.MAX_MOVES)
        
        pygame.draw.rect(self.screen, self.COLOR_MOVES_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_MOVES_BAR, (bar_x, bar_y, bar_width * moves_ratio, bar_height), border_radius=5)
        
        moves_text = self.font_small.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (bar_x + bar_width + 10, bar_y + 2))

        # Score
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Fossil Collection Panel
        panel_surf = pygame.Surface((130, 110), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_BG)
        
        fossil_text = self.font_small.render("Fossils", True, self.COLOR_UI_TEXT)
        panel_surf.blit(fossil_text, (10, 5))
        
        for i, f_type in enumerate(self.FOSSIL_COLORS.keys()):
            y_pos = 28 + i * 20
            self._draw_fossil(f_type, (20, y_pos - 10))
            
            count = self.collected_fossils[f_type]
            max_count = 3 if f_type != 4 else 1
            count_text = self.font_small.render(f"{count} / {max_count}", True, self.COLOR_UI_TEXT)
            panel_surf.blit(count_text, (40, y_pos))
        
        self.screen.blit(panel_surf, (self.WIDTH - 140, self.HEIGHT - 120))
        
    def _create_particles(self, pos, count, color, speed, lifetime, gravity):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            p_speed = self.np_random.uniform(speed * 0.5, speed * 1.5)
            vel = [math.cos(angle) * p_speed, math.sin(angle) * p_speed]
            self.particles.append([list(pos), vel, self.np_random.integers(lifetime[0], lifetime[1]), color, gravity])

    def _create_dig_particles(self, ix, iy):
        pos = self._iso_to_screen(ix, iy)
        pos = (pos[0], pos[1] + self.TILE_HEIGHT / 2)
        self._create_particles(pos, 30, self.COLOR_GROUND, 2, [20, 30], 0.1)

    def _create_move_particles(self, ix, iy):
        pos = self._iso_to_screen(ix, iy)
        pos = (pos[0], pos[1] + self.TILE_HEIGHT - 5)
        self._create_particles(pos, 15, (200, 180, 160, 100), 1.5, [15, 25], 0.05)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[1][1] += p[4]  # Gravity
            p[2] -= 1

    def _render_particles(self):
        for p in self.particles:
            pos, _, lifetime, color, _ = p
            radius = max(0, int(lifetime / 5))
            pygame.draw.circle(self.screen, color, [int(pos[0]), int(pos[1])], radius)

    def _create_message(self, text, pos, color, size, lifetime):
        font = pygame.font.Font(None, size)
        self.messages.append([text, list(pos), color, lifetime, font])
    
    def _update_messages(self):
        self.messages = [m for m in self.messages if m[3] > 0]
        for m in self.messages:
            m[1][1] -= 0.5 # Float up
            m[3] -= 1

    def _render_messages(self):
        for m in self.messages:
            text, pos, color, lifetime, font = m
            alpha = min(255, int(255 * (lifetime / 30)))
            text_surf = font.render(text, True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=pos)
            self.screen.blit(text_surf, text_rect)
            
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")