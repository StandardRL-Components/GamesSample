
# Generated: 2025-08-28T04:26:34.243943
# Source Brief: brief_05246.md
# Brief Index: 5246

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based stealth-horror game where the player must hunt down enemies.

    The player (white circle) navigates an 8x8 grid, some tiles of which are
    darkened "shadow tiles". The goal is to eliminate all 5 red "Shadow" enemies
    on the stage by striking them. The player strikes in the direction of their
    last movement.

    Enemies move randomly and will damage the player if they land on the same tile.
    Hiding on shadow tiles provides a small reward bonus. The game has 3 stages,
    with enemies moving more frequently in later stages. The episode ends if the
    player's health reaches zero, all 3 stages are cleared, or 3000 steps are taken.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to aim and move. Press Space to strike. Eliminate all red Shadows."
    )

    game_description = (
        "A grid-based horror game. Strategically move and strike from the shadows to eliminate lurking enemies."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = self.HEIGHT // self.GRID_SIZE
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.MAX_STEPS = 3000
        self.NUM_STAGES = 3
        self.SHADOWS_PER_STAGE = 5
        self.INITIAL_PLAYER_HEALTH = 3

        # --- Colors & Style ---
        self.COLOR_BG = (10, 15, 20)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_SHADOW_TILE = (20, 25, 35)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_SHADOW = (255, 0, 80)
        self.COLOR_TEXT = (200, 200, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_health = 0
        self.stage = 0
        self.player_pos = (0, 0)
        self.last_move_dir = (0, 1)
        self.shadows = []
        self.shadow_tiles = set()
        self.particles = []
        self.screen_flash = 0

        # self.validate_implementation() # For development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_health = self.INITIAL_PLAYER_HEALTH
        self.stage = 1
        self.last_move_dir = (0, 1)  # Default down
        self.particles = []
        self.screen_flash = 0

        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the grid, player, and shadows for the current stage."""
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        shuffled_coords = self.np_random.permutation(all_coords).tolist()

        num_shadow_tiles = int(self.GRID_SIZE * self.GRID_SIZE * 0.25)
        self.shadow_tiles = set(map(tuple, shuffled_coords[:num_shadow_tiles]))

        available_coords = shuffled_coords
        self.player_pos = tuple(available_coords.pop())

        self.shadows = []
        for _ in range(self.SHADOWS_PER_STAGE):
            if not available_coords:
                break
            pos = tuple(available_coords.pop())
            self.shadows.append({"pos": pos})

    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.1  # Time penalty

        self.steps += 1
        if self.screen_flash > 0:
            self.screen_flash -= 1

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # 1. Player Action
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.last_move_dir = (dx, dy)
            nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy

            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                is_occupied_by_shadow = any(s["pos"] == (nx, ny) for s in self.shadows)
                if not is_occupied_by_shadow:
                    self.player_pos = (nx, ny)
                    if self.player_pos in self.shadow_tiles:
                        reward += 0.2
                        # sfx: hide_sound
                        self._create_particles(self._grid_to_pixel(self.player_pos), 5, self.COLOR_PLAYER, 0.5)

        elif space_held:
            strike_pos = (self.player_pos[0] + self.last_move_dir[0], self.player_pos[1] + self.last_move_dir[1])
            # sfx: strike_whoosh
            self._create_particles(self._grid_to_pixel(self.player_pos), 15, self.COLOR_PLAYER, 2.0, self.last_move_dir)

            shadow_hit = next((s for s in self.shadows if s["pos"] == strike_pos), None)
            if shadow_hit:
                # sfx: enemy_die
                self._create_particles(self._grid_to_pixel(shadow_hit["pos"]), 30, self.COLOR_SHADOW, 4.0)
                self.shadows.remove(shadow_hit)
                reward += 10
                self.score += 10

        # 2. Enemy Turn
        occupied_cells = {self.player_pos} | {tuple(s["pos"]) for s in self.shadows}
        shadow_move_prob = min(1.0, 0.1 * self.stage)

        shadows_to_move = [s for s in self.shadows if self.np_random.random() < shadow_move_prob]
        for s in shadows_to_move:
            # sfx: shadow_move
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            self.np_random.shuffle(moves)
            for dx, dy in moves:
                nx, ny = s["pos"][0] + dx, s["pos"][1] + dy
                if (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in occupied_cells):
                    s["pos"] = (nx, ny)
                    occupied_cells.add((nx, ny))
                    break
        
        # 3. Check for Damage (after player and enemies move)
        if any(s["pos"] == self.player_pos for s in self.shadows):
            if self.player_health > 0:
                # sfx: player_hurt
                self.player_health -= 1
                reward -= 5
                self.score -= 5
                self.screen_flash = 2  # Flash for 2 steps

        # 4. Check for Termination/Progression
        terminated = False
        if not self.shadows and not self.game_won:
            # sfx: stage_clear
            reward += 100
            self.score += 100
            self.stage += 1
            if self.stage > self.NUM_STAGES:
                # sfx: game_win
                self.game_won = True
                terminated = True
                reward += 200
                self.score += 200
            else:
                self._setup_stage()

        if self.player_health <= 0:
            # sfx: game_over
            self.game_over = True
            terminated = True
            reward -= 100
            self.score -= 100

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw shadow tiles
        for x, y in self.shadow_tiles:
            rect = (self.GRID_OFFSET_X + x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SHADOW_TILE, rect)

        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X + i * self.CELL_SIZE, 0), (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.HEIGHT), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, i * self.CELL_SIZE), (self.GRID_OFFSET_X + self.GRID_WIDTH, i * self.CELL_SIZE), 1)

        # Draw shadows
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        shadow_radius = self.CELL_SIZE * 0.3 + pulse * 3
        for s in self.shadows:
            pos = self._grid_to_pixel(s["pos"])
            self._render_glow_circle(self.screen, pos, shadow_radius, self.COLOR_SHADOW)
        
        # Draw player
        player_radius = self.CELL_SIZE * 0.35
        player_pos_px = self._grid_to_pixel(self.player_pos)
        self._render_glow_circle(self.screen, player_pos_px, player_radius, self.COLOR_PLAYER)

        self._update_and_draw_particles()

        if self.screen_flash > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = 100 * (self.screen_flash / 2.0)
            flash_surface.fill((*self.COLOR_SHADOW, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Health
        for i in range(self.INITIAL_PLAYER_HEALTH):
            rect = pygame.Rect(15 + i * 25, 15, 20, 20)
            color = self.COLOR_PLAYER if i < self.player_health else self.COLOR_GRID
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 1, border_radius=3)

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 15, 15))

        # Shadow Count
        shadow_count_text = self.font_small.render(f"SHADOWS: {len(self.shadows)}", True, self.COLOR_TEXT)
        self.screen.blit(shadow_count_text, (self.WIDTH/2 - shadow_count_text.get_width()/2, self.HEIGHT - 30))

        if self.game_over:
            text = self.font_large.render("YOU ARE CAUGHT", True, self.COLOR_SHADOW)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)
        elif self.game_won:
            text = self.font_large.render("YOU ESCAPED", True, self.COLOR_PLAYER)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health, "stage": self.stage}

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _create_particles(self, pos, count, color, speed_mult=1.0, direction=None):
        for _ in range(count):
            if direction:
                angle = math.atan2(direction[1], direction[0]) + self.np_random.uniform(-0.4, 0.4)
            else:
                angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                
                alpha = int(255 * (p['life'] / p['max_life']))
                size = max(1, 3 * (p['life'] / p['max_life']))
                
                temp_surf = pygame.Surface((int(size*2), int(size*2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p['color'], alpha), (int(size), int(size)), int(size))
                self.screen.blit(temp_surf, (p['pos'][0] - size, p['pos'][1] - size), special_flags=pygame.BLEND_RGBA_ADD)
                active_particles.append(p)
        self.particles = active_particles

    def _render_glow_circle(self, surface, pos, radius, color, glow_layers=5):
        for i in range(glow_layers, 0, -1):
            alpha = int(100 / (i**1.5))
            glow_radius = radius + i * 2
            pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(glow_radius), (*color, alpha))
        pygame.gfxdraw.aacircle(surface, int(pos[0]), int(pos[1]), int(radius), color)
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), int(radius), color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")