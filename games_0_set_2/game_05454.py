
# Generated: 2025-08-28T05:04:16.274911
# Source Brief: brief_05454.md
# Brief Index: 5454

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑↓←→ to move one square at a time."

    # Must be a short, user-facing description of the game:
    game_description = "Navigate a crystal cavern, collect 10 crystals, and avoid the deadly traps. Each step costs a small amount of score."

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 8
        self.NUM_CRYSTALS_TARGET = 10
        self.NUM_TRAPS = 8
        self.MAX_STEPS = 200

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
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 18)
        except IOError:
            self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
            self.font_small = pygame.font.SysFont("monospace", 18)

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (40, 30, 60)
        self.COLOR_PLAYER = (60, 150, 255)
        self.COLOR_PLAYER_GLOW = (30, 75, 128)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_TRAP_GLOW = (128, 25, 25)
        self.CRYSTAL_COLORS = [
            (255, 255, 100), (100, 255, 255), (255, 100, 255),
            (150, 255, 150), (255, 150, 100)
        ]
        self.COLOR_TEXT = (240, 240, 240)

        # Grid layout calculation
        self.grid_render_size = self.HEIGHT - 40
        self.cell_size = self.grid_render_size // self.GRID_SIZE
        self.grid_offset_x = (self.WIDTH - self.grid_render_size) // 2
        self.grid_offset_y = (self.HEIGHT - self.grid_render_size) // 2
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        self.player_pos = (0, 0)
        self.crystals = []
        self.traps = []
        self.particles = []
        
        self.reset()
        
        # self.validate_implementation() # For development; commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        self.particles = []

        all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_cells)

        self.player_pos = all_cells.pop()
        self.traps = [all_cells.pop() for _ in range(self.NUM_TRAPS)]
        
        self.crystals = []
        for i in range(self.NUM_CRYSTALS_TARGET):
            pos = all_cells.pop()
            self.crystals.append({
                "pos": pos,
                "color": self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)],
                "anim_offset": self.np_random.random() * 2 * math.pi
            })
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        self.steps += 1
        reward = 0.0
        terminated = False

        # --- Handle Movement ---
        target_pos = list(self.player_pos)
        moved = True
        if movement == 1: target_pos[1] -= 1  # Up
        elif movement == 2: target_pos[1] += 1  # Down
        elif movement == 3: target_pos[0] -= 1  # Left
        elif movement == 4: target_pos[0] += 1  # Right
        else: moved = False
        
        # --- Update State & Rewards ---
        if moved and 0 <= target_pos[0] < self.GRID_SIZE and 0 <= target_pos[1] < self.GRID_SIZE:
            self.player_pos = tuple(target_pos)
            self._create_particles(self.player_pos, (200, 200, 200), 5, 3) # Movement poof
        
        # Step penalty
        reward -= 0.1
        self.score -= 0.1

        # Trap proximity penalty
        if self.traps:
            min_dist = min(abs(self.player_pos[0] - tx) + abs(self.player_pos[1] - ty) for tx, ty in self.traps)
            if min_dist > 0:
                proximity_penalty = 0.1 * (1 / min_dist)
                reward -= proximity_penalty
                self.score -= proximity_penalty

        # Check for trap collision
        if self.player_pos in self.traps:
            reward -= 100
            self.score -= 100
            self.game_over = True
            # // Play sound effect: Explosion/Fail
            self._create_particles(self.player_pos, self.COLOR_TRAP, 50, 15)

        # Check for crystal collection
        crystal_to_remove = next((i for i, c in enumerate(self.crystals) if c["pos"] == self.player_pos), None)
        if crystal_to_remove is not None:
            crystal = self.crystals.pop(crystal_to_remove)
            reward += 10
            self.score += 10
            self.crystals_collected += 1
            # // Play sound effect: Crystal collect
            self._create_particles(self.player_pos, crystal["color"], 20, 10)
        
        # Check for win/loss conditions
        if self.crystals_collected >= self.NUM_CRYSTALS_TARGET:
            reward += 100
            self.score += 100
            self.game_over = True
            # // Play sound effect: Win/Success
        elif self.steps >= self.MAX_STEPS or self.game_over:
            self.game_over = True
        
        terminated = self.game_over
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _grid_to_pixel(self, grid_pos):
        x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return x, y

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x + i * self.cell_size, self.grid_offset_y), (self.grid_offset_x + i * self.cell_size, self.grid_offset_y + self.grid_render_size), 1)
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y + i * self.cell_size), (self.grid_offset_x + self.grid_render_size, self.grid_offset_y + i * self.cell_size), 1)

        # Draw traps
        for trap_pos in self.traps:
            px, py = self._grid_to_pixel(trap_pos)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(self.cell_size * 0.3 + pulse * self.cell_size * 0.1)
            glow_radius = int(radius * 1.5)
            
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_TRAP_GLOW, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (px - glow_radius, py - glow_radius))
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_TRAP)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_TRAP)

        # Draw crystals
        for crystal in self.crystals:
            px, py = self._grid_to_pixel(crystal['pos'])
            size = self.cell_size * 0.3
            points = [(px, py - size), (px + size, py), (px, py + size), (px - size, py)]
            pygame.gfxdraw.filled_polygon(self.screen, points, crystal['color'])
            pygame.gfxdraw.aapolygon(self.screen, points, crystal['color'])

            anim_phase = self.steps * 0.1 + crystal['anim_offset']
            for i in range(3):
                angle = anim_phase + (i * 2 * math.pi / 3)
                sparkle_x = px + math.cos(angle) * size * 1.5
                sparkle_y = py + math.sin(angle) * size * 1.5
                sparkle_size = int(2 + math.sin(angle * 5) * 1)
                pygame.draw.circle(self.screen, (255, 255, 255), (sparkle_x, sparkle_y), sparkle_size)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], max(0, min(255, alpha)))
            pygame.draw.circle(self.screen, color, p['pos'], 2)

        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        size = self.cell_size * 0.35
        glow_size = size * 1.5
        
        glow_surf = pygame.Surface((int(glow_size * 2), int(glow_size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 100), (int(glow_size), int(glow_size)), int(glow_size))
        self.screen.blit(glow_surf, (int(px - glow_size), int(py - glow_size)))

        player_rect = pygame.Rect(px - size, py - size, size * 2, size * 2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_ui(self):
        score_text = f"Score: {int(self.score)}"
        crystal_text = f"Crystals: {self.crystals_collected} / {self.NUM_CRYSTALS_TARGET}"
        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"

        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        crystal_surf = self.font_small.render(crystal_text, True, self.COLOR_TEXT)
        steps_surf = self.font_small.render(steps_text, True, self.COLOR_TEXT)

        self.screen.blit(score_surf, (15, 10))
        self.screen.blit(crystal_surf, (15, 40))
        self.screen.blit(steps_surf, (15, 60))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "VICTORY!" if self.crystals_collected >= self.NUM_CRYSTALS_TARGET else "GAME OVER"
            color = (100, 255, 100) if self.crystals_collected >= self.NUM_CRYSTALS_TARGET else self.COLOR_TRAP
                
            end_surf = self.font_large.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_collected": self.crystals_collected
        }

    def _create_particles(self, pos_grid, color, count, max_life):
        px, py = self._grid_to_pixel(pos_grid)
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            life = self.np_random.integers(max_life // 2, max_life + 1)
            self.particles.append({
                "pos": [px, py], 
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed], 
                "life": life, 
                "max_life": life, 
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

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