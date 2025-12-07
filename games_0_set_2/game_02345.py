
# Generated: 2025-08-28T04:31:41.484919
# Source Brief: brief_02345.md
# Brief Index: 2345

        
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


# Helper classes defined outside to be cleaner
class Block:
    """Represents a single block in the tower."""
    def __init__(self, grid_x, grid_y):
        self.x = grid_x
        self.y = grid_y
        self.is_unstable = False
        self.wobble_phase = random.uniform(0, 2 * math.pi)

class Particle:
    """Represents a particle for the collapse effect."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2.5, 2.5)
        self.vy = random.uniform(-5, -1)
        self.life = random.randint(30, 60)
        self.color = color
        self.gravity = 0.25

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= 1
        return self.life > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (←/↑ for left, →/↓ for right) to move the block preview. Press space to place the block."
    )

    game_description = (
        "Build the tallest tower you can by strategically placing blocks. The tower will collapse if it becomes unstable! Win by reaching a height of 10."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_HEIGHT = 10
    MAX_STEPS = 1000
    GRID_WIDTH = 11

    # Visuals
    ISO_TILE_WIDTH_HALF = 20
    ISO_TILE_HEIGHT_HALF = 10
    BLOCK_RENDER_HEIGHT = 18
    
    # Colors
    COLOR_BG = (44, 62, 80)
    COLOR_GRID = (52, 73, 94)
    COLOR_BLOCK = (46, 204, 113)
    COLOR_BLOCK_OUTLINE = (39, 174, 96)
    COLOR_UNSTABLE = (231, 76, 60)
    COLOR_UNSTABLE_OUTLINE = (192, 57, 43)
    COLOR_PREVIEW = (52, 152, 219)
    COLOR_TEXT = (236, 240, 241)
    
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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT - 60

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.tower_height = 0
        self.preview_x_index = self.GRID_WIDTH // 2
        self.last_space_held = False
        self.wobble_tick = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held = action[0], action[1] == 1
        reward = 0.0
        
        self.wobble_tick += 1

        if not self.game_over:
            # Handle movement (1,3 -> left; 2,4 -> right)
            if movement in [1, 3]:
                self.preview_x_index -= 1
            elif movement in [2, 4]:
                self.preview_x_index += 1
            self.preview_x_index = np.clip(self.preview_x_index, 0, self.GRID_WIDTH - 1)

            # Handle placement on rising edge of space bar
            place_action = space_held and not self.last_space_held
            if place_action:
                # Sfx: Block place sound
                reward += 1.0
                
                landing_y = self._calculate_landing_y(self.preview_x_index)
                new_block = Block(self.preview_x_index, landing_y)
                self.blocks.append(new_block)

                if self._check_collapse():
                    # Sfx: Tower collapse sound
                    self.game_over = True
                    reward = -50.0
                    self._create_collapse_particles()
                else:
                    # Sfx: Success chime
                    current_height = self._get_tower_height()
                    if current_height > self.tower_height:
                        reward += 5.0
                        self.tower_height = current_height
                    
                    if self.tower_height >= self.WIN_HEIGHT:
                        # Sfx: Victory fanfare
                        self.win = True
                        self.game_over = True
                        reward += 50.0

        self.last_space_held = space_held
        
        self._update_particles()
        
        self.steps += 1
        self.score += reward
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_tower_height(self):
        if not self.blocks:
            return 0
        return max(b.y for b in self.blocks) + 1

    def _calculate_landing_y(self, grid_x):
        supported_y = [b.y for b in self.blocks if b.x == grid_x]
        return max(supported_y) + 1 if supported_y else 0

    def _check_collapse(self):
        max_h = self._get_tower_height()
        if max_h <= 1:
            return False

        for h in range(1, max_h):
            support_blocks = [b for b in self.blocks if b.y == h - 1]
            if not support_blocks: continue

            min_support_x = min(b.x for b in support_blocks)
            max_support_x = max(b.x for b in support_blocks) + 1.0

            mass_blocks = [b for b in self.blocks if b.y >= h]
            if not mass_blocks: continue

            com_x = sum(b.x + 0.5 for b in mass_blocks) / len(mass_blocks)

            if not (min_support_x <= com_x < max_support_x):
                for b in mass_blocks:
                    b.is_unstable = True
                return True
        return False
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _create_collapse_particles(self):
        for block in self.blocks:
            if block.is_unstable:
                cx, cy = self._get_iso_coords(block.x, block.y)
                cy -= self.BLOCK_RENDER_HEIGHT / 2
                for _ in range(15):
                    self.particles.append(Particle(cx, cy, self.COLOR_UNSTABLE))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.tower_height}

    def _get_iso_coords(self, grid_x, grid_y, offset_x=0, offset_y=0):
        screen_x = self.origin_x + (grid_x - grid_y) * self.ISO_TILE_WIDTH_HALF + offset_x
        screen_y = self.origin_y + (grid_x + grid_y) * self.ISO_TILE_HEIGHT_HALF - grid_y * self.BLOCK_RENDER_HEIGHT + offset_y
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_x, grid_y, color, outline_color, wobble_amp=0.0, alpha=255):
        wobble_x = math.sin(self.wobble_tick * 0.2 + (grid_x + grid_y) * 0.5) * wobble_amp
        wobble_y = math.cos(self.wobble_tick * 0.2 + (grid_x + grid_y) * 0.5) * wobble_amp * 0.5

        center_x, top_y = self._get_iso_coords(grid_x, grid_y, wobble_x, wobble_y)
        
        top_face_pts = [
            (center_x, top_y - self.ISO_TILE_HEIGHT_HALF),
            (center_x + self.ISO_TILE_WIDTH_HALF, top_y),
            (center_x, top_y + self.ISO_TILE_HEIGHT_HALF),
            (center_x - self.ISO_TILE_WIDTH_HALF, top_y)
        ]

        left_face_pts = [top_face_pts[3], top_face_pts[2], (top_face_pts[2][0], top_face_pts[2][1] + self.BLOCK_RENDER_HEIGHT), (top_face_pts[3][0], top_face_pts[3][1] + self.BLOCK_RENDER_HEIGHT)]
        right_face_pts = [top_face_pts[2], top_face_pts[1], (top_face_pts[1][0], top_face_pts[1][1] + self.BLOCK_RENDER_HEIGHT), (top_face_pts[2][0], top_face_pts[2][1] + self.BLOCK_RENDER_HEIGHT)]
        
        side_color = tuple(max(0, c - 30) for c in color)
        
        color_with_alpha = (*color, alpha)
        side_color_with_alpha = (*side_color, alpha)
        outline_color_with_alpha = (*outline_color, alpha)

        pygame.gfxdraw.filled_polygon(surface, left_face_pts, side_color_with_alpha)
        pygame.gfxdraw.aapolygon(surface, left_face_pts, outline_color_with_alpha)
        pygame.gfxdraw.filled_polygon(surface, right_face_pts, side_color_with_alpha)
        pygame.gfxdraw.aapolygon(surface, right_face_pts, outline_color_with_alpha)
        pygame.gfxdraw.filled_polygon(surface, top_face_pts, color_with_alpha)
        pygame.gfxdraw.aapolygon(surface, top_face_pts, outline_color_with_alpha)

    def _render_game(self):
        # Draw grid base platform
        base_depth = 2
        for y_offset in range(base_depth, -1, -1):
            color = tuple(max(0, c - y_offset * 10) for c in self.COLOR_GRID)
            outline = tuple(max(0, c - 10) for c in color)
            for i in range(self.GRID_WIDTH):
                self._draw_iso_cube(self.screen, i, -1-y_offset, color, outline)

        # Sort blocks for correct occlusion (painter's algorithm)
        sorted_blocks = sorted(self.blocks, key=lambda b: b.x + b.y)

        # Draw placed blocks
        for block in sorted_blocks:
            color = self.COLOR_UNSTABLE if block.is_unstable else self.COLOR_BLOCK
            outline = self.COLOR_UNSTABLE_OUTLINE if block.is_unstable else self.COLOR_BLOCK_OUTLINE
            wobble = 0.5 + block.y * 0.1
            self._draw_iso_cube(self.screen, block.x, block.y, color, outline, wobble_amp=wobble if not self.game_over else 0)

        # Draw preview block
        if not self.game_over:
            preview_y = self._calculate_landing_y(self.preview_x_index)
            temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            self._draw_iso_cube(temp_surface, self.preview_x_index, preview_y, self.COLOR_PREVIEW, self.COLOR_PREVIEW, alpha=150)
            self.screen.blit(temp_surface, (0, 0))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), 3, p.color)
            
    def _render_ui(self):
        height_text = self.font_large.render(f"{self.tower_height}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (20, 10))
        height_label = self.font_small.render("HEIGHT", True, self.COLOR_TEXT)
        self.screen.blit(height_label, (20, 50))
        
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))

        if self.game_over:
            message = "YOU WIN!" if self.win else "TOWER COLLAPSED"
            color = self.COLOR_BLOCK if self.win else self.COLOR_UNSTABLE
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 4))
            
            # Add a subtle shadow/background for the text
            shadow_text = self.font_large.render(message, True, (0,0,0))
            shadow_rect = shadow_text.get_rect(center=(self.SCREEN_WIDTH / 2 + 2, self.SCREEN_HEIGHT / 4 + 2))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    pygame.display.set_caption("Tower Builder")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            # This allows single press for movement
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_LEFT, pygame.K_a]:
                    movement = 3
                if event.key in [pygame.K_RIGHT, pygame.K_d]:
                    movement = 4
                if event.key in [pygame.K_UP, pygame.K_w]:
                    movement = 1
                if event.key in [pygame.K_DOWN, pygame.K_s]:
                    movement = 2

        # Hold to place
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)
        
    env.close()