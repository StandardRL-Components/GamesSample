
# Generated: 2025-08-27T19:57:35.727509
# Source Brief: brief_02304.md
# Brief Index: 2304

        
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
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling block. ↓ to speed up the fall. Space to drop instantly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build the tallest tower possible by stacking falling geometric tiles. A wobbly tower is an unstable one!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_HEIGHT = 30
        self.TILE_HEIGHT = 20
        self.FALL_SPEED_NORMAL = 1.5
        self.FALL_SPEED_FAST = 5.0
        self.MOVE_SPEED = 5.0
        self.MAX_TILES = 20
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (230, 230, 245)
        self.COLOR_GROUND = (80, 80, 90)
        self.COLOR_GRID = (210, 210, 225)
        self.COLOR_TEXT = (50, 50, 50)
        self.TILE_BASE_COLOR = (50, 100, 200)
        self.FALLING_TILE_COLOR = (255, 80, 80)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tiles_stacked = 0
        self.stacked_tiles = []
        self.falling_tile = None
        self.rng = None
        self.particles = []
        
        # For human rendering
        self.display_screen = None

    def _generate_new_tile(self):
        # Tiles get slightly smaller as you go up to make it harder
        min_width = max(40, 100 - self.tiles_stacked * 2)
        max_width = max(60, 140 - self.tiles_stacked * 3)
        width = self.rng.integers(min_width, max_width)
        
        # Start at a random x position
        x = self.rng.integers(0, self.SCREEN_WIDTH - width)
        
        return {
            "rect": pygame.Rect(x, -self.TILE_HEIGHT, width, self.TILE_HEIGHT),
            "wobble": 0,
            "wobble_speed": self.rng.uniform(0.1, 0.2)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.tiles_stacked = 0
        
        # Create the ground plane
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT)
        self.stacked_tiles = [{"rect": ground_rect, "wobble": 0, "wobble_speed": 0}]

        self.falling_tile = self._generate_new_tile()
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is not used per design brief
        
        # --- Game Logic ---
        # 1. Handle player input
        if movement == 3: # Left
            self.falling_tile["rect"].x -= self.MOVE_SPEED
        elif movement == 4: # Right
            self.falling_tile["rect"].x += self.MOVE_SPEED
        
        self.falling_tile["rect"].x = max(0, min(self.SCREEN_WIDTH - self.falling_tile["rect"].width, self.falling_tile["rect"].x))
        
        # 2. Update falling tile position
        fall_speed = self.FALL_SPEED_NORMAL
        if movement == 2: # Down (soft drop)
            fall_speed = self.FALL_SPEED_FAST
        
        if space_held: # Instant drop
            highest_y = self._get_landing_y(self.falling_tile["rect"])
            self.falling_tile["rect"].y = highest_y - self.TILE_HEIGHT
        else:
            self.falling_tile["rect"].y += fall_speed

        # 3. Check for landing
        landed = False
        for tile in self.stacked_tiles:
            if self.falling_tile["rect"].colliderect(tile["rect"]):
                landed = True
                self.falling_tile["rect"].bottom = tile["rect"].top # Snap to surface
                break
        
        if landed:
            # sfx: thud.wav
            self._create_particles(self.falling_tile["rect"].midbottom)
            
            support_base = self._get_support_base(self.falling_tile)
            
            if self._is_stable(self.falling_tile, support_base):
                # Stable placement
                self.stacked_tiles.append(self.falling_tile)
                self.tiles_stacked += 1
                self.score += 1
                reward += 1
                
                if self.tiles_stacked >= self.MAX_TILES:
                    self.game_over = True
                    reward += 100 # Win bonus
                    # sfx: win_fanfare.wav
                else:
                    self.falling_tile = self._generate_new_tile()
            else:
                # Unstable placement -> Tower collapses
                self.game_over = True
                reward -= 50 # Collapse penalty
                # sfx: tower_collapse.wav
                self.stacked_tiles.append(self.falling_tile)
                self.falling_tile = None
        
        self._update_animations()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_landing_y(self, rect):
        highest_y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT
        for tile in self.stacked_tiles:
            if rect.left < tile["rect"].right and rect.right > tile["rect"].left:
                highest_y = min(highest_y, tile["rect"].top)
        return highest_y

    def _get_support_base(self, new_tile):
        supporting_tiles = []
        for tile in self.stacked_tiles:
            if abs(new_tile["rect"].bottom - tile["rect"].top) < 5 and new_tile["rect"].colliderect(tile["rect"]):
                supporting_tiles.append(tile)

        if not supporting_tiles:
            return None, None

        min_x = min(t["rect"].left for t in supporting_tiles)
        max_x = max(t["rect"].right for t in supporting_tiles)
        return min_x, max_x

    def _is_stable(self, new_tile, support_base):
        min_x, max_x = support_base
        if min_x is None: return False
            
        center_x = new_tile["rect"].centerx
        tolerance = 2 
        return (min_x - tolerance) <= center_x <= (max_x + tolerance)

    def _update_animations(self):
        # Update tile wobble based on tower height and instability
        for i, tile in enumerate(self.stacked_tiles[1:]):
             instability_factor = (i + 1) / len(self.stacked_tiles)
             tile["wobble"] = math.sin(self.steps * tile["wobble_speed"]) * instability_factor * 0.5

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos):
        for _ in range(15):
            self.particles.append({
                "pos": list(pos),
                "vel": [self.rng.uniform(-1.5, 1.5), self.rng.uniform(-2, 0)],
                "life": self.rng.integers(15, 30),
                "color": (180, 200, 255, 150)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Draw background grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw stacked tiles
        tower_height = max(1, len(self.stacked_tiles) - 1)
        for i, tile in enumerate(self.stacked_tiles):
            if i == 0: # Ground
                pygame.draw.rect(self.screen, self.COLOR_GROUND, tile["rect"])
                continue

            lerp_factor = min(1.0, (i-1) / self.MAX_TILES)
            color = (
                int(self.TILE_BASE_COLOR[0] * (1 - lerp_factor) + 255 * lerp_factor),
                int(self.TILE_BASE_COLOR[1] * (1 - lerp_factor) + 200 * lerp_factor),
                int(self.TILE_BASE_COLOR[2] * (1 - lerp_factor) + 150 * lerp_factor)
            )

            angle = tile["wobble"]
            if self.game_over and self.falling_tile is None: # Collapse animation
                angle += (i / tower_height) * (self.steps % 30) * 0.2 * (-1 if i % 2 == 0 else 1)
                
            tile_surf = pygame.Surface((tile["rect"].width, tile["rect"].height), pygame.SRCALPHA)
            pygame.gfxdraw.box(tile_surf, pygame.Rect(0, 0, tile["rect"].width, tile["rect"].height), (*color, 255))
            pygame.gfxdraw.rectangle(tile_surf, pygame.Rect(0, 0, tile["rect"].width-1, tile["rect"].height-1), (*[c*0.8 for c in color], 255))
            
            rotated_surf = pygame.transform.rotate(tile_surf, angle)
            new_rect = rotated_surf.get_rect(center=tile["rect"].center)
            
            self.screen.blit(rotated_surf, new_rect.topleft)

        # Draw falling tile
        if self.falling_tile:
            rect = self.falling_tile["rect"]
            highest_y = self._get_landing_y(rect)
            shadow_rect = pygame.Rect(rect.x, highest_y - 5, rect.width, 5)
            pygame.gfxdraw.box(self.screen, shadow_rect, (0, 0, 0, 50))
            
            pygame.gfxdraw.box(self.screen, rect, (*self.FALLING_TILE_COLOR, 200))
            pygame.gfxdraw.rectangle(self.screen, rect, self.FALLING_TILE_COLOR)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), max(0, int(p["life"] / 5)), p["color"])

        self._render_ui()
        
        if self.render_mode == "human":
            self._render_to_display()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        tiles_text = self.font_large.render(f"Tiles: {self.tiles_stacked} / {self.MAX_TILES}", True, self.COLOR_TEXT)
        self.screen.blit(tiles_text, (self.SCREEN_WIDTH - tiles_text.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = ""
            if self.tiles_stacked >= self.MAX_TILES:
                end_text_str = "TOWER COMPLETE!"
            elif self.falling_tile is None:
                end_text_str = "TOWER COLLAPSED!"
            else:
                end_text_str = "TIME'S UP!"

            end_text = self.font_large.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tiles_stacked": self.tiles_stacked,
        }

    def render(self):
        if self.render_mode == "human":
            return self._render_to_display()
        return self._get_observation()

    def _render_to_display(self):
        if self.display_screen is None:
            self.display_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Tower Stacker")
        
        self.display_screen.blit(self.screen, (0, 0))
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    env.validate_implementation()
    obs, info = env.reset(seed=random.randint(0, 10000))
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # unused
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_DOWN]:
            movement = 2
        
        action = [movement, space_held, shift_held]
        
        # --- Event Handling for instant actions ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Press space to drop
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    total_reward = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Rendering ---
        env.render()
        
        # --- Frame Rate ---
        env.clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    env.close()