
# Generated: 2025-08-28T05:34:40.727142
# Source Brief: brief_05620.md
# Brief Index: 5620

        
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
    user_guide = (
        "Controls: ←→ to move the tile. Hold space to drop the tile quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically stack falling tiles to build the tallest tower possible."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (40, 45, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_FALLING_TILE = (0, 200, 255) # Bright Cyan
    COLOR_PARTICLE = (255, 255, 100)
    
    # Game Parameters
    TILE_HEIGHT = 15
    BASE_TILE_WIDTH = 200
    MIN_TILE_WIDTH = 50
    MAX_TILE_WIDTH = 100
    MOVE_SPEED = 8
    GRAVITY = 0.15
    FAST_DROP_MULTIPLIER = 10
    MAX_STEPS = 1000
    WIN_HEIGHT = 20
    STABILITY_THRESHOLD = 0.25 # Tile must have at least this much of its width supported
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.placed_tiles = []
        self.falling_tile = None
        self.fall_speed_initial = self.GRAVITY
        self.fall_speed_current = self.GRAVITY
        self.max_height = 0
        self.particles = []
        self.rng = None

        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            # Fallback to a default or unseeded generator if no seed is provided
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_height = 1
        self.fall_speed_current = self.fall_speed_initial
        self.particles.clear()
        
        # Create the base platform
        self.placed_tiles = [
            pygame.Rect(
                (self.SCREEN_WIDTH - self.BASE_TILE_WIDTH) / 2,
                self.SCREEN_HEIGHT - self.TILE_HEIGHT,
                self.BASE_TILE_WIDTH,
                self.TILE_HEIGHT,
            )
        ]
        
        self._spawn_tile()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed_current += 0.05
            
        self._handle_input(action)
        self._update_physics()
        
        placement_reward, placed_successfully = self._check_collisions_and_place()
        reward += placement_reward

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if placed_successfully:
            current_height = len(self.placed_tiles)
            if current_height > self.max_height:
                reward += 1.0 # Reward for increasing height
                self.max_height = current_height
            
            if self.max_height >= self.WIN_HEIGHT:
                reward += 100 # Win reward
                self.game_over = True
                terminated = True

        if self.game_over and not (self.max_height >= self.WIN_HEIGHT):
            reward = -100 # Collapse penalty
            
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3: # Left
            self.falling_tile['rect'].x -= self.MOVE_SPEED
        elif movement == 4: # Right
            self.falling_tile['rect'].x += self.MOVE_SPEED

        # Clamp to screen
        self.falling_tile['rect'].x = max(0, self.falling_tile['rect'].x)
        self.falling_tile['rect'].right = min(self.SCREEN_WIDTH, self.falling_tile['rect'].right)

        # Fast drop
        if space_held:
            self.falling_tile['vy'] = self.fall_speed_current * self.FAST_DROP_MULTIPLIER
        else:
            self.falling_tile['vy'] = self.fall_speed_current

    def _update_physics(self):
        # Update falling tile
        if self.falling_tile:
            self.falling_tile['pos_y'] += self.falling_tile['vy']
            self.falling_tile['rect'].y = int(self.falling_tile['pos_y'])
            
            # Check for falling off-screen (unsupported tile)
            if self.falling_tile['rect'].top > self.SCREEN_HEIGHT:
                self.game_over = True
                self.falling_tile = None # Stop processing it
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += self.GRAVITY / 2 # Particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_collisions_and_place(self):
        if not self.falling_tile:
            return 0, False

        for tile in self.placed_tiles:
            if self.falling_tile['rect'].colliderect(tile):
                # Collision detected, check for stability and place the tile
                
                # Check if the tile is supported
                supported_width = 0
                falling_rect = self.falling_tile['rect']
                
                # Find all supporting tiles directly underneath
                supporting_tiles = [t for t in self.placed_tiles if t.top == tile.top and falling_rect.bottom > t.top]
                
                for support in supporting_tiles:
                    overlap_left = max(falling_rect.left, support.left)
                    overlap_right = min(falling_rect.right, support.right)
                    supported_width += max(0, overlap_right - overlap_left)

                if supported_width / falling_rect.width >= self.STABILITY_THRESHOLD:
                    # Stable placement
                    # Snap tile into place
                    falling_rect.bottom = tile.top
                    self.placed_tiles.append(falling_rect)
                    
                    # sfx: tile_place.wav
                    self._spawn_particles(falling_rect.midbottom, 15)
                    self._spawn_tile()
                    
                    return 0.1, True # Base reward for placing a tile
                else:
                    # Unstable placement, tile will keep falling
                    # No reward, no placement
                    return 0, False
        return 0, False

    def _spawn_tile(self):
        width = self.rng.integers(self.MIN_TILE_WIDTH, self.MAX_TILE_WIDTH + 1)
        self.falling_tile = {
            'rect': pygame.Rect(
                (self.SCREEN_WIDTH - width) / 2, 0, width, self.TILE_HEIGHT
            ),
            'pos_y': 0.0,
            'vy': self.fall_speed_current
        }

    def _spawn_particles(self, pos, count):
        # sfx: particle_burst.wav
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': self.rng.integers(20, 40),
                'size': self.rng.uniform(2, 5),
            })
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw placed tiles
        num_tiles = len(self.placed_tiles)
        for i, tile in enumerate(self.placed_tiles):
            # Gradient color based on height (older = darker)
            lerp_factor = max(0, (i - max(0, num_tiles - 20)) / 20)
            color = (
                int(50 + lerp_factor * 50),
                int(100 + lerp_factor * 50),
                int(150 + lerp_factor * 100)
            )
            self._draw_rounded_rect(self.screen, tile, color, 4)

        # Draw falling tile
        if self.falling_tile:
            self._draw_rounded_rect(self.screen, self.falling_tile['rect'], self.COLOR_FALLING_TILE, 4)
            # Draw a "ghost" of where it would land
            if not self.game_over:
                self._draw_ghost_tile()

        # Draw particles
        for p in self.particles:
            size = p['size'] * (p['life'] / 40) # Fade size
            rect = pygame.Rect(p['pos'][0] - size/2, p['pos'][1] - size/2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PARTICLE, rect)

    def _draw_ghost_tile(self):
        if not self.falling_tile: return
        
        ghost_rect = self.falling_tile['rect'].copy()
        
        highest_y = self.SCREEN_HEIGHT
        for tile in self.placed_tiles:
            if ghost_rect.left < tile.right and ghost_rect.right > tile.left:
                highest_y = min(highest_y, tile.top)
        
        ghost_rect.bottom = highest_y
        
        # Use a transparent color
        ghost_color = (*self.COLOR_FALLING_TILE, 100)
        
        # Create a temporary surface for transparency
        temp_surface = pygame.Surface(ghost_rect.size, pygame.SRCALPHA)
        self._draw_rounded_rect(temp_surface, temp_surface.get_rect(), ghost_color, 4)
        self.screen.blit(temp_surface, ghost_rect.topleft)

    def _render_ui(self):
        height_text = f"HEIGHT: {self.max_height}/{self.WIN_HEIGHT}"
        score_text = f"SCORE: {self.score:.1f}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        
        self._draw_text(height_text, (10, 10), self.font_small)
        self._draw_text(score_text, (10, 35), self.font_small)
        self._draw_text(steps_text, (self.SCREEN_WIDTH - 10, 10), self.font_small, align="right")
        
        if self.game_over:
            if self.max_height >= self.WIN_HEIGHT:
                end_text = "TOWER COMPLETE!"
            else:
                end_text = "TOWER COLLAPSED"
            self._draw_text(end_text, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_large, align="center")

    def _draw_text(self, text, pos, font, align="left"):
        text_surface = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "right":
            text_rect.topright = pos
        else: # left
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_rounded_rect(self, surface, rect, color, corner_radius):
        """Draw a rectangle with rounded corners."""
        if rect.width < 2 * corner_radius or rect.height < 2 * corner_radius:
            pygame.draw.rect(surface, color, rect)
            return

        # Anti-aliased filled rounded rectangle

        # Main body
        pygame.draw.rect(surface, color, rect.inflate(-2 * corner_radius, 0))
        pygame.draw.rect(surface, color, rect.inflate(0, -2 * corner_radius))
        
        # Corners
        cr = corner_radius
        for p in [(rect.left+cr, rect.top+cr), (rect.right-cr-1, rect.top+cr),
                  (rect.left+cr, rect.bottom-cr-1), (rect.right-cr-1, rect.bottom-cr-1)]:
            pygame.gfxdraw.aacircle(surface, p[0], p[1], cr, color)
            pygame.gfxdraw.filled_circle(surface, p[0], p[1], cr, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.max_height,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a way to display the frames.
    # Pygame is used headlessly here, but we can create a window for testing.
    
    # Setup a display for manual play
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Stacker")
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_r]: # Reset key
            obs, info = env.reset()
            terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(30)
        
    env.close()