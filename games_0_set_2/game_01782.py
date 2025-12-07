
# Generated: 2025-08-27T18:16:48.810536
# Source Brief: brief_01782.md
# Brief Index: 1782

        
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
        "Controls: ←→ to move the falling tile. Hold space to drop it faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically stack falling tiles to build a stable tower and reach the target height."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    COLOR_BG = pygame.Color("#1E2733")
    COLOR_GRID = pygame.Color("#334155")
    COLOR_GROUND = pygame.Color("#475569")
    COLOR_FALLING_TILE = pygame.Color("#F59E0B") # Amber
    COLOR_TOWER_START = pygame.Color("#3B82F6") # Blue
    COLOR_TOWER_END = pygame.Color("#A78BFA") # Violet
    COLOR_TEXT = pygame.Color("#F1F5F9")
    COLOR_TARGET_LINE = pygame.Color("#FACC15") # Yellow
    COLOR_UNSTABLE_FLASH = pygame.Color("#EF4444") # Red

    TILE_WIDTH = 80
    TILE_HEIGHT = 15
    PLAYER_SPEED = 6.0
    BASE_GRAVITY = 0.5
    DROP_SPEED_MULTIPLIER = 8.0
    
    WIN_HEIGHT = 20
    MAX_TILES = 30
    MAX_STEPS = 3000 # Increased for a longer game

    PARTICLE_COUNT = 15
    PARTICLE_LIFESPAN = 30
    PARTICLE_SPEED = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.falling_tile = None
        self.stacked_tiles = []
        self.particles = []
        self.max_height = 0
        self.tiles_remaining = 0
        self.placed_count = 0
        self.gravity = 0
        self.unstable_flash_timer = 0
        
        # This will be seeded in reset()
        self.np_random = None
        
        # Initialize state by calling reset
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_height = 1
        self.tiles_remaining = self.MAX_TILES
        self.placed_count = 0
        self.gravity = self.BASE_GRAVITY
        self.unstable_flash_timer = 0
        
        self.stacked_tiles = []
        self.particles = []

        # Create the ground tile
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.TILE_HEIGHT, self.SCREEN_WIDTH, self.TILE_HEIGHT)
        self.stacked_tiles.append({"rect": ground_rect, "color": self.COLOR_GROUND})

        self._spawn_tile()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info()
            )

        self.steps += 1
        
        # --- Handle Input ---
        movement = action[0]
        space_held = action[1] == 1
        
        if self.falling_tile:
            if movement == 3: # Left
                self.falling_tile["rect"].x -= self.PLAYER_SPEED
            elif movement == 4: # Right
                self.falling_tile["rect"].x += self.PLAYER_SPEED
            
            # Clamp to screen
            self.falling_tile["rect"].left = max(0, self.falling_tile["rect"].left)
            self.falling_tile["rect"].right = min(self.SCREEN_WIDTH, self.falling_tile["rect"].right)

        # --- Update Game Logic ---
        self._update_particles()
        if self.unstable_flash_timer > 0:
            self.unstable_flash_timer -= 1

        if self.falling_tile:
            # Apply gravity
            fall_speed = self.gravity * (self.DROP_SPEED_MULTIPLIER if space_held else 1)
            self.falling_tile["rect"].y += fall_speed

            # Check for collision
            for tile in self.stacked_tiles:
                if self.falling_tile["rect"].colliderect(tile["rect"]):
                    # Collision detected, place the tile
                    self.falling_tile["rect"].bottom = tile["rect"].top
                    
                    # --- Stability Check ---
                    is_stable = self._check_stability(self.falling_tile["rect"])
                    
                    if is_stable:
                        # Sound: Place_Success.wav
                        new_height = len(self.stacked_tiles)
                        
                        # Calculate color based on height
                        color_ratio = min(1.0, (new_height - 1) / self.WIN_HEIGHT)
                        tile_color = self.COLOR_TOWER_START.lerp(self.COLOR_TOWER_END, color_ratio)
                        self.falling_tile["color"] = tile_color
                        
                        self.stacked_tiles.append(self.falling_tile)
                        self._spawn_particles(self.falling_tile["rect"].midbottom, self.falling_tile["color"])
                        self.falling_tile = None
                        
                        self.placed_count += 1
                        reward += 0.1 # Reward for stable placement
                        
                        if new_height > self.max_height:
                            reward += 1.0 # Reward for increasing height
                            self.max_height = new_height
                        
                        # Difficulty scaling
                        if self.placed_count > 0 and self.placed_count % 5 == 0:
                            self.gravity += 0.05
                            
                        # Win condition
                        if self.max_height >= self.WIN_HEIGHT:
                            reward += 100
                            self.game_over = True
                            
                    else:
                        # Sound: Collapse.wav
                        reward = -100 # Penalty for collapse
                        self.game_over = True
                        self._spawn_particles(self.falling_tile["rect"].midbottom, self.COLOR_UNSTABLE_FLASH, 2)
                        self.unstable_flash_timer = 15 # Flash red on failure

                    break # Stop checking collisions for this frame
        
        else: # If no falling tile, spawn one
            if not self.game_over and self.tiles_remaining > 0:
                self._spawn_tile()
            elif not self.game_over and self.tiles_remaining <= 0:
                # Ran out of tiles, game ends
                self.game_over = True

        # Check termination conditions
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_stability(self, falling_rect):
        center_x = falling_rect.centerx
        is_supported = False
        
        # Check against all tiles that are directly below the new one
        for tile in self.stacked_tiles:
            if tile["rect"].top == falling_rect.bottom:
                if center_x >= tile["rect"].left and center_x <= tile["rect"].right:
                    is_supported = True
                    break
        return is_supported

    def _spawn_tile(self):
        if self.tiles_remaining <= 0:
            return
            
        self.tiles_remaining -= 1
        start_x = self.np_random.integers(self.TILE_WIDTH // 2, self.SCREEN_WIDTH - self.TILE_WIDTH // 2)
        rect = pygame.Rect(0, 0, self.TILE_WIDTH, self.TILE_HEIGHT)
        rect.midtop = (start_x, 0)
        self.falling_tile = {"rect": rect, "color": self.COLOR_FALLING_TILE}

    def _spawn_particles(self, pos, color, speed_mult=1):
        for _ in range(self.PARTICLE_COUNT):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, self.PARTICLE_SPEED) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            particle = {
                "pos": list(pos),
                "vel": vel,
                "lifespan": self.PARTICLE_LIFESPAN,
                "color": color,
                "size": self.np_random.integers(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Target height line
        target_y = self.SCREEN_HEIGHT - self.TILE_HEIGHT * self.WIN_HEIGHT
        if target_y > 0:
            for x in range(0, self.SCREEN_WIDTH, 20):
                 pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 1)

        # Stacked tiles
        for tile in self.stacked_tiles:
            self._draw_tile(tile["rect"], tile["color"])

        # Falling tile
        if self.falling_tile:
            color = self.COLOR_UNSTABLE_FLASH if self.unstable_flash_timer > 0 else self.falling_tile["color"]
            self._draw_tile(self.falling_tile["rect"], color, is_falling=True)
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            color = p["color"]
            
            # Use a surface for transparency
            particle_surf = pygame.Surface((p["size"] * 2, p["size"] * 2), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, (*color, alpha), (0, 0, p["size"], p["size"]))
            self.screen.blit(particle_surf, (int(p["pos"][0] - p["size"]/2), int(p["pos"][1] - p["size"]/2)))

    def _draw_tile(self, rect, color, is_falling=False):
        # Main body
        pygame.draw.rect(self.screen, color, rect)
        
        # Border for definition
        border_color = color.lerp(pygame.Color(0,0,0), 0.4)
        pygame.draw.rect(self.screen, border_color, rect, 2)

        # Highlight for 3D effect
        highlight_color = color.lerp(pygame.Color(255,255,255), 0.3)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        
        # Shadow if falling
        if is_falling:
            shadow_pos = (rect.centerx, self.SCREEN_HEIGHT - self.TILE_HEIGHT)
            for tile in self.stacked_tiles:
                if rect.centerx > tile["rect"].left and rect.centerx < tile["rect"].right:
                    if tile["rect"].top < shadow_pos[1]:
                        shadow_pos = (rect.centerx, tile["rect"].top)
            
            shadow_rect = pygame.Rect(0, 0, rect.width, 5)
            shadow_rect.midbottom = shadow_pos
            shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
            shadow_surf.fill((0, 0, 0, 60))
            self.screen.blit(shadow_surf, shadow_rect)

    def _render_ui(self):
        # Height display
        height_text = f"Height: {self.max_height-1} / {self.WIN_HEIGHT-1}"
        height_surf = self.font_ui.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(height_surf, (10, 10))
        
        # Tiles remaining display
        tiles_text = f"Tiles: {self.tiles_remaining}"
        tiles_surf = self.font_ui.render(tiles_text, True, self.COLOR_TEXT)
        self.screen.blit(tiles_surf, (self.SCREEN_WIDTH - tiles_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            msg = "YOU WON!" if self.max_height >= self.WIN_HEIGHT else "GAME OVER"
            color = self.COLOR_TARGET_LINE if self.max_height >= self.WIN_HEIGHT else self.COLOR_UNSTABLE_FLASH
            
            over_surf = self.font_game_over.render(msg, True, color)
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            bg_rect = over_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect)
            
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.max_height - 1, # Ground doesn't count
            "tiles_remaining": self.tiles_remaining,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# --- Example usage to run and visualize the game ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for visualization ---
    pygame.display.set_caption("Tower Builder")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # --- Main game loop ---
    while not done:
        # --- Human Input ---
        # This part is for human play, an agent would call env.step(agent.get_action(obs))
        keys = pygame.key.get_pressed()
        move = 0 # No-op
        if keys[pygame.K_LEFT]:
            move = 3
        elif keys[pygame.K_RIGHT]:
            move = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- Rendering ---
        # The observation is already the rendered frame
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose the observation back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'R' key
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()