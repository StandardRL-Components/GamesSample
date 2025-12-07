
# Generated: 2025-08-28T01:23:33.326051
# Source Brief: brief_04094.md
# Brief Index: 4094

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your player (blue square) on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid to collect 10 green gems while avoiding red traps. Each move is a strategic choice."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_SIZE = (15, 15)
        self.TARGET_GEMS = 10
        self.NUM_TRAPS = 15
        self.MAX_STEPS = 1000

        # Visual constants
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16
        self.OFFSET_X = 640 // 2
        self.OFFSET_Y = 60

        # Colors
        self.COLOR_BG = (15, 18, 26)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (50, 150, 255, 50)
        self.COLOR_GEM = (0, 255, 150)
        self.COLOR_GEM_GLOW = (0, 255, 150, 60)
        self.COLOR_TRAP = (255, 50, 100)
        self.COLOR_TRAP_GLOW = (255, 50, 100, 70)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.gems = []
        self.traps = []
        self.gems_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gems_collected = 0
        
        # Generate level
        self._generate_level()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Generates a new layout of gems and traps."""
        grid_w, grid_h = self.GRID_SIZE
        all_coords = [(x, y) for x in range(grid_w) for y in range(grid_h)]
        self.np_random.shuffle(all_coords)

        # Player starts near the center
        self.player_pos = [grid_w // 2, grid_h // 2]
        if tuple(self.player_pos) in all_coords:
            all_coords.remove(tuple(self.player_pos))

        # Place traps
        self.traps = []
        for _ in range(self.NUM_TRAPS):
            if all_coords:
                self.traps.append(list(all_coords.pop(0)))

        # Place gems
        self.gems = []
        for _ in range(self.TARGET_GEMS):
            if all_coords:
                self.gems.append(list(all_coords.pop(0)))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        old_pos = list(self.player_pos)
        dist_before = self._get_dist_to_closest_gem()

        # Update player position based on action
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1

        # Clamp player position to grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_SIZE[1] - 1)
        
        moved = old_pos != self.player_pos

        # Proximity reward
        if moved:
            dist_after = self._get_dist_to_closest_gem()
            if dist_after < dist_before:
                reward += 1.0  # Moved closer to a gem
            elif dist_after > dist_before:
                reward -= 0.1 # Moved away from a gem
        
        # Check for events
        if self.player_pos in self.gems:
            # Gem collected
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            reward += 10.0
            # sfx: gem_collect.wav
        elif self.player_pos in self.traps:
            # Hit a trap
            reward -= 100.0
            self.game_over = True
            # sfx: trap_spring.wav

        self.steps += 1
        self.score += reward
        
        # Check termination conditions
        terminated = self.game_over
        if self.gems_collected >= self.TARGET_GEMS:
            reward += 100.0
            self.score += 100.0
            terminated = True
            # sfx: level_complete.wav
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_dist_to_closest_gem(self):
        if not self.gems:
            return float('inf')
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(player_x - gx) + abs(player_y - gy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen pixel coordinates."""
        screen_x = self.OFFSET_X + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.OFFSET_Y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_text(self, text, font, color, pos, shadow_color=None, shadow_offset=(1, 1)):
        text_surface = font.render(text, True, color)
        if shadow_color:
            shadow_surface = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surface, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
        self.screen.blit(text_surface, pos)

    def _draw_iso_tile(self, pos, color):
        """Draws a rhombus for an isometric tile."""
        x, y = self._iso_to_screen(pos[0], pos[1])
        points = [
            (x, y - self.TILE_HEIGHT // 2),
            (x + self.TILE_WIDTH // 2, y),
            (x, y + self.TILE_HEIGHT // 2),
            (x - self.TILE_WIDTH // 2, y)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_SIZE[0] + 1):
            start = self._iso_to_screen(x - 0.5, -0.5)
            end = self._iso_to_screen(x - 0.5, self.GRID_SIZE[1] - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_SIZE[1] + 1):
            start = self._iso_to_screen(-0.5, y - 0.5)
            end = self._iso_to_screen(self.GRID_SIZE[0] - 0.5, y - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw traps
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        trap_radius = int(self.TILE_WIDTH * 0.2 * (0.8 + pulse * 0.4))
        for trap_pos in self.traps:
            sx, sy = self._iso_to_screen(trap_pos[0], trap_pos[1])
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, trap_radius, self.COLOR_TRAP)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, trap_radius, self.COLOR_TRAP)
            
            # Glow
            glow_radius = int(trap_radius * (1.5 + pulse * 0.5))
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_TRAP_GLOW)


        # Draw gems
        sparkle = (math.sin(self.steps * 0.3) + 1) / 2
        gem_size = int(self.TILE_WIDTH * 0.3 * (0.9 + sparkle * 0.2))
        for gem_pos in self.gems:
            sx, sy = self._iso_to_screen(gem_pos[0], gem_pos[1])
            points = [
                (sx, sy - gem_size), (sx + gem_size, sy),
                (sx, sy + gem_size), (sx - gem_size, sy)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            
            # Glow
            glow_radius = int(gem_size * (1.8 + sparkle * 0.4))
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, glow_radius, self.COLOR_GEM_GLOW)


        # Draw player
        player_size = int(self.TILE_WIDTH * 0.3)
        px, py = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        player_rect = pygame.Rect(px - player_size, py - player_size, player_size * 2, player_size * 2)
        
        # Player Glow
        glow_radius = int(player_size * 2.5)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)

        # Player Body
        self._draw_iso_tile(self.player_pos, self.COLOR_PLAYER)

    def _render_ui(self):
        # Gem counter
        gem_text = f"Gems: {self.gems_collected} / {self.TARGET_GEMS}"
        self._draw_text(gem_text, self.font_main, self.COLOR_TEXT, (15, 15), self.COLOR_TEXT_SHADOW)
        
        # Step counter
        step_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        self._draw_text(step_text, self.font_small, self.COLOR_TEXT, (640 - 150, 15), self.COLOR_TEXT_SHADOW)

        # Game over message
        if self.game_over:
            if self.gems_collected >= self.TARGET_GEMS:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            text_surface = self.font_main.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(640 // 2, 400 // 2))
            
            shadow_surface = self.font_main.render(msg, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surface.get_rect(center=(640 // 2 + 2, 400 // 2 + 2))
            
            self.screen.blit(shadow_surface, shadow_rect)
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    # Note: Gymnasium's 'human' render mode is not used here.
    # We are manually updating a pygame window.
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Isometric Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)
    
    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if terminated:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    continue

                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # If a move key was pressed, step the environment
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                    if terminated:
                        print("Episode finished. Press 'R' to reset.")

        # Draw the observation from the environment to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()