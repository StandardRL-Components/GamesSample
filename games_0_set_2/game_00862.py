
# Generated: 2025-08-27T15:00:23.046881
# Source Brief: brief_00862.md
# Brief Index: 862

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move your character on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect all the gems in an isometric 2D world while avoiding deadly traps before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.NUM_GEMS = 15
        self.NUM_TRAPS = 15
        self.TIME_LIMIT = 600

        # Visual constants
        self.TILE_W, self.TILE_H = 30, 15
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 50

        # Colors
        self.COLOR_BG = (52, 78, 65) # Muted Green
        self.COLOR_GRID = (42, 62, 52)
        self.COLOR_PLAYER = (0, 166, 251)
        self.COLOR_PLAYER_GLOW = (173, 216, 230)
        self.COLOR_GEM = (255, 214, 10)
        self.COLOR_GEM_GLOW = (255, 255, 255)
        self.COLOR_TRAP = (230, 57, 70)
        self.COLOR_TRAP_GLOW = (168, 38, 49)
        self.COLOR_TEXT = (241, 250, 238)

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Game state variables (initialized in reset)
        self.player_pos = None
        self.gems = None
        self.traps = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.steps_since_last_gem = None
        self.particles = []
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.steps_since_last_gem = 0
        self.particles = []

        # Generate level layout
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        self.player_pos = list(all_coords.pop())
        
        gem_coords = [all_coords.pop() for _ in range(self.NUM_GEMS)]
        self.gems = [list(pos) for pos in gem_coords]
        
        trap_coords = [all_coords.pop() for _ in range(self.NUM_TRAPS)]
        self.traps = [list(pos) for pos in trap_coords]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Update game logic
        self.steps += 1
        self.steps_since_last_gem += 1
        reward = 0.1  # Base reward for surviving a step

        # Apply movement
        if movement == 1: # Up (grid coords)
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0]))
        self.player_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1]))

        # Check for gem collection
        if self.player_pos in self.gems:
            # sfx: gem_collect.wav
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10
            self._create_particles(self.player_pos, self.COLOR_GEM, 20)
            
            # Check for risky collection bonus
            is_risky = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    neighbor = [self.player_pos[0] + dx, self.player_pos[1] + dy]
                    if neighbor in self.traps:
                        is_risky = True
                        break
                if is_risky: break
            
            if is_risky:
                self.score += 5
                reward += 5

            self.steps_since_last_gem = 0

        # Penalty for not collecting gems
        if self.steps_since_last_gem > 5:
            reward -= 0.2

        # Check for termination conditions
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if len(self.gems) == 0: # Win condition
                # sfx: win_fanfare.wav
                self.score += 100
                reward = 100
            elif self.player_pos in self.traps: # Trap death
                # sfx: player_death.wav
                reward = -50
            elif self.steps >= self.TIME_LIMIT: # Timeout
                # sfx: timeout_buzzer.wav
                reward = -10
        
        self.score += reward # Add step reward to score (for display)
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if self.player_pos in self.traps:
            self.game_over = True
            return True
        if len(self.gems) == 0:
            self.game_over = True
            return True
        if self.steps >= self.TIME_LIMIT:
            self.game_over = True
            return True
        return False
        
    def _iso_to_screen(self, gx, gy):
        """Converts grid coordinates to screen coordinates."""
        sx = self.ORIGIN_X + (gx - gy) * (self.TILE_W / 2)
        sy = self.ORIGIN_Y + (gx + gy) * (self.TILE_H / 2)
        return int(sx), int(sy)

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
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Draw traps
        for trap_pos in self.traps:
            sx, sy = self._iso_to_screen(trap_pos[0], trap_pos[1])
            center = (sx, sy + self.TILE_H // 2)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 8, self.COLOR_TRAP_GLOW)
            # Main shape
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 6, self.COLOR_TRAP)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], 6, self.COLOR_TRAP)

        # Draw gems
        for gem_pos in self.gems:
            sx, sy = self._iso_to_screen(gem_pos[0], gem_pos[1])
            center = (sx, sy + self.TILE_H // 2)
            points = [
                (center[0], center[1] - 8),
                (center[0] + 6, center[1]),
                (center[0], center[1] + 8),
                (center[0] - 6, center[1])
            ]
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], 10, self.COLOR_GEM_GLOW)
            # Main shape
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)

        # Update and draw particles
        self._update_particles()
            
        # Draw player
        px, py = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
        player_center = (px, py + self.TILE_H // 2)
        points = [
            (player_center[0], player_center[1] - 10),
            (player_center[0] + 8, player_center[1]),
            (player_center[0], player_center[1] + 10),
            (player_center[0] - 8, player_center[1])
        ]
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], 14, self.COLOR_PLAYER_GLOW)
        # Main shape
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Time remaining
        time_left = self.TIME_LIMIT - self.steps
        time_text = self.font_small.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 45))
        
        # Gems remaining
        gems_left = len(self.gems)
        gems_text = self.font_small.render(f"Gems Left: {gems_left}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (10, 70))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if len(self.gems) == 0:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _create_particles(self, grid_pos, color, count):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        center = (sx, sy + self.TILE_H // 2)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(center), 'vel': vel, 'life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, color)
                
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_left": len(self.gems),
            "player_pos": self.player_pos,
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
    # This block allows you to play the game with keyboard controls
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and Shift are unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause for 2 seconds before restarting
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(10) # Control game speed for human play
        
    env.close()