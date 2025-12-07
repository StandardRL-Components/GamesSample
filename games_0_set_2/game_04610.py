
# Generated: 2025-08-28T02:54:44.836593
# Source Brief: brief_04610.md
# Brief Index: 4610

        
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

    # Short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character on the grid."
    )

    # Short, user-facing description of the game:
    game_description = (
        "Navigate a 10x10 grid to collect all 20 green gems while avoiding the red traps. Each move counts!"
    )

    # Frames advance only when an action is received.
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 1000
        self.NUM_GEMS = 20
        self.NUM_TRAPS = 5

        # Calculate grid rendering properties
        self.CELL_SIZE = min((self.HEIGHT - 80) // self.GRID_SIZE, (self.WIDTH - 40) // self.GRID_SIZE)
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.MARGIN_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.MARGIN_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (100, 220, 255, 50)
        self.COLOR_GEM = (0, 255, 150)
        self.COLOR_GEM_GLOW = (100, 255, 200, 60)
        self.COLOR_TRAP = (255, 50, 100)
        self.COLOR_TRAP_GLOW = (255, 100, 150, 70)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_WIN = (255, 223, 0)
        self.COLOR_LOSE = self.COLOR_TRAP

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
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_large = pygame.font.SysFont(None, 52)

        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win_condition = False
        
        self.player_pos = (0, 0)
        self.gems = []
        self.traps = []
        self.player_trail = []

        self.particles = []

        # --- Initialize State ---
        self.reset()

        # --- Self-Validation ---
        # self.validate_implementation() # Uncomment for debugging

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates for rendering."""
        x = self.MARGIN_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.MARGIN_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.win_condition = False
        self.player_trail.clear()
        self.particles.clear()
        
        # Generate item positions
        all_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_coords)
        
        self.player_pos = all_coords.pop(0)
        self.player_trail.append(self._grid_to_pixel(self.player_pos))
        
        self.traps = [all_coords.pop(0) for _ in range(self.NUM_TRAPS)]
        self.gems = [all_coords.pop(0) for _ in range(self.NUM_GEMS)]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0
        terminated = False

        old_pos = self.player_pos
        new_pos = list(old_pos)

        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1  # Down
        elif movement == 3: new_pos[0] -= 1  # Left
        elif movement == 4: new_pos[0] += 1  # Right
        
        # Clamp to grid boundaries
        new_pos[0] = max(0, min(self.GRID_SIZE - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.GRID_SIZE - 1, new_pos[1]))
        self.player_pos = tuple(new_pos)

        # --- Reward Calculation ---
        if self.gems:
            dist_to_gem_old = min([math.dist(old_pos, g) for g in self.gems])
            dist_to_gem_new = min([math.dist(self.player_pos, g) for g in self.gems])
            if dist_to_gem_new < dist_to_gem_old:
                reward += 1.0  # Moved closer to a gem
            elif dist_to_gem_new > dist_to_gem_old:
                reward -= 0.2 # Moved away from a gem
        
        if self.traps:
            dist_to_trap_old = min([math.dist(old_pos, t) for t in self.traps])
            dist_to_trap_new = min([math.dist(self.player_pos, t) for t in self.traps])
            if dist_to_trap_new < dist_to_trap_old:
                reward -= 0.1 # Moved closer to a trap

        # --- Event Handling ---
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            self.score += 10
            reward += 10
            # SFX: Gem collect sound
            self._spawn_particles(self._grid_to_pixel(self.player_pos), self.COLOR_GEM, 20)

        if self.player_pos in self.traps:
            self.score -= 100
            reward = -100
            self.game_over = True
            terminated = True
            self.win_condition = False
            # SFX: Explosion/Fail sound
            self._spawn_particles(self._grid_to_pixel(self.player_pos), self.COLOR_TRAP, 40, 5)

        # --- Termination Checks ---
        if self.gems_collected == self.NUM_GEMS:
            self.score += 100
            reward += 100
            self.game_over = True
            terminated = True
            self.win_condition = True
            # SFX: Win jingle

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        # Update player trail
        if old_pos != self.player_pos:
            self.player_trail.append(self._grid_to_pixel(self.player_pos))
            if len(self.player_trail) > 5:
                self.player_trail.pop(0)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_x = self.MARGIN_X + i * self.CELL_SIZE
            start_y, end_y = self.MARGIN_Y, self.MARGIN_Y + self.GRID_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (start_x, end_y))
            # Horizontal
            start_y = self.MARGIN_Y + i * self.CELL_SIZE
            start_x, end_x = self.MARGIN_X, self.MARGIN_X + self.GRID_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, start_y))

        # Draw traps
        for trap_pos in self.traps:
            px, py = self._grid_to_pixel(trap_pos)
            size = self.CELL_SIZE * 0.7
            glow_size = size * 1.5
            
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_TRAP_GLOW, glow_surf.get_rect(), border_radius=int(glow_size*0.2))
            self.screen.blit(glow_surf, (px - glow_size/2, py - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.draw.rect(self.screen, self.COLOR_TRAP, (px - size/2, py - size/2, size, size), border_radius=int(size*0.2))

        # Draw gems
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # 0 to 1
        for gem_pos in self.gems:
            px, py = self._grid_to_pixel(gem_pos)
            size = self.CELL_SIZE * 0.35 + pulse * self.CELL_SIZE * 0.1
            glow_size = size * 3.0
            
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, int(glow_size/2), int(glow_size/2), int(glow_size/2), self.COLOR_GEM_GLOW)
            self.screen.blit(glow_surf, (px - glow_size/2, py - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aacircle(self.screen, px, py, int(size), self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(size), self.COLOR_GEM)

        # Draw player trail
        for i, pos in enumerate(self.player_trail[:-1]):
            alpha = (i + 1) / len(self.player_trail) * 100
            trail_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            size = self.CELL_SIZE * 0.6
            pygame.draw.rect(trail_surf, (*self.COLOR_PLAYER, alpha), (self.CELL_SIZE/2 - size/2, self.CELL_SIZE/2 - size/2, size, size), border_radius=int(size*0.2))
            self.screen.blit(trail_surf, (pos[0] - self.CELL_SIZE/2, pos[1] - self.CELL_SIZE/2))
            
        # Draw player
        px, py = self._grid_to_pixel(self.player_pos)
        size = self.CELL_SIZE * 0.6
        glow_size = size * 2.5
        
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=int(glow_size*0.3))
        self.screen.blit(glow_surf, (px - glow_size/2, py - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - size/2, py - size/2, size, size), border_radius=int(size*0.2))

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        gems_text = self.font_main.render(f"GEMS: {self.gems_collected} / {self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gems_text, (self.WIDTH - gems_text.get_width() - 20, 15))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition:
                end_text_str = "YOU WIN!"
                end_color = self.COLOR_WIN
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_LOSE
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, pos, color, count, speed=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = self.np_random.uniform(0.5, 1) * speed
            vx = math.cos(angle) * vel
            vy = math.sin(angle) * vel
            life = self.np_random.integers(20, 40)
            size = self.np_random.uniform(2, 5)
            self.particles.append([list(pos), [vx, vy], life, size, color])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]  # pos.x += vel.x
            p[0][1] += p[1][1]  # pos.y += vel.y
            p[2] -= 1  # life -= 1
            
            if p[2] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = max(0, min(255, int(p[2] * (255 / 30))))
            color = (*p[4], alpha)
            
            temp_surf = pygame.Surface((p[3]*2, p[3]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p[3], p[3]), p[3])
            self.screen.blit(temp_surf, (p[0][0] - p[3], p[0][1] - p[3]), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                if terminated: continue

                # Map keys to actions
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                
                # Since auto_advance is False, we only step on keydown
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for the player window

    env.close()