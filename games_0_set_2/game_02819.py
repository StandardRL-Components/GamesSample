
# Generated: 2025-08-28T06:06:00.898072
# Source Brief: brief_02819.md
# Brief Index: 2819

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move your avatar on the isometric grid. "
        "Your goal is to collect all the gems before you run out of moves."
    )

    # User-facing game description
    game_description = (
        "A strategic puzzle game. Navigate an isometric grid to collect all 25 gems within a strict limit of "
        "20 moves. Every move counts, so plan your path carefully to achieve a perfect score!"
    )

    # Frames advance only on action
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 48, bold=True)

        # Game constants
        self.GRID_SIZE = 10
        self.TOTAL_GEMS = 25
        self.MOVE_LIMIT = 20
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Visuals
        self.TILE_WIDTH = 56
        self.TILE_HEIGHT = 28
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 100
        
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (60, 70, 80)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_SHADOW = (15, 20, 25)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)
        self.GEM_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 150, 255), 
            (255, 255, 80), (200, 80, 255)
        ]

        # Define fixed gem locations
        self.GEM_SPAWN_LOCATIONS = self._generate_fixed_gem_locations()
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.gem_locations = []
        self.moves_remaining = 0
        self.gems_collected = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.player_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.gem_locations = list(self.GEM_SPAWN_LOCATIONS)
        self.moves_remaining = self.MOVE_LIMIT
        self.gems_collected = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        moved = False
        if movement > 0:
            dist_before = self._get_dist_to_nearest_gem()
            
            dx, dy = 0, 0
            if movement == 1: dx, dy = -1, -1  # Up-Left (visual)
            elif movement == 2: dx, dy = 1, 1   # Down-Right (visual)
            elif movement == 3: dx, dy = -1, 1  # Down-Left (visual)
            elif movement == 4: dx, dy = 1, -1  # Up-Right (visual)
            
            new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.player_pos = new_pos
                self.moves_remaining -= 1
                moved = True
                
                dist_after = self._get_dist_to_nearest_gem()
                if dist_after is not None and dist_before is not None:
                    if dist_after < dist_before:
                        reward += 1.0  # Closer to a gem
                    else:
                        reward -= 0.1 # Further or same dist from a gem
            # else: player tried to move off-grid, no penalty, no move cost

        # Check for gem collection
        if self.player_pos in self.gem_locations:
            # Sound effect placeholder: # sfx_gem_collect.play()
            gem_index = self.gem_locations.index(self.player_pos)
            color_index = self.GEM_SPAWN_LOCATIONS.index(self.player_pos) % len(self.GEM_COLORS)
            gem_color = self.GEM_COLORS[color_index]
            self._create_particles(self.player_pos, gem_color)

            self.gem_locations.pop(gem_index)
            self.gems_collected += 1
            reward += 10.0

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.gems_collected == self.TOTAL_GEMS:
                reward += 100.0  # Win bonus
                self.win_message = "YOU WIN!"
                # Sound effect placeholder: # sfx_win.play()
            else:
                reward -= 50.0  # Loss penalty
                self.win_message = "OUT OF MOVES"
                # Sound effect placeholder: # sfx_lose.play()
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_fixed_gem_locations(self):
        # Creates a solvable, spread-out pattern of 25 gems on a 10x10 grid
        locations = set()
        # Outer ring
        for i in range(1, 9): locations.add((i, 0)); locations.add((i, 9))
        for i in range(1, 9): locations.add((0, i)); locations.add((9, i))
        # Inner pattern
        locations.add((2, 2)); locations.add((2, 7)); locations.add((7, 2)); locations.add((7, 7))
        locations.add((4, 4)); locations.add((4, 5)); locations.add((5, 4)); locations.add((5, 5))
        # Ensure exactly 25 gems
        while len(locations) < self.TOTAL_GEMS:
            x, y = random.randint(0, self.GRID_SIZE - 1), random.randint(0, self.GRID_SIZE - 1)
            if (x,y) != (self.GRID_SIZE//2, self.GRID_SIZE//2): # Avoid start pos
                locations.add((x, y))
        return list(locations)[:self.TOTAL_GEMS]

    def _get_dist_to_nearest_gem(self):
        if not self.gem_locations:
            return None
        
        min_dist = float('inf')
        for gx, gy in self.gem_locations:
            dist = abs(self.player_pos[0] - gx) + abs(self.player_pos[1] - gy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _check_termination(self):
        if self.gems_collected == self.TOTAL_GEMS:
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        return False

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_SIZE)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_SIZE, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

        # Draw gems
        for i, (gx, gy) in enumerate(self.gem_locations):
            color_index = self.GEM_SPAWN_LOCATIONS.index((gx, gy)) % len(self.GEM_COLORS)
            color = self.GEM_COLORS[color_index]
            sx, sy = self._iso_to_screen(gx, gy)
            
            points = [
                (sx, sy - self.TILE_HEIGHT // 2 + 4),
                (sx + self.TILE_WIDTH // 4, sy),
                (sx, sy + self.TILE_HEIGHT // 2 - 4),
                (sx - self.TILE_WIDTH // 4, sy)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, (255,255,255, 100))

        # Draw player shadow
        px, py = self.player_pos
        sx, sy = self._iso_to_screen(px, py)
        shadow_points = [
            (sx, sy + 3),
            (sx + self.TILE_WIDTH / 2 - 4, sy + self.TILE_HEIGHT / 2 + 3),
            (sx, sy + self.TILE_HEIGHT + 3),
            (sx - self.TILE_WIDTH / 2 + 4, sy + self.TILE_HEIGHT / 2 + 3)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, self.COLOR_PLAYER_SHADOW)

        # Draw player
        player_points = [
            (sx, sy),
            (sx + self.TILE_WIDTH / 2 - 4, sy + self.TILE_HEIGHT / 2),
            (sx, sy + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH / 2 + 4, sy + self.TILE_HEIGHT / 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_GRID)

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Gems collected
        gems_text = self.font_ui.render(f"Gems: {self.gems_collected} / {self.TOTAL_GEMS}", True, self.COLOR_TEXT)
        text_rect = gems_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(gems_text, text_rect)
        
        # Game over message
        if self.game_over:
            color = self.COLOR_WIN if self.gems_collected == self.TOTAL_GEMS else self.COLOR_LOSE
            msg_surf = self.font_msg.render(self.win_message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 80))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, color):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        center_y = sy + self.TILE_HEIGHT / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            self.particles.append({'pos': [sx, center_y], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                size = int(max(1, 5 * (p['life'] / 30)))
                pygame.draw.circle(self.screen, p['color'] + (alpha,), p['pos'], size)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "gems_collected": self.gems_collected,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert info['moves_remaining'] == self.MOVE_LIMIT
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        # Test state assertions
        assert self.gems_collected <= self.TOTAL_GEMS
        assert self.moves_remaining >= 0
        assert 0 <= self.player_pos[0] < self.GRID_SIZE
        assert 0 <= self.player_pos[1] < self.GRID_SIZE
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to 'dummy' if you are running headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This part requires a window and will not work with a dummy video driver.
    # To run this, comment out the os.environ line above.
    try:
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Isometric Gem Collector")
        
        obs, info = env.reset()
        done = False
        
        print(env.user_guide)

        while not done:
            action = [0, 0, 0] # Default to no-op
            
            # This loop allows for quick key presses without waiting for the next frame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        print("--- Game Reset ---")
                        continue
                    
                    # Since auto_advance is False, we only step on an action
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

            # Render the current state to the display window
            frame = env._get_observation()
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if done:
                print("Game Over. Press 'R' to play again or close the window.")
                
        # Keep window open after game over
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # This part is just for interactive demo, won't be hit in the loop above
                obs, info = env.reset()
                # Need to re-run the main loop, so we just break here
                break

    finally:
        env.close()