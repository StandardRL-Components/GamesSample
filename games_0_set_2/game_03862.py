
# Generated: 2025-08-28T00:40:47.857684
# Source Brief: brief_03862.md
# Brief Index: 3862

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Press space to collect adjacent gems."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid to collect 20 gems while dodging enemies. Get bonus points for risky grabs near enemies."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.CELL_SIZE

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
        
        # Fonts and Colors
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (100, 200, 255)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_GEM_GLOW = (255, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 120, 120)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEART = (220, 20, 60)

        # Game parameters
        self.INITIAL_LIVES = 3
        self.GEMS_TO_WIN = 20
        self.INITIAL_ENEMY_COUNT = 5
        self.MAX_STEPS = 1000
        self.INITIAL_GEM_COUNT = 5
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.gems = []
        self.enemies = []
        self.particles = []
        self.score = 0
        self.gems_collected = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.total_gems_ever_collected = 0
        self.enemy_speed_multiplier = 1.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_lives = self.INITIAL_LIVES
        self.gems_collected = 0
        self.win_message = ""
        self.particles = []

        # Reset progression-related state
        self.total_gems_ever_collected = 0
        self.enemy_speed_multiplier = 1.0

        # Place player in the center
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Spawn initial entities
        self.gems = []
        self.enemies = self._spawn_enemies(self.INITIAL_ENEMY_COUNT)
        
        # Ensure no initial overlap
        occupied_coords = {tuple(e['pos']) for e in self.enemies}
        occupied_coords.add(tuple(self.player_pos))

        for _ in range(self.INITIAL_GEM_COUNT):
            self._spawn_gem(occupied_coords)
            
        return self._get_observation(), self._get_info()

    def _spawn_enemies(self, count):
        enemies = []
        for i in range(count):
            path_type = self.np_random.integers(0, 3)
            w = self.np_random.integers(5, self.GRID_WIDTH // 2)
            h = self.np_random.integers(5, self.GRID_HEIGHT // 2)
            x = self.np_random.integers(1, self.GRID_WIDTH - w - 1)
            y = self.np_random.integers(1, self.GRID_HEIGHT - h - 1)
            
            if path_type == 0: # Rectangle
                path = self._generate_rect_path(x, y, w, h)
            elif path_type == 1: # Horizontal line
                path = self._generate_line_path(x, y, w, 'horizontal')
            else: # Vertical line
                path = self._generate_line_path(x, y, h, 'vertical')
            
            if not path: continue

            start_index = self.np_random.integers(0, len(path))
            enemies.append({
                'pos': path[start_index],
                'path': path,
                'path_index': start_index,
                'speed_accumulator': self.np_random.random() # Stagger start
            })
        return enemies

    def _generate_rect_path(self, x, y, w, h):
        path = []
        for i in range(w): path.append([x + i, y])
        for i in range(h): path.append([x + w, y + i])
        for i in range(w): path.append([x + w - i, y + h])
        for i in range(h): path.append([x, y + h - i])
        return path

    def _generate_line_path(self, x, y, length, direction):
        path = []
        if direction == 'horizontal':
            for i in range(length): path.append([x + i, y])
            for i in range(length): path.append([x + length - 1 - i, y])
        else: # vertical
            for i in range(length): path.append([x, y + i])
            for i in range(length): path.append([x, y + length - 1 - i])
        return path

    def _spawn_gem(self, occupied_coords):
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_coords:
                self.gems.append(pos)
                occupied_coords.add(tuple(pos))
                break
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for each step to encourage efficiency

        # 1. Unpack action and move player
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up
        elif movement == 2: py += 1  # Down
        elif movement == 3: px -= 1  # Left
        elif movement == 4: px += 1  # Right

        self.player_pos[0] = np.clip(px, 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(py, 0, self.GRID_HEIGHT - 1)

        # 2. Gem collection
        if space_held:
            collected_gem_pos = None
            for gem_pos in self.gems[:]:
                dist = abs(self.player_pos[0] - gem_pos[0]) + abs(self.player_pos[1] - gem_pos[1])
                if dist == 1: # Adjacent
                    collected_gem_pos = gem_pos
                    self.gems.remove(gem_pos)
                    self.gems_collected += 1
                    self.total_gems_ever_collected += 1
                    self.score += 10
                    reward += 10
                    # Placeholder: play gem collect sound
                    self._create_particles(gem_pos, self.COLOR_GEM, 15)
                    
                    # Check for risk bonus
                    is_risky = False
                    for enemy in self.enemies:
                        enemy_dist = abs(enemy['pos'][0] - gem_pos[0]) + abs(enemy['pos'][1] - gem_pos[1])
                        if enemy_dist <= 1:
                            is_risky = True
                            break
                    if is_risky:
                        self.score += 2
                        reward += 2

                    # Update enemy speed
                    if self.total_gems_ever_collected > 0 and self.total_gems_ever_collected % 20 == 0:
                        self.enemy_speed_multiplier = min(2.0, self.enemy_speed_multiplier + 0.02)

                    break # Only collect one gem per step
            
            if collected_gem_pos:
                occupied_coords = {tuple(e['pos']) for e in self.enemies}
                occupied_coords.add(tuple(self.player_pos))
                occupied_coords.update({tuple(g) for g in self.gems})
                self._spawn_gem(occupied_coords)

        # 3. Update particles
        self._update_particles()

        # 4. Move enemies and check for collision
        player_hit = False
        for enemy in self.enemies:
            enemy['speed_accumulator'] += self.enemy_speed_multiplier
            moves_to_make = int(enemy['speed_accumulator'])
            if moves_to_make > 0:
                for _ in range(moves_to_make):
                    if not enemy['path']: continue
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                    enemy['pos'] = enemy['path'][enemy['path_index']]
                    if enemy['pos'] == self.player_pos:
                        player_hit = True
                        break # Stop multi-move on hit
                enemy['speed_accumulator'] -= moves_to_make
            if player_hit:
                break
        
        if player_hit:
            self.player_lives -= 1
            reward -= 5
            self.score -= 5
            self._create_particles(self.player_pos, self.COLOR_PLAYER_GLOW, 20)
            # Placeholder: play hit sound
            # Reset player to center to avoid death loop
            self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]


        # 5. Update game state and check for termination
        self.steps += 1
        terminated = False
        
        if self.gems_collected >= self.GEMS_TO_WIN:
            terminated = True
            reward += 100
            self.score += 100
            self.win_message = "YOU WIN!"
        elif self.player_lives <= 0:
            terminated = True
            reward -= 100
            self.score -= 100
            self.win_message = "GAME OVER"
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_particles(self, grid_pos, color, count):
        px, py = (grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'color': color,
                'radius': self.np_random.uniform(1, 4)
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw gems
        for gx, gy in self.gems:
            center_x = int((gx + 0.5) * self.CELL_SIZE)
            center_y = int((gy + 0.5) * self.CELL_SIZE)
            radius = self.CELL_SIZE * 0.35
            glow_radius = radius * (1.2 + 0.2 * math.sin(self.steps * 0.2))
            
            points = [
                (center_x, center_y - radius),
                (center_x + radius, center_y),
                (center_x, center_y + radius),
                (center_x - radius, center_y),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(glow_radius), self.COLOR_GEM_GLOW + (50,))

        # Draw enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            center_x = int((ex + 0.5) * self.CELL_SIZE)
            center_y = int((ey + 0.5) * self.CELL_SIZE)
            radius = self.CELL_SIZE * 0.4
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(radius), self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(radius), self.COLOR_ENEMY)
            
            glow_radius = int(radius * (1.1 + 0.1 * math.sin(self.steps * 0.15 + id(enemy))))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, self.COLOR_ENEMY_GLOW + (60,))

        # Draw player
        px, py = self.player_pos
        center_x = int((px + 0.5) * self.CELL_SIZE)
        center_y = int((py + 0.5) * self.CELL_SIZE)
        size = self.CELL_SIZE * 0.8
        glow_size = size * (1.2 + 0.15 * math.sin(self.steps * 0.25))

        player_rect = pygame.Rect(center_x - size/2, center_y - size/2, size, size)
        glow_rect = pygame.Rect(center_x - glow_size/2, center_y - glow_size/2, glow_size, glow_size)
        
        # Draw glow
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PLAYER_GLOW + (80,), s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = p['color'] + (max(0, min(255, alpha)),)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Score and Gems
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        gems_text = self.font_main.render(f"GEMS: {self.gems_collected}/{self.GEMS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))
        self.screen.blit(gems_text, (10, 30))

        # Lives (Hearts)
        for i in range(self.player_lives):
            self._draw_heart(self.SCREEN_WIDTH - 30 - (i * 35), 25, 15)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _draw_heart(self, x, y, size):
        # Programmatically draw a heart shape
        points = [
            (x, y - size * 0.3),
            (x + size * 0.5, y - size * 0.8),
            (x + size, y - size * 0.3),
            (x, y + size * 0.7),
            (x - size, y - size * 0.3),
            (x - size * 0.5, y - size * 0.8),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_HEART)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_HEART)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
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
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            # Add a small delay before restarting
            pygame.time.wait(2000)

        clock.tick(10) # Control game speed for human play
        
    env.close()