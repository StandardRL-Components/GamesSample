
# Generated: 2025-08-28T04:00:53.872952
# Source Brief: brief_05116.md
# Brief Index: 5116

        
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
        "Controls: Arrow keys to move. Collect yellow gems and avoid the red enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect gems while dodging enemies in a fast-paced, top-down arcade environment. "
        "Collecting gems increases enemy speed. Getting too close to enemies is risky but can lead to faster collection."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.SCREEN_WIDTH // self.GRID_SIZE
        self.GRID_H = self.SCREEN_HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1500 # 50 seconds at 30fps
        self.WIN_GEM_COUNT = 10
        self.MAX_HEALTH = 3
        self.INITIAL_ENEMY_SPEED = 0.05
        self.ENEMY_SPEED_INCREASE = 0.01
        self.INVINCIBILITY_FRAMES = 60 # 2 seconds

        # Colors
        self.COLOR_BG = (15, 23, 42)
        self.COLOR_GRID = (30, 41, 59)
        self.COLOR_PLAYER = (74, 222, 128)
        self.COLOR_PLAYER_GLOW = (34, 197, 94)
        self.COLOR_GEM = (250, 204, 21)
        self.COLOR_GEM_GLOW = (234, 179, 8)
        self.COLOR_ENEMY = (239, 68, 68)
        self.COLOR_ENEMY_GLOW = (220, 38, 38)
        self.COLOR_TEXT = (226, 232, 240)
        self.COLOR_HEART = (244, 63, 94)

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        
        # Initialize state variables to avoid attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = [0, 0]
        self.player_health = 0
        self.gems_collected = 0
        self.gems = []
        self.enemies = []
        self.enemy_speed = 0.0
        self.particles = []
        self.invincibility_timer = 0
        self.last_dist_to_gem = float('inf')
        self.last_dist_to_enemy = float('inf')
        
        # Initialize state
        self.reset()

        # Run validation check
        # self.validate_implementation() # Comment out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_health = self.MAX_HEALTH
        self.gems_collected = 0
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        self.invincibility_timer = 0
        self.particles = []
        
        self.player_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.enemies = []
        self._spawn_enemies(4)

        self.gems = []
        for _ in range(3):
            self._spawn_gem()

        self.last_dist_to_gem = self._get_closest_distance(self.player_pos, self.gems)
        self.last_dist_to_enemy = self._get_closest_distance(self.player_pos, [e['pos'] for e in self.enemies])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Pre-update calculations for reward ---
        reward = 0
        
        # --- Update game logic ---
        self.steps += 1
        
        # Update timers and particles
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        self._update_particles()
        
        # Move player
        self._move_player(movement)
        
        # Move enemies
        self._move_enemies()
        
        # --- Continuous rewards ---
        current_dist_to_gem = self._get_closest_distance(self.player_pos, self.gems)
        if current_dist_to_gem < self.last_dist_to_gem:
            reward += 1.0
        self.last_dist_to_gem = current_dist_to_gem

        current_dist_to_enemy = self._get_closest_distance(self.player_pos, [e['pos'] for e in self.enemies])
        if current_dist_to_enemy < self.last_dist_to_enemy:
            reward -= 0.1
        self.last_dist_to_enemy = current_dist_to_enemy

        # --- Handle collisions and event-based rewards ---
        # Gem collection
        collided_gem_index = self._check_collision(self.player_pos, self.gems)
        if collided_gem_index is not None:
            # SFX: Gem collect sound
            reward += 10
            self.score += 10
            self.gems_collected += 1
            
            gem_pos_pixels = (self.gems[collided_gem_index][0] * self.GRID_SIZE + self.GRID_SIZE // 2, self.gems[collided_gem_index][1] * self.GRID_SIZE + self.GRID_SIZE // 2)
            self._create_particles(gem_pos_pixels, self.COLOR_GEM, 20)

            self.gems.pop(collided_gem_index)
            self._spawn_gem()
            
            if self.gems_collected > 0 and self.gems_collected % 2 == 0:
                self.enemy_speed += self.ENEMY_SPEED_INCREASE

        # Enemy collision
        enemy_positions = [e['pos'] for e in self.enemies]
        if self.invincibility_timer == 0 and self._check_collision(self.player_pos, enemy_positions) is not None:
            # SFX: Player hit sound
            reward -= 30
            self.player_health -= 1
            self.invincibility_timer = self.INVINCIBILITY_FRAMES
            
            player_pos_pixels = (self.player_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, self.player_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
            self._create_particles(player_pos_pixels, self.COLOR_ENEMY, 30)

        # --- Check termination conditions ---
        terminated = False
        if self.gems_collected >= self.WIN_GEM_COUNT:
            # SFX: Win sound
            reward += 100
            self.score += 100
            terminated = True
        elif self.player_health <= 0:
            # SFX: Lose sound
            reward -= 100
            terminated = True
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

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

        # Draw enemies
        for enemy in self.enemies:
            pos_px = (int(enemy['pos'][0] * self.GRID_SIZE) + self.GRID_SIZE // 2, int(enemy['pos'][1] * self.GRID_SIZE) + self.GRID_SIZE // 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE // 2 + 3, self.COLOR_ENEMY_GLOW + (100,))
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE // 2, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE // 2, self.COLOR_ENEMY)

        # Draw gems
        pulse = abs(math.sin(self.steps * 0.1)) * 3
        for gem_pos in self.gems:
            pos_px = (gem_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, gem_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], int(self.GRID_SIZE // 2 + pulse), self.COLOR_GEM_GLOW + (150,))
            pygame.draw.rect(self.screen, self.COLOR_GEM, (pos_px[0] - self.GRID_SIZE//3, pos_px[1] - self.GRID_SIZE//3, self.GRID_SIZE*2/3, self.GRID_SIZE*2/3))
        
        # Draw player
        pos_px = (self.player_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, self.player_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)
        is_invincible_flash = self.invincibility_timer > 0 and (self.steps // 3) % 2 == 0
        if not is_invincible_flash:
            pygame.gfxdraw.filled_circle(self.screen, pos_px[0], pos_px[1], self.GRID_SIZE // 2 + 5, self.COLOR_PLAYER_GLOW + (120,))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (pos_px[0] - self.GRID_SIZE//2, pos_px[1] - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Gems collected
        gem_text = self.font_small.render(f"GEMS: {self.gems_collected}/{self.WIN_GEM_COUNT}", True, self.COLOR_GEM)
        self.screen.blit(gem_text, (10, 35))

        # Health
        for i in range(self.MAX_HEALTH):
            heart_pos = (self.SCREEN_WIDTH - 30 - i * 25, 15)
            if i < self.player_health:
                color = self.COLOR_HEART
                pygame.gfxdraw.filled_circle(self.screen, heart_pos[0], heart_pos[1], 8, color)
                pygame.gfxdraw.filled_circle(self.screen, heart_pos[0] + 8, heart_pos[1], 8, color)
                pygame.gfxdraw.filled_polygon(self.screen, [(heart_pos[0]-8, heart_pos[1]), (heart_pos[0]+16, heart_pos[1]), (heart_pos[0]+4, heart_pos[1]+14)], color)
            else:
                color = self.COLOR_GRID
                pygame.gfxdraw.aacircle(self.screen, heart_pos[0], heart_pos[1], 8, color)
                pygame.gfxdraw.aacircle(self.screen, heart_pos[0] + 8, heart_pos[1], 8, color)
                pygame.draw.line(self.screen, color, (heart_pos[0]-8, heart_pos[1]), (heart_pos[0]+4, heart_pos[1]+14), 1)
                pygame.draw.line(self.screen, color, (heart_pos[0]+16, heart_pos[1]), (heart_pos[0]+4, heart_pos[1]+14), 1)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.gems_collected >= self.WIN_GEM_COUNT else "GAME OVER"
            end_text = self.font.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "gems_collected": self.gems_collected,
        }

    def _move_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp to screen bounds
        self.player_pos[0] = max(0, min(self.GRID_W - 1, self.player_pos[0]))
        self.player_pos[1] = max(0, min(self.GRID_H - 1, self.player_pos[1]))

    def _move_enemies(self):
        for enemy in self.enemies:
            target_pos = enemy['path'][enemy['path_index']]
            current_pos = enemy['pos']
            
            direction = [target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]]
            dist = math.sqrt(direction[0]**2 + direction[1]**2)

            if dist < self.enemy_speed:
                enemy['pos'] = list(target_pos)
                enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
            else:
                enemy['pos'][0] += (direction[0] / dist) * self.enemy_speed
                enemy['pos'][1] += (direction[1] / dist) * self.enemy_speed
    
    def _spawn_enemies(self, count):
        for i in range(count):
            quadrant = i % 4
            if quadrant == 0: # Top-left
                center_x, center_y = self.GRID_W // 4, self.GRID_H // 4
            elif quadrant == 1: # Top-right
                center_x, center_y = self.GRID_W * 3 // 4, self.GRID_H // 4
            elif quadrant == 2: # Bottom-left
                center_x, center_y = self.GRID_W // 4, self.GRID_H * 3 // 4
            else: # Bottom-right
                center_x, center_y = self.GRID_W * 3 // 4, self.GRID_H * 3 // 4
            
            path_size = self.np_random.integers(2, 5)
            path = [
                (center_x - path_size, center_y - path_size),
                (center_x + path_size, center_y - path_size),
                (center_x + path_size, center_y + path_size),
                (center_x - path_size, center_y + path_size),
            ]
            self.enemies.append({
                'pos': list(path[0]),
                'path': path,
                'path_index': 1
            })

    def _spawn_gem(self):
        while True:
            pos = [self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)]
            if self._check_collision(pos, self.gems) is None and \
               self._check_collision(pos, [e['pos'] for e in self.enemies], tolerance=3) is None and \
               self._check_collision(pos, [self.player_pos], tolerance=3) is None:
                self.gems.append(pos)
                break

    def _check_collision(self, pos1, pos_list, tolerance=0.5):
        for i, pos2 in enumerate(pos_list):
            dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            if dist < tolerance:
                return i
        return None

    def _get_closest_distance(self, pos, pos_list):
        if not pos_list:
            return float('inf')
        min_dist = float('inf')
        for p in pos_list:
            dist = math.sqrt((pos[0] - p[0])**2 + (pos[1] - p[1])**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['size'] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0.5]

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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        # --- Action mapping for human play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()