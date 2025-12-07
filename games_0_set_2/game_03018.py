
# Generated: 2025-08-27T22:06:05.374354
# Source Brief: brief_03018.md
# Brief Index: 3018

        
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

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your ship. "
        "Collect the yellow gems to score points."
    )

    game_description = (
        "Collect gems while dodging enemies in a fast-paced, top-down arcade environment. "
        "Gather 25 gems to win, but be careful! Colliding with a red enemy will end your run."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 12
    GEM_SIZE = 8
    ENEMY_SIZE = 14
    NUM_GEMS = 5
    NUM_ENEMIES = 4
    WIN_SCORE = 25
    MAX_STEPS = 2000 # Increased from 1000 to allow more time for collection

    # Physics
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.94
    PLAYER_MAX_SPEED = 6.0
    INITIAL_ENEMY_SPEED = 1.0
    ENEMY_SPEED_INCREASE = 0.05

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150, 64)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_SPARKLE = (255, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 100, 100, 100)
    COLOR_BOUNDARY = (100, 100, 255)
    COLOR_TEXT = (255, 255, 255)
    
    # Reward structure
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    REWARD_GEM_COLLECT = 1.0
    REWARD_RISKY_GEM_BONUS = 2.0
    REWARD_STEP_PENALTY = -0.01 # Reduced from -0.2 to encourage exploration
    REWARD_SURVIVAL = 0.0 # Removed +0.1 to simplify, step penalty covers this

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.player_pos = None
        self.player_vel = None
        self.gems = None
        self.enemies = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        
        self.particles = []
        
        self._spawn_enemies()
        self._spawn_gems(self.NUM_GEMS)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        reward = self.REWARD_STEP_PENALTY

        self._update_player(movement)
        self._update_enemies()
        self._update_particles()
        
        # --- Collision Detection & Rewards ---
        gem_collected_this_step = False
        for i, gem_pos in enumerate(self.gems):
            dist = np.linalg.norm(self.player_pos - gem_pos)
            if dist < self.PLAYER_SIZE + self.GEM_SIZE:
                # sfx: gem collect
                reward += self.REWARD_GEM_COLLECT
                self.score += 1
                gem_collected_this_step = True
                
                # Risky play bonus
                min_dist_to_enemy = min([np.linalg.norm(self.player_pos - e['pos']) for e in self.enemies])
                if min_dist_to_enemy < self.ENEMY_SIZE + self.PLAYER_SIZE + 50:
                    reward += self.REWARD_RISKY_GEM_BONUS
                
                self._create_particles(gem_pos, self.COLOR_GEM, 20)
                self.gems.pop(i)
                self._spawn_gems(1)
                break
        
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + self.ENEMY_SIZE:
                # sfx: player explosion
                self.game_over = True
                reward += self.REWARD_LOSE
                self._create_particles(self.player_pos, self.COLOR_ENEMY, 50, life=60, speed_mult=2.0)
                break

        # --- Termination Conditions ---
        self.steps += 1
        terminated = self.game_over
        
        if self.score >= self.WIN_SCORE:
            # sfx: win fanfare
            reward += self.REWARD_WIN
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _spawn_enemies(self):
        self.enemies = []
        for i in range(self.NUM_ENEMIES):
            path_radius = self.np_random.uniform(50, 150)
            center_x = self.np_random.uniform(path_radius, self.WIDTH - path_radius)
            center_y = self.np_random.uniform(path_radius, self.HEIGHT - path_radius)
            start_angle = self.np_random.uniform(0, 2 * math.pi)
            
            self.enemies.append({
                'center': np.array([center_x, center_y]),
                'radius': path_radius,
                'angle': start_angle,
                'speed': self.INITIAL_ENEMY_SPEED,
                'pos': np.array([center_x + path_radius * math.cos(start_angle), 
                                 center_y + path_radius * math.sin(start_angle)])
            })

    def _spawn_gems(self, num_to_spawn):
        if not hasattr(self, 'gems') or self.gems is None:
            self.gems = []
            
        for _ in range(num_to_spawn):
            while True:
                pos = self.np_random.uniform(
                    low=[self.GEM_SIZE, self.GEM_SIZE], 
                    high=[self.WIDTH - self.GEM_SIZE, self.HEIGHT - self.GEM_SIZE]
                )
                
                # Avoid spawning on player
                if np.linalg.norm(pos - self.player_pos) < 50:
                    continue
                
                # Avoid spawning on other gems
                if any(np.linalg.norm(pos - g_pos) < 20 for g_pos in self.gems):
                    continue
                
                self.gems.append(pos)
                break

    def _update_player(self, movement):
        accel = np.array([0.0, 0.0])
        if movement == 1: accel[1] = -self.PLAYER_ACCEL  # Up
        if movement == 2: accel[1] = self.PLAYER_ACCEL   # Down
        if movement == 3: accel[0] = -self.PLAYER_ACCEL  # Left
        if movement == 4: accel[0] = self.PLAYER_ACCEL   # Right

        self.player_vel += accel
        self.player_vel *= self.PLAYER_FRICTION
        
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_MAX_SPEED:
            self.player_vel = self.player_vel * (self.PLAYER_MAX_SPEED / speed)
            
        self.player_pos += self.player_vel
        
        # Boundary collision
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_enemies(self):
        difficulty_level = self.score // 5
        current_speed = self.INITIAL_ENEMY_SPEED + difficulty_level * self.ENEMY_SPEED_INCREASE
        
        for enemy in self.enemies:
            enemy['speed'] = current_speed
            enemy['angle'] += enemy['speed'] / enemy['radius']
            enemy['pos'][0] = enemy['center'][0] + enemy['radius'] * math.cos(enemy['angle'])
            enemy['pos'][1] = enemy['center'][1] + enemy['radius'] * math.sin(enemy['angle'])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.98 # Particle friction

    def _create_particles(self, pos, color, count, life=30, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'max_life': life,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos_int = p['pos'].astype(int)
            pygame.draw.circle(self.screen, color, pos_int, 2)

        # Gems
        sparkle_phase = math.sin(self.steps * 0.3)
        for gem_pos in self.gems:
            pos_int = gem_pos.astype(int)
            radius = int(self.GEM_SIZE + sparkle_phase * 2)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_GEM)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_GEM)
            if sparkle_phase > 0.8:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius+2, self.COLOR_GEM_SPARKLE + (100,))

        # Enemies
        for enemy in self.enemies:
            pos_int = enemy['pos'].astype(int)
            
            # Glow effect
            glow_surf = pygame.Surface((self.ENEMY_SIZE * 4, self.ENEMY_SIZE * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_ENEMY_GLOW, (self.ENEMY_SIZE * 2, self.ENEMY_SIZE * 2), self.ENEMY_SIZE * 1.5)
            self.screen.blit(glow_surf, (pos_int[0] - self.ENEMY_SIZE * 2, pos_int[1] - self.ENEMY_SIZE * 2), special_flags=pygame.BLEND_RGBA_ADD)

            # Body (2-frame animation)
            if (self.steps // 5) % 2 == 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            else:
                p1 = (pos_int[0] - self.ENEMY_SIZE, pos_int[1] + self.ENEMY_SIZE)
                p2 = (pos_int[0] + self.ENEMY_SIZE, pos_int[1] + self.ENEMY_SIZE)
                p3 = (pos_int[0], pos_int[1] - self.ENEMY_SIZE)
                pygame.gfxdraw.filled_trigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY)
                pygame.gfxdraw.aatrigon(self.screen, p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], self.COLOR_ENEMY)

        # Player
        if not self.game_over:
            pos_int = self.player_pos.astype(int)
            
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE * 4, self.PLAYER_SIZE * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), self.PLAYER_SIZE * 1.8)
            self.screen.blit(glow_surf, (pos_int[0] - self.PLAYER_SIZE * 2, pos_int[1] - self.PLAYER_SIZE * 2), special_flags=pygame.BLEND_RGBA_ADD)

            # Body
            player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
            player_rect.center = pos_int
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        score_text = f"GEMS: {self.score} / {self.WIN_SCORE}"
        text_surface = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display total reward for debugging
        reward_text = f"Total Reward: {total_reward:.2f}"
        text_surface = env.font.render(reward_text, True, (200, 200, 200))
        screen.blit(text_surface, (10, env.HEIGHT - 30))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

        clock.tick(30) # Match the auto_advance rate
        
    env.close()