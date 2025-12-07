import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:53:34.023001
# Source Brief: brief_00683.md
# Brief Index: 683
# """import gymnasium as gym
class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Navigate a futuristic vehicle through a winding fractal canyon, shooting enemies and avoiding collisions."
    user_guide = "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Press space to fire."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.CANYON_LENGTH = 12000
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_STARS = (200, 200, 255)
        self.COLOR_CANYON = (40, 30, 70)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_TRAIL = (100, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 180, 0)
        self.COLOR_ENEMY_PROJ = (255, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)

        # Player Physics
        self.PLAYER_ACCELERATION = 0.15
        self.PLAYER_BRAKE_FORCE = 0.2
        self.PLAYER_STEER_SPEED = 4.0
        self.PLAYER_DRAG = 0.985
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SHOOT_COOLDOWN = 8 # steps

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_big = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_big = pygame.font.SysFont("monospace", 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_health = 0
        self.player_shoot_timer = 0
        self.player_trail = None
        self.camera_x = 0
        self.canyon_top = []
        self.canyon_bottom = []
        self.stars = []
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.enemy_speed_multiplier = 1.0
        self.enemy_spawn_prob = 0.02
        self.last_distance_reward_pos = 0

    def _generate_canyon(self):
        def subdivide(points, level, roughness):
            if level <= 0:
                return points
            new_points = [points[0]]
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                offset = self.np_random.uniform(-roughness, roughness) * (p2[0] - p1[0])
                new_points.append((mid_x, mid_y + offset))
                new_points.append(p2)
            return subdivide(new_points, level - 1, roughness)

        initial_top = [(0, self.HEIGHT * 0.2), (self.CANYON_LENGTH, self.HEIGHT * 0.2)]
        initial_bottom = [(0, self.HEIGHT * 0.8), (self.CANYON_LENGTH, self.HEIGHT * 0.8)]
        
        top_points = subdivide(initial_top, 8, 0.2)
        bottom_points = subdivide(initial_bottom, 8, 0.2)
        
        self.canyon_top = np.interp(np.arange(self.CANYON_LENGTH), [p[0] for p in top_points], [p[1] for p in top_points])
        self.canyon_bottom = np.interp(np.arange(self.CANYON_LENGTH), [p[0] for p in bottom_points], [p[1] for p in bottom_points])

        # Ensure canyon is navigable
        for i in range(self.CANYON_LENGTH):
            if self.canyon_bottom[i] - self.canyon_top[i] < 100:
                mid = (self.canyon_bottom[i] + self.canyon_top[i]) / 2
                self.canyon_top[i] = mid - 50
                self.canyon_bottom[i] = mid + 50

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
                'depth': self.np_random.uniform(0.1, 0.8) # For parallax
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([150.0, self.HEIGHT / 2.0])
        self.player_vel = np.array([1.0, 0.0]) # Start with a little forward velocity
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_shoot_timer = 0
        self.player_trail = deque(maxlen=20)
        
        self.camera_x = 0
        self.last_distance_reward_pos = 0

        self._generate_canyon()
        self._generate_stars()
        
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.enemy_speed_multiplier = 1.0
        self.enemy_spawn_prob = 0.02

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            return obs, 0, True, False, self._get_info()
        
        reward = 0
        self.steps += 1
        
        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: # Accelerate
            self.player_vel[0] += self.PLAYER_ACCELERATION
        elif movement == 2: # Brake
            self.player_vel[0] -= self.PLAYER_BRAKE_FORCE
        if movement == 3: # Steer Left (Up on screen)
            self.player_pos[1] -= self.PLAYER_STEER_SPEED
        elif movement == 4: # Steer Right (Down on screen)
            self.player_pos[1] += self.PLAYER_STEER_SPEED

        if space_held and self.player_shoot_timer <= 0:
            # SFX: Player Shoot
            proj_pos = self.player_pos + np.array([20, 0])
            self.player_projectiles.append({'pos': proj_pos, 'vel': np.array([12.0, 0.0])})
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
        
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1

        # 2. Update Game State
        # Player movement
        self.player_vel[0] = max(0, self.player_vel[0] * self.PLAYER_DRAG)
        self.player_pos += self.player_vel
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        self.player_trail.append(self.player_pos.copy())
        
        # Camera
        self.camera_x = self.player_pos[0] - 150

        # Spawn enemies
        if self.np_random.random() < self.enemy_spawn_prob:
            spawn_x = self.camera_x + self.WIDTH + 50
            if 0 <= int(spawn_x % self.CANYON_LENGTH) < self.CANYON_LENGTH:
                spawn_y = self.np_random.uniform(self.canyon_top[int(spawn_x % self.CANYON_LENGTH)] + 30, 
                                                 self.canyon_bottom[int(spawn_x % self.CANYON_LENGTH)] - 30)
                self.enemies.append({
                    'pos': np.array([spawn_x, spawn_y]),
                    'base_y': spawn_y,
                    'type': self.np_random.choice(['sine', 'straight']),
                    'phase': self.np_random.uniform(0, 2 * math.pi),
                    'shoot_timer': self.np_random.integers(60, 120)
                })

        # Update entities
        self._update_list(self.player_projectiles)
        self._update_list(self.enemy_projectiles)
        self._update_list(self.particles, is_particle=True)
        self._update_enemies()

        # 3. Handle Collisions & Events
        reward += self._handle_collisions()

        # Distance reward
        dist_traveled = self.player_pos[0] - self.last_distance_reward_pos
        if dist_traveled >= 10:
            reward += 0.1
            self.last_distance_reward_pos = self.player_pos[0]
            
        # 4. Check Termination
        terminated = self.player_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            reward = -10.0 # Crash penalty
            self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)
        elif truncated:
            self.game_over = True
            reward = 10.0 # Win bonus
        
        # 5. Update Difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.enemy_speed_multiplier = min(2.5, self.enemy_speed_multiplier + 0.05)
            self.enemy_spawn_prob = min(0.05, self.enemy_spawn_prob + 0.01)
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, self._get_info()

    def _update_list(self, entity_list, is_particle=False):
        for e in entity_list[:]:
            e['pos'] += e['vel']
            if is_particle:
                e['lifetime'] -= 1
                if e['lifetime'] <= 0:
                    entity_list.remove(e)
            else:
                screen_x = e['pos'][0] - self.camera_x
                if not (0 < screen_x < self.WIDTH and 0 < e['pos'][1] < self.HEIGHT):
                    entity_list.remove(e)

    def _update_enemies(self):
        for e in self.enemies[:]:
            e['pos'][0] -= 1.5 * self.enemy_speed_multiplier
            if e['type'] == 'sine':
                e['pos'][1] = e['base_y'] + math.sin(e['phase'] + self.steps * 0.05) * 40
            
            e['shoot_timer'] -= 1
            if e['shoot_timer'] <= 0:
                # SFX: Enemy Shoot
                self.enemy_projectiles.append({'pos': e['pos'].copy(), 'vel': np.array([-8.0, 0.0])})
                e['shoot_timer'] = self.np_random.integers(100, 180)

            if e['pos'][0] < self.camera_x - 50:
                self.enemies.remove(e)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Canyon
        player_world_x = int(self.player_pos[0])
        if 0 <= player_world_x < self.CANYON_LENGTH:
            if self.player_pos[1] <= self.canyon_top[player_world_x] or \
               self.player_pos[1] >= self.canyon_bottom[player_world_x]:
                self.player_health = 0
                # SFX: Big Crash/Explosion
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj['pos'] - enemy['pos']) < 15:
                    self.player_projectiles.remove(proj)
                    self.enemies.remove(enemy)
                    self.score += 1
                    reward += 1.0
                    self._create_explosion(enemy['pos'], 30, self.COLOR_ENEMY)
                    # SFX: Enemy Explosion
                    break
        
        # Player vs Enemies
        for enemy in self.enemies[:]:
            if np.linalg.norm(self.player_pos - enemy['pos']) < 20:
                self.enemies.remove(enemy)
                self.player_health -= 35
                self._create_explosion(enemy['pos'], 30, self.COLOR_ENEMY)
                # SFX: Collision/Damage
                if self.player_health > 0:
                    # SFX: Player Damage
                    pass
        
        # Player vs Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            if np.linalg.norm(self.player_pos - proj['pos']) < 15:
                self.enemy_projectiles.remove(proj)
                self.player_health -= 15
                self._create_explosion(proj['pos'], 15, self.COLOR_ENEMY_PROJ)
                if self.player_health > 0:
                    # SFX: Player Damage
                    pass
        
        self.player_health = max(0, self.player_health)
        return reward

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars with parallax
        for star in self.stars:
            star_x = (star['pos'][0] - self.camera_x * star['depth']) % self.WIDTH
            pygame.draw.circle(self.screen, self.COLOR_STARS, (int(star_x), int(star['pos'][1])), 1)

        # Canyon
        visible_start = max(0, int(self.camera_x))
        visible_end = min(self.CANYON_LENGTH, int(self.camera_x + self.WIDTH + 2))
        if visible_end > visible_start:
            top_points = [(i - self.camera_x, self.canyon_top[i]) for i in range(visible_start, visible_end)]
            bottom_points = [(i - self.camera_x, self.canyon_bottom[i]) for i in range(visible_start, visible_end)]
            
            canyon_poly_top = [(0,0)] + top_points + [(self.WIDTH, 0)]
            canyon_poly_bottom = [(0, self.HEIGHT)] + bottom_points + [(self.WIDTH, self.HEIGHT)]
            
            pygame.gfxdraw.filled_polygon(self.screen, canyon_poly_top, self.COLOR_CANYON)
            pygame.gfxdraw.filled_polygon(self.screen, canyon_poly_bottom, self.COLOR_CANYON)

        # Player trail
        for i, p in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)))
            radius = int(8 * (i / len(self.player_trail)))
            if radius > 0:
                screen_pos = (int(p[0] - self.camera_x), int(p[1]))
                color = (*self.COLOR_TRAIL, alpha)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (screen_pos[0]-radius, screen_pos[1]-radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player bike
        if self.player_health > 0:
            px, py = int(self.player_pos[0] - self.camera_x), int(self.player_pos[1])
            bike_points = [(px + 15, py), (px - 10, py - 8), (px - 5, py), (px - 10, py + 8)]
            
            # Glow
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.polygon(glow_surf, self.COLOR_PLAYER_GLOW, [(p[0]-px+30, p[1]-py+30) for p in bike_points], 15)
            self.screen.blit(glow_surf, (px-30, py-30), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.gfxdraw.aapolygon(self.screen, bike_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, bike_points, self.COLOR_PLAYER)

        # Entities
        for e_list, color, radius in [(self.enemies, self.COLOR_ENEMY, 10), 
                                      (self.player_projectiles, self.COLOR_PLAYER_PROJ, 4),
                                      (self.enemy_projectiles, self.COLOR_ENEMY_PROJ, 4)]:
            for e in e_list:
                ex, ey = int(e['pos'][0] - self.camera_x), int(e['pos'][1])
                pygame.gfxdraw.aacircle(self.screen, ex, ey, radius, color)
                pygame.gfxdraw.filled_circle(self.screen, ex, ey, radius, color)

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['lifetime'] / 30.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,4,4))
            self.screen.blit(temp_surf, pos, special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

        # Speed
        speed = int(self.player_vel[0] * 20) # arbitrary unit
        speed_text = self.font_ui.render(f"SPEED: {speed}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, self.HEIGHT - 30))

        if self.game_over:
            msg = "CRASHED" if self.player_health <= 0 else "FINISH"
            end_text = self.font_big.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "health": self.player_health}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Un-comment the line below to run with display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    # This will fail if `SDL_VIDEODRIVER` is "dummy"
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Fractal Canyon Racer")
        human_play = True
    except pygame.error:
        print("Pygame display not available. Running in headless mode.")
        human_play = False
        
    clock = pygame.time.Clock()
    
    # Run a test episode
    obs, info = env.reset(seed=42)
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not terminated and not truncated:
        if human_play:
            # --- Action mapping for human ---
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1 # up
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2 # down
            
            # Steer with left/right
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3 # left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4 # right
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True # Exit loop
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- ENV RESET ---")
        else: # Agent play
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        if human_play:
            # --- Render to screen ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30) # Match assumed FPS

    print(f"Episode finished after {step_count} steps.")
    print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()