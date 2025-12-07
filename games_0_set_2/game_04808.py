
# Generated: 2025-08-28T03:03:55.584289
# Source Brief: brief_04808.md
# Brief Index: 4808

        
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

    user_guide = (
        "Controls: Arrow keys to move. Hold space to mine asteroids. Avoid red enemy ships."
    )

    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect minerals from asteroids while dodging enemy patrols. Your goal is to collect 50 minerals before your ship's hull is destroyed."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.W, self.H = 640, 400

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ASTEROID = [(100, 90, 80), (120, 110, 100), (80, 70, 60)]
        self.COLOR_MINERAL = (255, 220, 0)
        self.COLOR_LASER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_FG = (50, 200, 50)
        self.COLOR_HEALTH_BG = (100, 0, 0)
        
        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_MINERALS = 50
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 4.0
        self.PLAYER_RADIUS = 12
        self.ENEMY_BASE_SPEED = 1.5
        self.ENEMY_RADIUS = 10
        self.ASTEROID_MIN_RADIUS = 20
        self.ASTEROID_MAX_RADIUS = 40
        self.INITIAL_ASTEROIDS = 8
        self.INITIAL_ENEMIES = 3
        
        # State variables initialized in reset
        self.np_random = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_health = None
        self.minerals_collected = None
        self.asteroids = None
        self.enemies = None
        self.particles = None
        self.stars = None
        self.damage_flicker_timer = 0
        self.is_mining = False
        self.mining_target = None
        
        self.reset()
        self.validate_implementation()

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = self.np_random.integers(0, self.W)
            y = self.np_random.integers(0, self.H)
            size = self.np_random.integers(1, 4)
            brightness = self.np_random.integers(50, 150)
            self.stars.append(((x, y), size, (brightness, brightness, brightness)))

    def _get_random_spawn_pos(self, radius):
        for _ in range(100): # Max 100 attempts to find a free spot
            pos = pygame.Vector2(
                self.np_random.uniform(radius, self.W - radius),
                self.np_random.uniform(radius, self.H - radius)
            )
            
            # Check overlap with player
            if self.player_pos.distance_to(pos) < self.PLAYER_RADIUS + radius + 50:
                continue
            
            # Check overlap with other asteroids
            too_close = False
            for ast in self.asteroids:
                if pos.distance_to(ast['pos']) < ast['radius'] + radius + 20:
                    too_close = True
                    break
            if too_close: continue
            
            return pos
        return pygame.Vector2(self.np_random.uniform(radius, self.W - radius), self.np_random.uniform(radius, self.H - radius))

    def _spawn_asteroid(self):
        radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
        pos = self._get_random_spawn_pos(radius)
        
        num_vertices = self.np_random.integers(7, 12)
        angle_step = 2 * math.pi / num_vertices
        points = []
        for i in range(num_vertices):
            angle = i * angle_step + self.np_random.uniform(-0.1, 0.1)
            dist = self.np_random.uniform(0.8, 1.0) * radius
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))
            
        return {
            'pos': pos,
            'radius': radius,
            'minerals': int(radius),
            'max_minerals': int(radius),
            'angle': self.np_random.uniform(0, 2 * math.pi),
            'rotation_speed': self.np_random.uniform(-0.01, 0.01),
            'points': points,
            'color': self.np_random.choice(self.COLOR_ASTEROID),
        }

    def _spawn_enemy(self):
        radius = self.ENEMY_RADIUS
        pos = self._get_random_spawn_pos(radius)
        path_type = self.np_random.choice(['horizontal', 'vertical', 'circular'])
        
        vel = pygame.Vector2()
        if path_type == 'horizontal':
            vel.x = self.np_random.choice([-1, 1])
        elif path_type == 'vertical':
            vel.y = self.np_random.choice([-1, 1])
        else: # circular
            vel.x = self.np_random.choice([-1, 1])
        
        return {
            'pos': pos,
            'vel': vel,
            'path_type': path_type,
            'center': pygame.Vector2(pos),
            'radius': radius,
            'path_radius': self.np_random.uniform(50, 150)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.W / 2, self.H / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.minerals_collected = 0
        
        self.asteroids = []
        for _ in range(self.INITIAL_ASTEROIDS):
            self.asteroids.append(self._spawn_asteroid())
        
        self.enemies = []
        for _ in range(self.INITIAL_ENEMIES):
            self.enemies.append(self._spawn_enemy())
            
        self.particles = []
        self._generate_stars()
        
        self.damage_flicker_timer = 0
        self.is_mining = False
        self.mining_target = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Small penalty for every step
        self.steps += 1
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        player_vel = pygame.Vector2(0, 0)
        if movement == 1: player_vel.y = -1
        elif movement == 2: player_vel.y = 1
        elif movement == 3: player_vel.x = -1
        elif movement == 4: player_vel.x = 1
        
        if player_vel.length() > 0:
            player_vel.normalize_ip()
            player_vel *= self.PLAYER_SPEED
        
        self.player_pos += player_vel

        # World wrapping for player
        self.player_pos.x %= self.W
        self.player_pos.y %= self.H
        
        # --- Update Game State ---
        if self.damage_flicker_timer > 0:
            self.damage_flicker_timer -= 1

        # Update asteroids
        for ast in self.asteroids:
            ast['angle'] += ast['rotation_speed']

        # Update enemies
        enemy_speed_multiplier = 1.0 + (self.minerals_collected // 10) * 0.1
        current_enemy_speed = self.ENEMY_BASE_SPEED * enemy_speed_multiplier

        for enemy in self.enemies:
            if enemy['path_type'] == 'horizontal':
                enemy['pos'].x += enemy['vel'].x * current_enemy_speed
                if enemy['pos'].x > self.W or enemy['pos'].x < 0:
                    enemy['vel'].x *= -1
            elif enemy['path_type'] == 'vertical':
                enemy['pos'].y += enemy['vel'].y * current_enemy_speed
                if enemy['pos'].y > self.H or enemy['pos'].y < 0:
                    enemy['vel'].y *= -1
            elif enemy['path_type'] == 'circular':
                enemy['pos'].x = enemy['center'].x + math.cos(self.steps * 0.02 * enemy['vel'].x) * enemy['path_radius']
                enemy['pos'].y = enemy['center'].y + math.sin(self.steps * 0.02 * enemy['vel'].x) * enemy['path_radius']
            
            enemy['pos'].x %= self.W
            enemy['pos'].y %= self.H

        # --- Handle Mining ---
        self.is_mining = False
        self.mining_target = None
        if space_held:
            closest_ast = None
            min_dist = float('inf')
            for ast in self.asteroids:
                dist = self.player_pos.distance_to(ast['pos'])
                if dist < ast['radius'] + 60: # Mining range
                    if dist < min_dist:
                        min_dist = dist
                        closest_ast = ast
            
            if closest_ast:
                self.is_mining = True
                self.mining_target = closest_ast
                closest_ast['minerals'] -= 1
                
                # sfx: mining_beam_loop
                
                # Add mining particles
                if self.steps % 3 == 0:
                    self.minerals_collected = min(self.WIN_MINERALS, self.minerals_collected + 1)
                    reward += 1.0
                    self.score += 1
                    
                    # Create particle effect
                    for _ in range(2):
                        p_pos = pygame.Vector2(closest_ast['pos']) + pygame.Vector2(self.np_random.uniform(-closest_ast['radius'], closest_ast['radius']), self.np_random.uniform(-closest_ast['radius'], closest_ast['radius']))
                        p_vel = (self.player_pos - p_pos).normalize() * self.np_random.uniform(2, 4)
                        self.particles.append({'pos': p_pos, 'vel': p_vel, 'lifespan': 20, 'color': self.COLOR_MINERAL, 'size': 3})
                
                if closest_ast['minerals'] <= 0:
                    reward += 5.0
                    self.score += 50
                    # sfx: asteroid_explosion
                    self._create_explosion(closest_ast['pos'], closest_ast['color'], 30, int(closest_ast['radius']))
                    self.asteroids.remove(closest_ast)
                    self.asteroids.append(self._spawn_asteroid())


        # --- Handle Collisions ---
        if self.damage_flicker_timer == 0:
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_RADIUS + enemy['radius']:
                    self.player_health -= 25
                    self.damage_flicker_timer = 30 # Flicker for 1 second at 30fps
                    # sfx: player_damage
                    self._create_explosion(self.player_pos, self.COLOR_ENEMY, 15, 10)
                    break
        
        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
        
        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0:
            reward = -100.0
            self.game_over = True
            terminated = True
            # sfx: game_over
        elif self.minerals_collected >= self.WIN_MINERALS:
            reward = 100.0
            self.score += 1000
            self.game_over = True
            terminated = True
            # sfx: victory
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_explosion(self, pos, color, count, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5) * (speed_factor / 20.0)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            size = self.np_random.integers(2, 5)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'lifespan': lifespan, 'color': color, 'size': size})

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for pos, size, color in self.stars:
            pygame.draw.circle(self.screen, color, pos, size / 2)

    def _render_game(self):
        # Draw asteroids
        for ast in self.asteroids:
            points_rotated = []
            for p in ast['points']:
                x = p[0] * math.cos(ast['angle']) - p[1] * math.sin(ast['angle'])
                y = p[0] * math.sin(ast['angle']) + p[1] * math.cos(ast['angle'])
                points_rotated.append((x + ast['pos'].x, y + ast['pos'].y))
            
            if len(points_rotated) > 2:
                pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points_rotated], ast['color'])
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points_rotated], ast['color'])

        # Draw enemies
        for enemy in self.enemies:
            p1 = (enemy['pos'].x, enemy['pos'].y - enemy['radius'])
            p2 = (enemy['pos'].x - enemy['radius'] * 0.866, enemy['pos'].y + enemy['radius'] * 0.5)
            p3 = (enemy['pos'].x + enemy['radius'] * 0.866, enemy['pos'].y + enemy['radius'] * 0.5)
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Draw mining laser
        if self.is_mining and self.mining_target:
            start_pos = (int(self.player_pos.x), int(self.player_pos.y))
            end_pos = (int(self.mining_target['pos'].x), int(self.mining_target['pos'].y))
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 1)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30.0))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Draw player
        if self.damage_flicker_timer % 4 < 2:
            p1 = (self.player_pos.x, self.player_pos.y - self.PLAYER_RADIUS)
            p2 = (self.player_pos.x - self.PLAYER_RADIUS * 0.866, self.player_pos.y + self.PLAYER_RADIUS * 0.5)
            p3 = (self.player_pos.x + self.PLAYER_RADIUS * 0.866, self.player_pos.y + self.PLAYER_RADIUS * 0.5)
            points = [(int(p[0]), int(p[1])) for p in [p1, p2, p3]]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Minerals
        mineral_text = self.font_ui.render(f"MINERALS: {self.minerals_collected}/{self.WIN_MINERALS}", True, self.COLOR_TEXT)
        self.screen.blit(mineral_text, (10, 10))

        # Health bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.W - bar_w - 10, 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.W / 2, self.H - 20))
        self.screen.blit(score_text, score_rect)

        # Game Over / Win message
        if self.game_over:
            if self.minerals_collected >= self.WIN_MINERALS:
                msg = "MISSION COMPLETE"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            end_text = self.font_title.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(end_text, end_rect)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "minerals": self.minerals_collected,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment
if __name__ == '__main__':
    import time

    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.W, env.H))
    
    done = False
    total_reward = 0
    
    # Use a dummy action for auto-advancing frames
    action = env.action_space.sample() 
    action[0] = 0 # no-op movement
    action[1] = 0 # space released
    action[2] = 0 # shift released

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(2)
            obs, info = env.reset()
            total_reward = 0

    env.close()