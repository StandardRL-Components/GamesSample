
# Generated: 2025-08-28T02:17:04.762465
# Source Brief: brief_01655.md
# Brief Index: 1655

        
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

    user_guide = (
        "Controls: Arrow keys to move your ship. Hold Space to mine nearby asteroids. "
        "Hold Shift to engage thrusters for a visual effect."
    )

    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect ore by mining asteroids, "
        "but beware of hostile patrol drones. Collect 100 ore to win, but a single "
        "collision means game over."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 40)
    COLOR_ASTEROID = (150, 150, 150)
    COLOR_ASTEROID_GLOW = (150, 150, 150, 20)
    COLOR_ENEMY = (255, 80, 80)
    COLOR_ENEMY_GLOW = (255, 80, 80, 50)
    COLOR_ORE = (255, 220, 50)
    COLOR_EXPLOSION = (255, 150, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Screen & World
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    WORLD_WIDTH, WORLD_HEIGHT = 1600, 1200
    
    # Game Parameters
    FPS = 30
    MAX_STEPS = 2500
    WIN_SCORE = 100
    
    # Player
    PLAYER_ACCELERATION = 0.5
    PLAYER_FRICTION = 0.96
    PLAYER_MAX_SPEED = 7
    PLAYER_SIZE = 12
    PLAYER_MINING_RANGE = 80
    PLAYER_MINING_RATE = 0.2  # Ore per step
    
    # Asteroids
    NUM_ASTEROIDS = 15
    ASTEROID_MIN_ORE = 10
    ASTEROID_MAX_ORE = 30
    ASTEROID_RESPAWN_TIME = 600 # steps
    
    # Enemies
    NUM_ENEMIES = 5
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPEED_INCREASE_INTERVAL = 200
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    ENEMY_MAX_SPEED_MULTIPLIER = 2.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        self.stars = []
        self.player = {}
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.mining_target = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.game_over_timer = 0
        self.reward_this_step = 0
        
        self.enemy_speed_multiplier = 1.0

        # Player
        self.player = {
            "x": self.np_random.uniform(self.SCREEN_WIDTH / 2, self.WORLD_WIDTH - self.SCREEN_WIDTH / 2),
            "y": self.np_random.uniform(self.SCREEN_HEIGHT / 2, self.WORLD_HEIGHT - self.SCREEN_HEIGHT / 2),
            "vx": 0, "vy": 0, "angle": 0
        }

        # Stars
        self.stars = [{
            "x": self.np_random.uniform(0, self.WORLD_WIDTH),
            "y": self.np_random.uniform(0, self.WORLD_HEIGHT),
            "size": self.np_random.choice([1, 2]),
            "parallax": self.np_random.uniform(0.1, 0.5)
        } for _ in range(200)]

        # Asteroids
        self.asteroids = []
        for _ in range(self.NUM_ASTEROIDS):
            self.asteroids.append(self._create_asteroid())

        # Enemies
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            self.enemies.append(self._create_enemy())
            
        self.particles = []
        self.mining_target = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.game_over_timer -=1
            if self.game_over_timer <= 0:
                terminated = True
            else:
                terminated = False
            
            # Allow final frame rendering with effects
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.reward_this_step = -0.02 # Small penalty for existing
        self.steps += 1
        
        self._update_player(movement, space_held, shift_held)
        self._update_enemies()
        self._update_asteroids()
        self._update_particles()
        
        self._handle_collisions()

        reward = self.reward_this_step
        terminated = self._check_termination()

        if terminated:
            if self.game_won:
                reward += 100
                # sfx: game_win
            else:
                reward -= 50
                # sfx: player_explosion
                self._create_explosion(self.player['x'], self.player['y'])
            self.game_over = True
            self.game_over_timer = self.FPS * 2 # 2 second linger

        return self._get_observation(), reward, False, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def close(self):
        pygame.quit()
        
    # --- Update Logic ---
    
    def _update_player(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player['vy'] -= self.PLAYER_ACCELERATION
        if movement == 2: self.player['vy'] += self.PLAYER_ACCELERATION
        if movement == 3: self.player['vx'] -= self.PLAYER_ACCELERATION
        if movement == 4: self.player['vx'] += self.PLAYER_ACCELERATION
        
        self.player['vx'] *= self.PLAYER_FRICTION
        self.player['vy'] *= self.PLAYER_FRICTION
        
        speed = math.hypot(self.player['vx'], self.player['vy'])
        if speed > self.PLAYER_MAX_SPEED:
            self.player['vx'] = (self.player['vx'] / speed) * self.PLAYER_MAX_SPEED
            self.player['vy'] = (self.player['vy'] / speed) * self.PLAYER_MAX_SPEED
            
        self.player['x'] += self.player['vx']
        self.player['y'] += self.player['vy']
        
        # World boundaries
        self.player['x'] = np.clip(self.player['x'], 0, self.WORLD_WIDTH)
        self.player['y'] = np.clip(self.player['y'], 0, self.WORLD_HEIGHT)
        
        if speed > 0.1:
            self.player['angle'] = math.atan2(self.player['vy'], self.player['vx'])

        # Engine/Drift particles
        if speed > 1.0 or shift_held:
            # sfx: player_thruster_loop
            num_particles = 3 if shift_held else 1
            for _ in range(num_particles):
                self.particles.append(self._create_trail_particle(self.player))
        
        # Mining
        self.mining_target = None
        if space_held:
            closest_asteroid, min_dist = None, float('inf')
            for asteroid in self.asteroids:
                if asteroid['ore'] > 0:
                    dist = math.hypot(self.player['x'] - asteroid['x'], self.player['y'] - asteroid['y'])
                    if dist < self.PLAYER_MINING_RANGE and dist < min_dist:
                        min_dist = dist
                        closest_asteroid = asteroid
            
            if closest_asteroid:
                self.mining_target = closest_asteroid
                mined_amount = self.PLAYER_MINING_RATE
                
                if closest_asteroid['ore'] < mined_amount:
                    mined_amount = closest_asteroid['ore']
                
                closest_asteroid['ore'] -= mined_amount
                self.score += mined_amount
                
                self.reward_this_step += mined_amount * 0.1 # Reward for mining
                
                # sfx: mine_beam_loop
                if self.np_random.random() < 0.5: # Spawn ore particle
                    self.particles.append(self._create_ore_particle(closest_asteroid, self.player))

                if closest_asteroid['ore'] <= 0:
                    self.reward_this_step += 1.0 # Bonus for depleting
                    # sfx: asteroid_depleted
                    closest_asteroid['respawn_timer'] = self.ASTEROID_RESPAWN_TIME
    
    def _update_enemies(self):
        if self.steps > 0 and self.steps % self.ENEMY_SPEED_INCREASE_INTERVAL == 0:
            self.enemy_speed_multiplier = min(self.ENEMY_MAX_SPEED_MULTIPLIER, self.enemy_speed_multiplier + self.ENEMY_SPEED_INCREASE_AMOUNT)

        for enemy in self.enemies:
            speed = self.ENEMY_BASE_SPEED * self.enemy_speed_multiplier
            
            if enemy['type'] == 'sweeper':
                dx, dy = enemy['target_x'] - enemy['x'], enemy['target_y'] - enemy['y']
                dist = math.hypot(dx, dy)
                if dist < speed:
                    enemy['x'], enemy['y'] = enemy['target_x'], enemy['target_y']
                    enemy['target_x'], enemy['start_x'] = enemy['start_x'], enemy['target_x']
                    enemy['target_y'], enemy['start_y'] = enemy['start_y'], enemy['target_y']
                else:
                    enemy['x'] += (dx / dist) * speed
                    enemy['y'] += (dy / dist) * speed
            
            elif enemy['type'] == 'circular':
                enemy['angle'] += enemy['angular_speed']
                enemy['x'] = enemy['center_x'] + math.cos(enemy['angle']) * enemy['radius']
                enemy['y'] = enemy['center_y'] + math.sin(enemy['angle']) * enemy['radius']

            elif enemy['type'] == 'random_walk':
                enemy['timer'] -= 1
                if enemy['timer'] <= 0:
                    enemy['timer'] = self.np_random.integers(60, 120)
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    enemy['vx'] = math.cos(angle) * speed
                    enemy['vy'] = math.sin(angle) * speed
                enemy['x'] += enemy['vx']
                enemy['y'] += enemy['vy']
                # Bounce off world edges
                if not (0 < enemy['x'] < self.WORLD_WIDTH): enemy['vx'] *= -1
                if not (0 < enemy['y'] < self.WORLD_HEIGHT): enemy['vy'] *= -1
                enemy['x'] = np.clip(enemy['x'], 0, self.WORLD_WIDTH)
                enemy['y'] = np.clip(enemy['y'], 0, self.WORLD_HEIGHT)
    
    def _update_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['ore'] <= 0 and asteroid['respawn_timer'] > 0:
                asteroid['respawn_timer'] -= 1
                if asteroid['respawn_timer'] <= 0:
                    new_asteroid = self._create_asteroid()
                    asteroid.update(new_asteroid)
            else:
                asteroid['angle'] += asteroid['rot_speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['x'] += p['vx']
            p['y'] += p['vy']
            if p.get('gravity', False):
                p['vy'] += 0.1

    def _handle_collisions(self):
        for enemy in self.enemies:
            dist = math.hypot(self.player['x'] - enemy['x'], self.player['y'] - enemy['y'])
            if dist < self.PLAYER_SIZE + enemy['size']:
                self.game_over = True
                return

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.score = self.WIN_SCORE
            self.game_won = True
            return True
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    # --- Rendering ---

    def _render_game(self):
        cam_x = self.player['x'] - self.SCREEN_WIDTH / 2
        cam_y = self.player['y'] - self.SCREEN_HEIGHT / 2

        # Stars
        for star in self.stars:
            sx = (star['x'] - cam_x * star['parallax']) % self.WORLD_WIDTH
            sy = (star['y'] - cam_y * star['parallax']) % self.WORLD_HEIGHT
            if 0 <= sx < self.SCREEN_WIDTH and 0 <= sy < self.SCREEN_HEIGHT:
                alpha = 100 + 155 * star['parallax']
                pygame.draw.rect(self.screen, (alpha, alpha, alpha), (int(sx), int(sy), star['size'], star['size']))

        # Asteroids
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                sx, sy = int(asteroid['x'] - cam_x), int(asteroid['y'] - cam_y)
                size = int(asteroid['base_size'] * (asteroid['ore'] / asteroid['max_ore'])**0.5)
                if size > 2:
                    points = []
                    for i in range(asteroid['num_points']):
                        angle = asteroid['angle'] + 2 * math.pi * i / asteroid['num_points']
                        px = sx + math.cos(angle) * size * asteroid['shape'][i]
                        py = sy + math.sin(angle) * size * asteroid['shape'][i]
                        points.append((px, py))
                    
                    if self.screen.get_rect().colliderect(pygame.Rect(sx-size, sy-size, size*2, size*2)):
                        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ASTEROID)
                        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ASTEROID)
        
        # Particles
        for p in self.particles:
            sx, sy = int(p['x'] - cam_x), int(p['y'] - cam_y)
            alpha = p['alpha'] * (p['life'] / p['max_life'])
            color = (*p['color'], int(alpha))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, max(0, size), color)

        # Mining beam
        if self.mining_target:
            start_pos = (int(self.player['x'] - cam_x), int(self.player['y'] - cam_y))
            end_pos = (int(self.mining_target['x'] - cam_x), int(self.mining_target['y'] - cam_y))
            width = self.np_random.integers(1, 4)
            alpha = self.np_random.integers(100, 200)
            pygame.draw.line(self.screen, (*self.COLOR_ORE, alpha), start_pos, end_pos, width)

        # Enemies
        for enemy in self.enemies:
            sx, sy = int(enemy['x'] - cam_x), int(enemy['y'] - cam_y)
            size = enemy['size']
            if self.screen.get_rect().colliderect(pygame.Rect(sx-size, sy-size, size*2, size*2)):
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, size + 4, self.COLOR_ENEMY_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, size, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, size, self.COLOR_ENEMY)

        # Player
        if not (self.game_over and not self.game_won):
            sx, sy = int(self.player['x'] - cam_x), int(self.player['y'] - cam_y)
            size = self.PLAYER_SIZE
            angle = self.player['angle']
            p1 = (sx + math.cos(angle) * size, sy + math.sin(angle) * size)
            p2 = (sx + math.cos(angle + 2.2) * size * 0.8, sy + math.sin(angle + 2.2) * size * 0.8)
            p3 = (sx + math.cos(angle - 2.2) * size * 0.8, sy + math.sin(angle - 2.2) * size * 0.8)
            
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, size + 5, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Game Over / Win message
        if self.game_over and self.game_over_timer > 0:
            message = "MISSION COMPLETE" if self.game_won else "SHIP DESTROYED"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ENEMY
            
            alpha = min(255, 510 * (1 - self.game_over_timer / (self.FPS * 2)))
            
            text_surf = self.font_game_over.render(message, True, color)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    # --- Entity Creation ---
    
    def _create_asteroid(self):
        ore = self.np_random.integers(self.ASTEROID_MIN_ORE, self.ASTEROID_MAX_ORE + 1)
        num_points = self.np_random.integers(7, 12)
        return {
            "x": self.np_random.uniform(0, self.WORLD_WIDTH),
            "y": self.np_random.uniform(0, self.WORLD_HEIGHT),
            "ore": ore,
            "max_ore": ore,
            "base_size": self.np_random.uniform(15, 30),
            "angle": self.np_random.uniform(0, 2 * math.pi),
            "rot_speed": self.np_random.uniform(-0.02, 0.02),
            "respawn_timer": 0,
            "num_points": num_points,
            "shape": [self.np_random.uniform(0.8, 1.2) for _ in range(num_points)]
        }

    def _create_enemy(self):
        enemy_type = self.np_random.choice(['sweeper', 'circular', 'random_walk'])
        x = self.np_random.uniform(0, self.WORLD_WIDTH)
        y = self.np_random.uniform(0, self.WORLD_HEIGHT)
        base = {"x": x, "y": y, "type": enemy_type, "size": 8}

        if enemy_type == 'sweeper':
            start_x, start_y = x, y
            if self.np_random.random() > 0.5: # Horizontal
                target_x = start_x + self.np_random.uniform(200, 400) * self.np_random.choice([-1, 1])
                target_y = start_y
            else: # Vertical
                target_x = start_x
                target_y = start_y + self.np_random.uniform(200, 400) * self.np_random.choice([-1, 1])
            base.update({
                "start_x": start_x, "start_y": start_y,
                "target_x": np.clip(target_x, 0, self.WORLD_WIDTH),
                "target_y": np.clip(target_y, 0, self.WORLD_HEIGHT)
            })
        elif enemy_type == 'circular':
            radius = self.np_random.uniform(50, 150)
            base.update({
                "center_x": x, "center_y": y, "radius": radius,
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "angular_speed": self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
            })
        elif enemy_type == 'random_walk':
            base.update({"timer": 0, "vx": 0, "vy": 0})
        return base
        
    def _create_trail_particle(self, source):
        angle = source['angle'] + math.pi + self.np_random.uniform(-0.5, 0.5)
        speed = self.np_random.uniform(1, 3)
        return {
            "x": source['x'] - math.cos(source['angle']) * self.PLAYER_SIZE * 0.8,
            "y": source['y'] - math.sin(source['angle']) * self.PLAYER_SIZE * 0.8,
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "life": self.np_random.integers(10, 20), "max_life": 20,
            "size": self.np_random.integers(1, 4), "color": self.COLOR_PLAYER,
            "alpha": self.np_random.integers(50, 150)
        }
        
    def _create_ore_particle(self, source, target):
        dx, dy = target['x'] - source['x'], target['y'] - source['y']
        dist = math.hypot(dx, dy)
        speed = self.np_random.uniform(4, 8)
        return {
            "x": source['x'], "y": source['y'],
            "vx": (dx / dist) * speed if dist > 0 else 0, 
            "vy": (dy / dist) * speed if dist > 0 else 0,
            "life": int(dist / speed) if speed > 0 else 10, 
            "max_life": int(dist / speed) if speed > 0 else 10,
            "size": self.np_random.integers(2, 5), "color": self.COLOR_ORE,
            "alpha": 255
        }
        
    def _create_explosion(self, x, y):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 8)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(20, 50), "max_life": 50,
                "size": self.np_random.integers(2, 5), "color": self.COLOR_EXPLOSION,
                "alpha": self.np_random.integers(150, 255), "gravity": True
            })

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "dummy" for headless execution, or remove for visual debugging
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    env = GameEnv()
    
    # --- For human play ---
    # This requires a visual display
    if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
        obs, info = env.reset()
        done = False
        
        # Override screen for display
        env.screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Asteroid Miner")

        while not done:
            movement, space, shift = 0, 0, 0
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the display window
            display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            env.screen.blit(display_surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        
        print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")

    env.close()