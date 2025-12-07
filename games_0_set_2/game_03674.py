
# Generated: 2025-08-28T00:03:26.721080
# Source Brief: brief_03674.md
# Brief Index: 3674

        
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
        "Controls: ↑↓←→ to move. Hold space to mine nearby asteroids."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, dodging enemy lasers while collecting ore from asteroids in a procedurally generated asteroid field."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 1280, 800
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased from 1000 to allow more time for 100 ore
        self.WIN_SCORE = 100
        self.STARTING_LIVES = 3

        # Visual constants
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_SHIP = (60, 180, 255)
        self.COLOR_SHIP_GLOW = (60, 180, 255, 50)
        self.COLOR_ASTEROID = (100, 100, 110)
        self.COLOR_LASER = (255, 50, 50)
        self.COLOR_ORE = (255, 220, 50)
        self.COLOR_EXPLOSION = (255, 150, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.asteroids = []
        self.lasers = []
        self.particles = []
        self.stars = []
        self.laser_spawn_rate = 0
        self.laser_spawn_timer = 0
        self.invincibility_timer = 0

        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.STARTING_LIVES
        self.game_over = False
        self.win = False

        self.player_pos = np.array([self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        
        self.asteroids.clear()
        self.lasers.clear()
        self.particles.clear()
        self.stars.clear()

        self.laser_spawn_rate = 50
        self.laser_spawn_timer = 0
        self.invincibility_timer = 90 # Start with 3 seconds of invincibility

        self._spawn_stars(150)
        self._spawn_asteroids(15)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        if not self.game_over:
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_input(movement)
            self._update_player()
            self._update_asteroids()
            self._update_lasers()
            self._update_particles()
            self._update_difficulty()

            ore_reward = self._handle_mining(space_held)
            hit_penalty = self._handle_collisions()
            reward += ore_reward + hit_penalty

        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.win:
                reward += 100
            else:
                reward -= 100
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        acceleration = 0.5
        if movement == 1: # Up
            self.player_vel[1] -= acceleration
        elif movement == 2: # Down
            self.player_vel[1] += acceleration
        elif movement == 3: # Left
            self.player_vel[0] -= acceleration
        elif movement == 4: # Right
            self.player_vel[0] += acceleration

    def _update_player(self):
        # Apply friction
        self.player_vel *= 0.95
        
        # Limit speed
        speed = np.linalg.norm(self.player_vel)
        if speed > 8:
            self.player_vel = self.player_vel / speed * 8

        # Update position
        self.player_pos += self.player_vel

        # World wrapping
        self.player_pos[0] %= self.WORLD_WIDTH
        self.player_pos[1] %= self.WORLD_HEIGHT

        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360
            if asteroid['ore'] <= 0:
                # Respawn depleted asteroid
                self._respawn_asteroid(asteroid)

    def _update_lasers(self):
        self.laser_spawn_timer += 1
        if self.laser_spawn_timer >= self.laser_spawn_rate:
            self.laser_spawn_timer = 0
            self._spawn_laser()
            # sound: laser_spawn.wav

        for laser in self.lasers[:]:
            laser['pos'] += laser['vel']
            if not (0 < laser['pos'][0] < self.WORLD_WIDTH and 0 < laser['pos'][1] < self.WORLD_HEIGHT):
                self.lasers.remove(laser)
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            self.laser_spawn_rate = max(15, self.laser_spawn_rate - 4)

    def _handle_mining(self, space_held):
        reward = 0
        if not space_held:
            return reward

        mining_range = 100
        mining_rate = 5 # Mine every 5 frames
        
        # Find closest asteroid
        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            if asteroid['ore'] > 0:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
        
        if closest_asteroid and min_dist < mining_range:
            closest_asteroid['is_mined'] = True # For rendering beam
            if self.steps % mining_rate == 0:
                closest_asteroid['ore'] -= 1
                self.score += 1
                reward += 0.1
                # sound: ore_collect.wav
                
                # Spawn ore particle
                for _ in range(2):
                    self._spawn_particle(
                        pos=closest_asteroid['pos'].copy(),
                        type='ore',
                        target=self.player_pos
                    )
        return reward

    def _handle_collisions(self):
        if self.invincibility_timer > 0:
            return 0
        
        player_radius = 12
        hit_penalty = 0

        for laser in self.lasers[:]:
            dist = np.linalg.norm(self.player_pos - laser['pos'])
            if dist < player_radius + laser['radius']:
                self.lasers.remove(laser)
                self.lives -= 1
                hit_penalty -= 1.0
                self.invincibility_timer = 90 # 3 seconds of invincibility
                # sound: ship_hit.wav
                
                # Spawn explosion
                for _ in range(30):
                    self._spawn_particle(pos=self.player_pos.copy(), type='explosion')
                
                if self.lives <= 0:
                    # sound: game_over.wav
                    break
        return hit_penalty

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            return True
        if self.lives <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        cam_x = self.player_pos[0] - self.WIDTH / 2
        cam_y = self.player_pos[1] - self.HEIGHT / 2

        self._render_background(cam_x, cam_y)
        self._render_asteroids(cam_x, cam_y)
        self._render_lasers(cam_x, cam_y)
        self._render_mining_beam(cam_x, cam_y)
        self._render_particles(cam_x, cam_y)
        self._render_player()
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_x, cam_y):
        for star in self.stars:
            x = (star['pos'][0] - cam_x * star['depth']) % self.WIDTH
            y = (star['pos'][1] - cam_y * star['depth']) % self.HEIGHT
            pygame.draw.circle(self.screen, star['color'], (int(x), int(y)), star['size'])

    def _render_asteroids(self, cam_x, cam_y):
        for asteroid in self.asteroids:
            points = []
            for point in asteroid['shape']:
                rotated_point = self._rotate_point(point, asteroid['angle'])
                screen_pos = rotated_point + asteroid['pos'] - np.array([cam_x, cam_y])
                
                # Manual wrapping for objects on screen
                for i in range(2):
                    world_dim = self.WORLD_WIDTH if i == 0 else self.WORLD_HEIGHT
                    screen_dim = self.WIDTH if i == 0 else self.HEIGHT
                    player_coord = self.player_pos[i]
                    obj_coord = asteroid['pos'][i]
                    
                    dist = obj_coord - player_coord
                    if abs(dist) > world_dim / 2:
                        screen_pos[i] += world_dim * -np.sign(dist)

                points.append(screen_pos)
            
            if self._is_polygon_on_screen(points):
                pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_ASTEROID)
                pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_ASTEROID)

    def _render_lasers(self, cam_x, cam_y):
        for laser in self.lasers:
            start_pos = laser['pos'] - np.array([cam_x, cam_y])
            end_pos = laser['pos'] - laser['vel'] * 2 - np.array([cam_x, cam_y])
            pygame.draw.line(self.screen, self.COLOR_LASER, (int(start_pos[0]), int(start_pos[1])), (int(end_pos[0]), int(end_pos[1])), 3)

    def _render_mining_beam(self, cam_x, cam_y):
        mining_asteroid = None
        for asteroid in self.asteroids:
            if asteroid.get('is_mined', False):
                mining_asteroid = asteroid
                asteroid['is_mined'] = False # Reset for next frame
                break
        
        if mining_asteroid:
            start_pos = (self.WIDTH // 2, self.HEIGHT // 2)
            
            # Calculate wrapped position of asteroid relative to player
            dx = mining_asteroid['pos'][0] - self.player_pos[0]
            dy = mining_asteroid['pos'][1] - self.player_pos[1]
            if dx > self.WORLD_WIDTH / 2: dx -= self.WORLD_WIDTH
            if dx < -self.WORLD_WIDTH / 2: dx += self.WORLD_WIDTH
            if dy > self.WORLD_HEIGHT / 2: dy -= self.WORLD_HEIGHT
            if dy < -self.WORLD_HEIGHT / 2: dy += self.WORLD_HEIGHT

            end_pos = (start_pos[0] + dx, start_pos[1] + dy)
            
            alpha = 100 + random.randint(0, 50)
            beam_color = (*self.COLOR_ORE, alpha)
            
            # Create a temporary surface for the beam
            beam_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(beam_surf, beam_color, start_pos, end_pos, 4)
            self.screen.blit(beam_surf, (0, 0))

    def _render_player(self):
        player_screen_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        
        if self.invincibility_timer > 0 and self.steps % 6 < 3:
            return # Flicker effect

        # Draw glow
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], 18, self.COLOR_SHIP_GLOW)
        
        # Draw ship body (triangle)
        p1 = (player_screen_pos[0], player_screen_pos[1] - 14)
        p2 = (player_screen_pos[0] - 10, player_screen_pos[1] + 10)
        p3 = (player_screen_pos[0] + 10, player_screen_pos[1] + 10)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_SHIP)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_SHIP)

    def _render_particles(self, cam_x, cam_y):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            pos = p['pos'] - np.array([cam_x, cam_y])
            size = p['size'] * (p['life'] / p['max_life'])
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(pos[0] - size), int(pos[1] - size)))

    def _render_ui(self):
        score_text = self.font_ui.render(f"ORE: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        message = "MISSION COMPLETE" if self.win else "GAME OVER"
        text = self.font_game_over.render(message, True, self.COLOR_ORE if self.win else self.COLOR_LASER)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    # Helper methods
    def _spawn_stars(self, n):
        for _ in range(n):
            self.stars.append({
                'pos': np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                'size': self.np_random.integers(1, 3),
                'depth': self.np_random.uniform(0.1, 0.5),
                'color': random.choice([(100,100,100), (150,150,150), (200,200,255)])
            })

    def _spawn_asteroids(self, n):
        for _ in range(n):
            asteroid = {}
            self._respawn_asteroid(asteroid)
            self.asteroids.append(asteroid)

    def _respawn_asteroid(self, asteroid):
        asteroid['pos'] = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)])
        asteroid['size'] = self.np_random.uniform(20, 40)
        asteroid['shape'] = self._create_asteroid_shape(asteroid['size'])
        asteroid['angle'] = self.np_random.uniform(0, 360)
        asteroid['rot_speed'] = self.np_random.uniform(-0.5, 0.5)
        asteroid['ore'] = self.np_random.integers(10, 25)

    def _create_asteroid_shape(self, radius):
        points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = self.np_random.uniform(radius * 0.7, radius * 1.1)
            points.append(np.array([math.cos(angle) * dist, math.sin(angle) * dist]))
        return points

    def _rotate_point(self, point, angle):
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return np.array([
            point[0] * cos_a - point[1] * sin_a,
            point[0] * sin_a + point[1] * cos_a
        ])

    def _spawn_laser(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), 0.0])
            angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75)
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.WORLD_WIDTH), float(self.WORLD_HEIGHT)])
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        elif edge == 2: # Left
            pos = np.array([0.0, self.np_random.uniform(0, self.WORLD_HEIGHT)])
            angle = self.np_random.uniform(-math.pi * 0.25, math.pi * 0.25)
        else: # Right
            pos = np.array([float(self.WORLD_WIDTH), self.np_random.uniform(0, self.WORLD_HEIGHT)])
            angle = self.np_random.uniform(math.pi * 0.75, math.pi * 1.25)
        
        speed = 5
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        self.lasers.append({'pos': pos, 'vel': vel, 'radius': 2})

    def _spawn_particle(self, pos, type, target=None):
        if type == 'explosion':
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            color = self.COLOR_EXPLOSION
            size = self.np_random.integers(2, 5)
        elif type == 'ore':
            direction = target - pos
            dist = np.linalg.norm(direction)
            if dist == 0: return
            direction /= dist
            vel = direction * 4 + self.np_random.uniform(-0.5, 0.5, 2)
            life = int(dist / 4)
            color = self.COLOR_ORE
            size = self.np_random.integers(2, 4)
        
        self.particles.append({
            'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color, 'size': size
        })

    def _is_polygon_on_screen(self, points):
        # A simple bounding box check is sufficient and fast
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        return max_x > 0 and min_x < self.WIDTH and max_y > 0 and min_y < self.HEIGHT

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Miner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    pygame.quit()