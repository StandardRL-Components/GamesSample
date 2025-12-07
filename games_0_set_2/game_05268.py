
# Generated: 2025-08-28T04:29:54.573060
# Source Brief: brief_05268.md
# Brief Index: 5268

        
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
        "Controls: Arrow keys to move. Hold space to mine asteroids. Avoid the red enemy ships."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship in a dangerous asteroid field. Collect 100 ore to win, but watch out for hostile patrol drones. You have 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000
        self.WIN_ORE = 100
        self.INITIAL_LIVES = 3
        self.INITIAL_ENEMIES = 2
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ASTEROID = (120, 130, 140)
        self.COLOR_ORE = (255, 220, 50)
        self.COLOR_LASER = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_EXPLOSION = [(255, 50, 50), (255, 150, 50), (255, 220, 50)]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0
        self.lives = 0
        self.ore = 0
        self.steps = 0
        self.game_over = False
        
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.ore_particles = []
        self.stars = []
        self.last_reward = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = -90
        
        self.lives = self.INITIAL_LIVES
        self.ore = 0
        self.steps = 0
        self.game_over = False
        self.last_reward = 0
        
        self.asteroids = []
        self.enemies = []
        self.particles = []
        self.ore_particles = []
        self.stars = []
        
        self._generate_stars(200)
        for _ in range(10):
            self._spawn_asteroid()
            
        for _ in range(self.INITIAL_ENEMIES):
            self._spawn_enemy()
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01 # Small time penalty to encourage action

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement)
        self._update_player()
        self._update_enemies()
        self._update_asteroids()
        
        if space_held:
            reward += self._handle_mining()
        
        reward += self._handle_collisions()
        reward += self._update_particles()

        self.steps += 1
        self._update_difficulty()
        
        terminated = (
            self.lives <= 0 or self.ore >= self.WIN_ORE or self.steps >= self.MAX_STEPS
        )
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.ore >= self.WIN_ORE:
                reward += 100
            if self.lives <= 0:
                reward -= 100
        
        self.last_reward = reward
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
            self.player_vel += pygame.Vector2(0, -acceleration)
        elif movement == 2: # Down
            self.player_vel += pygame.Vector2(0, acceleration)
        elif movement == 3: # Left
            self.player_vel += pygame.Vector2(-acceleration, 0)
        elif movement == 4: # Right
            self.player_vel += pygame.Vector2(acceleration, 0)
        
        if self.player_vel.length() > 0:
            self.player_angle = self.player_vel.angle_to(pygame.Vector2(1, 0))
            self._create_engine_particles()

    def _update_player(self):
        max_speed = 5
        friction = 0.96
        self.player_vel *= friction
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)
        
        self.player_pos += self.player_vel
        
        # World wrapping
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['pos'] += enemy['vel']
            enemy['pos'].x %= self.WIDTH
            enemy['pos'].y %= self.HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360

    def _handle_mining(self):
        reward = 0
        target_asteroid = None
        min_dist = float('inf')

        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < asteroid['radius'] + 100: # Mining range
                if dist < min_dist:
                    min_dist = dist
                    target_asteroid = asteroid
        
        if target_asteroid:
            # sfx: mining_laser_loop.wav
            self._create_mining_particles(target_asteroid['pos'])
            target_asteroid['ore'] -= 1
            if target_asteroid['ore'] <= 0:
                # sfx: asteroid_explosion.wav
                reward += 5
                for _ in range(target_asteroid['initial_ore']):
                    self._spawn_ore_particle(target_asteroid['pos'])
                self.asteroids.remove(target_asteroid)
                self._spawn_asteroid()
        return reward

    def _handle_collisions(self):
        reward = 0
        # Player vs Enemy
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < 10 + 10: # Player radius + Enemy radius
                self.lives -= 1
                reward -= 5
                # sfx: player_explosion.wav
                self._create_explosion(self.player_pos)
                self.enemies.remove(enemy)
                self._spawn_enemy()
                
                # Respawn player in center after a brief moment of invulnerability (implicit)
                self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
                self.player_vel = pygame.Vector2(0, 0)
                if self.lives > 0:
                    # sfx: respawn.wav
                    pass
                break # Only one collision per frame

        # Player vs Asteroid (bounce)
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < 10 + asteroid['radius']:
                # Simple bounce physics
                normal = (self.player_pos - asteroid['pos']).normalize()
                self.player_vel.reflect_ip(normal)
                self.player_vel *= 0.8
                # Push player out of asteroid
                overlap = 10 + asteroid['radius'] - dist
                self.player_pos += normal * overlap

        # Player vs Ore
        for ore_p in self.ore_particles[:]:
            if self.player_pos.distance_to(ore_p['pos']) < 15:
                self.ore += 1
                reward += 1
                self.ore_particles.remove(ore_p)
                # sfx: ore_collect.wav
        return reward
    
    def _update_particles(self):
        # Update and remove dead particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update ore particles (attracted to player)
        for p in self.ore_particles:
            direction = (self.player_pos - p['pos'])
            if direction.length() < 100: # Attraction range
                 p['vel'] += direction.normalize() * 0.5
            p['vel'] *= 0.95 # friction
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.ore_particles.remove(p)
        return 0

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % 200 == 0:
            for enemy in self.enemies:
                enemy['vel'] *= 1.01

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._draw_stars()
        for p in self.particles: self._draw_particle(p)
        for p in self.ore_particles: self._draw_ore_particle(p)
        for asteroid in self.asteroids: self._draw_asteroid(asteroid)
        for enemy in self.enemies: self._draw_enemy(enemy)
        self._draw_player()

    def _render_ui(self):
        # Ore display
        ore_text = self.font_small.render(f"ORE: {self.ore}/{self.WIN_ORE}", True, self.COLOR_ORE)
        self.screen.blit(ore_text, (10, 10))

        # Lives display
        for i in range(self.lives):
            self._draw_ship_icon(self.WIDTH - 30 - i * 25, 20)
        
        if self.game_over:
            msg = "YOU WIN!" if self.ore >= self.WIN_ORE else "GAME OVER"
            color = self.COLOR_PLAYER if self.ore >= self.WIN_ORE else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"ore": self.ore, "lives": self.lives, "steps": self.steps}

    # --- Spawning Methods ---
    def _spawn_asteroid(self):
        while True:
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
            if pos.distance_to(self.player_pos) > 100: break # Don't spawn on player
        
        size = self.np_random.uniform(15, 40)
        ore_amount = int(size)
        
        num_vertices = self.np_random.integers(5, 9)
        vertices = []
        for i in range(num_vertices):
            angle = i * (360 / num_vertices)
            dist = self.np_random.uniform(size * 0.8, size)
            vertices.append(pygame.Vector2(dist, 0).rotate(angle))
            
        self.asteroids.append({
            "pos": pos, "radius": size, "ore": ore_amount, "initial_ore": ore_amount,
            "angle": self.np_random.uniform(0, 360), "rot_speed": self.np_random.uniform(-1.5, 1.5),
            "vertices": vertices
        })

    def _spawn_enemy(self):
        while True:
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
            if pos.distance_to(self.player_pos) > 150: break
            
        speed = self.np_random.uniform(1, 2)
        angle = self.np_random.uniform(0, 360)
        vel = pygame.Vector2(speed, 0).rotate(angle)
        self.enemies.append({"pos": pos, "vel": vel})

    def _spawn_ore_particle(self, pos):
        angle = self.np_random.uniform(0, 360)
        speed = self.np_random.uniform(1, 4)
        vel = pygame.Vector2(speed, 0).rotate(angle)
        self.ore_particles.append({
            "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(100, 150)
        })

    def _generate_stars(self, num_stars):
        for _ in range(num_stars):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "depth": self.np_random.uniform(0.1, 0.6),
                "brightness": self.np_random.integers(50, 150)
            })

    # --- Particle Creation ---
    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            color = random.choice(self.COLOR_EXPLOSION)
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "life": self.np_random.integers(20, 40),
                "color": color, "size": self.np_random.uniform(2, 5)
            })

    def _create_engine_particles(self):
        if self.np_random.random() < 0.8: # Don't spawn every frame
            angle = self.player_angle + 180 + self.np_random.uniform(-15, 15)
            speed = self.np_random.uniform(1, 3)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            pos = self.player_pos - pygame.Vector2(10, 0).rotate(self.player_angle)
            self.particles.append({
                "pos": pos, "vel": vel, "life": self.np_random.integers(10, 20),
                "color": (255, 150, 50), "size": self.np_random.uniform(1, 3)
            })

    def _create_mining_particles(self, target_pos):
        if self.np_random.random() < 0.7:
            direction = (self.player_pos - target_pos).normalize()
            pos = target_pos + direction * self.np_random.uniform(10, 30)
            vel = direction.rotate(self.np_random.uniform(-90, 90)) * self.np_random.uniform(0.5, 2)
            self.particles.append({
                "pos": pos, "vel": vel, "life": self.np_random.integers(5, 15),
                "color": self.COLOR_LASER, "size": self.np_random.uniform(1, 2)
            })

    # --- Drawing Methods ---
    def _draw_player(self):
        p1 = self.player_pos + pygame.Vector2(12, 0).rotate(self.player_angle)
        p2 = self.player_pos + pygame.Vector2(-8, 7).rotate(self.player_angle)
        p3 = self.player_pos + pygame.Vector2(-8, -7).rotate(self.player_angle)
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_ship_icon(self, x, y):
        points = [(x, y), (x-12, y+6), (x-12, y-6)]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _draw_enemy(self, enemy):
        pos = enemy['pos']
        angle = enemy['vel'].angle_to(pygame.Vector2(1, 0))
        p1 = pos + pygame.Vector2(10, 0).rotate(angle)
        p2 = pos + pygame.Vector2(-10, 8).rotate(angle)
        p3 = pos + pygame.Vector2(-5, 0).rotate(angle)
        p4 = pos + pygame.Vector2(-10, -8).rotate(angle)
        points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
        
    def _draw_asteroid(self, asteroid):
        points = [(v.rotate(asteroid['angle']) + asteroid['pos']) for v in asteroid['vertices']]
        int_points = [(int(p.x), int(p.y)) for p in points]
        if len(int_points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _draw_particle(self, p):
        size = max(0, p['size'] * (p['life'] / 20.0))
        pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(size))

    def _draw_ore_particle(self, p):
        size = 2
        pygame.draw.circle(self.screen, self.COLOR_ORE, (int(p['pos'].x), int(p['pos'].y)), size)

    def _draw_stars(self):
        # Parallax effect
        player_offset = self.player_pos - pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)
        for star in self.stars:
            pos_x = (star['pos'].x - player_offset.x * star['depth']) % self.WIDTH
            pos_y = (star['pos'].y - player_offset.y * star['depth']) % self.HEIGHT
            color_val = star['brightness']
            pygame.gfxdraw.pixel(self.screen, int(pos_x), int(pos_y), (color_val, color_val, color_val))

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
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Miner")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score (Ore): {info['ore']}, Total Reward: {total_reward:.2f}")
    env.close()