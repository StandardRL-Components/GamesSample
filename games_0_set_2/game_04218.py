import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to jump, ←→ to move. Avoid asteroids and collect coins "
        "before you run out of fuel or time!"
    )

    game_description = (
        "Guide your hopping spaceship through a dangerous asteroid field, "
        "collecting coins to win while managing your limited fuel supply."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game Constants ---
        self.FPS = 30
        self.MAX_STEPS = 60 * self.FPS # 60 seconds
        self.WIN_COINS = 50
        
        # Physics
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -8
        self.HORIZONTAL_ACCEL = 0.8
        self.DRAG = 0.92
        self.MAX_VEL_X = 6

        # Entities
        self.NUM_ASTEROIDS = 15
        self.NUM_COINS = 8
        self.NUM_STARS = 150

        # Fuel
        self.MAX_FUEL = 100
        self.JUMP_FUEL_COST = 8
        self.ASTEROID_COLLISION_FUEL_PENALTY = 30
        
        # Rewards
        self.REWARD_PER_STEP = 0.01
        self.REWARD_COIN = 10.0
        self.REWARD_ASTEROID_HIT = -5.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 50)
        self.COLOR_ASTEROID = (100, 110, 120)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_COIN_GLOW = (255, 223, 0, 60)
        self.COLOR_UI_TEXT = (150, 255, 150)
        self.COLOR_FUEL_GREEN = (0, 255, 0)
        self.COLOR_FUEL_YELLOW = (255, 255, 0)
        self.COLOR_FUEL_RED = (255, 0, 0)
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.coins_collected = 0
        self.game_over = False
        
        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT * 0.8], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.can_jump = True
        
        # Resources
        self.fuel = self.MAX_FUEL
        
        # Entity lists
        self.asteroids = []
        self.coins = []
        self.particles = []
        self.stars = []
        
        # Procedural generation
        self.asteroid_speed = 2.0
        self._generate_stars()
        for _ in range(self.NUM_ASTEROIDS):
            self._spawn_asteroid(at_top=False)
        for _ in range(self.NUM_COINS):
            self._spawn_coin()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = self.REWARD_PER_STEP

        # --- Update Logic ---
        self._handle_input(action)
        self._update_player_physics()
        self._update_asteroids()
        self._update_particles()
        
        # --- Collisions & Events ---
        reward += self._handle_coin_collection()
        reward += self._handle_asteroid_collisions()
        
        # --- State Updates ---
        self.steps += 1
        self.score += reward # Cumulative reward for this episode
        self._update_difficulty()
        
        # --- Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.coins_collected >= self.WIN_COINS:
                reward = self.REWARD_WIN
            else:
                reward = self.REWARD_LOSE
        
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]

        # Horizontal movement
        if movement == 3: # Left
            self.player_vel[0] -= self.HORIZONTAL_ACCEL
            self._create_thruster_sparks(self.player_pos + [5, 2], angle_offset=math.pi/2)
        elif movement == 4: # Right
            self.player_vel[0] += self.HORIZONTAL_ACCEL
            self._create_thruster_sparks(self.player_pos + [-5, 2], angle_offset=-math.pi/2)
        
        # Vertical movement (jump)
        if movement == 1 and self.can_jump and self.fuel > self.JUMP_FUEL_COST:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.fuel -= self.JUMP_FUEL_COST
            self.can_jump = False
            # sfx: jump_sound()
            self._create_thruster_sparks(self.player_pos + [0, 10], num_particles=20)
    
    def _update_player_physics(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        
        # Apply drag
        self.player_vel[0] *= self.DRAG
        
        # Clamp horizontal velocity
        self.player_vel[0] = np.clip(self.player_vel[0], -self.MAX_VEL_X, self.MAX_VEL_X)
        
        # Update position
        self.player_pos += self.player_vel
        
        # Floor collision
        if self.player_pos[1] > self.HEIGHT - 20:
            self.player_pos[1] = self.HEIGHT - 20
            self.player_vel[1] = 0
            self.can_jump = True
            
        # Wall collisions (wrap around)
        if self.player_pos[0] < 0:
            self.player_pos[0] = self.WIDTH
        elif self.player_pos[0] > self.WIDTH:
            self.player_pos[0] = 0

    def _update_asteroids(self):
        asteroids_to_keep = []
        num_to_respawn = 0
        for asteroid in self.asteroids:
            asteroid['pos'][1] += self.asteroid_speed
            if asteroid['pos'][1] - asteroid['radius'] > self.HEIGHT:
                num_to_respawn += 1
            else:
                # Update the absolute points for rendering based on the new position
                asteroid['points'] = [(int(asteroid['pos'][0] + pt[0]), int(asteroid['pos'][1] + pt[1])) for pt in asteroid['points_template']]
                asteroids_to_keep.append(asteroid)

        self.asteroids = asteroids_to_keep
        for _ in range(num_to_respawn):
            self._spawn_asteroid(at_top=True)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _update_difficulty(self):
        # Asteroid speed increases every 10 seconds
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
            self.asteroid_speed += 0.2

    def _handle_coin_collection(self):
        reward = 0
        player_radius = 10
        coins_to_keep = []
        num_collected = 0
        
        for coin in self.coins:
            dist = np.linalg.norm(self.player_pos - coin['pos'])
            if dist < player_radius + coin['radius']:
                num_collected += 1
                reward += self.REWARD_COIN
                self._create_explosion(coin['pos'], 15, self.COLOR_COIN)
                # sfx: coin_collect_sound()
            else:
                coins_to_keep.append(coin)
        
        if num_collected > 0:
            self.coins_collected += num_collected
            self.coins = coins_to_keep
            for _ in range(num_collected):
                self._spawn_coin()
        
        return reward

    def _handle_asteroid_collisions(self):
        reward = 0
        player_radius = 8 # Smaller hitbox for player
        
        collided_asteroid_index = -1
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < player_radius + asteroid['radius']:
                collided_asteroid_index = i
                break # Process only the first collision

        if collided_asteroid_index != -1:
            collided_asteroid = self.asteroids[collided_asteroid_index]
            
            self.fuel = max(0, self.fuel - self.ASTEROID_COLLISION_FUEL_PENALTY)
            reward += self.REWARD_ASTEROID_HIT
            self.player_vel += (self.player_pos - collided_asteroid['pos']) * 0.2 # Knockback
            
            # Remove the collided asteroid by index
            self.asteroids.pop(collided_asteroid_index)
            
            self._spawn_asteroid(at_top=True)
            self._create_explosion(self.player_pos, 30, (255, 100, 100))
            # sfx: explosion_sound()
            return reward # Return early, as in the original logic
        
        return reward

    def _check_termination(self):
        if self.fuel <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.coins_collected >= self.WIN_COINS: return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.coins_collected,
            "steps": self.steps,
            "fuel": self.fuel,
        }

    def _render_game(self):
        self._render_stars()
        self._render_asteroids()
        self._render_coins()
        self._render_particles()
        self._render_player()

    def _render_stars(self):
        for star in self.stars:
            pos, size, layer = star
            # Parallax effect
            scroll_pos_y = (pos[1] + self.steps * 0.1 * layer) % self.HEIGHT
            color_val = 50 + 100 * layer
            color = (color_val, color_val, color_val + 50 * layer)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(scroll_pos_y)), int(size))

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        points = [(x, y - 12), (x - 8, y + 8), (x + 8, y + 8)]
        
        # Glow effect
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER_GLOW)
        
        # Main ship
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            pygame.gfxdraw.aapolygon(self.screen, asteroid['points'], self.COLOR_ASTEROID)
            pygame.gfxdraw.filled_polygon(self.screen, asteroid['points'], self.COLOR_ASTEROID)
    
    def _render_coins(self):
        for coin in self.coins:
            pos = (int(coin['pos'][0]), int(coin['pos'][1]))
            radius = int(coin['radius'])
            
            # Pulsing glow
            pulse = abs(math.sin(self.steps * 0.1 + coin['pos'][0]))
            glow_radius = radius + int(pulse * 5)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_COIN_GLOW)

            # Coin body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_COIN)
            # Glint
            glint_radius = max(1, int(radius * 0.4))
            pygame.gfxdraw.filled_circle(self.screen, pos[0]-2, pos[1]-2, glint_radius, (255,255,200))

    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"COINS: {self.coins_collected}/{self.WIN_COINS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Fuel Gauge
        fuel_percent = self.fuel / self.MAX_FUEL
        bar_width = 200
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - bar_height - 10
        
        # Interpolate color
        if fuel_percent > 0.5:
            color = self._lerp_color(self.COLOR_FUEL_YELLOW, self.COLOR_FUEL_GREEN, (fuel_percent - 0.5) * 2)
        else:
            color = self._lerp_color(self.COLOR_FUEL_RED, self.COLOR_FUEL_YELLOW, fuel_percent * 2)

        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width * fuel_percent, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 2)

    def _generate_stars(self):
        self.stars = []
        for _ in range(self.NUM_STARS):
            layer = self.np_random.uniform(0.1, 1.0)
            self.stars.append((
                [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)], # pos
                self.np_random.uniform(0.5, 1.5) * layer, # size
                layer # parallax layer
            ))

    def _spawn_asteroid(self, at_top=False):
        radius = self.np_random.uniform(10, 30)
        x = self.np_random.uniform(0, self.WIDTH)
        y = -radius if at_top else self.np_random.uniform(0, self.HEIGHT)
        pos = np.array([x, y])
        
        # Generate irregular polygon shape
        num_vertices = self.np_random.integers(7, 12)
        angles = np.linspace(0, 2 * math.pi, num_vertices, endpoint=False)
        noise = self.np_random.uniform(0.7, 1.1, num_vertices)
        
        # Store shape relative to origin (0,0)
        points_template = []
        for i, angle in enumerate(angles):
            r = radius * noise[i]
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            points_template.append((px, py))

        # Calculate absolute points for rendering
        points = [(int(pos[0] + pt[0]), int(pos[1] + pt[1])) for pt in points_template]

        self.asteroids.append({'pos': pos, 'radius': radius, 'points_template': points_template, 'points': points})


    def _spawn_coin(self):
        self.coins.append({
            'pos': self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50]),
            'radius': 8
        })

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(4, 8),
                'color': color,
                'life': self.np_random.integers(15, 30)
            })
    
    def _create_thruster_sparks(self, pos, num_particles=5, angle_offset=0):
        for _ in range(num_particles):
            angle = self.np_random.normal(math.pi / 2 + angle_offset, 0.3)
            speed = self.np_random.uniform(2, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(1, 4),
                'color': (255, 180, 50),
                'life': self.np_random.integers(10, 20)
            })

    def _lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Astro Hopper")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        # Pygame uses a different coordinate system for surfaces, so we need to transpose
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score (Coins): {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause for 2 seconds on game over

        # --- Tick Clock ---
        clock.tick(env.FPS)
        
    env.close()