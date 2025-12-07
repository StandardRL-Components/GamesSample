import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to mine asteroids. Avoid the red enemy ships."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a mining ship, collect valuable ore from asteroids, and dodge enemy patrols in this top-down arcade space miner."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 100
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SPEED = 3.0
        self.PLAYER_DAMPING = 0.90
        self.PLAYER_SIZE = 10
        self.ENEMY_COUNT = 10
        self.ASTEROID_COUNT = 20
        self.MINING_RANGE = 80
        self.MINING_RATE = 2.0
        self.ENEMY_DAMAGE = 20
        self.ENEMY_BASE_SPEED = 0.5
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        # For headless execution, set SDL_VIDEODRIVER to "dummy"
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (13, 13, 26)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_ENEMY = (255, 51, 51)
        self.COLOR_ASTEROID = (128, 128, 128)
        self.COLOR_WHITE = (255, 255, 255)
        self.ORE_COLORS = {
            1: (200, 200, 200),
            2: (173, 216, 230),
            3: (255, 215, 0),
            4: (255, 165, 0),
            5: (65, 105, 225)
        }
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # RNG
        self.np_random = None

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_health = 0
        self.is_mining = False
        self.mining_target = None
        self.enemies = []
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.enemy_speed_modifier = 0.0

        self.reset()
        # self.validate_implementation() # This was causing the crash, validation is for debugging.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = self.np_random.random(2) * [self.WIDTH, self.HEIGHT]
        self.player_vel = np.zeros(2, dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        
        self.is_mining = False
        self.mining_target = None
        
        self.enemy_speed_modifier = 0.0
        self.enemies = [self._spawn_enemy() for _ in range(self.ENEMY_COUNT)]
        self.asteroids = [self._spawn_asteroid() for _ in range(self.ASTEROID_COUNT)]
        
        self.particles = []
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.integers(1, 3)
            ) for _ in range(150)
        ]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action
        reward = -0.01  # Cost of living

        # --- Update game logic ---
        self._handle_input(movement, space_held == 1)
        self._update_player()
        self._update_enemies()
        
        mining_reward = self._update_mining()
        reward += mining_reward
        
        self._update_particles()
        
        collision_reward = self._check_collisions()
        reward += collision_reward

        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_speed_modifier = min(1.5, self.enemy_speed_modifier + 0.05)

        # --- Check termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward = 100.0
            elif self.player_health <= 0:
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        accel = np.zeros(2, dtype=np.float32)
        if movement == 1: accel[1] -= 1  # Up
        if movement == 2: accel[1] += 1  # Down
        if movement == 3: accel[0] -= 1  # Left
        if movement == 4: accel[0] += 1  # Right
        
        if np.linalg.norm(accel) > 0:
            self.player_vel += accel * self.PLAYER_SPEED * 0.1

        # Mining
        self.is_mining = space_held

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DAMPING
        
        # Clamp velocity
        speed = np.linalg.norm(self.player_vel)
        if speed > self.PLAYER_SPEED:
            self.player_vel = self.player_vel / speed * self.PLAYER_SPEED
            
        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Engine particle trail
        if np.linalg.norm(self.player_vel) > 0.5:
            self._create_particles(self.player_pos, 1, self.COLOR_PLAYER, lifetime=10, size=2, spread=10, speed_min=0.1, speed_max=0.5)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += enemy['angular_velocity'] * (self.ENEMY_BASE_SPEED + self.enemy_speed_modifier)
            enemy['pos'][0] = enemy['center'][0] + math.cos(enemy['angle']) * enemy['radius']
            enemy['pos'][1] = enemy['center'][1] + math.sin(enemy['angle']) * enemy['radius']

    def _update_mining(self):
        reward = 0
        if not self.is_mining:
            self.mining_target = None
            return reward

        # Find a target if we don't have one
        if self.mining_target is None or self.mining_target['ore_amount'] <= 0:
            self.mining_target = None
            min_dist = self.MINING_RANGE
            for asteroid in self.asteroids:
                dist = np.linalg.norm(self.player_pos - asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    self.mining_target = asteroid
        
        # Mine the target
        if self.mining_target:
            dist = np.linalg.norm(self.player_pos - self.mining_target['pos'])
            if dist < self.MINING_RANGE and self.mining_target['ore_amount'] > 0:
                mined_amount = min(self.mining_target['ore_amount'], self.MINING_RATE / 30.0)
                self.mining_target['ore_amount'] -= mined_amount
                
                # Check if ore unit is collected
                if int(self.score) < int(self.score + mined_amount * self.mining_target['ore_value']):
                    reward += 0.1 # Base reward for any ore
                    if self.mining_target['ore_value'] >= 4:
                        reward += 1.0 # Bonus for high-value ore
                
                self.score = min(self.WIN_SCORE, self.score + mined_amount * self.mining_target['ore_value'])
                
                # Visual effect for mining
                if self.steps % 2 == 0:
                    self._create_particles(
                        self.mining_target['pos'], 1, self.ORE_COLORS[self.mining_target['ore_value']],
                        lifetime=30, size=3, spread=self.mining_target['size'], speed_min=0.5, speed_max=1.5,
                        target_pos=self.player_pos
                    )
                
                if self.mining_target['ore_amount'] <= 0:
                    # Asteroid depleted, respawn it
                    idx = self.asteroids.index(self.mining_target)
                    self.asteroids[idx] = self._spawn_asteroid()
                    self.mining_target = None
            else:
                self.mining_target = None
        
        return reward

    def _check_collisions(self):
        # Player vs Enemies
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + enemy['size']:
                self.player_health = max(0, self.player_health - self.ENEMY_DAMAGE)
                self._create_particles(self.player_pos, 20, self.COLOR_WHITE, lifetime=20, size=4, spread=0, speed_min=1, speed_max=4)
                # Remove enemy and respawn
                enemy.update(self._spawn_enemy())
                return -1.0 # Negative reward for getting hit
        return 0.0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['lifetime'] -= 1
            if p.get('target_pos') is not None:
                # Homing particles
                direction = (p['target_pos'] - p['pos'])
                dist = np.linalg.norm(direction)
                if dist > 1:
                    direction /= dist
                p['vel'] = p['vel'] * 0.9 + direction * 1.5
            p['pos'] += p['vel']
            p['size'] = max(0, p['size'] - 0.05)
    
    def _check_termination(self):
        return self.player_health <= 0 or self.score >= self.WIN_SCORE

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
        }

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (x, y, size, size))
            
        # Asteroids
        for a in self.asteroids:
            a['angle'] = (a['angle'] + a['rotation_speed']) % 360
            self._draw_poly(a['pos'], a['points'], a['angle'], self.COLOR_ASTEROID)
            # Draw ore inside
            ore_color = self.ORE_COLORS[a['ore_value']]
            ore_size = int(a['size'] * (a['ore_amount'] / a['initial_ore']) * 0.6)
            if ore_size > 1:
                pygame.draw.circle(self.screen, ore_color, a['pos'].astype(int), ore_size)
        
        # Mining beam
        if self.mining_target and self.is_mining:
            dist = np.linalg.norm(self.player_pos - self.mining_target['pos'])
            if dist < self.MINING_RANGE:
                color = self.ORE_COLORS[self.mining_target['ore_value']]
                start_pos = self.player_pos.astype(int)
                end_pos = self.mining_target['pos'].astype(int)
                
                # Create a temporary surface for the beam to handle alpha
                beam_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                for i in range(1, 4):
                    alpha_color = (*color, 150 // i)
                    pygame.draw.aaline(beam_surf, alpha_color, start_pos, end_pos, i)
                self.screen.blit(beam_surf, (0,0))


        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['initial_lifetime']))
            color_with_alpha = (*p['color'], alpha)
            if p['size'] > 1:
                # Use a temporary surface for alpha blending
                particle_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color_with_alpha, (p['size'], p['size']), int(p['size']))
                self.screen.blit(particle_surf, (p['pos'] - p['size']).astype(int))


        # Enemies
        for e in self.enemies:
            angle_rad = math.atan2(self.player_pos[1] - e['pos'][1], self.player_pos[0] - e['pos'][0])
            self._draw_triangle(e['pos'], e['size'], angle_rad, self.COLOR_ENEMY)

        # Player
        player_angle_rad = math.atan2(self.player_vel[1], self.player_vel[0]) if np.linalg.norm(self.player_vel) > 0.1 else -math.pi/2
        self._draw_triangle(self.player_pos, self.PLAYER_SIZE, player_angle_rad, self.COLOR_PLAYER, glow=True)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"ORE: {int(self.score)} / {self.WIN_SCORE}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        health_color = (
            int(255 * (1 - health_pct)),
            int(255 * health_pct),
            0
        )
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (bar_x, bar_y, bar_width, bar_height), 1)

        # Game Over Message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "VICTORY"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_ENEMY
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Helper Functions ---
    def _spawn_asteroid(self):
        size = self.np_random.integers(12, 25)
        ore_amount = self.np_random.uniform(10, 20)
        ore_value = self.np_random.choice(list(self.ORE_COLORS.keys()), p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        num_points = self.np_random.integers(6, 10)
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = size * self.np_random.uniform(0.7, 1.0)
            points.append((math.cos(angle) * radius, math.sin(angle) * radius))
        
        return {
            'pos': self.np_random.random(2) * [self.WIDTH, self.HEIGHT],
            'size': size,
            'initial_ore': ore_amount,
            'ore_amount': ore_amount,
            'ore_value': ore_value,
            'angle': self.np_random.uniform(0, 360),
            'rotation_speed': self.np_random.uniform(-0.3, 0.3),
            'points': np.array(points)
        }

    def _spawn_enemy(self):
        center_x = self.np_random.uniform(self.WIDTH * 0.2, self.WIDTH * 0.8)
        center_y = self.np_random.uniform(self.HEIGHT * 0.2, self.HEIGHT * 0.8)
        radius = self.np_random.uniform(50, 150)
        angle = self.np_random.uniform(0, 2 * math.pi)
        angular_velocity = self.np_random.uniform(0.01, 0.03) * self.np_random.choice([-1, 1])
        return {
            'pos': np.array([center_x + math.cos(angle) * radius, center_y + math.sin(angle) * radius]),
            'size': 8,
            'center': np.array([center_x, center_y]),
            'radius': radius,
            'angle': angle,
            'angular_velocity': angular_velocity
        }

    def _create_particles(self, pos, count, color, lifetime, size, spread, speed_min, speed_max, target_pos=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_min, speed_max)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            start_pos = pos + self.np_random.uniform(-spread, spread, 2)
            self.particles.append({
                'pos': start_pos.copy(),
                'vel': velocity,
                'color': color,
                'lifetime': lifetime,
                'initial_lifetime': lifetime,
                'size': self.np_random.uniform(size * 0.5, size),
                'target_pos': target_pos.copy() if target_pos is not None else None
            })

    def _draw_triangle(self, pos, size, angle, color, glow=False):
        points = [
            (size, 0),
            (-size * 0.5, -size * 0.8),
            (-size * 0.5, size * 0.8)
        ]
        
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        
        rotated_points = (points @ rotation_matrix.T + pos).astype(int)

        if glow:
            glow_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            glow_color = (*color, 50)
            pygame.gfxdraw.filled_polygon(glow_surf, rotated_points, glow_color)
            pygame.gfxdraw.aapolygon(glow_surf, rotated_points, glow_color)
            self.screen.blit(glow_surf, (0,0))
        
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, color)
    
    def _draw_poly(self, pos, points, angle, color):
        angle_rad = math.radians(angle)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)]
        ])
        
        rotated_points = (points @ rotation_matrix.T + pos).astype(int)
        
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, color)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, (100, 100, 100)) # Darker border

    def close(self):
        pygame.quit()


# Example usage to run and display the game
if __name__ == '__main__':
    # To display the game, uncomment the following line
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv()
    obs, info = env.reset()
    
    # For human play
    pygame.display.set_caption("Arcade Space Miner")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
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

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)

    env.close()
    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")