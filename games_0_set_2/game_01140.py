
# Generated: 2025-08-27T16:10:14.121138
# Source Brief: brief_01140.md
# Brief Index: 1140

        
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
    """
    A Gymnasium environment for an arcade-style asteroid mining game.

    The player controls a spaceship and must collect ore from asteroids while
    avoiding collisions. The game rewards strategic mining (closer proximity
    yields more ore but increases risk) and penalizes collisions and inaction.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing documentation
    user_guide = (
        "Controls: ↑ to drive forward, ←→ to turn, and ↓ to brake. "
        "Hold space to activate the mining beam. Get close to asteroids to mine them."
    )
    game_description = (
        "Pilot a mining ship through a dangerous asteroid field. Extract ore by "
        "getting close to asteroids and using your mining beam. Reach the ore quota "
        "to win, but watch your ship's integrity!"
    )

    # Frame advance setting
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Game constants
        self.MAX_STEPS = 5000
        self.WIN_ORE = 100
        self.MAX_HEALTH = 3
        self.NUM_ASTEROIDS = 12
        self.PLAYER_TURN_RATE = 5
        self.PLAYER_THRUST = 0.25
        self.PLAYER_DRAG = 0.98
        self.PLAYER_BRAKE_DRAG = 0.9
        self.PLAYER_RADIUS = 10
        self.MINE_RANGE = 120
        self.MINE_RATE = 0.2
        self.MINE_DAMAGE_PROB = 0.005 # Probability of damage per frame per pixel closer than MINE_RANGE
        
        # Color palette
        self.COLOR_BG = (15, 20, 40)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (150, 255, 150)
        self.COLOR_ASTEROID = (120, 120, 120)
        self.COLOR_ASTEROID_OUTLINE = (160, 160, 160)
        self.COLOR_BEAM = (255, 255, 100)
        self.COLOR_ORE_PARTICLE = (255, 220, 50)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_HEALTH_FG = (50, 200, 50)
        self.COLOR_HEALTH_BG = (200, 50, 50)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.player_health = None
        self.ore_collected = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.game_over = False
        self.win = False
        self.asteroid_base_speed = 1.0

        # Pre-generate stars for a consistent background
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(150)
        ]

        # Initial reset to populate state
        self.reset()
        
        # Validate implementation after full initialization
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_vel = np.array([0.0, 0.0], dtype=float)
        self.player_angle = -90.0  # Pointing up
        self.player_health = self.MAX_HEALTH

        # Game state
        self.ore_collected = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.asteroid_base_speed = 1.0

        # World state
        self.particles.clear()
        self.asteroids = [self._create_asteroid(on_screen=False) for _ in range(self.NUM_ASTEROIDS)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        # Unpack actions
        movement, space_held, _ = action
        space_held = space_held == 1

        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_asteroids()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        mining_reward, mined_something = self._handle_mining(space_held)
        reward += mining_reward
        
        # Small penalty for inaction
        if not mined_something:
            reward -= 0.01

        self._update_particles()
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.asteroid_base_speed = min(3.0, self.asteroid_base_speed + 0.05)

        # --- Check Termination ---
        terminated = False
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            self.win = False
            reward -= 100  # Large penalty for losing
            self._create_explosion(self.player_pos, 50, (255,100,0))
        elif self.ore_collected >= self.WIN_ORE:
            self.ore_collected = self.WIN_ORE
            self.game_over = True
            terminated = True
            self.win = True
            reward += 100  # Large reward for winning
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            self.win = False

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 3:  # Turn left
            self.player_angle -= self.PLAYER_TURN_RATE
        if movement == 4:  # Turn right
            self.player_angle += self.PLAYER_TURN_RATE

        rad_angle = math.radians(self.player_angle)
        if movement == 1:  # Thrust
            self.player_vel[0] += math.cos(rad_angle) * self.PLAYER_THRUST
            self.player_vel[1] += math.sin(rad_angle) * self.PLAYER_THRUST
            # sfx_thrust
            # Add thrust particles
            if self.steps % 2 == 0:
                self._create_thrust_particles()

        # Apply drag
        drag = self.PLAYER_BRAKE_DRAG if movement == 2 else self.PLAYER_DRAG
        self.player_vel *= drag

        # Update position and wrap around screen
        self.player_pos += self.player_vel
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            asteroid['pos'] += asteroid['vel'] * self.asteroid_base_speed
            asteroid['pos'][0] %= self.WIDTH
            asteroid['pos'][1] %= self.HEIGHT
            asteroid['angle'] = (asteroid['angle'] + asteroid['rot_speed']) % 360

    def _handle_collisions(self):
        reward = 0
        for i, asteroid in enumerate(self.asteroids):
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                # sfx_collision
                self.player_health -= 1
                reward -= 5
                self._create_explosion(asteroid['pos'], 30, (200, 200, 200))
                self.asteroids[i] = self._create_asteroid()
                # Apply knockback to player
                knockback = (self.player_pos - asteroid['pos']) / dist * 5
                self.player_vel += knockback
        return reward

    def _handle_mining(self, space_held):
        reward = 0
        mined_something = False
        if not space_held:
            return reward, mined_something

        # Find closest asteroid
        closest_asteroid = None
        min_dist = float('inf')
        for asteroid in self.asteroids:
            dist = np.linalg.norm(self.player_pos - asteroid['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_asteroid = asteroid

        if closest_asteroid and min_dist < self.MINE_RANGE:
            # sfx_mining_beam
            mined_something = True
            
            # Add mining beam particles
            self._create_beam_particles(closest_asteroid['pos'])

            # Calculate ore yield and damage risk
            proximity_factor = max(0, 1 - (min_dist / self.MINE_RANGE))
            ore_yield = proximity_factor**2 * self.MINE_RATE
            
            if closest_asteroid['ore'] > 0:
                mined_ore = min(ore_yield, closest_asteroid['ore'])
                self.ore_collected += mined_ore
                closest_asteroid['ore'] -= mined_ore
                reward += mined_ore * 0.1 # Continuous reward for mining
                
                # Create ore collection particles
                if self.steps % 4 == 0:
                    self._create_ore_particles(closest_asteroid['pos'])

                if closest_asteroid['ore'] <= 0:
                    # sfx_asteroid_depleted
                    # Bonus reward for depleting an asteroid
                    if closest_asteroid['size_cat'] == 'large': reward += 1.0
                    elif closest_asteroid['size_cat'] == 'medium': reward += 0.5
                    else: reward += 0.2
                    
                    idx = self.asteroids.index(closest_asteroid)
                    self.asteroids[idx] = self._create_asteroid()


            # Risk of damage from mining too close
            damage_chance = proximity_factor * self.MINE_DAMAGE_PROB
            if random.random() < damage_chance:
                # sfx_mining_damage
                self.player_health -= 1
                reward -= 5
                self._create_explosion(self.player_pos, 15, (255, 150, 0))

        return reward, mined_something

    def _create_asteroid(self, on_screen=True):
        size_cat = random.choices(['small', 'medium', 'large'], weights=[0.5, 0.35, 0.15], k=1)[0]
        if size_cat == 'small':
            radius = random.uniform(12, 18)
            ore = random.uniform(5, 10)
        elif size_cat == 'medium':
            radius = random.uniform(20, 30)
            ore = random.uniform(15, 25)
        else: # large
            radius = random.uniform(35, 45)
            ore = random.uniform(30, 50)
            
        if on_screen:
            pos = np.array([random.uniform(0, self.WIDTH), random.uniform(0, self.HEIGHT)], dtype=float)
        else: # Spawn off-screen
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top': pos = np.array([random.uniform(0, self.WIDTH), -radius], dtype=float)
            elif edge == 'bottom': pos = np.array([random.uniform(0, self.WIDTH), self.HEIGHT + radius], dtype=float)
            elif edge == 'left': pos = np.array([-radius, random.uniform(0, self.HEIGHT)], dtype=float)
            else: pos = np.array([self.WIDTH + radius, random.uniform(0, self.HEIGHT)], dtype=float)

        angle = random.uniform(0, 2 * math.pi)
        vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * random.uniform(0.3, 0.8)

        # Generate points for a jagged polygon
        num_vertices = random.randint(7, 12)
        points = []
        for i in range(num_vertices):
            angle = 2 * math.pi * i / num_vertices
            dist = random.uniform(radius * 0.8, radius * 1.2)
            points.append((math.cos(angle) * dist, math.sin(angle) * dist))

        return {
            'pos': pos, 'vel': vel, 'radius': radius, 'ore': ore, 'initial_ore': ore,
            'angle': random.uniform(0, 360), 'rot_speed': random.uniform(-0.5, 0.5),
            'points': points, 'size_cat': size_cat
        }

    # --- Particle System ---
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p.get('fade', False):
                p['size'] = max(0, p['size'] - p['fade_rate'])

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': random.randint(15, 30),
                'color': color, 'size': random.uniform(2, 5), 'type': 'circle'
            })
    
    def _create_thrust_particles(self):
        angle = math.radians(self.player_angle + 180 + random.uniform(-15, 15))
        speed = random.uniform(2, 4)
        vel = np.array([math.cos(angle), math.sin(angle)]) * speed + self.player_vel
        pos_offset_angle = math.radians(self.player_angle + 180)
        pos = self.player_pos + np.array([math.cos(pos_offset_angle), math.sin(pos_offset_angle)]) * self.PLAYER_RADIUS
        self.particles.append({
            'pos': pos, 'vel': vel, 'life': random.randint(10, 20),
            'color': random.choice([(255, 150, 0), (255, 200, 50)]),
            'size': random.uniform(3, 6), 'type': 'circle', 'fade': True, 'fade_rate': 0.2
        })

    def _create_beam_particles(self, target_pos):
        direction = target_pos - self.player_pos
        dist = np.linalg.norm(direction)
        if dist == 0: return
        direction /= dist
        
        start_pos = self.player_pos + direction * self.PLAYER_RADIUS
        vel = direction * 8
        self.particles.append({
            'pos': start_pos, 'vel': vel, 'life': int(dist / 8),
            'color': self.COLOR_BEAM, 'size': 2, 'type': 'line', 'target': target_pos
        })
    
    def _create_ore_particles(self, asteroid_pos):
        direction = self.player_pos - asteroid_pos
        dist = np.linalg.norm(direction)
        if dist == 0: return
        direction /= dist
        
        vel = direction * random.uniform(2, 4)
        self.particles.append({
            'pos': asteroid_pos.copy() + np.random.uniform(-5, 5, 2), 
            'vel': vel, 'life': int(dist / 3),
            'color': self.COLOR_ORE_PARTICLE, 'size': 2, 'type': 'circle'
        })


    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)

    def _render_asteroids(self):
        for a in self.asteroids:
            self._draw_rotated_polygon(a['pos'], a['points'], a['angle'], self.COLOR_ASTEROID_OUTLINE, self.COLOR_ASTEROID)

    def _render_player(self):
        rad = math.radians(self.player_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        p1 = (self.player_pos[0] + self.PLAYER_RADIUS * cos_a, self.player_pos[1] + self.PLAYER_RADIUS * sin_a)
        p2 = (self.player_pos[0] + self.PLAYER_RADIUS * 0.8 * math.cos(rad + 2.3), self.player_pos[1] + self.PLAYER_RADIUS * 0.8 * math.sin(rad + 2.3))
        p3 = (self.player_pos[0] + self.PLAYER_RADIUS * 0.8 * math.cos(rad - 2.3), self.player_pos[1] + self.PLAYER_RADIUS * 0.8 * math.sin(rad - 2.3))
        
        points = [p1, p2, p3]
        
        # Draw glow effect
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_PLAYER_GLOW)
        
        # Draw main ship body
        inner_scale = 0.7
        p1_in = (self.player_pos[0] + self.PLAYER_RADIUS * inner_scale * cos_a, self.player_pos[1] + self.PLAYER_RADIUS * inner_scale * sin_a)
        p2_in = (self.player_pos[0] + self.PLAYER_RADIUS * 0.6 * math.cos(rad + 2.1), self.player_pos[1] + self.PLAYER_RADIUS * 0.6 * math.sin(rad + 2.1))
        p3_in = (self.player_pos[0] + self.PLAYER_RADIUS * 0.6 * math.cos(rad - 2.1), self.player_pos[1] + self.PLAYER_RADIUS * 0.6 * math.sin(rad - 2.1))
        
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in [p1_in, p2_in, p3_in]], self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if p['type'] == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), p['color'])
            elif p['type'] == 'line':
                end_pos = p['pos'] + p['vel']
                pygame.draw.line(self.screen, p['color'], pos, (int(end_pos[0]), int(end_pos[1])), int(p['size']))

    def _render_ui(self):
        # Ore counter
        ore_text = self.font_small.render(f"ORE: {int(self.ore_collected)} / {self.WIN_ORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ore_text, (10, 10))

        # Health bar
        health_bar_width = 150
        health_bar_height = 15
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (self.WIDTH - health_bar_width - 10, 10, health_bar_width, health_bar_height))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (self.WIDTH - health_bar_width - 10, 10, health_bar_width * health_ratio, health_bar_height))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_HEALTH_BG
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _draw_rotated_polygon(self, pos, points, angle, outline_color, fill_color):
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        
        rotated_points = []
        for x, y in points:
            rx = x * cos_a - y * sin_a + pos[0]
            ry = x * sin_a + y * cos_a + pos[1]
            rotated_points.append((int(rx), int(ry)))
            
        if len(rotated_points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, rotated_points, fill_color)
            pygame.gfxdraw.aapolygon(self.screen, rotated_points, outline_color)

    def _get_info(self):
        return {
            "score": self.ore_collected, # using ore as score for simplicity
            "steps": self.steps,
            "health": self.player_health,
            "ore": self.ore_collected,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0 # Not used in this game
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        # Update display title with info
        pygame.display.set_caption(f"Ore: {info['ore']:.1f} | Health: {info['health']} | Reward: {total_reward:.2f}")

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for 'r' to reset
            wait_for_reset = True
            while wait_for_reset:
                 for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        running = False
                        wait_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()