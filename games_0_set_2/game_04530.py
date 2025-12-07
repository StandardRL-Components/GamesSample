import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in a headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to mine nearby asteroids. "
        "Avoid collisions and manage your fuel."
    )

    game_description = (
        "Pilot a spaceship to mine asteroids for valuable minerals. "
        "Larger asteroids have stronger gravity. Collect 20 minerals to win, but don't run out of fuel!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_MARGIN = 500  # How far outside the screen things can exist
        self.WORLD_WIDTH = self.WIDTH + 2 * self.WORLD_MARGIN
        self.WORLD_HEIGHT = self.HEIGHT + 2 * self.WORLD_MARGIN
        self.WIN_MINERALS = 20
        self.MAX_STEPS = 2500
        self.INITIAL_ASTEROIDS = 5

        # --- Colors ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.COLOR_LASER = (255, 200, 0)
        self.COLOR_FUEL_FG = (0, 255, 100)
        self.COLOR_FUEL_BG = (100, 0, 0)
        self.COLOR_UI_TEXT = (220, 220, 255)
        self.COLOR_UI_BG = (25, 30, 50, 150) # RGBA

        # --- Physics & Gameplay ---
        self.PLAYER_ACCELERATION = 0.25
        self.PLAYER_DRAG = 0.98
        self.PLAYER_MAX_SPEED = 5
        self.PLAYER_RADIUS = 12
        self.MAX_FUEL = 1000
        self.FUEL_COST_MOVE = 0.05
        self.FUEL_COST_MINE = 0.5
        self.FUEL_PENALTY_COLLISION = 100
        self.MINING_RANGE = 100
        self.MINING_RATE = 1  # Minerals per step
        self.GRAVITY_CONSTANT = 700

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 60)

        # --- Internal State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_fuel = None
        self.asteroids = []
        self.particles = []
        self.stars = []
        self.mining_target = None
        self.np_random = None

        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() is for debugging and not part of the standard __init__

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.math.Vector2(self.WORLD_WIDTH / 2, self.WORLD_HEIGHT / 2)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_fuel = self.MAX_FUEL

        self.asteroids = []
        self._spawn_asteroids(self.INITIAL_ASTEROIDS)

        self.particles = []
        self.stars = [
            (
                self.np_random.uniform(0, self.WORLD_WIDTH),
                self.np_random.uniform(0, self.WORLD_HEIGHT),
                self.np_random.uniform(0.5, 1.5), # parallax factor
                self.np_random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 200)])
            )
            for _ in range(200)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Player Movement ---
        accel = pygame.math.Vector2(0, 0)
        if movement == 1: accel.y -= self.PLAYER_ACCELERATION
        if movement == 2: accel.y += self.PLAYER_ACCELERATION
        if movement == 3: accel.x -= self.PLAYER_ACCELERATION
        if movement == 4: accel.x += self.PLAYER_ACCELERATION
        
        self.player_vel += accel
        
        # --- Gravity ---
        for asteroid in self.asteroids:
            dist_vec = asteroid['pos'] - self.player_pos
            dist_sq = dist_vec.length_squared()
            if dist_sq > 1: # Avoid division by zero
                force_mag = self.GRAVITY_CONSTANT * asteroid['size'] / dist_sq
                self.player_vel += dist_vec.normalize() * force_mag

        # --- Drag & Speed Limit ---
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        self.player_vel *= self.PLAYER_DRAG
        
        # --- Fuel for Movement ---
        move_fuel_cost = self.player_vel.length() * self.FUEL_COST_MOVE
        self.player_fuel -= move_fuel_cost
        reward -= move_fuel_cost * 0.1 # Small penalty for fuel use

        # --- Update Position & Engine Particles ---
        self.player_pos += self.player_vel
        if accel.length() > 0: # Engine trail
            self._create_particles(self.player_pos - self.player_vel, 1, self.COLOR_PLAYER, 3, 0.5, -accel.normalize() * 2)

        # --- World Boundaries ---
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WORLD_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.WORLD_HEIGHT)

        # --- Mining ---
        self.mining_target = None
        if space_held and self.player_fuel > 0:
            closest_asteroid, min_dist = None, float('inf')
            for asteroid in self.asteroids:
                dist = self.player_pos.distance_to(asteroid['pos'])
                if dist < min_dist:
                    min_dist = dist
                    closest_asteroid = asteroid
            
            if closest_asteroid and min_dist < self.MINING_RANGE + closest_asteroid['radius']:
                self.mining_target = closest_asteroid
                # sfx: mining_laser_loop.wav
                self.player_fuel -= self.FUEL_COST_MINE
                
                mined_amount = min(self.MINING_RATE, closest_asteroid['minerals'])
                closest_asteroid['minerals'] -= mined_amount
                self.score += mined_amount
                reward += mined_amount * 1.0 # Reward for each mineral unit

                self._create_particles(closest_asteroid['pos'], 1, self.COLOR_LASER, 4, 0.8, (self.player_pos - closest_asteroid['pos']).normalize() * 3)

                if closest_asteroid['minerals'] <= 0:
                    reward += 10 # Bonus for depleting an asteroid
                    # sfx: explosion.wav
                    self._create_particles(closest_asteroid['pos'], 30, (200, 200, 200), 8, 1.5)
                    self.asteroids.remove(closest_asteroid)
                    self._spawn_asteroids(1)
                    
                    # Difficulty scaling
                    if self.score // 5 > (self.score - mined_amount) // 5:
                        self._spawn_asteroids(1)


        # --- Collisions ---
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < self.PLAYER_RADIUS + asteroid['radius']:
                # sfx: collision_thud.wav
                self.player_fuel -= self.FUEL_PENALTY_COLLISION
                reward -= 20 # Penalty for collision
                
                # Knockback
                knockback_vec = (self.player_pos - asteroid['pos']).normalize()
                self.player_vel += knockback_vec * 5
                
                self._create_particles(self.player_pos, 15, (255, 255, 100), 5, 0.8)
                asteroid['hit_flash'] = 10 # frames to flash


        # --- Update Particles & Asteroids ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        for asteroid in self.asteroids:
            asteroid['angle'] += asteroid['rot_speed']
            if 'hit_flash' in asteroid:
                asteroid['hit_flash'] -= 1
                if asteroid['hit_flash'] <= 0:
                    del asteroid['hit_flash']

        # --- Termination Conditions ---
        terminated = False
        truncated = False
        if self.score >= self.WIN_MINERALS:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.player_fuel <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.player_fuel = 0
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # --- Camera ---
        camera_offset = self.player_pos - pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)

        # --- Render Background ---
        self.screen.fill(self.COLOR_BG)
        for x, y, parallax, color in self.stars:
            sx = (x - camera_offset.x * parallax) % self.WIDTH
            sy = (y - camera_offset.y * parallax) % self.HEIGHT
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), int(parallax))

        # --- Render Asteroids ---
        for asteroid in self.asteroids:
            screen_pos = asteroid['pos'] - camera_offset
            if -50 < screen_pos.x < self.WIDTH + 50 and -50 < screen_pos.y < self.HEIGHT + 50:
                self._draw_asteroid(self.screen, asteroid, screen_pos)

        # --- Render Particles ---
        for p in self.particles:
            screen_pos = p['pos'] - camera_offset
            size = p['size'] * (p['life'] / p['max_life'])
            if size > 1:
                pygame.draw.circle(self.screen, p['color'], (int(screen_pos.x), int(screen_pos.y)), int(size))

        # --- Render Mining Laser ---
        if self.mining_target:
            start_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
            end_pos = self.mining_target['pos'] - camera_offset
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
            # Add glow
            pygame.draw.aaline(self.screen, (*self.COLOR_LASER, 50), start_pos, end_pos, 6)

        # --- Render Player ---
        player_screen_pos = (self.WIDTH // 2, self.HEIGHT // 2)
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS + 4, (*self.COLOR_PLAYER_GLOW, 100))
        # Body
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        # Cockpit
        pygame.gfxdraw.filled_circle(self.screen, player_screen_pos[0], player_screen_pos[1], self.PLAYER_RADIUS // 2, (200, 255, 255))

        # --- Render UI ---
        self._render_ui()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # UI Background
        ui_surf = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        pygame.draw.rect(ui_surf, self.COLOR_UI_BG, (0, 0, self.WIDTH, 50))
        self.screen.blit(ui_surf, (0, 0))

        # Mineral Score
        score_text = self.font_ui.render(f"MINERALS: {int(self.score)} / {self.WIN_MINERALS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 12))

        # Fuel Gauge
        fuel_percent = self.player_fuel / self.MAX_FUEL if self.MAX_FUEL > 0 else 0
        bar_width = 200
        bar_height = 20
        fuel_bar_x = self.WIDTH - bar_width - 15
        
        # Color gradient for fuel
        fuel_color = (
            int(self.COLOR_FUEL_BG[0] + (self.COLOR_FUEL_FG[0] - self.COLOR_FUEL_BG[0]) * fuel_percent),
            int(self.COLOR_FUEL_BG[1] + (self.COLOR_FUEL_FG[1] - self.COLOR_FUEL_BG[1]) * fuel_percent),
            int(self.COLOR_FUEL_BG[2] + (self.COLOR_FUEL_FG[2] - self.COLOR_FUEL_BG[2]) * fuel_percent),
        )

        pygame.draw.rect(self.screen, self.COLOR_FUEL_BG, (fuel_bar_x, 15, bar_width, bar_height))
        pygame.draw.rect(self.screen, fuel_color, (fuel_bar_x, 15, int(bar_width * fuel_percent), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (fuel_bar_x, 15, bar_width, bar_height), 1)
        
        fuel_text = self.font_ui.render("FUEL", True, self.COLOR_UI_TEXT)
        self.screen.blit(fuel_text, (fuel_bar_x - fuel_text.get_width() - 10, 12))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_MINERALS:
                end_text = self.font_big.render("MISSION COMPLETE", True, self.COLOR_FUEL_FG)
            else:
                end_text = self.font_big.render("GAME OVER", True, self.COLOR_FUEL_BG)
                
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fuel": self.player_fuel,
        }

    def _spawn_asteroids(self, count):
        for _ in range(count):
            size = self.np_random.uniform(0.8, 2.0)
            radius = int(size * 15)
            
            # Ensure asteroids don't spawn on top of each other or the player
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(self.WORLD_MARGIN / 2, self.WORLD_WIDTH - self.WORLD_MARGIN / 2),
                    self.np_random.uniform(self.WORLD_MARGIN / 2, self.WORLD_HEIGHT - self.WORLD_MARGIN / 2)
                )
                
                too_close = False
                if self.player_pos and pos.distance_to(self.player_pos) < 200:
                    too_close = True
                
                for other in self.asteroids:
                    if pos.distance_to(other['pos']) < other['radius'] + radius + 50:
                        too_close = True
                        break
                if not too_close:
                    break

            num_vertices = self.np_random.integers(7, 12)
            base_vertices = []
            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices
                dist = radius * self.np_random.uniform(0.8, 1.2)
                base_vertices.append(pygame.math.Vector2(dist, 0).rotate_rad(angle))

            self.asteroids.append({
                'pos': pos,
                'size': size,
                'radius': radius,
                'minerals': int(size * 50),
                'angle': self.np_random.uniform(0, 360),
                'rot_speed': self.np_random.uniform(-0.5, 0.5),
                'base_vertices': base_vertices,
                'color1': (
                    self.np_random.integers(80, 120),
                    self.np_random.integers(70, 110),
                    self.np_random.integers(60, 100)
                ),
                'color2': (
                    self.np_random.integers(50, 90),
                    self.np_random.integers(40, 80),
                    self.np_random.integers(30, 70)
                ),
            })

    def _draw_asteroid(self, surface, asteroid, screen_pos):
        rotated_vertices = [v.rotate(asteroid['angle']) + screen_pos for v in asteroid['base_vertices']]
        
        color = asteroid['color1']
        if 'hit_flash' in asteroid:
            color = (255, 255, 200)

        pygame.gfxdraw.aapolygon(surface, rotated_vertices, asteroid['color2'])
        pygame.gfxdraw.filled_polygon(surface, rotated_vertices, color)

    def _create_particles(self, pos, count, color, max_size, max_life_s, base_vel=None):
        for _ in range(count):
            if base_vel:
                vel = base_vel + pygame.math.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            else:
                vel = pygame.math.Vector2(0,0).from_polar((self.np_random.uniform(1, 4), self.np_random.uniform(0, 360)))
            
            life = int(max_life_s * 30) # 30 FPS
            self.particles.append({
                'pos': pygame.math.Vector2(pos),  # FIX: Create a copy of the Vector2 object
                'vel': vel,
                'color': color,
                'size': self.np_random.uniform(1, max_size),
                'life': life,
                'max_life': life
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    # Setup Pygame display for human play
    pygame.display.set_caption("Asteroid Miner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        # --- Handle User Input ---
        keys = pygame.key.get_pressed()
        
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset(seed=42)
                    terminated = False
                    truncated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Display the rendered frame ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()