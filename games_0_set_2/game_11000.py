import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


# Set the SDL video driver to "dummy" to run Pygame headlessly.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# It's good practice to import these after setting the environment variable,
# although it's not strictly necessary for all systems.
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a rotating arrow.
    The goal is to collect all stars while avoiding moving obstacles.
    Colliding with obstacles shrinks the arrow, and if it becomes too small, the game ends.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing metadata ---
    game_description = (
        "Control a rotating arrow to collect all stars while avoiding moving obstacles "
        "that shrink you on contact."
    )
    user_guide = "Controls: Use ← and → arrow keys to rotate the arrow. Collect all stars to win."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG_TOP = (10, 0, 20)
    COLOR_BG_BOTTOM = (40, 0, 60)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_STAR = (255, 255, 0)
    COLOR_STAR_GLOW = (150, 150, 0)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (150, 0, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (20, 20, 20)

    # Player settings
    PLAYER_BASE_SIZE = 20
    PLAYER_INITIAL_SIZE_MULTIPLIER = 1.0
    PLAYER_SPEED = 3.0
    PLAYER_ROTATION_SPEED = math.radians(5) # 5 degrees per step
    PLAYER_SHRINK_RATE = 0.20 # Shrinks by 20% on hit
    MIN_PLAYER_SIZE_MULTIPLIER = 0.20 # Loss condition

    # Game settings
    NUM_STARS = 10
    STAR_RADIUS = 8
    NUM_OBSTACLES = 5
    OBSTACLE_MIN_RADIUS = 15
    OBSTACLE_MAX_RADIUS = 30
    OBSTACLE_MAX_SPEED = 1.5
    MAX_STEPS = 2000
    
    # Reward structure
    REWARD_SURVIVAL = 0.01
    REWARD_STAR_COLLECT = 10.0
    REWARD_OBSTACLE_HIT = -5.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
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
        self.font = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_angle = None
        self.player_size_multiplier = None
        self.stars = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.stars_collected = None
        self.game_over = None
        self.last_obstacle_hit_step = -10 # Cooldown for obstacle hit penalty
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_angle = self.np_random.uniform(0, 2 * math.pi)
        self.player_size_multiplier = self.PLAYER_INITIAL_SIZE_MULTIPLIER
        
        # Initialize game state
        self.steps = 0
        self.stars_collected = 0
        self.game_over = False
        self.last_obstacle_hit_step = -10

        # Generate obstacles and stars
        self.obstacles = self._generate_obstacles()
        self.stars = self._generate_stars()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        reward = self.REWARD_SURVIVAL

        # --- Update Game Logic ---
        self._handle_action(movement)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        
        # --- Handle Collisions & Events ---
        collision_reward = self._check_collisions()
        reward += collision_reward

        # --- Check Termination ---
        self.steps += 1
        terminated = False
        win = self.stars_collected >= self.NUM_STARS
        loss = self.player_size_multiplier < self.MIN_PLAYER_SIZE_MULTIPLIER
        timeout = self.steps >= self.MAX_STEPS

        if win:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif loss:
            reward += self.REWARD_LOSS
            terminated = True
            self.game_over = True
        elif timeout:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False, # Truncated is not used in this environment
            self._get_info()
        )

    def _handle_action(self, movement):
        if movement == 1 or movement == 3: # 1=left, 3=right
            rotation_dir = -1 if movement == 1 else 1
            self.player_angle += rotation_dir * self.PLAYER_ROTATION_SPEED
        # Other movement actions are no-ops for rotation

    def _update_player(self):
        self.player_pos.x += self.PLAYER_SPEED * math.cos(self.player_angle)
        self.player_pos.y += self.PLAYER_SPEED * math.sin(self.player_angle)

        # Screen wrap
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle['pos'] += obstacle['vel']
            # Bounce off walls
            if obstacle['pos'].x - obstacle['radius'] < 0 or obstacle['pos'].x + obstacle['radius'] > self.WIDTH:
                obstacle['vel'].x *= -1
            if obstacle['pos'].y - obstacle['radius'] < 0 or obstacle['pos'].y + obstacle['radius'] > self.HEIGHT:
                obstacle['vel'].y *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_collisions(self):
        reward = 0.0
        
        # Star collisions
        for star in self.stars[:]:
            if star['pos'].distance_to(self.player_pos) < self.PLAYER_BASE_SIZE * self.player_size_multiplier + self.STAR_RADIUS:
                self.stars.remove(star)
                self.stars_collected += 1
                reward += self.REWARD_STAR_COLLECT
                self._create_particles(star['pos'], self.COLOR_STAR, 30)
        
        # Obstacle collisions (with cooldown)
        if self.steps > self.last_obstacle_hit_step + 5: # 5-step invulnerability
            player_points = self._get_player_points()
            for obstacle in self.obstacles:
                if self._is_triangle_colliding_with_circle(player_points, obstacle['pos'], obstacle['radius']):
                    self.player_size_multiplier *= (1.0 - self.PLAYER_SHRINK_RATE)
                    reward += self.REWARD_OBSTACLE_HIT
                    self.last_obstacle_hit_step = self.steps
                    self._create_particles(self.player_pos, self.COLOR_OBSTACLE, 50)
                    break # Only one hit per frame
        
        return reward

    def _get_observation(self):
        self._render_background()
        self._render_obstacles()
        self._render_stars()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "stars_collected": self.stars_collected,
            "player_size": self.player_size_multiplier,
            "steps": self.steps,
        }

    # --- Generation Methods ---
    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            radius = self.np_random.uniform(self.OBSTACLE_MIN_RADIUS, self.OBSTACLE_MAX_RADIUS)
            pos = pygame.Vector2(
                self.np_random.uniform(radius, self.WIDTH - radius),
                self.np_random.uniform(radius, self.HEIGHT - radius)
            )
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, self.OBSTACLE_MAX_SPEED)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            obstacles.append({'pos': pos, 'radius': radius, 'vel': vel})
        return obstacles

    def _generate_stars(self):
        stars = []
        min_dist_from_obstacle = self.OBSTACLE_MAX_RADIUS + self.STAR_RADIUS + 10
        for _ in range(self.NUM_STARS):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(20, self.WIDTH - 20),
                    self.np_random.uniform(20, self.HEIGHT - 20)
                )
                # Ensure stars are not inside obstacles at start
                valid_pos = True
                for obs in self.obstacles:
                    if pos.distance_to(obs['pos']) < min_dist_from_obstacle:
                        valid_pos = False
                        break
                if valid_pos:
                    stars.append({'pos': pos})
                    break
        return stars

    # --- Rendering Methods ---
    def _render_background(self):
        # Efficient gradient by drawing wide lines
        for y in range(self.HEIGHT):
            color_ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[0] * color_ratio),
                int(self.COLOR_BG_TOP[1] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[1] * color_ratio),
                int(self.COLOR_BG_TOP[2] * (1 - color_ratio) + self.COLOR_BG_BOTTOM[2] * color_ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_player(self):
        points = self._get_player_points()
        # Glow effect: draw a larger, semi-transparent triangle first
        glow_points = self._get_player_points(scale=1.8)
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in glow_points], self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in glow_points], self.COLOR_PLAYER_GLOW)
        
        # Main player triangle
        pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], self.COLOR_PLAYER)

    def _render_stars(self):
        for star in self.stars:
            self._draw_star(self.screen, star['pos'], self.STAR_RADIUS, self.COLOR_STAR, self.COLOR_STAR_GLOW)

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            pos = (int(obstacle['pos'].x), int(obstacle['pos'].y))
            radius = int(obstacle['radius'])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 4, self.COLOR_OBSTACLE_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 4, self.COLOR_OBSTACLE_GLOW)
            # Main circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_OBSTACLE)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            color = (*p['color'], alpha)
            # Simple rect particle for performance
            rect = pygame.Rect(int(p['pos'].x), int(p['pos'].y), p['size'], p['size'])
            shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, rect)

    def _render_ui(self):
        score_text = f"Stars: {self.stars_collected} / {self.NUM_STARS}"
        shadow_surf = self.font.render(score_text, True, self.COLOR_UI_SHADOW)
        text_surf = self.font.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(shadow_surf, (12, 12))
        self.screen.blit(text_surf, (10, 10))

    # --- Helper & Utility Methods ---
    def _get_player_points(self, scale=1.0):
        size = self.PLAYER_BASE_SIZE * self.player_size_multiplier * scale
        p1 = self.player_pos + pygame.Vector2(size, 0).rotate_rad(self.player_angle)
        p2 = self.player_pos + pygame.Vector2(-size / 2, size / 2).rotate_rad(self.player_angle)
        p3 = self.player_pos + pygame.Vector2(-size / 2, -size / 2).rotate_rad(self.player_angle)
        return [p1, p2, p3]
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'color': color,
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(2, 4)
            })

    @staticmethod
    def _draw_star(surface, center, radius, color, glow_color):
        points = []
        for i in range(10):
            angle = math.pi / 5 * i
            r = radius if i % 2 == 0 else radius / 2
            points.append((center.x + r * math.cos(angle), center.y + r * math.sin(angle)))
        
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.aapolygon(surface, int_points, glow_color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)

    @staticmethod
    def _is_triangle_colliding_with_circle(triangle_points, circle_center, circle_radius):
        # 1. Check if any vertex is inside the circle
        for point in triangle_points:
            if point.distance_squared_to(circle_center) < circle_radius ** 2:
                return True
        
        # 2. Check distance from circle center to each edge
        for i in range(3):
            p1 = triangle_points[i]
            p2 = triangle_points[(i + 1) % 3]
            
            line_vec = p2 - p1
            if line_vec.length_squared() == 0: continue
            
            point_vec = circle_center - p1
            t = point_vec.dot(line_vec) / line_vec.dot(line_vec)
            t = max(0, min(1, t)) # Clamp to segment
            
            closest_point_on_segment = p1 + t * line_vec
            if closest_point_on_segment.distance_squared_to(circle_center) < circle_radius ** 2:
                return True
        return False
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage for Human Play ---
    # This block allows you to play the game with a real window.
    # It will not run in a headless environment.
    
    # Unset the dummy driver to allow a real window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Arrow Dodger")
    clock = pygame.time.Clock()

    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 1 # Rotate left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 3 # Rotate right
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Info: {info}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS

    env.close()