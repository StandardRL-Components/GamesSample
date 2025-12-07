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



# Helper class for the player (Hopper)
class Hopper:
    def __init__(self, pos, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.jump_charge = 0.0
        self.on_ground = False

# Helper class for obstacles
class Obstacle:
    def __init__(self, pos, size, rot_speed, rng):
        self.pos = pygame.Vector2(pos)
        self.size = size
        self.angle = rng.uniform(0, 2 * math.pi)
        self.rot_speed = rot_speed
        self.passed = False
        self.color = (255, 50, 50)

    def update(self, dt):
        self.angle += self.rot_speed * dt

    def get_rotated_corners(self):
        half_size = self.size / 2
        points = [
            pygame.Vector2(-half_size, -half_size),
            pygame.Vector2(half_size, -half_size),
            pygame.Vector2(half_size, half_size),
            pygame.Vector2(-half_size, half_size),
        ]
        rotated_points = [p.rotate_rad(self.angle) + self.pos for p in points]
        return rotated_points

# Helper class for particles
class Particle:
    def __init__(self, pos, vel, life, color, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = radius

    def update(self, dt):
        self.life -= dt
        self.pos += self.vel * dt


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    user_guide = (
        "Controls: Hold [Space] to charge a jump, release to leap. Avoid the red obstacles."
    )

    game_description = (
        "A minimalist arcade game. Control a space hopper, jumping to new heights while dodging rotating obstacles. The higher you go, the harder it gets."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    DT = 1.0 / FPS

    # Colors
    COLOR_BG_BOTTOM = (10, 0, 30)
    COLOR_BG_TOP = (40, 0, 80)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_OBSTACLE_GLOW = (255, 20, 20)
    COLOR_PARTICLE = (220, 220, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BAR = (100, 100, 255)
    COLOR_UI_BAR_BG = (50, 50, 80)
    COLOR_SUCCESS_FLASH = (50, 255, 50)
    
    # Physics & Gameplay
    GRAVITY = -350
    JUMP_CHARGE_RATE = 2.0  # units per second
    MIN_JUMP_VEL = 150
    MAX_JUMP_VEL = 300
    TARGET_HEIGHT = 100 # meters
    MAX_STEPS = 10000
    PIXELS_PER_METER = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        if self.render_mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        
        # FIX: Set the display mode to initialize Pygame's video system.
        # This is required even for headless rendering. The "dummy" driver handles
        # the case where no window should appear.
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        self.background_surf = self._create_gradient_background()

        # State variables are initialized in reset()
        self.hopper = None
        self.obstacles = []
        self.particles = []
        self.camera_y = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.was_space_held = False
        self.max_height_achieved = 0.0
        self.last_reward_height = 0.0
        self.milestone_flash_timer = 0
        self.last_milestone = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.hopper = Hopper(pos=(self.WIDTH / 2, 50), radius=10)
        self.obstacles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.was_space_held = False
        self.camera_y = 0.0
        self.max_height_achieved = 0.0
        self.last_reward_height = 0.0
        self.milestone_flash_timer = 0
        self.last_milestone = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Player Input ---
        if space_held:
            self.hopper.jump_charge = min(1.0, self.hopper.jump_charge + self.JUMP_CHARGE_RATE * self.DT)
        
        jump_triggered = not space_held and self.was_space_held and self.hopper.jump_charge > 0
        if jump_triggered:
            jump_power = self.MIN_JUMP_VEL + self.hopper.jump_charge * (self.MAX_JUMP_VEL - self.MIN_JUMP_VEL)
            self.hopper.vel.y = jump_power
            self.hopper.jump_charge = 0.0
            # sfx: jump sound
            self._spawn_jump_particles(20)

        self.was_space_held = space_held

        # --- Physics and Game Logic ---
        # Apply gravity
        self.hopper.vel.y += self.GRAVITY * self.DT
        self.hopper.pos += self.hopper.vel * self.DT
        
        # Ground constraint
        if self.hopper.pos.y < self.hopper.radius:
            self.hopper.pos.y = self.hopper.radius
            self.hopper.vel.y = 0

        # Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update(self.DT)

        # Update obstacles
        for o in self.obstacles:
            o.update(self.DT)

        # --- Difficulty and Spawning ---
        current_height_m = self.max_height_achieved / self.PIXELS_PER_METER
        spawn_chance = (0.02 + 0.001 * current_height_m) # Base chance + scaling
        rot_speed_range = 1.0 + 0.1 * (current_height_m // 25)

        if self.np_random.random() < spawn_chance:
            spawn_y = self.camera_y + self.HEIGHT + 50
            spawn_x = self.np_random.uniform(50, self.WIDTH - 50)
            rot_speed = self.np_random.uniform(-rot_speed_range, rot_speed_range)
            self.obstacles.append(Obstacle((spawn_x, spawn_y), 30, rot_speed, self.np_random))

        # Despawn old obstacles
        self.obstacles = [o for o in self.obstacles if o.pos.y > self.camera_y - 50]
        
        # --- Collision Detection ---
        for obs in self.obstacles:
            if self._check_collision(self.hopper, obs):
                self.game_over = True
                reward -= 10
                # sfx: explosion sound
                self._spawn_collision_particles(50, obs.color)
                break
        
        # --- Update Camera ---
        target_camera_y = self.hopper.pos.y - 120
        self.camera_y += (target_camera_y - self.camera_y) * 0.08

        # --- Reward Calculation ---
        height_in_meters = self.hopper.pos.y / self.PIXELS_PER_METER
        if height_in_meters > self.max_height_achieved:
            height_gain = height_in_meters - self.max_height_achieved
            reward += height_gain * 0.1
            self.max_height_achieved = height_in_meters

        if not jump_triggered and not space_held:
            reward -= 0.005 # smaller penalty than brief, feels better

        # Near-miss reward
        for obs in self.obstacles:
            if not obs.passed and self.hopper.pos.y > obs.pos.y:
                obs.passed = True
                if not self._check_collision(self.hopper, obs):
                    # Check for vertical proximity
                    hopper_bottom = self.hopper.pos.y - self.hopper.radius
                    obs_top_y = max(c.y for c in obs.get_rotated_corners())
                    gap = hopper_bottom - obs_top_y
                    if 0 < gap < 5:
                        reward += 1
                        # sfx: near miss whoosh

        # --- Termination ---
        terminated = self.game_over or self.max_height_achieved >= self.TARGET_HEIGHT or self.steps >= self.MAX_STEPS
        if self.max_height_achieved >= self.TARGET_HEIGHT:
            reward += 100

        self.score += reward
        self.steps += 1
        
        # --- UI Effects ---
        if self.milestone_flash_timer > 0:
            self.milestone_flash_timer -= self.DT
        
        current_milestone = int(self.max_height_achieved / self.PIXELS_PER_METER / 25)
        if current_milestone > self.last_milestone:
            self.last_milestone = current_milestone
            self.milestone_flash_timer = 0.2
            # sfx: milestone reached fanfare

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_collision(self, hopper, obstacle):
        # Circle vs Rotated Rectangle collision
        # 1. Get obstacle corners
        corners = obstacle.get_rotated_corners()
        # 2. Check circle-polygon collision
        # Check if circle center is inside polygon
        if self._point_in_polygon(hopper.pos, corners):
            return True
        # Check distance from circle center to each edge
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            if self._dist_point_to_segment(hopper.pos, p1, p2) < hopper.radius:
                return True
        return False

    def _point_in_polygon(self, point, polygon_corners):
        # Ray-casting algorithm
        x, y = point.x, point.y
        n = len(polygon_corners)
        inside = False
        p1x, p1y = polygon_corners[0].x, polygon_corners[0].y
        for i in range(n + 1):
            p2x, p2y = polygon_corners[i % n].x, polygon_corners[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _dist_point_to_segment(self, p, a, b):
        # Distance from point p to line segment ab
        ab = b - a
        ap = p - a
        if ab.length_squared() == 0:
            return ap.length()
        t = ab.dot(ap) / ab.length_squared()
        t = max(0, min(1, t))
        closest_point = a + t * ab
        return (p - closest_point).length()

    def _world_to_screen(self, pos):
        x = pos[0]
        y = self.HEIGHT - (pos[1] - self.camera_y)
        return int(x), int(y)

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = max(0, p.life / p.max_life)
            color = (*p.color, int(alpha * 255))
            screen_pos = self._world_to_screen(p.pos)
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(p.radius), color)

        # Obstacles
        for o in self.obstacles:
            screen_corners = [self._world_to_screen(p) for p in o.get_rotated_corners()]
            pygame.gfxdraw.aapolygon(self.screen, screen_corners, self.COLOR_OBSTACLE_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, screen_corners, self.COLOR_OBSTACLE)

        # Player
        hopper_screen_pos = self._world_to_screen(self.hopper.pos)
        # Charge indicator
        if self.hopper.jump_charge > 0:
            charge_radius = int(self.hopper.radius + 2 + self.hopper.jump_charge * 8)
            charge_alpha = int(100 + self.hopper.jump_charge * 155)
            pygame.gfxdraw.aacircle(self.screen, hopper_screen_pos[0], hopper_screen_pos[1], charge_radius, (*self.COLOR_PLAYER_GLOW, charge_alpha))
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, hopper_screen_pos[0], hopper_screen_pos[1], self.hopper.radius + 3, (*self.COLOR_PLAYER_GLOW, 100))
        # Body
        pygame.gfxdraw.aacircle(self.screen, hopper_screen_pos[0], hopper_screen_pos[1], self.hopper.radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, hopper_screen_pos[0], hopper_screen_pos[1], self.hopper.radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Milestone Flash
        if self.milestone_flash_timer > 0:
            alpha = int(200 * (self.milestone_flash_timer / 0.2))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_SUCCESS_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Height Bar
        bar_height = self.HEIGHT - 40
        bar_x = 20
        bar_y = 20
        
        progress = min(1.0, self.max_height_achieved / self.TARGET_HEIGHT)
        fill_height = int(bar_height * progress)

        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, 15, bar_height))
        if fill_height > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y + bar_height - fill_height, 15, fill_height))
        
        # Height Text
        height_text = f"{int(self.max_height_achieved)}m"
        text_surf = self.font_large.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (bar_x + 25, bar_y))
        
        # Score Text
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.max_height_achieved,
        }
        
    def _create_gradient_background(self):
        surf = pygame.Surface((1, self.HEIGHT)).convert_alpha()
        for y in range(self.HEIGHT):
            progress = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_BOTTOM[i] * (1 - progress) + self.COLOR_BG_TOP[i] * progress)
                for i in range(3)
            ]
            surf.set_at((0, y), color)
        return pygame.transform.scale(surf, (self.WIDTH, self.HEIGHT))

    def _spawn_jump_particles(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
            speed = self.np_random.uniform(50, 150)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            pos = self.hopper.pos - pygame.Vector2(0, self.hopper.radius)
            life = self.np_random.uniform(0.3, 0.8)
            radius = self.np_random.uniform(1, 3)
            self.particles.append(Particle(pos, vel, life, self.COLOR_PARTICLE, radius))

    def _spawn_collision_particles(self, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 250)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            pos = self.hopper.pos.copy()
            life = self.np_random.uniform(0.5, 1.5)
            radius = self.np_random.uniform(1, 4)
            p_color = color if self.np_random.random() > 0.3 else self.COLOR_PLAYER
            self.particles.append(Particle(pos, vel, life, p_color, radius))

    def close(self):
        pygame.font.quit()
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="human") # Change to "human" to see it
    
    # --- Human player setup ---
    if env.render_mode == "human":
        pygame.display.set_caption("Space Hopper")
        # Get the screen surface that was created inside the environment
        screen = pygame.display.get_surface()
        
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    while not terminated:
        # Human controls
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = [0, 1 if space_held else 0, 0] # Movement=None, Space, Shift=None

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render for human player
        if env.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            # The observation is already a rendered frame, so we just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        env.clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {info['score']:.2f}, Height: {info['height']:.2f}m")
    env.close()