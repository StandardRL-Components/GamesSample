
# Generated: 2025-08-28T06:34:46.562788
# Source Brief: brief_02971.md
# Brief Index: 2971

        
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
        "Controls: Use ←→ to adjust power and ↑↓ to adjust angle. Press space to launch the jumper."
    )

    game_description = (
        "A ski jumping arcade game. Set your launch angle and power to achieve the maximum distance. "
        "Accumulate 100m total distance to win, but watch out for trees on the slope!"
    )

    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    FPS = 30

    # Colors
    COLOR_BG = pygame.Color("#74BDEB")  # Sky Blue
    COLOR_SNOW = pygame.Color("#F0F8FF") # Alice Blue
    COLOR_MOUNTAIN = pygame.Color("#495A65")
    COLOR_MOUNTAIN_SNOW = pygame.Color("#D8E6F2")
    COLOR_JUMPER = pygame.Color("#FFD700") # Gold
    COLOR_OBSTACLE = pygame.Color("#8B4513") # Saddle Brown
    COLOR_OBSTACLE_LEAVES = pygame.Color("#228B22") # Forest Green
    COLOR_UI_TEXT = pygame.Color("#1E2D38")
    COLOR_UI_BG = pygame.Color(255, 255, 255, 128)
    COLOR_POWER_BAR = pygame.Color("#FF4500") # OrangeRed
    COLOR_POWER_BAR_BG = pygame.Color("#555555")
    COLOR_ANGLE_INDICATOR = pygame.Color("#E53935")

    # Game States
    STATE_SETUP = 0
    STATE_FLIGHT = 1
    STATE_LANDED = 2
    STATE_CRASHED = 3

    # Physics
    GRAVITY = 1200.0  # pixels/s^2
    MAX_POWER = 100.0
    MIN_POWER = 10.0
    POWER_TO_VEL_SCALE = 4.5
    MIN_ANGLE = -45  # degrees from horizontal
    MAX_ANGLE = 0    # degrees from horizontal

    # World
    RAMP_HEIGHT = 150
    RAMP_LENGTH = 180
    SLOPE_START_Y = HEIGHT - 80
    SLOPE_GRADIENT = 0.3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        self.game_state = self.STATE_SETUP
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.jumper_pos = pygame.Vector2(0, 0)
        self.jumper_vel = pygame.Vector2(0, 0)
        self.launch_angle = -15.0
        self.launch_power = 50.0
        self.total_distance = 0.0
        self.current_jump_distance = 0.0
        self.camera_x = 0.0
        self.obstacles = []
        self.particles = []
        self.mountains = []
        self.post_jump_timer = 0
        self.last_reward = 0.0

        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.total_distance = 0.0
        self.game_over = False
        self.last_reward = 0.0
        self._reset_jump()
        self._generate_mountains()
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0.0
        terminated = False

        dt = self.clock.tick(self.FPS) / 1000.0

        if self.post_jump_timer > 0:
            self.post_jump_timer -= dt
            if self.post_jump_timer <= 0:
                self._reset_jump()

        elif self.game_state == self.STATE_SETUP:
            self._handle_setup_action(movement, space_held)
        
        elif self.game_state == self.STATE_FLIGHT:
            self._update_flight(dt)
            
            # Check for landing or crash
            slope_y = self._get_slope_y(self.jumper_pos.x)
            if self.jumper_pos.y >= slope_y:
                if self._check_collision():
                    # sfx: crash_sound.wav
                    self.game_state = self.STATE_CRASHED
                    self.post_jump_timer = 2.0
                    reward = -10.0
                    terminated = True
                else:
                    # sfx: land_soft.wav
                    self.game_state = self.STATE_LANDED
                    self.post_jump_timer = 2.0
                    reward = self.current_jump_distance * 0.1
                    self.total_distance += self.current_jump_distance
                    self.score = self.total_distance
                    self._create_landing_particles(20)
        
        self._update_particles(dt)

        if not terminated and self.total_distance >= 100:
            reward += 100.0
            terminated = True
        
        if not terminated and self.steps >= 1000:
            terminated = True
        
        self.game_over = terminated
        self.steps += 1
        self.last_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_setup_action(self, movement, space_held):
        if movement == 1: # Up
            self.launch_angle = min(self.MAX_ANGLE, self.launch_angle + 0.5)
        elif movement == 2: # Down
            self.launch_angle = max(self.MIN_ANGLE, self.launch_angle - 0.5)
        elif movement == 3: # Left
            self.launch_power = max(self.MIN_POWER, self.launch_power - 1.0)
        elif movement == 4: # Right
            self.launch_power = min(self.MAX_POWER, self.launch_power + 1.0)

        if space_held:
            # sfx: launch_whoosh.wav
            self.game_state = self.STATE_FLIGHT
            angle_rad = math.radians(self.launch_angle)
            initial_velocity = self.launch_power * self.POWER_TO_VEL_SCALE
            self.jumper_vel = pygame.Vector2(
                initial_velocity * math.cos(angle_rad),
                initial_velocity * math.sin(angle_rad)
            )

    def _update_flight(self, dt):
        self.jumper_vel.y += self.GRAVITY * dt
        self.jumper_pos += self.jumper_vel * dt
        self.current_jump_distance = max(0, (self.jumper_pos.x - self.RAMP_LENGTH) / 10) # in meters
        self.camera_x = self.jumper_pos.x - self.WIDTH / 3

    def _reset_jump(self):
        self.game_state = self.STATE_SETUP
        self.jumper_pos = pygame.Vector2(self.RAMP_LENGTH, self.RAMP_HEIGHT)
        self.jumper_vel = pygame.Vector2(0, 0)
        self.current_jump_distance = 0.0
        self.camera_x = 0
        if self.game_state != self.STATE_CRASHED:
            self._generate_obstacles()

    def _get_slope_y(self, x):
        return self.SLOPE_START_Y + self.SLOPE_GRADIENT * max(0, x - self.RAMP_LENGTH)

    def _check_collision(self):
        jumper_rect = pygame.Rect(self.jumper_pos.x - 5, self.jumper_pos.y - 15, 10, 15)
        for obs in self.obstacles:
            if jumper_rect.colliderect(obs):
                return True
        return False

    def _generate_obstacles(self):
        self.obstacles.clear()
        # Density increases by 0.004 per meter jumped after 20m
        # Let's convert this to probability per 100 pixels
        base_density = 0.004 * 10 # per meter -> per 100 pixels
        density = base_density * max(0, self.total_distance - 20)
        
        for x in range(self.RAMP_LENGTH + 200, 5000, self.np_random.integers(80, 200)):
            if self.np_random.random() < density:
                slope_y = self._get_slope_y(x)
                tree_height = self.np_random.integers(30, 70)
                tree_width = 15
                obstacle = pygame.Rect(x - tree_width / 2, slope_y - tree_height, tree_width, tree_height)
                self.obstacles.append(obstacle)

    def _generate_mountains(self):
        self.mountains.clear()
        x = -self.WIDTH
        while x < self.WIDTH * 3:
            width = self.np_random.integers(200, 400)
            height = self.np_random.integers(100, 250)
            self.mountains.append((x, width, height))
            x += width * self.np_random.uniform(0.6, 0.9)

    def _create_landing_particles(self, count):
        slope_y = self._get_slope_y(self.jumper_pos.x)
        for _ in range(count):
            vel = pygame.Vector2(self.np_random.uniform(-150, 50), self.np_random.uniform(-200, -50))
            pos = pygame.Vector2(self.jumper_pos.x, slope_y)
            size = self.np_random.uniform(2, 5)
            lifetime = self.np_random.uniform(0.5, 1.5)
            self.particles.append([pos, vel, size, lifetime])

    def _update_particles(self, dt):
        for p in self.particles:
            p[0] += p[1] * dt # pos
            p[1].y += self.GRAVITY * 0.5 * dt # vel
            p[3] -= dt # lifetime
        self.particles = [p for p in self.particles if p[3] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Parallax mountains
        for x, w, h in self.mountains:
            points = [
                (x - self.camera_x * 0.2, self.HEIGHT),
                (x - self.camera_x * 0.2 + w/2, self.HEIGHT - h),
                (x - self.camera_x * 0.2 + w, self.HEIGHT)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_MOUNTAIN, points)
            snow_points = [
                (points[1][0] - w*0.1, points[1][1] + h*0.2),
                points[1],
                (points[1][0] + w*0.1, points[1][1] + h*0.2)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_MOUNTAIN_SNOW, snow_points)

        # Slope
        slope_points = []
        for x_offset in range(-10, self.WIDTH + 10, 20):
            world_x = self.camera_x + x_offset
            slope_points.append((x_offset, self._get_slope_y(world_x)))
        
        if len(slope_points) > 1:
            render_points = [(p[0], p[1]) for p in slope_points]
            pygame.draw.polygon(self.screen, self.COLOR_SNOW, 
                [(0, self.HEIGHT), (self.WIDTH, self.HEIGHT)] + render_points[::-1])
        
        # Ramp
        ramp_end_x_on_screen = self.RAMP_LENGTH - self.camera_x
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (0, self.RAMP_HEIGHT + 20), (ramp_end_x_on_screen, self.RAMP_HEIGHT), 5)
        pygame.draw.polygon(self.screen, pygame.Color(150,150,150), [(0, self.HEIGHT), (0, self.RAMP_HEIGHT+20), (ramp_end_x_on_screen, self.RAMP_HEIGHT), (ramp_end_x_on_screen, self.HEIGHT)])

        # Obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(-self.camera_x, 0)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
                leaves_rect = pygame.Rect(screen_rect.x - 10, screen_rect.y - 20, 35, 30)
                pygame.draw.ellipse(self.screen, self.COLOR_OBSTACLE_LEAVES, leaves_rect)

        # Jumper
        jumper_screen_pos = self.jumper_pos - pygame.Vector2(self.camera_x, 0)
        if 0 < jumper_screen_pos.x < self.WIDTH:
            p = (int(jumper_screen_pos.x), int(jumper_screen_pos.y))
            pygame.draw.circle(self.screen, self.COLOR_JUMPER, p, 8)
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1], 8, self.COLOR_JUMPER)
            
            # Simple ski
            angle = self.jumper_vel.angle_to(pygame.Vector2(1,0)) if self.game_state == self.STATE_FLIGHT else -10
            x1 = p[0] + 15 * math.cos(math.radians(angle))
            y1 = p[1] - 15 * math.sin(math.radians(angle))
            x2 = p[0] - 15 * math.cos(math.radians(angle))
            y2 = p[1] + 15 * math.sin(math.radians(angle))
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (x1, y1), (x2, y2), 3)

        # Particles
        for pos, vel, size, lifetime in self.particles:
            screen_pos = pos - pygame.Vector2(self.camera_x, 0)
            alpha = max(0, min(255, int(255 * (lifetime / 1.5))))
            color = self.COLOR_SNOW.lerp(self.COLOR_BG, 1 - alpha/255)
            pygame.draw.circle(self.screen, color, (int(screen_pos.x), int(screen_pos.y)), int(size))

    def _render_ui(self):
        # --- Info Panel ---
        info_surface = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
        info_surface.fill((255, 255, 255, 100))
        
        # Total Distance
        dist_text = self.font_medium.render(f"Total: {self.total_distance:.1f}m / 100m", True, self.COLOR_UI_TEXT)
        info_surface.blit(dist_text, (10, 5))
        
        # Current Jump
        jump_text = self.font_medium.render(f"Jump: {self.current_jump_distance:.1f}m", True, self.COLOR_UI_TEXT)
        info_surface.blit(jump_text, (10, 30))
        
        # Last Reward
        reward_color = pygame.Color("green") if self.last_reward > 0 else (pygame.Color("red") if self.last_reward < 0 else self.COLOR_UI_TEXT)
        reward_text = self.font_medium.render(f"Reward: {self.last_reward:+.1f}", True, reward_color)
        info_surface.blit(reward_text, (self.WIDTH - reward_text.get_width() - 10, 5))

        # Steps
        steps_text = self.font_medium.render(f"Step: {self.steps}", True, self.COLOR_UI_TEXT)
        info_surface.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 30))

        self.screen.blit(info_surface, (0, 0))

        # --- Setup UI ---
        if self.game_state == self.STATE_SETUP:
            # Power Bar
            power_ratio = (self.launch_power - self.MIN_POWER) / (self.MAX_POWER - self.MIN_POWER)
            bar_width = 200
            bar_height = 20
            fill_width = int(bar_width * power_ratio)
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (self.WIDTH / 2 - bar_width / 2, self.HEIGHT - 40, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR, (self.WIDTH / 2 - bar_width / 2, self.HEIGHT - 40, fill_width, bar_height))
            power_text = self.font_small.render("POWER", True, self.COLOR_UI_TEXT)
            self.screen.blit(power_text, (self.WIDTH/2 - power_text.get_width()/2, self.HEIGHT - 60))

            # Angle Indicator
            jumper_screen_pos = self.jumper_pos - pygame.Vector2(self.camera_x, 0)
            angle_rad = math.radians(self.launch_angle)
            end_pos = (jumper_screen_pos.x + 50 * math.cos(angle_rad), jumper_screen_pos.y + 50 * math.sin(angle_rad))
            pygame.draw.line(self.screen, self.COLOR_ANGLE_INDICATOR, (jumper_screen_pos.x, jumper_screen_pos.y), end_pos, 3)
            angle_text = self.font_small.render(f"{-self.launch_angle:.1f}°", True, self.COLOR_UI_TEXT)
            self.screen.blit(angle_text, (end_pos[0] + 5, end_pos[1] - 5))

        # --- Post-Jump Messages ---
        if self.game_state == self.STATE_LANDED:
            msg = f"LANDED! {self.current_jump_distance:.1f}m"
            text = self.font_big.render(msg, True, pygame.Color("darkgreen"))
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2))
        elif self.game_state == self.STATE_CRASHED:
            text = self.font_big.render("CRASHED!", True, pygame.Color("darkred"))
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 - text.get_height()/2))
        
        if self.total_distance >= 100:
            text = self.font_big.render("VICTORY!", True, pygame.Color("gold"))
            self.screen.blit(text, (self.WIDTH/2 - text.get_width()/2, self.HEIGHT/2 + 40))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "total_distance": self.total_distance,
            "current_jump_distance": self.current_jump_distance,
            "game_state": self.game_state
        }
        
    def close(self):
        pygame.quit()

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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Total Distance: {info['total_distance']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # --- Pygame Event Handling & Rendering ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # The environment's observation is the rendered screen
        # We just need to get it to the display
        display_surface = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_surface.blit(draw_surface, (0, 0))
        pygame.display.flip()

    env.close()