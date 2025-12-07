import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Pressure Wave', a sci-fi arcade survival game.
    The player pilots a momentum-based drone, using pressure waves to destroy
    obstacles while escaping a rising pressure field.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot a momentum-based drone, using pressure waves to destroy obstacles "
        "while escaping a rising deadly pressure field."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to accelerate the drone. "
        "Press space to release a pressure wave that destroys obstacles."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 20  # Steps per second
    MAX_STEPS = 120 * TARGET_FPS  # 120 seconds

    # Colors
    COLOR_BG_SAFE = (10, 20, 40)
    COLOR_BG_WARN = (80, 20, 40)
    COLOR_DRONE = (255, 255, 255)
    COLOR_DRONE_GLOW = (200, 200, 255)
    COLOR_OBSTACLE = (0, 0, 0)
    COLOR_WAVE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_COOLDOWN_BAR_BG = (50, 50, 80)
    COLOR_COOLDOWN_BAR_FG = (120, 120, 255)

    # Drone Physics
    DRONE_ACCELERATION = 0.7
    DRONE_DRAG = 0.92
    DRONE_MAX_SPEED = 8.0
    DRONE_SIZE = 12

    # Pressure Wave
    WAVE_COOLDOWN_STEPS = 60  # 3 seconds
    WAVE_MAX_RADIUS = 100
    WAVE_EXPANSION_SPEED = 4

    # Pressure Field
    PRESSURE_INITIAL_RATE = 1.0 / 250.0  # units per step (Adjusted for stability)
    PRESSURE_ACCELERATED_RATE = 1.5 / 100.0
    PRESSURE_ACCELERATION_STEP = 1200 # 60 seconds

    # Obstacles
    OBSTACLE_INITIAL_SPAWN_RATE = 0.05 # probability per step
    OBSTACLE_SPAWN_RATE_INCREASE = 0.001 / 100.0
    OBSTACLE_SPEED = 2.5
    OBSTACLE_MIN_SIZE = 20
    OBSTACLE_MAX_SIZE = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.render_mode = render_mode
        self._initialize_state()


    def _initialize_state(self):
        """Initializes all game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.drone_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT * 0.75)
        self.drone_vel = pygame.Vector2(0, 0)

        self.pressure_y = self.SCREEN_HEIGHT
        self.pressure_rate = self.PRESSURE_INITIAL_RATE

        self.obstacles = []
        self.obstacle_spawn_rate = self.OBSTACLE_INITIAL_SPAWN_RATE

        self.pressure_waves = []
        self.wave_cooldown = 0
        self.prev_space_held = False

        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        # --- Update Game Logic ---
        self._handle_input(movement, space_held)
        self._update_drone()
        self._update_pressure_field()
        self._update_obstacles()
        self._update_waves()
        self._update_particles()
        
        # --- Handle Collisions and Events ---
        obstacles_destroyed = self._handle_collisions()
        
        # --- Calculate Reward ---
        reward += 0.1  # Survival reward
        reward += obstacles_destroyed * 5.0

        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.win:
            reward = -100.0 # Death penalty
        elif truncated and self.win:
            reward = 100.0 # Win bonus
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.drone_vel.y -= self.DRONE_ACCELERATION # Up
        if movement == 2: self.drone_vel.y += self.DRONE_ACCELERATION # Down
        if movement == 3: self.drone_vel.x -= self.DRONE_ACCELERATION # Left
        if movement == 4: self.drone_vel.x += self.DRONE_ACCELERATION # Right
        
        # Pressure Wave (on press, not hold)
        if space_held and not self.prev_space_held and self.wave_cooldown <= 0:
            self.wave_cooldown = self.WAVE_COOLDOWN_STEPS
            self.pressure_waves.append({
                "pos": self.drone_pos.copy(),
                "radius": 0,
                "max_radius": self.WAVE_MAX_RADIUS
            })
        self.prev_space_held = space_held

    def _update_drone(self):
        self.drone_vel *= self.DRONE_DRAG
        if self.drone_vel.length() > self.DRONE_MAX_SPEED:
            self.drone_vel.scale_to_length(self.DRONE_MAX_SPEED)
        self.drone_pos += self.drone_vel

        # Screen boundaries
        self.drone_pos.x = np.clip(self.drone_pos.x, self.DRONE_SIZE, self.SCREEN_WIDTH - self.DRONE_SIZE)
        self.drone_pos.y = np.clip(self.drone_pos.y, self.DRONE_SIZE, self.SCREEN_HEIGHT - self.DRONE_SIZE)
        
    def _update_pressure_field(self):
        if self.steps == self.PRESSURE_ACCELERATION_STEP:
            self.pressure_rate = self.PRESSURE_ACCELERATED_RATE
        self.pressure_y -= self.pressure_rate * self.SCREEN_HEIGHT
        self.pressure_y = max(0, self.pressure_y)

    def _update_obstacles(self):
        # Spawn new obstacles
        if self.np_random.random() < self.obstacle_spawn_rate:
            width = self.np_random.integers(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
            height = self.np_random.integers(self.OBSTACLE_MIN_SIZE, self.OBSTACLE_MAX_SIZE)
            x = self.np_random.uniform(0, self.SCREEN_WIDTH - width)
            self.obstacles.append(pygame.Rect(int(x), -height, width, height))

        # Update existing obstacles
        for obstacle in self.obstacles:
            obstacle.y += self.OBSTACLE_SPEED
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs.top < self.SCREEN_HEIGHT]
        
        # Increase spawn rate over time
        self.obstacle_spawn_rate += self.OBSTACLE_SPAWN_RATE_INCREASE
        
    def _update_waves(self):
        for wave in self.pressure_waves:
            wave["radius"] += self.WAVE_EXPANSION_SPEED
        self.pressure_waves = [w for w in self.pressure_waves if w["radius"] < w["max_radius"]]
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        drone_rect = pygame.Rect(
            self.drone_pos.x - self.DRONE_SIZE / 2,
            self.drone_pos.y - self.DRONE_SIZE / 2,
            self.DRONE_SIZE, self.DRONE_SIZE
        )
        
        # Drone vs Obstacles
        for obstacle in self.obstacles:
            if drone_rect.colliderect(obstacle):
                self.game_over = True
                self._create_explosion(self.drone_pos, self.COLOR_DRONE, 50)
                return 0

        # Waves vs Obstacles
        obstacles_destroyed_this_step = 0
        remaining_obstacles = []
        for obstacle in self.obstacles:
            destroyed = False
            for wave in self.pressure_waves:
                # Check if any corner of the obstacle is inside the wave circle
                corners = [obstacle.topleft, obstacle.topright, obstacle.bottomleft, obstacle.bottomright]
                if any(pygame.Vector2(c).distance_to(wave["pos"]) < wave["radius"] for c in corners):
                    obstacles_destroyed_this_step += 1
                    destroyed = True
                    self._create_explosion(pygame.Vector2(obstacle.center), self.COLOR_OBSTACLE, 20)
                    break 
            if not destroyed:
                remaining_obstacles.append(obstacle)
        self.obstacles = remaining_obstacles
        return obstacles_destroyed_this_step

    def _check_termination(self):
        if self.game_over:
            return True
        
        # Drone vs Pressure Field
        if self.drone_pos.y > self.pressure_y:
            self.game_over = True
            self._create_explosion(self.drone_pos, self.COLOR_DRONE, 30)
            return True
            
        # Win condition (now handled by truncation)
        if self.steps >= self.MAX_STEPS:
            self.win = True
            self.game_over = True # End game on win
            return False # Not a failure termination
            
        return False
        
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "drone_pos": tuple(self.drone_pos),
            "pressure_y": self.pressure_y,
            "wave_cooldown": self.wave_cooldown,
        }
        
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _render_all(self):
        self._render_background()
        self._render_obstacles()
        self._render_waves()
        self._render_particles()
        if not self.game_over or self.win:
            self._render_drone()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG_SAFE)
        warn_zone_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, int(self.pressure_y))
        pygame.draw.rect(self.screen, self.COLOR_BG_WARN, warn_zone_rect)


    def _render_obstacles(self):
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)

    def _render_waves(self):
        for wave in self.pressure_waves:
            alpha = int(255 * (1 - wave["radius"] / wave["max_radius"]))
            if alpha > 0:
                # Using gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, int(wave["pos"].x), int(wave["pos"].y), int(wave["radius"]), (*self.COLOR_WAVE, alpha))
                pygame.gfxdraw.aacircle(self.screen, int(wave["pos"].x), int(wave["pos"].y), int(wave["radius"]-1), (*self.COLOR_WAVE, alpha))

    def _render_particles(self):
        for p in self.particles:
            size = int(p['life'] / 4)
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), size, size))

    def _render_drone(self):
        pos_int = (int(self.drone_pos.x), int(self.drone_pos.y))
        size = self.DRONE_SIZE
        
        # Glow effect
        glow_radius = int(size * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_DRONE_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Core
        drone_rect = pygame.Rect(pos_int[0] - size / 2, pos_int[1] - size / 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_DRONE, drone_rect)

    def _render_ui(self):
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.TARGET_FPS
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Pressure Level
        pressure_percent = 100 * (self.SCREEN_HEIGHT - self.pressure_y) / self.SCREEN_HEIGHT
        pressure_text = self.font_ui.render(f"PRESSURE: {max(0, pressure_percent):.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(pressure_text, (self.SCREEN_WIDTH - pressure_text.get_width() - 10, 10))
        
        # Cooldown Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        
        cooldown_ratio = self.wave_cooldown / self.WAVE_COOLDOWN_STEPS
        fill_width = bar_width * (1 - cooldown_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_COOLDOWN_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_COOLDOWN_BAR_FG, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win:
            text = self.font_game_over.render("SURVIVED", True, (150, 255, 150))
        else:
            text = self.font_game_over.render("DESTROYED", True, (255, 150, 150))
            
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # This block is for human play and visualization, not used by the tests
    # It requires a display, so we unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pressure Wave")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop for human play
    running = True
    while running:
        # --- Human Input ---
        movement_action = 0 # none
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it by transposing it back for pygame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        clock.tick(GameEnv.TARGET_FPS)

    env.close()