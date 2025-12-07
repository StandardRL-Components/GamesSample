import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    # --- METADATA ---
    game_description = (
        "Climb a skyscraper while dodging falling barrels and battling strong winds. "
        "Manage your grip and brace yourself to reach the top."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Hold space to grip the wall and slow your fall. "
        "Hold shift to brace against the wind."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BUILDING_HEIGHT = 4000
    GOAL_Y = 100
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (25, 30, 45)
    COLOR_BUILDING = (40, 45, 60)
    COLOR_WINDOW = (60, 65, 80)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 80, 80)
    COLOR_PLAYER_BRACE = (80, 200, 255)
    COLOR_BARREL = (200, 100, 30)
    COLOR_BARREL_GLOW = (200, 100, 30)
    COLOR_WIND = (180, 200, 255)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_GRIP_BAR = (80, 255, 80)
    COLOR_BRACE_BAR = (80, 200, 255)
    COLOR_BAR_BG = (70, 70, 70)

    # Player Physics
    PLAYER_SPEED = 1.8
    GRAVITY = 0.18
    FRICTION = 0.90
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 30

    # Grip Mechanics
    MAX_GRIP = 100
    GRIP_DRAIN = 1.5
    GRIP_RECHARGE = 0.5
    GRIP_FORCE_MULTIPLIER = 0.7

    # Brace Mechanics
    MAX_BRACE = 100
    BRACE_DRAIN = 2.5
    BRACE_RECHARGE = 0.4
    BRACE_EFFECTIVENESS = 0.8

    # Wind
    WIND_CHANGE_INTERVAL = 120 # steps
    INITIAL_MAX_WIND_SPEED = 1.0

    # Barrels
    INITIAL_BARREL_SPAWN_INTERVAL = 90 # steps (3 seconds at 30fps)
    BARREL_RADIUS = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = None
        self.player_vel = None
        self.grip = None
        self.brace = None
        self.is_bracing = False
        
        self.wind_speed = None
        self.wind_direction = None
        self.wind_timer = None
        self.max_wind_speed = None
        
        self.barrels = None
        self.barrel_spawn_timer = None
        self.current_barrel_spawn_interval = None
        self.barrel_id_counter = None
        self.dodged_barrels = None
        
        self.wind_particles = None
        self.camera_y = None
        self.last_player_y = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.BUILDING_HEIGHT - 100.0])
        self.player_vel = np.array([0.0, 0.0])
        self.last_player_y = self.player_pos[1]
        
        self.grip = self.MAX_GRIP
        self.brace = self.MAX_BRACE
        self.is_bracing = False

        self.wind_speed = 0.0
        self.wind_direction = 1
        self.wind_timer = self.WIND_CHANGE_INTERVAL
        self.max_wind_speed = self.INITIAL_MAX_WIND_SPEED

        self.barrels = []
        self.barrel_spawn_timer = 0
        self.current_barrel_spawn_interval = self.INITIAL_BARREL_SPAWN_INTERVAL
        self.barrel_id_counter = 0
        self.dodged_barrels = set()

        self.wind_particles = []
        for _ in range(100):
            self._spawn_particle(random_y=True)

        self.camera_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.7

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- UPDATE GAME LOGIC ---
        self._update_wind()
        self._update_difficulty()
        
        self._update_player(action)
        barrel_reward = self._update_barrels()
        reward += barrel_reward
        
        self._update_particles()
        self._update_camera()

        # --- REWARD & TERMINATION ---
        y_change = self.last_player_y - self.player_pos[1]
        reward += y_change * 0.02 # Small reward for climbing
        self.last_player_y = self.player_pos[1]
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over: # Reached goal or fell
            if self.player_pos[1] <= self.GOAL_Y:
                reward += 100 # Reached the top
            else: # Fell off
                reward -= 100 
        
        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_wind(self):
        self.wind_timer -= 1
        if self.wind_timer <= 0:
            self.wind_timer = self.WIND_CHANGE_INTERVAL + self.np_random.integers(-30, 30)
            target_speed = self.np_random.uniform(0, self.max_wind_speed)
            self.wind_speed = target_speed
            if self.np_random.random() < 0.5:
                self.wind_direction *= -1
    
    def _update_difficulty(self):
        # Wind increases every ~16s
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_wind_speed += 0.05
        # Barrels spawn faster every ~33s
        if self.steps > 0 and self.steps % 1000 == 0:
            self.current_barrel_spawn_interval = max(30, self.current_barrel_spawn_interval - 0.1 * 30) # 0.1s faster

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Apply Action ---
        # Movement
        if movement == 1: self.player_vel[1] -= self.PLAYER_SPEED # Up
        elif movement == 2: self.player_vel[1] += self.PLAYER_SPEED * 0.5 # Down
        elif movement == 3: self.player_vel[0] -= self.PLAYER_SPEED # Left
        elif movement == 4: self.player_vel[0] += self.PLAYER_SPEED # Right

        # Grip
        is_gripping = space_held and self.grip > 0
        if is_gripping:
            self.grip = max(0, self.grip - self.GRIP_DRAIN)
            self.player_vel[1] *= self.GRIP_FORCE_MULTIPLIER # Slows fall
        else:
            self.grip = min(self.MAX_GRIP, self.grip + self.GRIP_RECHARGE)

        # Brace
        self.is_bracing = shift_held and self.brace > 0
        if self.is_bracing:
            self.brace = max(0, self.brace - self.BRACE_DRAIN)
            wind_force = self.wind_speed * self.wind_direction * (1 - self.BRACE_EFFECTIVENESS)
        else:
            self.brace = min(self.MAX_BRACE, self.brace + self.BRACE_RECHARGE)
            wind_force = self.wind_speed * self.wind_direction

        # --- Apply Physics ---
        # FIX: Player holds position by default (no vertical input and not gripping)
        if movement == 0 and not is_gripping:
            self.player_vel[1] = 0 # Annihilate vertical velocity
        else:
            self.player_vel[1] += self.GRAVITY

        self.player_vel[0] += wind_force
        self.player_vel *= self.FRICTION
        
        self.player_pos += self.player_vel

        # --- Boundary checks ---
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)

    def _update_barrels(self):
        reward = 0
        self.barrel_spawn_timer -= 1
        if self.barrel_spawn_timer <= 0:
            self.barrel_spawn_timer = self.current_barrel_spawn_interval
            spawn_x = self.np_random.uniform(self.BARREL_RADIUS, self.SCREEN_WIDTH - self.BARREL_RADIUS)
            spawn_y = self.camera_y - self.BARREL_RADIUS
            new_barrel = {
                "id": self.barrel_id_counter,
                "pos": np.array([spawn_x, spawn_y]),
                "vel": np.array([0.0, 0.0])
            }
            self.barrels.append(new_barrel)
            self.barrel_id_counter += 1

        kept_barrels = []
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_WIDTH/2, self.player_pos[1] - self.PLAYER_HEIGHT/2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

        for b in self.barrels:
            # Physics
            b['vel'][1] += self.GRAVITY * 0.8 # Barrels are lighter
            b['vel'][0] += self.wind_speed * self.wind_direction * 1.2 # More affected by wind
            b['vel'] *= 0.98 # Barrel friction
            b['pos'] += b['vel']

            # Collision check
            if player_rect.collidepoint(b['pos']):
                self.game_over = True
                reward -= 10
                continue

            # Reward for dodging
            if b['id'] not in self.dodged_barrels and b['pos'][1] > self.player_pos[1] + self.PLAYER_HEIGHT:
                reward += 1
                self.dodged_barrels.add(b['id'])

            # Keep if on screen
            if b['pos'][1] < self.camera_y + self.SCREEN_HEIGHT + 50:
                kept_barrels.append(b)
        
        self.barrels = kept_barrels
        return reward

    def _update_particles(self):
        for p in self.wind_particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        
        self.wind_particles = [p for p in self.wind_particles if p['life'] > 0]

        if self.wind_speed > 0.1:
            num_new_particles = int(self.wind_speed * 2)
            for _ in range(num_new_particles):
                self._spawn_particle()
    
    def _spawn_particle(self, random_y=False):
        if self.wind_direction > 0:
            x = -10
            vx = self.np_random.uniform(3, 6) * self.wind_speed
        else:
            x = self.SCREEN_WIDTH + 10
            vx = -self.np_random.uniform(3, 6) * self.wind_speed

        if random_y:
            y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
        else:
            y = self.np_random.uniform(self.camera_y, self.camera_y + self.SCREEN_HEIGHT)
            
        self.wind_particles.append({
            "pos": np.array([x, y]),
            "vel": np.array([vx, self.np_random.uniform(-0.1, 0.1)]),
            "life": self.np_random.integers(100, 200),
            "radius": self.np_random.uniform(1, 3) * (0.5 + self.wind_speed)
        })

    def _update_camera(self):
        target_cam_y = self.player_pos[1] - self.SCREEN_HEIGHT * 0.7
        self.camera_y = 0.9 * self.camera_y + 0.1 * target_cam_y

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player_pos[1] <= self.GOAL_Y:
            return True # Victory
        if self.player_pos[1] > self.BUILDING_HEIGHT - 50:
            return True # Fell off the bottom
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        height_climbed = max(0, self.BUILDING_HEIGHT - self.player_pos[1])
        return {
            "score": self.score,
            "steps": self.steps,
            "height": height_climbed,
            "grip": self.grip,
            "brace": self.brace,
        }

    # --- RENDERING ---
    def _render_game(self):
        self._render_building()
        self._render_wind_particles()
        self._render_barrels()
        self._render_player()

    def _render_building(self):
        pygame.draw.rect(self.screen, self.COLOR_BUILDING, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        window_size = 20
        window_spacing_y = 60
        window_spacing_x = 80
        start_y = - (self.camera_y % window_spacing_y)
        
        for row in range(int(self.SCREEN_HEIGHT / window_spacing_y) + 2):
            y = start_y + row * window_spacing_y
            world_y = y + self.camera_y
            offset = math.sin(world_y / 200) * 20
            
            for col in range(int(self.SCREEN_WIDTH / window_spacing_x) + 1):
                x = col * window_spacing_x + offset
                pygame.draw.rect(self.screen, self.COLOR_WINDOW, (int(x), int(y), window_size, window_size))

    def _render_wind_particles(self):
        for p in self.wind_particles:
            screen_pos = (int(p['pos'][0]), int(p['pos'][1] - self.camera_y))
            alpha = int(max(0, min(255, (p['life'] / 50) * 150 * (0.2 + self.wind_speed))))
            if alpha > 10:
                color = (*self.COLOR_WIND, alpha)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], int(p['radius']), color)
    
    def _render_barrels(self):
        for b in self.barrels:
            screen_pos = (int(b['pos'][0]), int(b['pos'][1] - self.camera_y))
            radius = int(self.BARREL_RADIUS)
            
            glow_radius = int(radius * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_BARREL_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            pygame.draw.circle(self.screen, self.COLOR_BARREL, screen_pos, radius)

    def _render_player(self):
        screen_pos = (int(self.player_pos[0]), int(self.player_pos[1] - self.camera_y))
        
        color = self.COLOR_PLAYER_BRACE if self.is_bracing else self.COLOR_PLAYER
        glow_color = self.COLOR_PLAYER_BRACE if self.is_bracing else self.COLOR_PLAYER_GLOW
        
        glow_radius = int(self.PLAYER_HEIGHT * 0.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        player_rect.center = screen_pos
        pygame.draw.rect(self.screen, color, player_rect, border_radius=4)
        
        eye_y = player_rect.top + 7
        eye_x_offset = 4
        pygame.draw.circle(self.screen, (255,255,255), (player_rect.centerx - eye_x_offset, eye_y), 2)
        pygame.draw.circle(self.screen, (255,255,255), (player_rect.centerx + eye_x_offset, eye_y), 2)


    def _render_ui(self):
        height_climbed = max(0, self.BUILDING_HEIGHT - self.player_pos[1])
        height_text = f"HEIGHT: {int(height_climbed)}m / {self.BUILDING_HEIGHT - self.GOAL_Y}m"
        text_surf = self.font_small.render(height_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        if self.wind_speed < 0.05:
            wind_str = "- 0km/h -"
        elif self.wind_direction > 0:
            wind_str = f"{'>>' * int(1 + self.wind_speed * 2)} {int(self.wind_speed * 20)}km/h"
        else:
            wind_str = f"{int(self.wind_speed * 20)}km/h {'<<' * int(1 + self.wind_speed * 2)}"
        wind_text = self.font_small.render(wind_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wind_text, (self.SCREEN_WIDTH - wind_text.get_width() - 10, 10))

        bar_width = 150
        bar_height = 15
        
        grip_ratio = self.grip / self.MAX_GRIP
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_GRIP_BAR, (10, self.SCREEN_HEIGHT - bar_height - 10, bar_width * grip_ratio, bar_height))
        grip_text = self.font_small.render("GRIP", True, self.COLOR_UI_TEXT)
        self.screen.blit(grip_text, (15, self.SCREEN_HEIGHT - bar_height - 30))

        brace_ratio = self.brace / self.MAX_BRACE
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (self.SCREEN_WIDTH - bar_width - 10, self.SCREEN_HEIGHT - bar_height - 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BRACE_BAR, (self.SCREEN_WIDTH - bar_width - 10, self.SCREEN_HEIGHT - bar_height - 10, bar_width * brace_ratio, bar_height))
        brace_text = self.font_small.render("BRACE", True, self.COLOR_UI_TEXT)
        self.screen.blit(brace_text, (self.SCREEN_WIDTH - brace_text.get_width() - 15, self.SCREEN_HEIGHT - bar_height - 30))

    def close(self):
        pygame.quit()