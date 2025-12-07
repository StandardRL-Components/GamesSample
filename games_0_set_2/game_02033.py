
# Generated: 2025-08-28T03:28:05.628050
# Source Brief: brief_02033.md
# Brief Index: 2033

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space for a normal jump or shift for a high jump. "
        "Avoid the red obstacles and reach the spaceship!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a vibrant alien planet by jumping over procedurally generated obstacles "
        "to reach your spaceship before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_title = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 18)
            self.font_title = pygame.font.SysFont("arial", 48)

        # Game Constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.WORLD_WIDTH = 6000
        self.GROUND_Y = 350
        self.MAX_STEPS = 1200  # Approx 40 seconds at 30fps

        # Colors
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_OBSTACLE = (255, 50, 50)
        self.COLOR_GROUND = (60, 140, 90)
        self.COLOR_SPACESHIP = (255, 220, 0)
        self.COLOR_UI = (230, 230, 230)
        self.PARALLAX_COLORS = [
            (25, 20, 50),
            (35, 30, 70),
            (50, 40, 90),
        ]
        
        # State variables (initialized in reset)
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.camera_x = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.base_obstacle_speed = None
        self.last_space_held = False
        self.last_shift_held = False
        self.np_random = None
        
        # Initialize state
        self.reset()
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Player state
        self.player_pos = np.array([100.0, self.GROUND_Y])
        self.player_vel_y = 0.0
        self.on_ground = True

        # World state
        self.camera_x = 0
        self.base_obstacle_speed = 3.0
        self._generate_obstacles()

        # System state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()

    def _generate_obstacles(self):
        self.obstacles = []
        current_x = 800
        while current_x < self.WORLD_WIDTH - 800:
            # Enforce minimum gap to prevent unjumpable sections
            gap = self.np_random.integers(250, 450)
            current_x += gap
            obstacle_height = self.np_random.integers(20, 70)
            obstacle_width = self.np_random.integers(30, 90)
            self.obstacles.append({
                "rect": pygame.Rect(current_x, self.GROUND_Y - obstacle_height, obstacle_width, obstacle_height),
                "cleared": False
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Survival reward

        self._handle_input(space_held, shift_held)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        self._update_camera()
        
        collision, new_reward = self._check_events()
        reward += new_reward
        
        terminated = False
        if collision:
            self.game_over = True
            reward = -10  # Collision penalty overrides other rewards
            # sfx: explosion_sound
            terminated = True
        elif self.player_pos[0] >= self.WORLD_WIDTH - 200:
            self.game_over = True
            self.game_won = True
            reward = 100  # Win bonus
            # sfx: win_sound
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            reward = -1 # Small penalty for timeout
            # sfx: timeout_sound
            terminated = True
        
        self.steps += 1
        self.score += reward
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, space_held, shift_held):
        # Trigger jump only on the first frame the key is pressed
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        if self.on_ground:
            if shift_pressed:
                self.player_vel_y = -13.5  # High jump
                self.on_ground = False
                self._create_particles(self.player_pos, 20, (200, 200, 255), "jump")
                # sfx: high_jump_sound
            elif space_pressed:
                self.player_vel_y = -10.5  # Normal jump
                self.on_ground = False
                self._create_particles(self.player_pos, 15, (200, 200, 255), "jump")
                # sfx: normal_jump_sound

    def _update_player(self):
        self.player_pos[0] += 4  # Constant forward movement

        if not self.on_ground:
            self.player_vel_y += 0.5  # Gravity
            self.player_pos[1] += self.player_vel_y

        if self.player_pos[1] >= self.GROUND_Y:
            if not self.on_ground: # Just landed
                self._create_particles((self.player_pos[0], self.GROUND_Y), 10, (180, 180, 180), "land")
                # sfx: land_sound
            self.player_pos[1] = self.GROUND_Y
            self.player_vel_y = 0
            self.on_ground = True

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_WIDTH)

    def _update_obstacles(self):
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_obstacle_speed = min(8.0, self.base_obstacle_speed + 0.05)
        for obs in self.obstacles:
            obs['rect'].x -= self.base_obstacle_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['vel'][1] += p['gravity']
            p['pos'] += p['vel']
            p['life'] -= 1

    def _update_camera(self):
        target_x = self.player_pos[0] - self.SCREEN_WIDTH / 3
        self.camera_x += (target_x - self.camera_x) * 0.1 # Smooth camera
        self.camera_x = np.clip(self.camera_x, 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)

    def _check_events(self):
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 20, 16, 20)
        reward = 0
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                self._create_particles(player_rect.center, 50, self.COLOR_OBSTACLE, "explosion")
                return True, 0  # Collision detected

            if not obs['cleared'] and obs['rect'].right < player_rect.left:
                obs['cleared'] = True
                reward += 1.0  # Reward for clearing an obstacle
                # sfx: clear_obstacle_sound
        return False, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_parallax()

        ground_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        
        ship_x = self.WORLD_WIDTH - 200 - self.camera_x
        if -100 < ship_x < self.SCREEN_WIDTH + 100:
            self._render_spaceship(ship_x)

        for obs in self.obstacles:
            obs_screen_rect = obs['rect'].move(-self.camera_x, 0)
            if obs_screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs_screen_rect, border_radius=3)
                pygame.draw.rect(self.screen, (255, 150, 150), obs_screen_rect, 2, border_radius=3)

        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                color = (*p['color'], alpha)
                radius = int(p['size'] * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        player_screen_x = int(self.player_pos[0] - self.camera_x)
        player_screen_y = int(self.player_pos[1])
        player_rect = pygame.Rect(player_screen_x - 8, player_screen_y - 20, 16, 20)
        
        glow_radius = 18
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_rect.centerx - glow_radius, player_rect.centery - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_parallax(self):
        for i, color in enumerate(self.PARALLAX_COLORS):
            speed = 0.1 + 0.2 * i
            offset = - (self.camera_x * speed)
            
            # Use a fixed seed for each layer for consistent "flora"
            layer_seed = 42 + i
            self.np_random_render = np.random.default_rng(layer_seed)

            for j in range(50): # Number of glowing flora
                star_world_x = self.np_random_render.integers(0, self.WORLD_WIDTH)
                star_y = self.np_random_render.integers(0, self.GROUND_Y - 10)
                star_size = self.np_random_render.integers(1, 4)
                star_screen_x = int(star_world_x + offset)
                
                if -star_size < star_screen_x < self.SCREEN_WIDTH + star_size:
                    alpha = 50 + i*30
                    pygame.gfxdraw.filled_circle(self.screen, star_screen_x, star_y, star_size, (*(200,200,255), alpha))

    def _render_spaceship(self, x):
        x, y = int(x), int(self.GROUND_Y)
        ship_color = self.COLOR_SPACESHIP
        trim_color = (200, 180, 0)
        window_color = (150, 200, 255)
        
        # Main body
        pygame.draw.polygon(self.screen, ship_color, [(x, y - 60), (x + 80, y - 40), (x + 80, y), (x, y-10)])
        # Outline
        pygame.draw.polygon(self.screen, trim_color, [(x, y - 60), (x + 80, y - 40), (x + 80, y), (x, y-10)], 3)
        # Cockpit window
        pygame.gfxdraw.filled_circle(self.screen, x + 55, y - 42, 12, window_color)
        pygame.gfxdraw.aacircle(self.screen, x + 55, y - 42, 12, (200, 220, 255))
        # Fins
        pygame.draw.polygon(self.screen, trim_color, [(x, y-10), (x - 20, y), (x, y)])
        pygame.draw.polygon(self.screen, trim_color, [(x+80, y), (x + 100, y), (x+80, y-15)])

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(text_surf, (10, 10))

        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_surf, score_rect)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "MISSION COMPLETE" if self.game_won else "MISSION FAILED"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        
        text_surf = self.font_title.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, count, color, p_type="jump"):
        for _ in range(count):
            if p_type == "jump":
                angle = self.np_random.uniform(-math.pi * 0.8, -math.pi * 0.2)
                speed = self.np_random.uniform(1, 4)
                gravity = 0.1
            elif p_type == "land":
                angle = self.np_random.uniform(math.pi * 0.9, math.pi * 2.1)
                speed = self.np_random.uniform(0.5, 2.5)
                gravity = 0.05
            else: # explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 8)
                gravity = 0.2

            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.uniform(1, 4),
                'gravity': gravity
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos[0],
            "player_y": self.player_pos[1],
            "won": self.game_won,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")